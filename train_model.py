import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Model
from keras.layers import (LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization, 
                          MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, 
                          Conv1D, Concatenate, Add, Activation, GRU, GlobalMaxPooling1D, Lambda)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import AdamW
from keras.regularizers import l1_l2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
import warnings
import os
import config
import data_processor

warnings.filterwarnings('ignore')

@tf.keras.utils.register_keras_serializable(package="Custom", name="PositionEmbedding")
class PositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_steps, max_dims, **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.max_steps = max_steps
        self.max_dims = max_dims
        
    def build(self, input_shape):
        self.pos_emb = self.add_weight(
            name="pos_emb", 
            shape=(self.max_steps, self.max_dims), 
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True
        )
        super(PositionEmbedding, self).build(input_shape)

    def call(self, inputs):
        return inputs + self.pos_emb

    def get_config(self):
        config = super(PositionEmbedding, self).get_config()
        config.update({
            "max_steps": self.max_steps, 
            "max_dims": self.max_dims
        })
        return config

def generate_improved_labels(df, horizon=2, method='adaptive', threshold=config.DATA_THRESHOLD):
    data = df.copy()
    future_return = (data['Close'].shift(-horizon) - data['Close']) / data['Close']
    if method == 'adaptive':
        window = 60
        volatility = data['Close'].pct_change().rolling(window).std()
        upper_threshold = volatility * 1.5
        lower_threshold = -volatility * 1.5
    else:
        upper_threshold = threshold
        lower_threshold = -threshold
    
    conditions = [future_return > upper_threshold, future_return < lower_threshold]
    choices = [2, 0] # 漲, 跌
    data['label'] = np.select(conditions, choices, default=1)
    data['future_return'] = future_return
    return data.dropna()

def create_enhanced_transformer_block(inputs, d_model=64, num_heads=4, dropout_rate=0.2):
    input_dim = int(inputs.shape[-1])
    if input_dim % num_heads != 0:
        d_model = ((input_dim // num_heads) + 1) * num_heads
        adjusted_inputs = Dense(d_model, activation='linear')(inputs)
    else:
        d_model = input_dim
        adjusted_inputs = inputs
    
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout_rate)(adjusted_inputs, adjusted_inputs)
    attention_output = Dropout(dropout_rate)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(Add()([adjusted_inputs, attention_output]))
    
    ffn = Dense(d_model * 2, activation='relu')(attention_output)
    ffn = Dense(d_model)(ffn)
    final_output = LayerNormalization(epsilon=1e-6)(Add()([attention_output, ffn]))
    return final_output

def create_ultimate_model(input_shape, num_classes=3):
    inputs = Input(shape=input_shape)
    
    # 1. Multi-scale CNN
    cnn_outputs = []
    for kernel_size in [3, 5, 7, 11]:
        cnn = Conv1D(64, kernel_size, padding='same', activation='relu')(inputs)
        cnn = BatchNormalization()(cnn)
        cnn = Dropout(0.2)(cnn)
        cnn_outputs.append(cnn)
    cnn_merged = Concatenate()(cnn_outputs)
    
    # 2. RNN Branch
    lstm_out = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2))(inputs)
    gru_out = Bidirectional(GRU(48, return_sequences=True, dropout=0.2))(inputs)
    
    # 3. Transformer Branch
    transformer_input = Dense(64, activation='linear')(inputs)
    # 加入位置編碼 (Positional Encoding)，讓 Transformer 具備序列順序感知能力
    transformer_input = PositionEmbedding(max_steps=30, max_dims=64)(transformer_input)
    transformer_out = create_enhanced_transformer_block(transformer_input, d_model=64)
    
    # 4. Feature Fusion
    feature_dim = 48
    cnn_adj = Dense(feature_dim)(cnn_merged)
    lstm_adj = Dense(feature_dim)(lstm_out)
    gru_adj = Dense(feature_dim)(gru_out)
    trans_adj = Dense(feature_dim)(transformer_out)
    
    # Simple Attention Weighting
    def apply_attn(inputs):
        f, w = inputs
        return f * tf.expand_dims(w, axis=1)

    merged = Concatenate()([cnn_adj, lstm_adj, gru_adj, trans_adj])
    
    x = MultiHeadAttention(num_heads=8, key_dim=24)(merged, merged)
    x = GlobalAveragePooling1D()(x)
    
    x = Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)

def create_sequences(data, labels, time_steps=30, overlap=0.6):
    step_size = max(1, int(time_steps * (1 - overlap)))
    X, y = [], []
    for i in range(0, len(data) - time_steps, step_size):
        X.append(data[i:i+time_steps])
        y.append(labels[i+time_steps])
    return np.array(X), np.array(y)

def main():
    print("🚀 啟動模型訓練流程")
    stock_data, sox_data, taiwan_index, usd_index, usdtwd, vix_data = data_processor.fetch_market_data(config.FETCH_DATA_START_DATE, config.FETCH_DATA_END_DATE)
    
    combined_data = data_processor.generate_all_features(stock_data, sox_data, taiwan_index, usd_index, usdtwd, vix_data)
    combined_data = generate_improved_labels(combined_data)
    
    features = combined_data.drop(['label', 'future_return'], axis=1, errors='ignore')
    labels = combined_data['label']
    
    print(f"訓練特徵數量: {len(features.columns)}")
    
    # 計算序列切分點以決定 Scaler 的擬合範圍 (避免 Data Leakage)
    time_steps = 30
    overlap = 0.6
    step_size = max(1, int(time_steps * (1 - overlap)))
    
    total_sequences = len(range(0, len(features) - time_steps, step_size))
    split = int(total_sequences * 0.8)
    
    # 確保 Scaler 只能看到訓練集範圍內的資料
    max_train_idx = (split - 1) * step_size + time_steps
    
    print(f"訓練集將使用前 {max_train_idx} 筆資料進行 Scaler 擬合，嚴格防止未來數據洩露")
    
    scaler = StandardScaler()
    scaler.fit(features.iloc[:max_train_idx])
    scaled_features = scaler.transform(features)
    
    X, y = create_sequences(scaled_features, labels.values, time_steps=time_steps, overlap=overlap)
    
    # Split
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    
    model = create_ultimate_model(input_shape=(30, X.shape[2]))
    model.compile(optimizer=AdamW(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5),
        ModelCheckpoint(config.MODEL_PATH, save_best_only=True)
    ]
    
    print("開始訓練...")
    model.fit(X_train, y_train, validation_split=0.1, epochs=100, batch_size=32, class_weight=class_weight_dict, callbacks=callbacks)
    
    print("儲存成果...")
    joblib.dump(scaler, config.SCALER_PATH)
    model.save(config.MODEL_PATH)
    print(f"✅ 訓練完成！模型已儲存至: {config.MODEL_PATH}")

if __name__ == "__main__":
    main()
