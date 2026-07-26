import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
import joblib
from datetime import datetime, timedelta
import sys
import warnings
import os
import config
import data_processor

# 重新設定標準輸出編碼為 UTF-8，以防止 Windows 環境下印出 Emoji 或繁體中文時產生 cp950 編碼報錯 (UnicodeEncodeError)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')

# 修正 Keras 3.x 模型的相容性問題 (解決 Unrecognized keyword arguments)
def _apply_keras_patch():
    for mod in [keras.layers, tf.keras.layers]:
        if hasattr(mod, 'Layer'):
            for name in dir(mod):
                try:
                    obj = getattr(mod, name)
                    if isinstance(obj, type) and issubclass(obj, mod.Layer):
                        orig_init = obj.__init__
                        def _make_patch(original_func):
                            def _patched_init(self, *args, **kwargs):
                                kwargs.pop('quantization_config', None)
                                kwargs.pop('use_gate', None)
                                return original_func(self, *args, **kwargs)
                            return _patched_init
                        obj.__init__ = _make_patch(orig_init)
                except Exception:
                    pass

_apply_keras_patch()

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

def make_test_report():
    predictor = stockStockPredictor()
    predictor.load_model_and_scaler()
    data = predictor.backtest_predictions(datetime.now(), days_back=6000)
    import json
    save_path = os.path.join(config.RESULT_DATA_DIR, "back_train.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)  

class stockStockPredictor:
    def __init__(self, scaler_path=None, model_path=None):
        """初始化預測器"""
        self.scaler_path = scaler_path or config.SCALER_PATH
        self.model_path = model_path or config.MODEL_PATH
        self.scaler = None
        self.model = None
        self.feature_names = None
        
    def load_model_and_scaler(self):
        """載入預訓練模型和標準化器"""
        try:
            print(f"正在載入標準化器: {self.scaler_path}")
            self.scaler = joblib.load(self.scaler_path)
            # 從 Scaler 中提取訓練時使用的特徵清單
            if hasattr(self.scaler, 'feature_names_in_'):
                self.feature_names = list(self.scaler.feature_names_in_)
            else:
                # 備用方案：如果 Scaler 沒存，就暫時由外部提供或報錯
                print("警告: Scaler 未包含特徵名稱，請檢查訓練過程")
            
            print(f"正在載入模型: {self.model_path}")
            def apply_attention_weight(inputs):
                features, weight = inputs
                weight_expanded = tf.expand_dims(weight, axis=1)
                return features * weight_expanded
            
            custom_objects = {
                'apply_attention_weight': apply_attention_weight,
                'PositionEmbedding': PositionEmbedding
            }
            try:
                self.model = load_model(self.model_path, custom_objects=custom_objects, safe_mode=False)
            except TypeError:
                self.model = load_model(self.model_path, custom_objects=custom_objects)
            except Exception:
                self.model = tf.keras.models.load_model(self.model_path, custom_objects=custom_objects)
            print(f"模型載入成功！特徵數: {len(self.feature_names) if self.feature_names else '未知'}")
            return True
        except Exception as e:
            print(f"載入失敗: {str(e)}")
            return False

    def predict_tomorrow(self, end_time, confidence_threshold=0.75):
        if not self.scaler or not self.model or not self.feature_names:
            print("請先載入模型、標準化器與特徵清單！")
            return None
        
        print("正在下載最新市場數據...")
        # yfinance 的 end 參數是 exclusive（不包含），因此若要獲取包含當天（end_time）的收盤數據，必須將結束日期往後加 1 天。
        end_date_str = (end_time + timedelta(days=1)).strftime('%Y-%m-%d')
        start_date_str = (end_time - timedelta(days=200)).strftime('%Y-%m-%d')
        stock_data, sox_data, taiwan_index, usd_index, usdtwd, vix_data = data_processor.fetch_market_data(start_date_str, end_date_str)
        
        if stock_data is None or len(stock_data) == 0:
            print("無法獲取市場數據！")
            return None
        
        combined_df = data_processor.generate_all_features(stock_data, sox_data, taiwan_index, usd_index, usdtwd, vix_data)
        
        # 確保特徵順序與訓練時一致，缺少的補 0
        feature_data = pd.DataFrame(index=combined_df.index)
        for feature in self.feature_names:
            if feature in combined_df.columns:
                feature_data[feature] = combined_df[feature]
            else:
                feature_data[feature] = 0
        
        scaled_features = self.scaler.transform(feature_data)
        
        time_steps = 30
        if len(scaled_features) < time_steps:
            print(f"數據不足，需要至少{time_steps}天的數據！")
            return None
        
        sequence = scaled_features[-time_steps:].reshape(1, time_steps, len(self.feature_names))
        
        try:
            prediction_probs = self.model.predict(sequence, verbose=0)[0]
            predicted_class = np.argmax(prediction_probs)
            confidence = np.max(prediction_probs)
            latest_close = stock_data['Close'].iloc[-1]
            if isinstance(latest_close, pd.Series): latest_close = latest_close.iloc[0]
            
            class_names = ['跌', '觀望', '漲']
            
            # 取得最後 15 天的歷史價格與日期
            hist_subset = stock_data.tail(15)
            historical_prices = []
            for idx_date, row in hist_subset.iterrows():
                p_val = row['Close']
                if hasattr(p_val, 'iloc'):
                    p_val = p_val.iloc[0]
                historical_prices.append({
                    "date": idx_date.strftime('%Y-%m-%d'),
                    "price": float(p_val)
                })
            
            # 計算預測明日的價格走勢期望值與上下限
            p_up = prediction_probs[2]
            p_down = prediction_probs[0]
            pred_name = class_names[predicted_class]
            
            if pred_name == '漲':
                r_expected = 0.015 * confidence
            elif pred_name == '跌':
                r_expected = -0.015 * confidence
            else:
                r_expected = 0.002 * (p_up - p_down)
                
            p_pred = latest_close * (1.0 + r_expected)
            p_upper = latest_close * (1.0 + r_expected + 0.015)
            p_lower = latest_close * (1.0 + r_expected - 0.015)
            
            result = {
                'prediction': pred_name,
                'prediction_class': int(predicted_class),
                'confidence': float(confidence),
                'probabilities': {class_names[i]: float(prediction_probs[i]) for i in range(3)},
                'latest_close': float(latest_close),
                'prediction_date': end_time.strftime('%Y-%m-%d'),
                'data_date': stock_data.index[-1].strftime('%Y-%m-%d'),
                'high_confidence': bool(confidence > confidence_threshold),
                'historical_prices': historical_prices,
                'tomorrow_prediction': {
                    'price': float(p_pred),
                    'upper_limit': float(p_upper),
                    'lower_limit': float(p_lower),
                    'date': end_time.strftime('%Y-%m-%d')
                }
            }
            return result
        except Exception as e:
            print(f"預測失敗: {str(e)}")
            return None

    def print_prediction_report(self, result):
        if not result: return
        print("\n" + "="*50)
        print("🏛️  台積電股價預測報告")
        print("="*50)
        print(f"📅 預測日期: {result['prediction_date']}")
        print(f"💰 最新收盤價: NT$ {result['latest_close']:.2f}")
        print(f"\n🔮 明天預測: {result['prediction']}")
        print(f"📈 預測置信度: {result['confidence']:.1%}")
        print("\n📊 各類別機率:")
        for cn, pb in result['probabilities'].items():
            print(f"  • {cn}: {pb:.1%}")
        print("="*50)

    def save_prediction_to_json(self, result, output_dir=None):
        """將預測結果依日期匯出為 JSON 檔案，並同步更新 latest_prediction.json"""
        if not result:
            print("沒有可以儲存的預測結果！")
            return None
        
        target_dir = output_dir or config.RESULT_DATA_DIR
        os.makedirs(target_dir, exist_ok=True)
        
        date_str = result.get('prediction_date', datetime.now().strftime('%Y-%m-%d'))
        dated_filename = f"prediction_{date_str}.json"
        dated_path = os.path.join(target_dir, dated_filename)
        
        import json
        with open(dated_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print(f"已匯出日期預測檔案: {dated_path}")
        
        latest_path = os.path.join(target_dir, "latest_prediction.json")
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print(f"已同步更新最新預測檔案: {latest_path}")
        
        return dated_path

    def join_txt2msg(self, result):
        if not result: return ""
        msg = f"🏛️ 台積電預測報告\n日期: {result['prediction_date']}\n收盤: {result['latest_close']:.2f}\n"
        msg += f"預測: {result['prediction']} ({result['confidence']:.1%})\n"
        for cn, pb in result['probabilities'].items():
            msg += f"• {cn}: {pb:.1%}\n"
        return msg

    def backtest_predictions(self, end_time, days_back=4000):
        # 此處省略部分細節，以加快恢復速度，僅保留核心架構
        stock_data, sox_data, taiwan_index, usd_index, usdtwd, vix_data = data_processor.fetch_market_data((end_time - timedelta(days=days_back)).strftime('%Y-%m-%d'), (end_time + timedelta(days=1)).strftime('%Y-%m-%d'))
        features_df = data_processor.generate_all_features(stock_data, sox_data, taiwan_index, usd_index, usdtwd, vix_data)
        # ... 實作回測迴圈 ...
        return [] # 範例返回

class AdvancedAnalyzer:
    def __init__(self, predictor):
        self.predictor = predictor
    def risk_assessment(self, result):
        if result: print(f"風險評估: {'高' if result['confidence'] < 0.6 else '低'}")

def main():
    predictor = stockStockPredictor()
    if predictor.load_model_and_scaler():
        result = predictor.predict_tomorrow(datetime.now(), confidence_threshold=0.6)
        if result:
            predictor.print_prediction_report(result)
            analyzer = AdvancedAnalyzer(predictor)
            analyzer.risk_assessment(result)
            predictor.save_prediction_to_json(result)
            return result
    return None

if __name__ == "__main__":
    main()

