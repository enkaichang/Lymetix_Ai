# 🏛️ Lymetix_Ai: 混合深度學習台積電股價預測與回測系統

本專案是一個基於混合深度學習架構（Hybrid Deep Learning Architecture）的台積電（2330.TW）股價趨勢預測系統。結合了多尺度 CNN、雙向 LSTM/GRU 以及 Transformer 機制，整合多源市場與情緒特徵（包含美股半導體、台股加權指數、美元指數、VIX 恐慌指數、匯率等），並支援自動化 Line API 預測推播與資產回測模擬。

大部分程式邏輯、指標由 **Gemini 3 Flash** 及 **Gemini 3.1 Pro** 生成。

---

## 🌟 核心特色

1. **多維度特徵工程 (`data_processor.py`)**：
   - 自動下載 Yahoo Finance 多源市場數據（`2330.TW`, `SOXX`, `^TWII`, `DX-Y.NYB`, `USDTWD=X`, `^VIX`）。
   - 提取技術指標（RSI, MACD, Bollinger Bands, KDJ, Williams %R, CCI, Stochastic RSI, ADX, Parabolic SAR, OBV, ATR）。
   - 包含蠟燭形態識別（Doji, Hammer, Engulfing）與支持/阻力距離與背離檢測。
2. **混合深度學習架構 (`train_model.py`)**：
   - **多尺度 CNN 模組**：捕捉不同時間窗口的空間特徵。
   - **雙向 RNN 模組（BiLSTM + BiGRU）**：捕捉序列前後的短期與中期時序特徵。
   - **Transformer 模組（Multi-Head Attention）**：捕捉長距離時間相關性。
   - **注意力融合層（Feature Fusion）**：將所有模組特徵進行權重調配，提升決策解釋力。
3. **自動化 Line 推播 (`Line_api.py`)**：
   - 支援將每日模型預測的「漲、跌、觀望」與置信度以 Line Bot 自動廣播給用戶。
4. **回測與交易模擬 (`back_test.py`)**：
   - 模擬實際交易策略，並根據模型置信度動態調整買賣比例與部位，自動生成 `bought_history.txt` 交易明細。

---

## 📂 檔案結構

```bash
Lymetix_Ai/
├── config.py             # 全局參數設定（標的、日期、特徵閾值與路徑）
├── data_processor.py     # 多源數據抓取、清洗與高階特徵工程
├── train_model.py        # 混合深度學習模型架構搭建與訓練流程
├── use_model.py          # 模型與標準化器載入、單日預測與回測數據生成
├── Line_api.py           # 串接 Line Messaging API，實現每日自動推播
├── back_test.py          # 基於預測概率的交易模擬與資產回測
├── requirements.txt      # 專案相依套件清單
├── model/                # 存放訓練好的模型檔 (.h5)
├── scaler/               # 存放特徵標準化器 (.pkl)
└── result_data/          # 存放回測結果與交易歷史紀錄 (.json / .txt)
```

---

## 🛠️ 安裝與快速開始

### 1. 安裝環境與套件
請確保系統已安裝 Python 3.8+，並執行以下指令安裝相依套件：

```bash
pip install -r requirements.txt
```

> [!NOTE]
> 主要相依套件包含：`tensorflow`, `keras`, `pandas`, `numpy`, `yfinance`, `scikit-learn`, `joblib`, `requests`。

### 2. 步驟一：訓練模型
執行 `train_model.py` 下載歷史數據、產生特徵並訓練混合深度學習模型：

```bash
python train_model.py
```
*訓練完成後，模型將儲存於 `model/enhanced_model.h5`，標準化器存於 `scaler/enhanced_model_scaler.pkl`。*

### 3. 步驟二：執行單日預測
使用訓練好的模型預測明天的趨勢（漲/跌/觀望）：

```bash
python use_model.py
```

### 4. 步驟三：啟動 Line 自動化推送
在 `Line_api.py` 中填入您的 `Channel Access Token`，並執行以發送最新預測訊息：

```bash
python Line_api.py
```

### 5. 步驟四：運行回測模擬
評估模型在歷史數據上的交易表現：

```bash
python back_test.py
```
*回測結果與資產變化紀錄會輸出到 `result_data/bought_history.txt` 中。*

---

## 📈 參數調整說明 (`config.py`)

您可以在 `config.py` 中自訂以下參數：
- `STOCK_SYMBOL`: 預測股票標的（預設為 `"2330.TW"`）
- `FETCH_DATA_START_DATE` / `END_DATE`: 訓練資料的時間範圍
- `DATA_THRESHOLD`: 生成標籤時的漲跌變動閾值預設值
