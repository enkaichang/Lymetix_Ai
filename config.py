import os

# 基礎路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 檔案路徑設定
MODEL_PATH = os.path.join(BASE_DIR, "model", "enhanced_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler", "enhanced_model_scaler.pkl")
RESULT_DATA_DIR = os.path.join(BASE_DIR, "result_data")

# 股票與資料參數
STOCK_SYMBOL = "2330.TW"
FETCH_DATA_START_DATE = "2015-01-01"
# 訓練時用的 end date, 也可以是目前時間以取得最新
FETCH_DATA_END_DATE = "2026-05-17"

# 生成特徵/標籤時的閾值參數
DATA_THRESHOLD = 0.015

def get_base_dir():
    return BASE_DIR
