# -*- coding: utf-8 -*-
import os
import sys
import json
from datetime import datetime
from flask import Flask, render_template, jsonify, request, send_file
import pandas as pd

# 確保專案根目錄在 Python 模組搜尋路徑中，以便能夠載入 config, use_model, back_test 等模組
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config
import use_model
import back_test

# 解決 Windows 下印出中文的編碼問題
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

app = Flask(
    __name__,
    static_folder=os.path.join(current_dir, 'static'),
    template_folder=os.path.join(current_dir, 'templates')
)

def run_prediction_flow():
    """載入模型並計算最新預測，並儲存快取到 latest_prediction.json"""
    try:
        predictor = use_model.stockStockPredictor()
        if predictor.load_model_and_scaler():
            result = predictor.predict_tomorrow(datetime.now(), confidence_threshold=0.6)
            if result:
                # 確保結果目錄存在
                os.makedirs(config.RESULT_DATA_DIR, exist_ok=True)
                save_path = os.path.join(config.RESULT_DATA_DIR, "latest_prediction.json")
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=4)
                return result
    except Exception as e:
        print(f"運行 AI 預測流程時發生錯誤: {str(e)}")
    return None

# ==============================================================================
# Flask Web 路由 (Flask Routes)
# ==============================================================================

@app.route('/')
def home():
    """首頁路由：渲染前端儀表板"""
    return render_template('index.html')

@app.route('/api/prediction', methods=['GET'])
def get_prediction():
    """取得最新 AI 預測結果。支援優先讀取快取以達到極速載入"""
    cache_path = os.path.join(config.RESULT_DATA_DIR, "latest_prediction.json")
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 確保快取包含明天預報走勢與歷史價格特徵，否則強行重新推理
            if "tomorrow_prediction" in data and "historical_prices" in data:
                return jsonify({"status": "success", "data": data, "cached": True})
        except Exception as e:
            print(f"讀取預測快取失敗: {str(e)}")
            
    # 若無快取或快取過期，則現場計算一次
    print("未發現完整預測快取，正在進行首次模型預測推理...")
    result = run_prediction_flow()
    if result:
        return jsonify({"status": "success", "data": result, "cached": False})
    else:
        return jsonify({"status": "error", "message": "無法生成預測數據"}), 500

@app.route('/api/predict/run', methods=['POST'])
def force_prediction():
    """強制後台重新下載最新市場數據，並調用深度學習模型重新推理"""
    print("使用者觸發：強制重新運行 AI 預測...")
    result = run_prediction_flow()
    if result:
        return jsonify({"status": "success", "data": result})
    else:
        return jsonify({"status": "error", "message": "重新運行預測失敗"}), 500

if __name__ == '__main__':
    # 啟動 Flask 開發伺服器，預設連接埠為 5000，使用 0.0.0.0 使區域網路內可造訪
    app.run(host='0.0.0.0', port=5000, debug=True)
