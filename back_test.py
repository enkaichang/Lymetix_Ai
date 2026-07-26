import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import config

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ==============================================================================
# 核心回測框架 (Core Backtester)
# ==============================================================================
class Backtester:
    def __init__(self, initial_capital=100000.0):
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.shares = 0
        
        # 交易手續費率 (台股標準: 買 0.1425%, 賣 0.1425% + 證交稅 0.3%)
        self.buy_fee_rate = 0.001425
        self.sell_fee_rate = 0.004425
        
        self.history = []
        self.win_trades = 0
        self.total_trades = 0
        self.last_buy_price = 0.0
        self.baseline_shares = 0
        self.baseline_balance = 0
        
    def execute_trade(self, date, price, predict_class, confidence):
        # 紀錄資產基準 (Buy & Hold 對比)
        if len(self.history) == 0:
            self.baseline_shares = self.initial_capital // (price * (1 + self.buy_fee_rate))
            self.baseline_cost = self.baseline_shares * price * (1 + self.buy_fee_rate)
            self.baseline_balance = self.initial_capital - self.baseline_cost
            
        baseline_value = self.baseline_balance + self.baseline_shares * price * (1 - self.sell_fee_rate)
        
        action = "Hold"
        trade_shares = 0
        
        # 策略邏輯: 2 (漲) 買入, 0 (跌) 賣出
        if predict_class == 2 and self.balance >= price * (1 + self.buy_fee_rate):
            # 動態部位控管：信心愈高，買入比例愈高
            allocation_ratio = 0.9 if confidence > 0.85 else 0.5
            allocated_funds = self.balance * allocation_ratio
            
            trade_shares = int(allocated_funds // (price * (1 + self.buy_fee_rate)))
            if trade_shares > 0:
                cost = trade_shares * price * (1 + self.buy_fee_rate)
                self.balance -= cost
                self.shares += trade_shares
                self.last_buy_price = price
                action = "Buy"
                
        elif predict_class == 0 and self.shares > 0:
            # 動態部位控管：信心高全出，信心普通出一半
            sell_ratio = 1.0 if confidence > 0.85 else 0.5
            trade_shares = int(self.shares * sell_ratio)
            
            if trade_shares > 0:
                revenue = trade_shares * price * (1 - self.sell_fee_rate)
                self.balance += revenue
                self.shares -= trade_shares
                
                # 計算是否獲利
                if price * (1 - self.sell_fee_rate) > self.last_buy_price * (1 + self.buy_fee_rate):
                    self.win_trades += 1
                self.total_trades += 1
                action = "Sell"
                
        # 總資產估算 (現金 + 股票變現價值)
        total_value = self.balance + self.shares * price * (1 - self.sell_fee_rate)
        
        self.history.append({
            "Date": date,
            "Price": price,
            "Action": action,
            "TradeShares": trade_shares,
            "Shares": self.shares,
            "Balance": self.balance,
            "TotalValue": total_value,
            "BaselineValue": baseline_value,
            "Confidence": confidence
        })
        
    def get_performance_metrics(self):
        df = pd.DataFrame(self.history)
        if df.empty: return {}
        
        total_return = (df["TotalValue"].iloc[-1] - self.initial_capital) / self.initial_capital
        baseline_return = (df["BaselineValue"].iloc[-1] - self.initial_capital) / self.initial_capital
        
        # 計算 MDD (Maximum Drawdown)
        df["Peak"] = df["TotalValue"].cummax()
        df["Drawdown"] = (df["TotalValue"] - df["Peak"]) / df["Peak"]
        mdd = df["Drawdown"].min()
        
        win_rate = self.win_trades / self.total_trades if self.total_trades > 0 else 0
        
        return {
            "TotalReturn": total_return,
            "BaselineReturn": baseline_return,
            "MDD": mdd,
            "WinRate": win_rate,
            "TotalTrades": self.total_trades,
            "FinalValue": df["TotalValue"].iloc[-1]
        }
        
    def plot_performance(self):
        df = pd.DataFrame(self.history)
        if df.empty: return
        
        df["Date"] = pd.to_datetime(df["Date"])
        
        plt.figure(figsize=(12, 6))
        plt.plot(df["Date"], df["TotalValue"], label="AI Strategy", color="blue", linewidth=2)
        plt.plot(df["Date"], df["BaselineValue"], label="Buy & Hold Baseline", color="gray", linestyle="--")
        
        # 標記買賣點
        buy_signals = df[df["Action"] == "Buy"]
        sell_signals = df[df["Action"] == "Sell"]
        
        plt.scatter(buy_signals["Date"], buy_signals["TotalValue"], marker="^", color="green", s=100, label="Buy", zorder=5)
        plt.scatter(sell_signals["Date"], sell_signals["TotalValue"], marker="v", color="red", s=100, label="Sell", zorder=5)
        
        plt.title("AI Trading Strategy vs. Buy & Hold (Friction Costs Included)")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value (NT$)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(config.RESULT_DATA_DIR, "portfolio_performance.png")
        plt.savefig(save_path, dpi=300)
        print(f"📊 資金曲線圖已儲存至: {save_path}")
        
    def save_history(self):
        df = pd.DataFrame(self.history)
        if df.empty: return
        
        save_path = os.path.join(config.RESULT_DATA_DIR, "bought_history.csv")
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"💾 交易歷史已儲存至: {save_path}")

# ==============================================================================
# 主要流程
# ==============================================================================
def print_confusion_matrix(data):
    print("\n" + "="*50)
    print("🎯 模型預測準確度報告")
    print("="*50)
    
    correct = center = wrong = total = 0
    
    for item in data:
        if "probabilities" not in item:
            continue
        probs = list(item["probabilities"].values())
        pred_class = np.argmax(probs)
        actual_class = item.get("actual_class", -1)
        
        if actual_class == -1: continue
        
        total += 1
        if pred_class == actual_class:
            correct += 1
        elif (pred_class == 0 and actual_class == 2) or (pred_class == 2 and actual_class == 0):
            wrong += 1
        else:
            center += 1
            
    if total == 0: return
    
    print(f"預測總數: {total} 天")
    print(f"✅ 完全命中率: {correct/total:.1%}")
    print(f"⚠️ 模糊誤差率: {center/total:.1%} (一方為觀望)")
    print(f"❌ 致命錯誤率: {wrong/total:.1%} (看漲卻跌，看跌卻漲)")
    print("="*50 + "\n")

def run_backtest():
    file_path = os.path.join(config.RESULT_DATA_DIR, "back_train.json")
    if not os.path.exists(file_path):
        print("找不到 back_train.json，請先執行模型回測生成資料。")
        return
        
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # 分析預測準確度
    print_confusion_matrix(data)
    
    print("🚀 開始執行全區間交易回測...")
    # 使用 10 萬元台幣作為初始資金
    bt = Backtester(initial_capital=100000.0)
    
    for item in data:
        # 確認所需的欄位是否存在
        if "probabilities" not in item:
            continue
            
        probs = list(item["probabilities"].values())
        pred_class = np.argmax(probs)
        confidence = np.max(probs)
        # 支持兩種格式 (actual_close_today 或是 Close)
        price = item.get("actual_close_today", item.get("Close", 0))
        if price == 0: continue
        date = item.get("date", "Unknown")
        
        bt.execute_trade(date, price, pred_class, confidence)
        
    metrics = bt.get_performance_metrics()
    if metrics:
        print("="*50)
        print("💼 量化回測績效報告 (已扣除手續費與證交稅)")
        print("="*50)
        print(f"總交易次數: {metrics['TotalTrades']} 次")
        print(f"策略勝率:   {metrics['WinRate']:.1%}")
        print(f"最大回撤:   {metrics['MDD']:.1%}")
        print(f"最終總資產: NT$ {metrics['FinalValue']:,.0f}")
        print(f"策略總報酬: {metrics['TotalReturn']:.1%}")
        print(f"基準總報酬: {metrics['BaselineReturn']:.1%} (Buy & Hold)")
        print("="*50)
        
        bt.plot_performance()
        bt.save_history()

# ==============================================================================
# Legacy / 實驗性模組 (Secondary LSTM Meta-Model)
# ==============================================================================
def legacy_train_meta_model(data, timesteps=10):
    from keras import layers, Sequential
    model = Sequential([
        layers.LSTM(16, activation='relu', input_shape=(timesteps, len(data[0]["probabilities"]))),
        layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    X, y = [], []
    pro_data = [list(item["probabilities"].values()) for item in data]
    for index in range(len(data)-timesteps):
        X.append(pro_data[index:index+timesteps])
    
    X = np.array(X)
    y = np.array([item["actual_class"] for item in data][timesteps:])
    split = int(len(X)*0.8)
    model.fit(X[:split], y[:split], epochs=10, batch_size=32, validation_data=(X[split:], y[split:]))
    model.save("back_edit_model.h5")

if __name__ == "__main__":
    run_backtest()
