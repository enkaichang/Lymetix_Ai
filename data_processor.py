import yfinance as yf
import pandas as pd
import numpy as np
import config
import warnings
warnings.filterwarnings("ignore")

# --- 基礎工具 ---
def safe_divide(numerator, denominator):
    if isinstance(numerator, np.ndarray): numerator = pd.Series(numerator)
    if isinstance(denominator, np.ndarray): denominator = pd.Series(denominator)
    result = numerator / (denominator.abs() + 1e-6)
    result = result.replace([np.inf, -np.inf], np.nan)
    return result

def winsorize_data(data, lower_percentile=5, upper_percentile=95):
    if isinstance(data, np.ndarray): data = pd.Series(data)
    # 使用擴展窗口 (Expanding Window) 來避免未來數據洩露 (Look-ahead Bias)
    # min_periods=30 確保初期有足夠樣本才開始去除極值，並使用 bfill 填補最初的 30 天
    lower_bound = data.expanding(min_periods=30).quantile(lower_percentile / 100).bfill()
    upper_bound = data.expanding(min_periods=30).quantile(upper_percentile / 100).bfill()
    return data.clip(lower=lower_bound, upper=upper_bound)

# --- 技術指標計算工具 ---
def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = safe_divide(avg_gain, avg_loss)
    rsi = 100 - safe_divide(100, (1 + rs))
    return rsi.fillna(50)

def compute_MACD(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def compute_bollinger_bands(series, period=20, std_dev=2):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + (std * std_dev)
    lower = ma - (std * std_dev)
    return upper, lower, ma

def compute_KDJ(df, n=9, k_period=3, d_period=3):
    low_min = df['Low'].rolling(n).min()
    high_max = df['High'].rolling(n).max()
    rsv = 100 * safe_divide((df['Close'] - low_min), (high_max - low_min))
    k = rsv.ewm(com=(k_period - 1), adjust=False).mean()
    d = k.ewm(com=(d_period - 1), adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j

def compute_williams_r(df, period=14):
    high_max = df['High'].rolling(period).max()
    low_min = df['Low'].rolling(period).min()
    wr = -100 * safe_divide((high_max - df['Close']), (high_max - low_min))
    return wr

def compute_cci(df, period=20):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    return safe_divide((tp - ma), (0.015 * mad))

def compute_stoch_rsi(df, period=14):
    rsi = compute_RSI(df['Close'], period)
    return safe_divide((rsi - rsi.rolling(period).min()), (rsi.rolling(period).max() - rsi.rolling(period).min())) * 100

def compute_adx(df, period=14):
    plus_dm = df['High'].diff().clip(lower=0)
    minus_dm = (df['Low'].diff() * -1).clip(lower=0)
    tr = compute_ATR(df, 1).rolling(period).sum()
    plus_di = 100 * (plus_dm.rolling(period).mean() / tr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / tr)
    dx = 100 * safe_divide((plus_di - minus_di).abs(), (plus_di + minus_di))
    return dx.rolling(period).mean()

def compute_parabolic_sar(df):
    sar = df['Close'].copy()
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]: sar.iloc[i] = min(df['Low'].iloc[i-1], sar.iloc[i-1])
        else: sar.iloc[i] = max(df['High'].iloc[i-1], sar.iloc[i-1])
    return sar

def compute_obv(df):
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return obv

def compute_ATR(df, period=14):
    tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_trend_strength(series, period=20):
    ret = series.pct_change()
    pos, neg = ret.clip(lower=0).rolling(period).mean(), ret.clip(upper=0).abs().rolling(period).mean()
    return safe_divide(pos, pos + neg).fillna(0.5)

def compute_resistance_distance(df, window=20):
    return safe_divide((df['High'].rolling(window).max() - df['Close']), df['Close'])

def compute_support_distance(df, window=20):
    return safe_divide((df['Close'] - df['Low'].rolling(window).min()), df['Close'])

def detect_divergence(price, indicator, window=20):
    p_t = price.rolling(window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    i_t = indicator.rolling(window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    return np.where((p_t < 0) & (i_t > 0), 1, np.where((p_t > 0) & (i_t < 0), -1, 0))

def identify_hammer(df):
    body = (df['Close'] - df['Open']).abs()
    lower_shadow = df[['Close', 'Open']].min(axis=1) - df['Low']
    upper_shadow = df['High'] - df[['Close', 'Open']].max(axis=1)
    return ((lower_shadow > 2 * body) & (upper_shadow < 0.1 * body)).astype(int)

def identify_engulfing_pattern(df):
    pb, cb = (df['Close'].shift() - df['Open'].shift()).abs(), (df['Close'] - df['Open']).abs()
    bull = (df['Close'].shift() < df['Open'].shift()) & (df['Close'] > df['Open']) & (cb > pb)
    bear = (df['Close'].shift() > df['Open'].shift()) & (df['Close'] < df['Open']) & (cb > pb)
    return np.where(bull, 1, np.where(bear, -1, 0))

# --- 主要資料抓取與處理 ---
def fetch_market_data(start_date=config.FETCH_DATA_START_DATE, end_date=config.FETCH_DATA_END_DATE):
    print(f"正在下載{config.STOCK_SYMBOL}數據...")
    stock_data = yf.download(config.STOCK_SYMBOL, start=start_date, end=end_date)
    sox_data = yf.download("SOXX", start=start_date, end=end_date)
    taiwan_index = yf.download("^TWII", start=start_date, end=end_date)
    usd_index = yf.download("DX-Y.NYB", start=start_date, end=end_date)
    vix_data = yf.download("^VIX", start=start_date, end=end_date)
    try: usdtwd = yf.download("USDTWD=X", start=start_date, end=end_date)
    except: usdtwd = yf.download("TWD=X", start=start_date, end=end_date)
    
    for d in [stock_data, sox_data, taiwan_index, usd_index, usdtwd, vix_data]:
        if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
    return stock_data, sox_data, taiwan_index, usd_index, usdtwd, vix_data

def compute_market_correlation_features(stock_data, sox_data, taiwan_index, usd_index, usdtwd, vix_data):
    common = stock_data.index.intersection(sox_data.index).intersection(taiwan_index.index).intersection(usd_index.index).intersection(usdtwd.index).intersection(vix_data.index)
    sr, sxr, tr, ur, twr, vr = [d.loc[common, 'Close'].pct_change() for d in [stock_data, sox_data, taiwan_index, usd_index, usdtwd, vix_data]]
    
    mf = pd.DataFrame(index=common)
    mf['VIX_Close'] = vix_data.loc[common, 'Close']
    mf['VIX_Returns_1'] = vr
    mf['VIX_MA5'] = mf['VIX_Close'].rolling(5).mean()
    mf['VIX_Sentiment'] = np.where(mf['VIX_Close'] > 25, -1, np.where(mf['VIX_Close'] < 15, 1, 0)) # 恐慌/貪婪
    mf['SOX_Returns_1'], mf['SOX_Returns_5'], mf['SOX_RSI'] = sxr, sox_data.loc[common, 'Close'].pct_change(5), compute_RSI(sox_data.loc[common, 'Close'])
    mf['SOX_Momentum'] = sox_data.loc[common, 'Close'] / sox_data.loc[common, 'Close'].rolling(20).mean() - 1
    mf['SOX_Volatility'] = sxr.rolling(20).std()
    # 台股加權指數特徵
    mf['TWII_Returns_1'], mf['TWII_Returns_5'], mf['TWII_RSI'] = tr, taiwan_index.loc[common, 'Close'].pct_change(5), compute_RSI(taiwan_index.loc[common, 'Close'])
    mf['TWII_Momentum'] = taiwan_index.loc[common, 'Close'] / taiwan_index.loc[common, 'Close'].rolling(20).mean() - 1
    mf['TWII_Volatility'] = tr.rolling(20).std()
    
    # 方案 A: 自建台股波動率指標 (類 Taiwan VIX)
    # 計算 20 日年化波動率
    mf['TW_VIX_Proxy'] = tr.rolling(20).std() * np.sqrt(252) * 100
    mf['TW_VIX_Change'] = mf['TW_VIX_Proxy'].pct_change()
    
    mf['USD_Returns_1'], mf['USD_Volatility'] = ur, ur.rolling(20).std()
    mf['USDTWD_Returns_1'] = twr
    
    mf['stock_SOX_Corr'], mf['stock_TWII_Corr'] = sr.rolling(20).corr(sxr), sr.rolling(20).corr(tr)
    mf['stock_USD_Corr'], mf['stock_TW_Corr'] = sr.rolling(20).corr(ur), sr.rolling(20).corr(twr)
    
    mf['stock_SOX_Ratio'] = safe_divide(stock_data.loc[common, 'Close'], sox_data.loc[common, 'Close'])
    mf['stock_TWII_Ratio'] = safe_divide(stock_data.loc[common, 'Close'], taiwan_index.loc[common, 'Close'])
    mf['stock_vs_SOX'], mf['stock_vs_TWII'] = sr - sxr, sr - tr
    return mf

def compute_advanced_technical_indicators(df):
    data = df.copy()
    data['Returns_1'], data['Returns_3'], data['Returns_5'], data['Returns_10'] = [data['Close'].pct_change(i) for i in [1, 3, 5, 10]]
    for p in [5, 10, 20, 50, 100]:
        data[f'MA{p}'] = data['Close'].rolling(p).mean()
        data[f'EMA{p}'] = data['Close'].ewm(span=p).mean()
        data[f'MA{p}_slope'] = data[f'MA{p}'].diff(5)
        data[f'Price_MA{p}_ratio'] = safe_divide(data['Close'], data[f'MA{p}'])
        data[f'Price_EMA{p}_ratio'] = safe_divide(data['Close'], data[f'EMA{p}'])
    
    data['MA5_MA20_cross'] = np.where(data['MA5'] > data['MA20'], 1, -1)
    data['MA10_MA50_cross'] = np.where(data['MA10'] > data['MA50'], 1, -1)
    data['MA20_MA100_cross'] = np.where(data['MA20'] > data['MA100'], 1, -1)

    for p in [7, 14, 21, 28]:
        data[f'RSI{p}'] = compute_RSI(data['Close'], p)
        data[f'RSI{p}_signal'] = np.where(data[f'RSI{p}'] > 70, -1, np.where(data[f'RSI{p}'] < 30, 1, 0))
        data[f'RSI{p}_divergence'] = detect_divergence(data['Close'], data[f'RSI{p}'])

    for f, s, sig in [(12, 26, 9), (5, 35, 5), (19, 39, 9)]:
        macd, sl, hist = compute_MACD(data['Close'], f, s, sig)
        data[f'MACD_{f}_{s}_{sig}'], data[f'MACD_signal_{f}_{s}_{sig}'], data[f'MACD_histogram_{f}_{s}_{sig}'] = macd, sl, hist
        data[f'MACD_cross_{f}_{s}_{sig}'] = np.where(macd > sl, 1, -1)

    for p, sd in [(20, 2), (20, 2.5), (10, 1.5)]:
        up, lo, mid = compute_bollinger_bands(data['Close'], p, sd)
        data[f'BB_upper_{p}_{sd}'], data[f'BB_lower_{p}_{sd}'], data[f'BB_middle_{p}_{sd}'] = up, lo, mid
        data[f'BB_width_{p}_{sd}'] = safe_divide(up - lo, mid)
        data[f'BB_position_{p}_{sd}'] = safe_divide(data['Close'] - lo, up - lo)
        data[f'BB_squeeze_{p}_{sd}'] = (data[f'BB_width_{p}_{sd}'] < data[f'BB_width_{p}_{sd}'].rolling(20).mean() * 0.8).astype(int)

    for n, kp, dp in [(9, 3, 3), (14, 5, 5), (21, 7, 7)]:
        k, d, j = compute_KDJ(data, n, kp, dp)
        data[f'K_{n}_{kp}_{dp}'], data[f'D_{n}_{kp}_{dp}'], data[f'J_{n}_{kp}_{dp}'] = k, d, j
        data[f'KD_cross_{n}_{kp}_{dp}'] = np.where(k > d, 1, -1)

    data['Williams_R'], data['CCI'], data['Stoch_RSI'], data['ADX'], data['PSAR'] = compute_williams_r(data), compute_cci(data), compute_stoch_rsi(data), compute_adx(data), compute_parabolic_sar(data)
    data['Volume_ratio'] = safe_divide(data['Volume'], data['Volume'].rolling(20).mean())
    data['Volume_trend'] = data['Volume'].rolling(5).mean() / data['Volume'].rolling(20).mean() - 1
    data['OBV'] = compute_obv(data)
    data['OBV_signal'] = np.where(data['OBV'] > data['OBV'].rolling(20).mean(), 1, -1)
    data['Price_Volume_correlation'] = data['Close'].rolling(10).corr(data['Volume']).fillna(0)
    data['Volume_price_trend'] = np.where((data['Returns_5'] > 0) & (data['Volume_ratio'] > 1.2), 1, np.where((data['Returns_5'] < 0) & (data['Volume_ratio'] > 1.2), -1, 0))

    for p in [7, 14, 21, 30]:
        data[f'ATR{p}_ratio'] = safe_divide(compute_ATR(data, p), data['Close'])
        data[f'Volatility{p}'] = data['Returns_1'].rolling(p).std()

    data['High_Low_ratio'] = safe_divide(data['High'] - data['Low'], data['Close'])
    data['Close_position'] = safe_divide(data['Close'] - data['Low'], data['High'] - data['Low'])
    data['Upper_shadow'] = safe_divide(data['High'] - data[['Open', 'Close']].max(axis=1), data['Close'])
    data['Lower_shadow'] = safe_divide(data[['Open', 'Close']].min(axis=1) - data['Low'], data['Close'])
    data['Trend_strength'], data['Price_velocity'] = compute_trend_strength(data['Close']), data['Close'].diff() / data['Close'].shift()
    data['Price_acceleration'] = data['Price_velocity'].diff()

    for w in [10, 20, 50]:
        data[f'Resistance_distance_{w}'], data[f'Support_distance_{w}'] = compute_resistance_distance(data, w), compute_support_distance(data, w)

    data['Doji'], data['Hammer'], data['Engulfing'] = ((data['Close'] - data['Open']).abs() / (data['High'] - data['Low']) < 0.1).astype(int), identify_hammer(data), identify_engulfing_pattern(data)

    for col in data.select_dtypes(include=[np.number]).columns:
        if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
            data[col] = winsorize_data(data[col])
            data[col] = data[col].ffill().fillna(0)
    return data

def generate_all_features(stock_data, sox_data, taiwan_index, usd_index, usdtwd, vix_data):
    print(f"正在計算{config.STOCK_SYMBOL}技術指標...")
    stock_data = compute_advanced_technical_indicators(stock_data)
    print("正在計算市場相關與情緒特徵 (含VIX)...")
    market_features = compute_market_correlation_features(stock_data, sox_data, taiwan_index, usd_index, usdtwd, vix_data)
    common = stock_data.index.intersection(market_features.index)
    combined = pd.concat([stock_data.loc[common], market_features.loc[common]], axis=1)
    for col in combined.columns:
        combined[col] = combined[col].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    return combined
