import os
import time
import requests
import pandas as pd
import numpy as np
import ta

# Telegram 环境变量
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# 监控币种
coins = ["btcusdt","ethusdt","xrpusdt","bnbusdt","solusdt",
         "dogeusdt","trxusdt","adausdt","hypeusdt","linkusdt"]

# K 线周期
periods = ["15min", "30min", "60min", "4hour"]

# 记录上一次信号，避免重复发送
last_signal = {coin: {p: None for p in periods} for coin in coins}

# 获取 Huobi K 线数据
def get_kline(symbol, period="60min", size=100):
    url = "https://api.huobi.pro/market/history/kline"
    params = {"symbol": symbol, "period": period, "size": size}
    try:
        r = requests.get(url, params=params, timeout=10)
        res = r.json()
        if "data" not in res or res.get("status") != "ok":
            print(f"获取 {symbol} {period} K线失败:", res)
            return None
        df = pd.DataFrame(res["data"])
        df = df.sort_values("id")
        for col in ['close','open','high','low','vol']:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        print(f"请求 {symbol} {period} K线异常:", e)
        return None

# 计算指标并判断信号
def check_signal(df):
    close = df['close']
    high = df['high']
    low = df['low']

    # EMA
    df['EMA5'] = close.ewm(span=5, adjust=False).mean()
    df['EMA10'] = close.ewm(span=10, adjust=False).mean()
    df['EMA30'] = close.ewm(span=30, adjust=False).mean()

    # MACD
    macd = ta.trend.MACD(close)
    df['MACD'] = macd.macd_diff()

    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()

    # KDJ
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=9, smooth_window=3)
    df['K'] = stoch.stoch()
    df['D'] = stoch.stoch_signal()
    df['J'] = 3*df['K'] - 2*df['D']

    # WR
    df['WR'] = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r()

    latest = df.iloc[-1]
    price = latest['close']

    long_signal = (latest['EMA5'] > latest['EMA10'] > latest['EMA30']) and (latest['MACD'] > 0) and (latest['RSI'] < 70)
    short_signal = (latest['EMA5'] < latest['EMA10'] < latest['EMA30']) and (latest['MACD'] < 0) and (latest['RSI'] > 30)

    if long_signal:
        buy_price = price
        sell_price = price * 1.02  # 假设止盈 2%
        return "做多", buy_price, sell_price
    elif short_signal:
        buy_price = price
        sell_price = price * 0.98  # 假设止盈 2%
        return "做空", buy_price, sell_price
    else:
        return None, None, None

# 发送 Telegram 消息
def send_telegram_message(message):
    if TOKEN and CHAT_ID:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": message}
        try:
            r = requests.post(url, data=data)
            print(f"消息发送状态: {r.status_code} -> {message}")
        except Exception as e:
            print("发送消息失败:", e)

# 主循环
while True:
    for coin in coins:
        for period in periods:
            df = get_kline(coin, period)
            if df is None:
                continue
            signal, buy_price, sell_price = check_signal(df)
            if signal and signal != last_signal[coin][period]:
                msg = f"{coin.upper()} {period} 信号: {signal}\n买入价: {buy_price:.4f}\n卖出价: {sell_price:.4f}"
                send_telegram_message(msg)
            last_signal[coin][period] = signal
    # 每 30 分钟检查一次
    time.sleep(1800)
