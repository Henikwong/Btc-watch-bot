import os
import time
import requests
import pandas as pd
import numpy as np
import ta  # pip install ta

# Telegram 环境变量
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# 币种列表
coins = ["btcusdt","ethusdt","xrpusdt","bnbusdt","solusdt","dogeusdt","trxusdt","adausdt","hypeusdt","linkusdt"]

# 获取 Huobi K 线数据
def get_kline(symbol, period="60min", size=100):
    url = "https://api.huobi.pro/market/history/kline"
    params = {"symbol": symbol, "period": period, "size": size}
    r = requests.get(url, params=params)
    data = r.json()["data"]
    df = pd.DataFrame(data)
    df = df.sort_values("id")
    df['close'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['vol'] = df['vol'].astype(float)
    return df

# 计算技术指标并判断信号
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
    df['MACD'] = macd.macd_diff()  # DIF - DEA

    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()

    # KDJ
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=9, smooth_window=3)
    df['K'] = stoch.stoch()
    df['D'] = stoch.stoch_signal()
    df['J'] = 3*df['K'] - 2*df['D']

    # WR
    df['WR'] = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r()

    # 简单判断做多/做空信号
    latest = df.iloc[-1]

    long_signal = (latest['EMA5'] > latest['EMA10'] > latest['EMA30']) and (latest['MACD'] > 0) and (latest['RSI'] < 70)
    short_signal = (latest['EMA5'] < latest['EMA10'] < latest['EMA30']) and (latest['MACD'] < 0) and (latest['RSI'] > 30)

    if long_signal:
        return "做多"
    elif short_signal:
        return "做空"
    else:
        return None

# 发送 Telegram 消息
def send_telegram_message(message):
    if TOKEN and CHAT_ID:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": message}
        try:
            r = requests.post(url, data=data)
            print(f"消息发送状态: {r.status_code}")
        except Exception as e:
            print("发送消息失败:", e)

# 主循环
while True:
    for coin in coins:
        df = get_kline(coin)
        signal = check_signal(df)
        if signal:
            send_telegram_message(f"{coin.upper()} 信号: {signal}")
    time.sleep(60)  # 每分钟检查一次
