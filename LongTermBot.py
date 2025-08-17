import os
import time
import requests
import pandas as pd
import ta
from datetime import datetime, timedelta
import pytz

# Telegram 环境变量
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

coins = ["btcusdt","ethusdt","xrpusdt","bnbusdt","solusdt",
         "dogeusdt","trxusdt","adausdt","hypeusdt","linkusdt"]

# 存储上一次周线信号，避免重复发
last_week_signal = {coin: None for coin in coins}

# 获取周线数据
def get_week_kline(symbol, size=52):  # 默认取过去一年周线
    url = "https://api.huobi.pro/market/history/kline"
    params = {"symbol": symbol, "period": "1week", "size": size}
    try:
        r = requests.get(url, params=params, timeout=10)
        res = r.json()
        if "data" not in res or res.get("status") != "ok":
            print(f"获取 {symbol} 周线失败:", res)
            return None
        df = pd.DataFrame(res["data"])
        df = df.sort_values("id")
        for col in ['close','open','high','low','vol']:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        print(f"请求 {symbol} 周线异常:", e)
        return None

# 信号判断
def check_signal(df):
    close = df['close']
    high = df['high']
    low = df['low']

    df['EMA5'] = close.ewm(span=5, adjust=False).mean()
    df['EMA10'] = close.ewm(span=10, adjust=False).mean()
    df['EMA30'] = close.ewm(span=30, adjust=False).mean()
    macd = ta.trend.MACD(close)
    df['MACD'] = macd.macd_diff()
    df['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=9, smooth_window=3)
    df['K'] = stoch.stoch()
    df['D'] = stoch.stoch_signal()
    df['J'] = 3*df['K'] - 2*df['D']
    df['WR'] = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r()

    latest = df.iloc[-1]
    price = latest['close']

    long_signal = (latest['EMA5'] > latest['EMA10'] > latest['EMA30']) and (latest['MACD'] > 0) and (latest['RSI'] < 70)
    short_signal = (latest['EMA5'] < latest['EMA10'] < latest['EMA30']) and (latest['MACD'] < 0) and (latest['RSI'] > 30)

    if long_signal:
        return "做多", price, price*1.05  # 止盈 5%
    elif short_signal:
        return "做空", price, price*0.95  # 止盈 5%
    else:
        return "观望", price, price

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

# 主循环：每天美国时间9点推送
while True:
    # 获取美国东部时间
    tz = pytz.timezone("US/Eastern")
    now = datetime.now(tz)

    # 判断是否是周一~周日9点整（可调整到每天9点）
    if now.hour == 9 and now.minute == 0:
        messages = []
        for coin in coins:
            df = get_week_kline(coin)
            if df is None:
                continue
            signal, buy_price, sell_price = check_signal(df)
            # 只有信号变化才发送
            if signal != last_week_signal[coin]:
                last_week_signal[coin] = signal
            messages.append(f"{coin.upper()} 周线: {signal}\n买入价: {buy_price:.4f}\n卖出价: {sell_price:.4f}")

        # 整合消息一次性发送
        if messages:
            send_telegram_message("\n\n".join(messages))

        # 等 61 秒，避免重复触发
        time.sleep(61)
    else:
        # 每分钟检查一次时间
        time.sleep(60)
