import os
import time
import requests
import pandas as pd
import ta
from datetime import datetime, timedelta

# ================== Telegram ==================
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ================== 币种列表 ==================
coins = ["btcusdt","ethusdt","xrpusdt","bnbusdt","solusdt",
         "dogeusdt","trxusdt","adausdt","linkusdt"]

# ================== 工具函数 ==================
def format_price(price):
    try:
        price = float(price)
        if price >= 100:
            return f"{price:.2f}"
        elif price >= 1:
            return f"{price:.4f}"
        elif price >= 0.01:
            return f"{price:.6f}"
        else:
            return f"{price:.8f}"
    except:
        return "-"

def send_telegram_message(message):
    if TOKEN and CHAT_ID:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        try:
            requests.post(url, data={"chat_id": CHAT_ID, "text": message})
            print("✅ Telegram 已发送")
        except Exception as e:
            print("❌ Telegram 发送失败:", e)

# ================== K线获取 ==================
def get_kline_huobi(symbol, period="60min", size=50):
    url = "https://api.huobi.pro/market/history/kline"
    try:
        r = requests.get(url, params={"symbol": symbol, "period": period, "size": size}, timeout=10)
        res = r.json()
        if "data" not in res or res.get("status") != "ok":
            print(f"获取 {symbol} {period} K线失败:", res)
            return None
        df = pd.DataFrame(res["data"]).sort_values("id")
        for col in ['open','high','low','close','vol']:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        print(f"请求 {symbol} K线异常:", e)
        return None

# ================== 信号计算 ==================
def calc_signal(df):
    close = df['close']
    ema5 = close.ewm(span=5, adjust=False).mean()
    ema10 = close.ewm(span=10, adjust=False).mean()
    latest = df.iloc[-1]

    # 简化信号：只看 EMA5 与 EMA10
    if ema5.iloc[-1] > ema10.iloc[-1]:
        return "做多", latest['close']
    elif ema5.iloc[-1] < ema10.iloc[-1]:
        return "做空", latest['close']
    else:
        return "观望", latest['close']

# ================== 主循环 ==================
last_signals = {coin: None for coin in coins}

while True:
    messages = []
    for coin in coins:
        df = get_kline_huobi(coin, period="60min", size=50)
        if df is None or df.empty:
            continue
        signal, price = calc_signal(df)
        # 打印调试信息
        print(f"{datetime.utcnow()} | {coin} 信号: {signal}, 当前价: {format_price(price)}")

        # 只有信号变化才发送 Telegram
        if signal != last_signals[coin]:
            last_signals[coin] = signal
            msg = f"📊 {coin.upper()} 当前信号: {signal}\n当前价格: {format_price(price)}"
            messages.append(msg)

    if messages:
        send_telegram_message("\n\n".join(messages))

    # 每分钟循环一次
    time.sleep(60)
