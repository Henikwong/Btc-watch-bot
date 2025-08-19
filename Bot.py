import os
import requests
import pandas as pd
import ta

# ================== Telegram ==================
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message):
    if TOKEN and CHAT_ID:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": message}
        try:
            requests.post(url, data=data)
            print("✅ 已发送信号到 Telegram")
        except Exception as e:
            print(f"消息发送失败: {e}")

# ================== K线获取 ==================
def get_kline(symbol, period="60min", size=120):
    url = "https://api.huobi.pro/market/history/kline"
    try:
        r = requests.get(url, params={"symbol": symbol, "period": period, "size": size})
        data = r.json()
        if "data" not in data:
            return None
        df = pd.DataFrame(data["data"])
        df = df.sort_values("id")
        for col in ["open","high","low","close","vol"]:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        print(f"获取K线失败 {symbol} {period}: {e}")
        return None

# ================== 信号计算 ==================
def calc_signal(df):
    close = df["close"]
    high = df["high"]
    low = df["low"]

    ema5 = close.ewm(span=5).mean()
    ema10 = close.ewm(span=10).mean()
    ema30 = close.ewm(span=30).mean()
    macd = ta.trend.MACD(close)
    macd_diff = macd.macd_diff()
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=9, smooth_window=3)
    k = stoch.stoch()
    d = stoch.stoch_signal()
    j = 3*k - 2*d

    entry = close.iloc[-1]

    long_signal = (ema5.iloc[-1] > ema10.iloc[-1] > ema30.iloc[-1]) and (macd_diff.iloc[-1] > 0) and (rsi.iloc[-1] < 70) and (j.iloc[-1] > d.iloc[-1])
    short_signal = (ema5.iloc[-1] < ema10.iloc[-1] < ema30.iloc[-1]) and (macd_diff.iloc[-1] < 0) and (rsi.iloc[-1] > 30) and (j.iloc[-1] < d.iloc[-1])

    if long_signal:
        return "做多", entry
    elif short_signal:
        return "做空", entry
    else:
        return None, entry

# ================== 主程序（只跑一次） ==================
main_coins = ["btcusdt","ethusdt","bnbusdt","solusdt"]

msgs = []
for coin in main_coins:
    df = get_kline(coin, "60min")
    if df is None or len(df) < 35:
        continue
    signal, entry = calc_signal(df)
    if signal:
        stop_loss = df["low"].tail(10).min() if "多" in signal else df["high"].tail(10).max()
        target = entry * (1.01 if "多" in signal else 0.99)
        msgs.append(f"⏰ 测试 1h {coin.upper()}\n信号:{signal}\n入场:{entry:.2f}\n目标:{target:.2f}\n止损:{stop_loss:.2f}")

if msgs:
    send_telegram_message("\n\n".join(msgs))
else:
    send_telegram_message("⏰ 测试完成：当前没有 1h 信号")
