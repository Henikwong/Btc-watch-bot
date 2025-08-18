import os
import time
import requests
import pandas as pd
import ta
from datetime import datetime

# ================== Telegram ==================
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ================== 币种 ==================
main_coins = ["btcusdt","ethusdt","xrpusdt","bnbusdt","solusdt","dogeusdt","trxusdt","adausdt","ltcusdt","linkusdt"]
meme_coins = ["dogeusdt","shibusdt","pepeusdt","penguusdt","bonkusdt","trumpusdt","spkusdt","flokusdt"]

main_periods = ["60min","4hour","1day"]
MODE = "新闻主导"  # 可选 ["普通","新闻主导","突破放量","时间窗口","汇总表格"]

# MEME 币时间窗口
MEME_START = 0  # UTC 小时
MEME_END = 8

# ================== 工具函数 ==================
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
        print(f"获取K线失败: {e}")
        return None

def calc_signal(df):
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["vol"]

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
    volume = vol.iloc[-1]

    long_signal = (ema5.iloc[-1] > ema10.iloc[-1] > ema30.iloc[-1]) and (macd_diff.iloc[-1] > 0) and (rsi.iloc[-1] < 70) and (j.iloc[-1] > d.iloc[-1])
    short_signal = (ema5.iloc[-1] < ema10.iloc[-1] < ema30.iloc[-1]) and (macd_diff.iloc[-1] < 0) and (rsi.iloc[-1] > 30) and (j.iloc[-1] < d.iloc[-1])

    # 突破放量模式
    if MODE == "突破放量":
        avg_vol = vol.tail(20).mean()
        if long_signal and volume > avg_vol * 1.5:
            return "突破放量做多", entry
        elif short_signal and volume > avg_vol * 1.5:
            return "跌破放量做空", entry
        else:
            return None, entry

    if long_signal:
        return "做多", entry
    elif short_signal:
        return "做空", entry
    else:
        return None, entry

def calc_stop_loss(df, signal, entry, lookback=10):
    support = df["low"].tail(lookback).min()
    resistance = df["high"].tail(lookback).max()
    if signal and "多" in signal:
        return support
    elif signal and "空" in signal:
        return resistance
    return None

# 新闻情绪分析示例（真实使用需接 RSS/API）
def get_btc_news_sentiment():
    trusted_sources = ["coindesk.com", "cointelegraph.com", "theblock.co", "reuters.com", "bloomberg.com"]
    news = [
        {"title": "Bitcoin drops after SEC delay", "source": "reuters.com", "sentiment": -1},
        {"title": "BTC adoption grows in Asia", "source": "cointelegraph.com", "sentiment": 1},
    ]
    filtered = [n for n in news if any(src in n["source"] for src in trusted_sources)]
    if not filtered:
        return 0
    avg_sentiment = sum(n["sentiment"] for n in filtered) / len(filtered)
    if avg_sentiment > 0.2:
        return 1
    elif avg_sentiment < -0.2:
        return -1
    else:
        return 0

def send_telegram_message(message):
    if TOKEN and CHAT_ID:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": message}
        try:
            requests.post(url, data=data)
        except Exception as e:
            print(f"消息发送失败: {e}")

# ================== 主循环 ==================
while True:
    main_msgs = []
    meme_msgs = []

    sentiment = get_btc_news_sentiment()
    hour_now = datetime.utcnow().hour
    allow_meme = (MODE != "时间窗口") or (MEME_START <= hour_now <= MEME_END)

    # 主流币
    for coin in main_coins:
        signals_by_period = {"60min": [], "4hour": [], "1day": []}
        for period in main_periods:
            df = get_kline(coin, period)
            if df is None or len(df) < 35:
                continue
            signal, entry = calc_signal(df)
            if signal:
                if sentiment == -1 and "多" in signal:
                    signal = None
                elif sentiment == -1:
                    signal = f"{signal} ⚠️ (新闻利空，可信度0.8)"
                elif sentiment == 1:
                    signal = f"{signal} ✅ (新闻利好加持)"
            if signal:
                target = entry * (1.01 if "多" in signal else 0.99)
                stop_loss = calc_stop_loss(df, signal, entry)
                signals_by_period[period].append(
                    f"{coin.upper()} {period}\n信号：{signal}\n入场价：{entry:.6f}\n目标价：{target:.6f}\n止损价：{stop_loss:.6f}\n——"
                )

        coin_msg = []
        for p in ["60min","4hour","1day"]:
            if signals_by_period[p]:
                coin_msg.append(f"{p} 信号:\n" + "\n".join(signals_by_period[p]))
        if coin_msg:
            main_msgs.append(f"📊 {coin.upper()} 技术信号\n" + "\n".join(coin_msg) + "\n")

    # MEME 币
    if allow_meme:
        for coin in meme_coins:
            df = get_kline(coin, "60min")
            if df is None or len(df) < 35:
                continue
            signal, entry = calc_signal(df)
            if signal:
                if sentiment == -1 and "多" in signal:
                    signal = None
                elif sentiment == -1:
                    signal = f"{signal} ⚠️ (新闻利空，可信度0.8)"
                elif sentiment == 1:
                    signal = f"{signal} ✅ (新闻利好加持)"
            if signal:
                target = entry * (1.08 if "多" in signal else 0.92)
                stop_loss = calc_stop_loss(df, signal, entry)
                meme_msgs.append(
                    f"🔥 MEME 币 {coin.upper()} 出现信号！\n信号：{signal}\n入场价：{entry:.6f}\n目标价：{target:.6f}\n止损价：{stop_loss:.6f}"
                )

    # 推送
    if MODE == "汇总表格":
        if main_msgs or meme_msgs:
            table_msg = "📊 今日交易信号汇总\n\n" + "\n\n".join(main_msgs + meme_msgs)
            send_telegram_message(table_msg)
    else:
        if main_msgs:
            send_telegram_message("\n\n".join(main_msgs))
        if meme_msgs:
            send_telegram_message("\n\n".join(meme_msgs))

    time.sleep(3600)  # 每小时运行一次
