import os
import time
import requests
import pandas as pd
import ta  # pip install ta

# Telegram 变量
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# 币种
main_coins = ["btcusdt","ethusdt","xrpusdt","bnbusdt","solusdt","dogeusdt","trxusdt","adausdt","ltcusdt","linkusdt"]
meme_coins = ["dogeusdt","shibusdt","pepeusdt","penguusdt","bonkusdt","trumpusdt","spkusdt","flokusdt"]

# 周期
main_periods = ["60min","4hour","1day"]

# 获取 K 线数据
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
        print("获取K线失败:", e)
        return None

# 技术指标 & 信号
def calc_signal(df):
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # EMA
    ema5 = close.ewm(span=5).mean()
    ema10 = close.ewm(span=10).mean()
    ema30 = close.ewm(span=30).mean()

    # MACD
    macd = ta.trend.MACD(close)
    macd_diff = macd.macd_diff()

    # RSI
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi()

    # KDJ
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

# 支撑 / 阻力止损计算
def calc_stop_loss(df, signal, entry, lookback=10):
    support = df["low"].tail(lookback).min()
    resistance = df["high"].tail(lookback).max()
    if signal == "做多":
        return support  # 多单止损在支撑位
    elif signal == "做空":
        return resistance  # 空单止损在阻力位
    return None

# 获取新闻情绪
def get_news_sentiment(symbol):
    coin = symbol.replace("usdt", "").upper()
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token=demo&currencies={coin}"
    try:
        r = requests.get(url)
        data = r.json()
        if "results" not in data:
            return "中性"

        # 只取最新3条新闻
        headlines = [item["title"] for item in data["results"][:3]]
        text = " ".join(headlines).lower()

        # 简单情绪规则
        positive_words = ["bullish","surge","up","positive","rise","gain","partnership","adopt"]
        negative_words = ["bearish","down","drop","fall","loss","hack","ban","scam"]

        pos = sum(word in text for word in positive_words)
        neg = sum(word in text for word in negative_words)

        if pos > neg:
            return "利好"
        elif neg > pos:
            return "利空"
        else:
            return "中性"
    except Exception as e:
        print("获取新闻失败:", e)
        return "中性"

# Telegram 消息
def send_telegram_message(message):
    if TOKEN and CHAT_ID:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": message}
        try:
            requests.post(url, data=data)
        except Exception as e:
            print("消息发送失败:", e)

# 主循环
while True:
    main_msgs = []
    meme_msgs = []

    # -------- 主流币 --------
    for coin in main_coins:
        signals_by_period = {"60min": [], "4hour": [], "1day": []}
        news_sentiment = get_news_sentiment(coin)

        for period in main_periods:
            df = get_kline(coin, period)
            if df is None or len(df) < 35:
                continue
            signal, entry = calc_signal(df)
            if signal:
                if period == "60min":
                    target = entry * (1.01 if signal=="做多" else 0.99)
                elif period == "4hour":
                    target = entry * (1.02 if signal=="做多" else 0.98)
                else:
                    target = entry * (1.03 if signal=="做多" else 0.97)

                stop_loss = calc_stop_loss(df, signal, entry)

                signals_by_period[period].append(
                    f"{coin.upper()} {period}\n信号：{signal}\n入场价：{entry:.6f}\n目标价：{target:.6f}\n止损价：{stop_loss:.6f}\n新闻情绪：{news_sentiment}\n——"
                )

        coin_msg = []
        if signals_by_period["60min"]:
            coin_msg.append("⏱ 1H 信号\n" + "\n".join(signals_by_period["60min"]))
        if signals_by_period["4hour"]:
            coin_msg.append("⏰ 4H 信号\n" + "\n".join(signals_by_period["4hour"]))
        if signals_by_period["1day"]:
            coin_msg.append("📅 1D 信号\n" + "\n".join(signals_by_period["1day"]))
        if coin_msg:
            main_msgs.append(f"📊 {coin.upper()} 技术 + 新闻信号\n" + "\n".join(coin_msg) + "\n")

    # -------- MEME 币 --------
    for coin in meme_coins:
        df = get_kline(coin, "60min")
        if df is None or len(df) < 35:
            continue
        signal, entry = calc_signal(df)
        news_sentiment = get_news_sentiment(coin)
        if signal:
            target = entry * (1.08 if signal=="做多" else 0.92)
            stop_loss = calc_stop_loss(df, signal, entry)
            meme_msgs.append(
                f"🔥 MEME 币 {coin.upper()} 出现信号！\n信号：{signal}\n入场价：{entry:.6f}\n目标价：{target:.6f}\n止损价：{stop_loss:.6f}\n新闻情绪：{news_sentiment}"
            )

    # -------- 推送 --------
    if main_msgs:
        send_telegram_message("\n\n".join(main_msgs))
    if meme_msgs:
        send_telegram_message("\n\n".join(meme_msgs))

    time.sleep(3600)  # 每小时运行一次
