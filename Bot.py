import os
import time
import requests
import pandas as pd
import ta
import feedparser
from datetime import datetime

# ================== Telegram ==================
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ================== 币种 ==================
main_coins = ["btcusdt","ethusdt","xrpusdt","bnbusdt","solusdt","dogeusdt","trxusdt","adausdt","ltcusdt","linkusdt"]
meme_coins = ["dogeusdt","shibusdt","pepeusdt","penguusdt","bonkusdt","trumpusdt","spkusdt","flokusdt"]

main_periods = ["60min","4hour","1day"]
MODE = "新闻主导"  

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
        print(f"❌ 获取K线失败: {e}")
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

# ================== 新闻情绪 ==================
def get_btc_news_sentiment():
    feeds = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://www.theblock.co/rss",
        "https://www.reuters.com/markets/cryptocurrency/rss/",
        "https://www.bloomberg.com/feed/podcast/etf-report.xml"
    ]
    sentiments = []
    for url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                title = entry.title.lower()
                if "drop" in title or "fall" in title or "delay" in title:
                    sentiments.append(-1)
                elif "rise" in title or "adoption" in title or "growth" in title:
                    sentiments.append(1)
        except Exception as e:
            print(f"❌ 新闻获取失败 {url}: {e}")
    if not sentiments:
        return 0
    avg_sentiment = sum(sentiments)/len(sentiments)
    if avg_sentiment > 0.2: return 1
    elif avg_sentiment < -0.2: return -1
    else: return 0

# ================== 挂单监控 ==================
def get_orderbook(symbol, exchange="binance"):
    try:
        if exchange == "binance":
            url = f"https://api.binance.com/api/v3/depth?symbol={symbol.upper()}&limit=50"
            data = requests.get(url).json()
            return data
        elif exchange == "okx":
            url = f"https://www.okx.com/api/v5/market/books?instId={symbol.upper()}&sz=50"
            data = requests.get(url).json()
            return data
        elif exchange == "bybit":
            url = f"https://api.bybit.com/v2/public/orderBook/L2?symbol={symbol.upper()}"
            data = requests.get(url).json()
            return data
        elif exchange == "bitget":
            url = f"https://api.bitget.com/api/spot/v1/market/depth?symbol={symbol.upper()}&limit=50"
            data = requests.get(url).json()
            return data
        elif exchange == "huobi":
            url = f"https://api.huobi.pro/market/depth?symbol={symbol.lower()}&type=step0"
            data = requests.get(url).json()
            return data
    except Exception as e:
        print(f"❌ 获取挂单失败 {exchange} {symbol}: {e}")
    return None

def detect_large_orders(symbol, threshold_usdt=5_000_000):
    exchanges = ["binance","okx","bybit","bitget","huobi"]
    alerts = []
    for ex in exchanges:
        data = get_orderbook(symbol, ex)
        if not data: continue
        try:
            if ex == "binance":
                bids = [(float(p), float(q)) for p,q in data["bids"]]
                asks = [(float(p), float(q)) for p,q in data["asks"]]
            elif ex == "okx":
                bids = [(float(p[0]), float(p[1])) for p in data["data"][0]["bids"]]
                asks = [(float(p[0]), float(p[1])) for p in data["data"][0]["asks"]]
            elif ex == "bybit":
                bids = [(float(p["price"]), float(p["size"])) for p in data["result"] if p["side"]=="Buy"]
                asks = [(float(p["price"]), float(p["size"])) for p in data["result"] if p["side"]=="Sell"]
            elif ex == "bitget":
                bids = [(float(p[0]), float(p[1])) for p in data["data"]["bids"]]
                asks = [(float(p[0]), float(p[1])) for p in data["data"]["asks"]]
            elif ex == "huobi":
                bids = [(float(p[0]), float(p[1])) for p in data["tick"]["bids"]]
                asks = [(float(p[0]), float(p[1])) for p in data["tick"]["asks"]]
            for price, qty in bids+asks:
                notional = price * qty
                if notional >= threshold_usdt:
                    alerts.append(f"🚨 {ex.upper()} {symbol.upper()} 大额挂单 {notional/1e6:.2f}M USDT @ {price}")
        except Exception as e:
            print(f"⚠️ 挂单解析失败 {ex} {symbol}: {e}")
    return alerts

# ================== Telegram ==================
def send_telegram_message(message):
    if TOKEN and CHAT_ID:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": message}
        try:
            requests.post(url, data=data)
            print(f"📨 已推送到 Telegram: {message[:50]}...")
        except Exception as e:
            print(f"❌ 消息发送失败: {e}")

# ================== 主循环 ==================
print("✅ Bot 启动成功，开始运行...")

while True:
    main_msgs, meme_msgs, order_msgs = [], [], []

    sentiment = get_btc_news_sentiment()
    hour_now = datetime.utcnow().hour
    allow_meme = (MODE != "时间窗口") or (MEME_START <= hour_now <= MEME_END)

    # 主流币
    for coin in main_coins:
        signals_by_period = {}
        for period in main_periods:
            df = get_kline(coin, period)
            if df is None or len(df) < 35: continue
            signal, entry = calc_signal(df)
            if signal:
                if sentiment == -1 and "多" in signal:
                    signal = None
                elif sentiment == -1:
                    signal = f"{signal} ⚠️ (新闻利空)"
                elif sentiment == 1:
                    signal = f"{signal} ✅ (新闻利好)"
            if signal:
                target = entry * (1.01 if "多" in signal else 0.99)
                stop_loss = calc_stop_loss(df, signal, entry)
                signals_by_period[period] = f"{coin.upper()} {period}\n信号：{signal}\n入场价：{entry:.2f}\n目标价：{target:.2f}\n止损价：{stop_loss:.2f}\n——"
        if signals_by_period:
            msg = "📊 "+coin.upper()+" 技术信号\n" + "\n".join(signals_by_period.values())
            main_msgs.append(msg)

        # 检查挂单
        alerts = detect_large_orders(coin)
        if alerts:
            order_msgs.extend(alerts)

    # MEME 币
    if allow_meme:
        for coin in meme_coins:
            for period in ["60min","4hour"]:
                df = get_kline(coin, period)
                if df is None or len(df) < 35: continue
                signal, entry = calc_signal(df)
                if signal:
                    if sentiment == -1 and "多" in signal:
                        signal = None
                    elif sentiment == -1:
                        signal = f"{signal} ⚠️ (新闻利空)"
                    elif sentiment == 1:
                        signal = f"{signal} ✅ (新闻利好)"
                if signal:
                    target = entry * (1.08 if "多" in signal else 0.92)
                    stop_loss = calc_stop_loss(df, signal, entry)
                    meme_msgs.append(f"🔥 MEME {coin.upper()} {period}\n信号：{signal}\n入场价：{entry:.2f}\n目标价：{target:.2f}\n止损价：{stop_loss:.2f}")

            # 检查挂单
            alerts = detect_large_orders(coin, threshold_usdt=2_000_000)
            if alerts:
                order_msgs.extend(alerts)

    # 推送
    if main_msgs: send_telegram_message("\n\n".join(main_msgs))
    if meme_msgs: send_telegram_message("\n\n".join(meme_msgs))
    if order_msgs: send_telegram_message("\n".join(order_msgs))

    print("⏳ 等待 1 小时后再次运行...")
    time.sleep(3600)
