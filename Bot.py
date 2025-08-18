import os
import time
import requests
import pandas as pd
import ta
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# ================== Telegram ==================
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ================== 币种 ==================
main_coins = ["btcusdt","ethusdt","xrpusdt","bnbusdt","solusdt","dogeusdt","trxusdt","adausdt","ltcusdt","linkusdt"]
meme_coins = ["dogeusdt","shibusdt","pepeusdt","penguusdt","bonkusdt","trumpusdt","spkusdt","flokusdt"]

main_periods = ["60min","4hour","1day"]
meme_periods = ["60min","4hour"]

MODE = "新闻主导"  # 可选 ["普通","新闻主导","突破放量","时间窗口","汇总表格"]

# MEME 币时间窗口
MEME_START = 0  # UTC 小时
MEME_END = 8

# ================== 工具函数 ==================
def get_kline(symbol, period="60min", size=120):
    url = "https://api.huobi.pro/market/history/kline"
    try:
        r = requests.get(url, params={"symbol": symbol, "period": period, "size": size}, timeout=10)
        data = r.json()
        if "data" not in data:
            return None
        df = pd.DataFrame(data["data"])
        df = df.sort_values("id")
        for col in ["open","high","low","close","vol"]:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        print(f"获取K线失败 {symbol}: {e}")
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

# ================== 新闻情绪 ==================
def fetch_rss_news(url, limit=10):
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.content, features="xml")
        items = soup.findAll("item")[:limit]
        news = [{"title": i.title.text, "link": i.link.text} for i in items]
        return news
    except Exception as e:
        print(f"抓取新闻失败 {url}: {e}")
        return []

def sentiment_score(text):
    text = text.lower()
    positives = ["surge","bull","rise","gain","adoption","positive","approval"]
    negatives = ["drop","fall","crash","ban","delay","lawsuit","negative","reject"]
    score = 0
    for w in positives:
        if w in text: score += 1
    for w in negatives:
        if w in text: score -= 1
    return score

def get_btc_news_sentiment():
    rss_feeds = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://www.reuters.com/finance/markets/rss",
        "https://www.bloomberg.com/feed/podcast/etf-report.xml"
    ]
    news = []
    for feed in rss_feeds:
        news.extend(fetch_rss_news(feed, limit=5))

    if not news:
        return 0

    total_score = sum(sentiment_score(n["title"]) for n in news)
    avg = total_score / len(news)

    if avg > 0.2:
        return 1
    elif avg < -0.2:
        return -1
    else:
        return 0

# ================== 订单簿挂单观察 ==================
def get_orderbook(exchange, symbol="BTCUSDT", depth=50):
    try:
        if exchange == "binance":
            url = f"https://api.binance.com/api/v3/depth?symbol={symbol.upper()}&limit={depth}"
            data = requests.get(url, timeout=10).json()
            return data["bids"], data["asks"]

        elif exchange == "bybit":
            url = f"https://api.bybit.com/v5/market/orderbook?category=spot&symbol={symbol.upper()}&limit={depth}"
            data = requests.get(url, timeout=10).json()
            return data["result"]["b"], data["result"]["a"]

        elif exchange == "okx":
            url = f"https://www.okx.com/api/v5/market/books?instId={symbol.upper()}&sz={depth}"
            data = requests.get(url, timeout=10).json()
            return data["data"][0]["bids"], data["data"][0]["asks"]

        elif exchange == "bitget":
            url = f"https://api.bitget.com/api/spot/v1/market/depth?symbol={symbol.upper()}&limit={depth}"
            data = requests.get(url, timeout=10).json()
            return data["data"]["bids"], data["data"]["asks"]

        elif exchange == "huobi":
            url = f"https://api.huobi.pro/market/depth?symbol={symbol.lower()}&type=step0"
            data = requests.get(url, timeout=10).json()
            return data["tick"]["bids"], data["tick"]["asks"]

    except Exception as e:
        print(f"订单簿获取失败 {exchange}: {e}")
    return [], []

def detect_large_orders(symbol="BTCUSDT", threshold_usdt=5_000_000):
    alerts = []
    for ex in ["binance","bybit","okx","bitget","huobi"]:
        bids, asks = get_orderbook(ex, symbol)
        for px, qty in bids:
            try:
                px = float(px); qty = float(qty)
                notional = px * qty
                if notional > threshold_usdt:
                    alerts.append(f"{ex.upper()} 买单 {qty:.2f} @ {px} ≈ ${notional/1e6:.2f}M")
            except: pass
        for px, qty in asks:
            try:
                px = float(px); qty = float(qty)
                notional = px * qty
                if notional > threshold_usdt:
                    alerts.append(f"{ex.upper()} 卖单 {qty:.2f} @ {px} ≈ ${notional/1e6:.2f}M")
            except: pass
    return alerts

# ================== Telegram ==================
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
        signals_by_period = {p: [] for p in main_periods}
        for period in main_periods:
            df = get_kline(coin, period)
            if df is None or len(df) < 35:
                continue
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
                signals_by_period[period].append(
                    f"{coin.upper()} {period}\n信号：{signal}\n入场价：{entry:.2f}\n目标价：{target:.2f}\n止损价：{stop_loss:.2f}\n——"
                )

        coin_msg = []
        for p in main_periods:
            if signals_by_period[p]:
                coin_msg.append(f"{p} 信号:\n" + "\n".join(signals_by_period[p]))
        if coin_msg:
            main_msgs.append(f"📊 {coin.upper()} 技术信号\n" + "\n".join(coin_msg) + "\n")

    # MEME 币
    if allow_meme:
        for coin in meme_coins:
            signals_by_period = {p: [] for p in meme_periods}
            for period in meme_periods:
                df = get_kline(coin, period)
                if df is None or len(df) < 35:
                    continue
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
                    signals_by_period[period].append(
                        f"{coin.upper()} {period}\n信号：{signal}\n入场价：{entry:.6f}\n目标价：{target:.6f}\n止损价：{stop_loss:.6f}\n——"
                    )

            coin_msg = []
            for p in meme_periods:
                if signals_by_period[p]:
                    coin_msg.append(f"{p} 信号:\n" + "\n".join(signals_by_period[p]))
            if coin_msg:
                meme_msgs.append(f"🔥 MEME 币 {coin.upper()} 信号\n" + "\n".join(coin_msg))

    # 大额挂单监控
    large_orders = detect_large_orders("BTCUSDT")
    if large_orders:
        main_msgs.append("🚨 大额挂单提醒:\n" + "\n".join(large_orders))

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
