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
        print(f"获取K线失败 {symbol} {period}: {e}")
        return None

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
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    ]
    news = []
    for f in feeds:
        try:
            d = feedparser.parse(f)
            for entry in d.entries[:3]:
                news.append(entry.title.lower())
        except Exception as e:
            print(f"新闻源读取失败 {f}: {e}")

    score = 0
    for n in news:
        if "drop" in n or "fall" in n or "bear" in n:
            score -= 1
        if "rise" in n or "adoption" in n or "bull" in n:
            score += 1

    return 1 if score > 0 else -1 if score < 0 else 0

# ================== 各交易所挂单 ==================
def get_orderbook_binance(symbol):
    url = f"https://api.binance.com/api/v3/depth"
    try:
        r = requests.get(url, params={"symbol": symbol.upper(), "limit": 50})
        data = r.json()
        bids = [(float(p), float(q)) for p,q in data["bids"]]
        asks = [(float(p), float(q)) for p,q in data["asks"]]
        return bids, asks
    except Exception as e:
        print(f"❌ Binance挂单获取失败 {symbol}: {e}")
        return [], []

def get_orderbook_huobi(symbol):
    url = f"https://api.huobi.pro/market/depth"
    try:
        r = requests.get(url, params={"symbol": symbol, "type": "step0"})
        data = r.json()
        bids = [(float(p), float(q)) for p,q in data["tick"]["bids"][:50]]
        asks = [(float(p), float(q)) for p,q in data["tick"]["asks"][:50]]
        return bids, asks
    except Exception as e:
        print(f"❌ Huobi挂单获取失败 {symbol}: {e}")
        return [], []

def get_orderbook_bybit(symbol):
    url = "https://api.bybit.com/v5/market/orderbook"
    try:
        r = requests.get(url, params={"category":"spot", "symbol":symbol.upper(), "limit":50})
        j = r.json()
        if j.get("retCode") != 0:
            return [], []
        bids = [(float(p), float(q)) for p,q in j["result"]["b"]]
        asks = [(float(p), float(q)) for p,q in j["result"]["a"]]
        return bids, asks
    except Exception as e:
        print(f"❌ Bybit挂单获取失败 {symbol}: {e}")
        return [], []

def get_orderbook_okx(symbol):
    url = "https://www.okx.com/api/v5/market/books"
    try:
        r = requests.get(url, params={"instId":symbol.upper().replace("USDT","-USDT"), "sz":50})
        j = r.json()
        if j.get("code") != "0":
            return [], []
        data = j.get("data", [])[0]
        bids = [(float(p[0]), float(p[1])) for p in data.get("bids", [])]
        asks = [(float(p[0]), float(p[1])) for p in data.get("asks", [])]
        return bids, asks
    except Exception as e:
        print(f"❌ OKX挂单获取失败 {symbol}: {e}")
        return [], []

def get_orderbook_coinbase(symbol):
    url = f"https://api.exchange.coinbase.com/products/{symbol.replace('usdt','-usd')}/book"
    try:
        r = requests.get(url, params={"level":2})
        j = r.json()
        if "bids" not in j or "asks" not in j:
            return [], []
        bids = [(float(p[0]), float(p[1])) for p in j["bids"]]
        asks = [(float(p[0]), float(p[1])) for p in j["asks"]]
        return bids, asks
    except Exception as e:
        print(f"❌ Coinbase挂单获取失败 {symbol}: {e}")
        return [], []

# ================== 观察挂单 ==================
def check_large_walls(symbol, threshold=5_000_000):
    messages = []
    exch_funcs = [
        ("Binance", get_orderbook_binance, symbol),
        ("Huobi", get_orderbook_huobi, symbol),
        ("Bybit", get_orderbook_bybit, symbol),
        ("OKX", get_orderbook_okx, symbol),
        ("Coinbase", get_orderbook_coinbase, symbol),
    ]
    for ex, func, sym in exch_funcs:
        bids, asks = func(sym)
        if not bids and not asks:
            continue
        for price, qty in bids + asks:
            notional = price * qty
            if notional > threshold:
                side = "买单墙" if (price, qty) in bids else "卖单墙"
                messages.append(f"💎 {ex} {symbol.upper()} 出现{side}\n价格: {price} 数量: {qty:.2f} 总额: {notional/1e6:.2f}M USDT")
    return messages

# ================== 推送 ==================
def send_telegram_message(message):
    if TOKEN and CHAT_ID:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": message}
        try:
            requests.post(url, data=data)
            print("✅ 已发送信号到 Telegram")
        except Exception as e:
            print(f"消息发送失败: {e}")

# ================== 主循环 ==================
while True:
    try:
        sentiment = get_btc_news_sentiment()
        print(f"📰 当前新闻情绪: {sentiment}")

        main_msgs, meme_msgs = [], []

        # 主流币
        for coin in main_coins:
            for period in main_periods:
                df = get_kline(coin, period)
                if df is None or len(df) < 35:
                    continue
                signal, entry = calc_signal(df)

                # 情绪修正
                if signal:
                    if sentiment == -1 and "多" in signal:
                        signal = None
                    elif sentiment == -1:
                        signal += " ⚠️ (新闻利空)"
                    elif sentiment == 1:
                        signal += " ✅ (新闻利好)"

                if signal:
                    stop_loss = calc_stop_loss(df, signal, entry)
                    target = entry * (1.01 if "多" in signal else 0.99)

                    # === 过滤 <1% 的信号 ===
                    if abs(target - entry) / entry < 0.01:
                        continue

                    # === 只推送 4h 和 1d ===
                    if period in ["4hour", "1day"]:
                        main_msgs.append(f"📊 {coin.upper()} {period}\n信号:{signal}\n入场:{entry:.2f}\n目标:{target:.2f}\n止损:{stop_loss:.2f}")

            # 观察挂单
            walls = check_large_walls(coin)
            if walls:
                main_msgs.extend(walls)

        # MEME 币
        for coin in meme_coins:
            for period in ["4hour"]:  # 只保留4h
                df = get_kline(coin, period)
                if df is None or len(df) < 35:
                    continue
                signal, entry = calc_signal(df)
                if signal:
                    stop_loss = calc_stop_loss(df, signal, entry)
                    target = entry * (1.08 if "多" in signal else 0.92)
                    if abs(target - entry) / entry >= 0.01:
                        meme_msgs.append(f"🔥 MEME {coin.upper()} {period}\n信号:{signal}\n入场:{entry:.6f}\n目标:{target:.6f}\n止损:{stop_loss:.6f}")

            walls = check_large_walls(coin)
            if walls:
                meme_msgs.extend(walls)

        if main_msgs:
            send_telegram_message("\n\n".join(main_msgs))
        if meme_msgs:
            send_telegram_message("\n\n".join(meme_msgs))

    except Exception as e:
        print(f"循环错误: {e}")

    time.sleep(3600)
