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
meme_periods = ["60min","4hour"]

MODE = "新闻主导"  # 可选 ["普通","新闻主导","突破放量","时间窗口","汇总表格"]

MEME_START = 0
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
def get_btc_news_sentiment():
    feeds = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://www.theblock.co/rss",
    ]
    news = []
    for url in feeds:
        try:
            d = feedparser.parse(url)
            for entry in d.entries[:5]:
                title = entry.title.lower()
                if "bitcoin" in title or "btc" in title:
                    sentiment = 0
                    if any(word in title for word in ["rise","bull","surge","adoption","positive","growth"]):
                        sentiment = 1
                    elif any(word in title for word in ["fall","drop","bear","regulation","ban","negative","delay"]):
                        sentiment = -1
                    news.append(sentiment)
        except:
            continue
    if not news:
        return 0
    avg_sentiment = sum(news) / len(news)
    if avg_sentiment > 0.2:
        return 1
    elif avg_sentiment < -0.2:
        return -1
    else:
        return 0

# ================== 订单簿观察 ==================
def get_orderbook(symbol, limit=50):
    apis = {
        "binance": f"https://api.binance.com/api/v3/depth?symbol={symbol.upper()}&limit={limit}",
        "okx": f"https://www.okx.com/api/v5/market/books?instId={symbol.upper()}&sz={limit}",
        "bybit": f"https://api.bybit.com/v5/market/orderbook?category=spot&symbol={symbol.upper()}&limit={limit}",
        "bitget": f"https://api.bitget.com/api/spot/v1/market/depth?symbol={symbol.upper()}&limit={limit}",
        "huobi": f"https://api.huobi.pro/market/depth?symbol={symbol.lower()}&type=step0",
    }
    large_walls = []
    for ex, url in apis.items():
        try:
            r = requests.get(url, timeout=5)
            data = r.json()
            if "binance" in ex and "bids" in data:
                bids = [(float(p), float(q)) for p,q in data["bids"]]
                asks = [(float(p), float(q)) for p,q in data["asks"]]
            elif "okx" in ex and "data" in data:
                bids = [(float(p[0]), float(p[1])) for p in data["data"][0]["bids"]]
                asks = [(float(p[0]), float(p[1])) for p in data["data"][0]["asks"]]
            elif "bybit" in ex and "result" in data:
                bids = [(float(p[0]), float(p[1])) for p in data["result"]["b"]]
                asks = [(float(p[0]), float(p[1])) for p in data["result"]["a"]]
            elif "bitget" in ex and "data" in data:
                bids = [(float(p[0]), float(p[1])) for p in data["data"]["bids"]]
                asks = [(float(p[0]), float(p[1])) for p in data["data"]["asks"]]
            elif "huobi" in ex and "tick" in data:
                bids = [(float(data["tick"]["bids"][i]), float(data["tick"]["bids"][i+1])) for i in range(0, len(data["tick"]["bids"]), 2)]
                asks = [(float(data["tick"]["asks"][i]), float(data["tick"]["asks"][i+1])) for i in range(0, len(data["tick"]["asks"]), 2)]
            else:
                continue

            for price, qty in bids + asks:
                if qty > 5_000_000:  # 大挂单阈值
                    large_walls.append(f"{ex.upper()} {symbol.upper()} 大额挂单 {qty:.2f} @ {price}")
        except:
            continue
    return large_walls

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
    orderbook_msgs = []

    sentiment = get_btc_news_sentiment()
    hour_now = datetime.utcnow().hour
    allow_meme = (MODE != "时间窗口") or (MEME_START <= hour_now <= MEME_END)

    # 主流币
    for coin in main_coins:
        signals_by_period = {}
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
                signals_by_period[period] = f"{coin.upper()} {period}\n信号：{signal}\n入场价：{entry:.6f}\n目标价：{target:.6f}\n止损价：{stop_loss:.6f}\n——"

        if signals_by_period:
            msg = "📊 " + coin.upper() + " 技术信号\n" + "\n".join(signals_by_period.values()) + "\n"
            main_msgs.append(msg)

    # MEME 币
    if allow_meme:
        for coin in meme_coins:
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
                    meme_msgs.append(
                        f"🔥 MEME 币 {coin.upper()} {period} 出现信号！\n信号：{signal}\n入场价：{entry:.6f}\n目标价：{target:.6f}\n止损价：{stop_loss:.6f}"
                    )

    # 订单簿监控
    for coin in ["btcusdt"]:  # 只监控BTC，避免请求过多
        large_orders = get_orderbook(coin)
        if large_orders:
            orderbook_msgs.extend(large_orders)

    # 推送
    if MODE == "汇总表格":
        if main_msgs or meme_msgs or orderbook_msgs:
            table_msg = "📊 今日交易信号汇总\n\n" + "\n\n".join(main_msgs + meme_msgs + orderbook_msgs)
            send_telegram_message(table_msg)
    else:
        for block in [main_msgs, meme_msgs, orderbook_msgs]:
            if block:
                send_telegram_message("\n\n".join(block))

    time.sleep(3600)  # 每小时运行一次
