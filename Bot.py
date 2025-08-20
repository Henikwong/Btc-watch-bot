# Bot.py - 完整升级版交易信号与分析脚本
import os
import time
import requests
import pandas as pd
import numpy as np
import ta
import feedparser
from datetime import datetime, timedelta

# ================== CONFIG ==================
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

POLL_INTERVAL = 900  # 15分钟轮询
PUSH_INTERVAL = 3600  # 每小时推送一次
GPT_SUMMARY_INTERVAL = 4 * 3600  # 每4小时 GPT 综合分析
WALL_THRESHOLD = 5_000_000  # 大单墙阈值
ATR_MULTIPLIER = 1.5  # ATR止盈倍数

# ================== SYMBOL LIST ==================
main_coins = ["btcusdt","ethusdt","xrpusdt","bnbusdt","solusdt","dogeusdt","trxusdt","adausdt","ltcusdt","linkusdt"]
meme_coins = ["dogeusdt","shibusdt","pepeusdt","penguusdt","bonkusdt","trumpusdt","spkusdt","flokusdt"]
main_periods = ["60min","4hour","1day"]

# ================== UTILITIES ==================
def format_price(price):
    try:
        if price is None or np.isnan(price):
            return "-"
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

def compute_atr(df, period=14):
    try:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr.iloc[-1]
    except:
        return None

# ================== NEWS SENTIMENT ==================
NEWS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://www.theblock.co/rss",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
]

def get_btc_news_sentiment():
    news_titles = []
    for feed in NEWS_FEEDS:
        try:
            d = feedparser.parse(feed)
            for e in d.entries[:4]:
                if hasattr(e, "title"):
                    news_titles.append(e.title.lower())
        except:
            continue
    score = 0
    for t in news_titles:
        if any(k in t for k in ["drop","falls","fall","decline","bear","sell-off","delist","lawsuit","scam","liquidation","regulator"]):
            score -= 1
        if any(k in t for k in ["rise","rally","adoption","record high","institutional","bull","approval"]):
            score += 1
    return 1 if score > 0 else -1 if score < 0 else 0

# ================== KLINE FETCHERS ==================
def get_kline_huobi(symbol, period="60min", size=200):
    try:
        r = requests.get("https://api.huobi.pro/market/history/kline",
                         params={"symbol": symbol, "period": period, "size": size}, timeout=15)
        j = r.json()
        if not j or "data" not in j:
            return None
        df = pd.DataFrame(j["data"]).sort_values("id")
        for col in ["open","high","low","close","vol"]:
            df[col] = df[col].astype(float)
        return df
    except:
        return None

def get_kline_binance(symbol, period="1h", limit=200):
    try:
        r = requests.get("https://api.binance.com/api/v3/klines",
                         params={"symbol": symbol.upper(), "interval": period, "limit": limit}, timeout=15)
        j = r.json()
        if not isinstance(j, list):
            return None
        df = pd.DataFrame(j, columns=[
            "open_time","open","high","low","close","vol","close_time",
            "qav","num_trades","taker_base_vol","taker_quote_vol","ignore"])
        for col in ["open","high","low","close","vol"]:
            df[col] = df[col].astype(float)
        return df
    except:
        return None

def get_kline_bybit(symbol, period="60", limit=200):
    try:
        r = requests.get("https://api.bybit.com/v2/public/kline/list",
                         params={"symbol": symbol.upper(), "interval": period, "limit": limit}, timeout=15)
        j = r.json()
        if not j or j.get("ret_code") not in (0, None):
            return None
        res = j.get("result") or []
        if not res:
            return None
        df = pd.DataFrame(res)
        for c in ["open","high","low","close","volume"]:
            if c in df.columns:
                df[c] = df[c].astype(float)
        if "volume" in df.columns:
            df = df.rename(columns={"volume":"vol"})
        return df
    except:
        return None

# ================== ORDERBOOK / WALLS ==================
def get_orderbook_binance(symbol):
    try:
        r = requests.get("https://api.binance.com/api/v3/depth",
                         params={"symbol": symbol.upper(), "limit": 50}, timeout=10)
        j = r.json()
        bids = [(float(p[0]), float(p[1])) for p in j.get("bids", [])]
        asks = [(float(p[0]), float(p[1])) for p in j.get("asks", [])]
        return bids, asks
    except:
        return [], []

def get_orderbook_huobi(symbol):
    try:
        r = requests.get("https://api.huobi.pro/market/depth",
                         params={"symbol": symbol, "type": "step0"}, timeout=10)
        j = r.json()
        tick = j.get("tick") or {}
        bids = [(float(p[0]), float(p[1])) for p in tick.get("bids", [])]
        asks = [(float(p[0]), float(p[1])) for p in tick.get("asks", [])]
        return bids, asks
    except:
        return [], []

def get_orderbook_bybit(symbol):
    try:
        r = requests.get("https://api.bybit.com/v5/market/orderbook",
                         params={"category":"spot", "symbol":symbol.upper(), "limit":50}, timeout=10)
        j = r.json()
        if j.get("retCode") != 0:
            return [], []
        bids = [(float(p[0]), float(p[1])) for p in j["result"].get("b",[])]
        asks = [(float(p[0]), float(p[1])) for p in j["result"].get("a",[])]
        return bids, asks
    except:
        return [], []

def check_large_walls(symbol, threshold=WALL_THRESHOLD):
    messages = []
    exch_list = [
        ("Binance", get_orderbook_binance),
        ("Huobi", get_orderbook_huobi),
        ("Bybit", get_orderbook_bybit),
    ]
    for ex, func in exch_list:
        try:
            bids, asks = func(symbol)
            for price, qty in bids + asks:
                notional = price * qty
                if notional > threshold:
                    side = "买单墙" if (price, qty) in bids else "卖单墙"
                    messages.append(f"💎 {ex} {symbol.upper()} 出现{side} 价格:{format_price(price)} 数量:{qty:.2f} 总额:{notional/1e6:.2f}M")
        except:
            continue
    return messages

# ================== SIGNAL CALCULATION ==================
def calc_signal(df):
    try:
        df_work = df.iloc[:-1] if len(df) > 1 else df.copy()
        close = df_work["close"]
        high = df_work["high"]
        low = df_work["low"]

        # EMA
        ema5 = close.ewm(span=5).mean()
        ema10 = close.ewm(span=10).mean()
        ema30 = close.ewm(span=30).mean()

        # MACD
        macd_diff = ta.trend.MACD(close).macd_diff()

        # RSI
        rsi = ta.momentum.RSIIndicator(close,14).rsi()

        # KDJ
        low_min = low.rolling(9).min()
        high_max = high.rolling(9).max()
        rsv = (close - low_min) / (high_max - low_min) * 100
        k = rsv.ewm(alpha=1/3).mean()
        d = k.ewm(alpha=1/3).mean()
        j = 3*k - 2*d

        long_signal = (ema5.iloc[-1] > ema10.iloc[-1] > ema30.iloc[-1]) \
                      and (macd_diff.iloc[-1] > 0) \
                      and (rsi.iloc[-1] < 70) \
                      and (j.iloc[-1] > j.iloc[-2])
        short_signal = (ema5.iloc[-1] < ema10.iloc[-1] < ema30.iloc[-1]) \
                       and (macd_diff.iloc[-1] < 0) \
                       and (rsi.iloc[-1] > 30) \
                       and (j.iloc[-1] < j.iloc[-2])

        return "做多" if long_signal else "做空" if short_signal else None, float(close.iloc[-1])
    except Exception as e:
        print("calc_signal error:", e)
        return None, None

# ================== STOP LOSS / TARGET ==================
def calc_stop_loss(df, signal, lookback=10):
    try:
        if signal == "做多":
            return float(df["low"].tail(lookback).min())
        elif signal == "做空":
            return float(df["high"].tail(lookback).max())
    except:
        return None

def calc_dynamic_target_by_atr(df, signal, entry):
    try:
        atr = compute_atr(df)
        if atr is None or entry is None:
            return None, None
        if signal == "做多":
            target = entry + ATR_MULTIPLIER * atr
            stop = max(calc_stop_loss(df, signal), entry - ATR_MULTIPLIER * atr)
        elif signal == "做空":
            target = entry - ATR_MULTIPLIER * atr
            stop = min(calc_stop_loss(df, signal), entry + ATR_MULTIPLIER * atr)
        else:
            return None, None
        return float(target), float(stop)
    except:
        return None, None

# ================== TELEGRAM SENDER ==================
def send_telegram_message(msg):
    try:
        if not TOKEN or not CHAT_ID:
            print("⚠️ Telegram TOKEN 或 CHAT_ID 未配置")
            return
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": msg, "parse_mode":"HTML"}
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        print("send_telegram_message error:", e)
