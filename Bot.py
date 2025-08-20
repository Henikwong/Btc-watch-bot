# Bot.py - 完整升级版交易信号 & 分析 Bot
import os
import time
import requests
import pandas as pd
import numpy as np
import ta
import feedparser
from datetime import datetime, timedelta

# ================== CONFIG (ENV) ==================
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # optional

# ================== TIMING ==================
POLL_INTERVAL = 900            # 15分钟轮询
PUSH_INTERVAL = 3600           # 每小时推送
GPT_SUMMARY_INTERVAL = 4*3600  # 每4小时GPT综合分析
WALL_THRESHOLD = 5_000_000     # 大单墙阈值 (USD)
ATR_MULTIPLIER = 1.5           # ATR乘数

# ================== COINS & PERIODS ==================
main_coins = ["btcusdt","ethusdt","xrpusdt","bnbusdt","solusdt","dogeusdt","trxusdt","adausdt","ltcusdt","linkusdt"]
meme_coins = ["dogeusdt","shibusdt","pepeusdt","penguusdt","bonkusdt","trumpusdt","spkusdt","flokusdt"]
main_periods = ["60min","4hour","1day"]

# ================== UTILITIES ==================
def format_price(price):
    try:
        if price is None or (isinstance(price,float) and np.isnan(price)):
            return "-"
        price = float(price)
        if price >= 100: return f"{price:.2f}"
        elif price >= 1: return f"{price:.4f}"
        elif price >= 0.01: return f"{price:.6f}"
        else: return f"{price:.8f}"
    except: return "-"

def compute_atr(df, period=14):
    try:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr = pd.concat([high - low,
                        (high - close.shift(1)).abs(),
                        (low - close.shift(1)).abs()], axis=1).max(axis=1)
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
    score = 0
    for feed in NEWS_FEEDS:
        try:
            d = feedparser.parse(feed)
            for e in d.entries[:4]:
                t = e.title.lower() if hasattr(e, "title") else ""
                if any(k in t for k in ["drop","falls","fall","decline","bear","sell-off","delist","lawsuit","scam","liquidation","regulator"]):
                    score -= 1
                if any(k in t for k in ["rise","rally","adoption","record high","institutional","bull","approval"]):
                    score += 1
        except: continue
    return 1 if score>0 else (-1 if score<0 else 0)

# ================== KLINE FETCHERS ==================
def get_kline_huobi(symbol, period="60min", size=200):
    try:
        r = requests.get("https://api.huobi.pro/market/history/kline",
                         params={"symbol":symbol,"period":period,"size":size},timeout=15)
        j = r.json()
        if not j or "data" not in j: return None
        df = pd.DataFrame(j["data"]).sort_values("id")
        for col in ["open","high","low","close","vol"]: df[col]=df[col].astype(float)
        return df
    except: return None

def get_kline_binance(symbol, period="1h", limit=200):
    try:
        r = requests.get("https://api.binance.com/api/v3/klines",
                         params={"symbol":symbol.upper(),"interval":period,"limit":limit},timeout=15)
        j = r.json()
        if not isinstance(j,list): return None
        df = pd.DataFrame(j, columns=[
            "open_time","open","high","low","close","vol","close_time",
            "qav","num_trades","taker_base_vol","taker_quote_vol","ignore"])
        for col in ["open","high","low","close","vol"]: df[col]=df[col].astype(float)
        return df
    except: return None

def get_kline_bybit(symbol, period="60", limit=200):
    try:
        r = requests.get("https://api.bybit.com/v2/public/kline/list",
                         params={"symbol":symbol.upper(),"interval":period,"limit":limit},timeout=15)
        j = r.json()
        if not j or j.get("ret_code") not in (0,None): return None
        res = j.get("result") or []
        if not res: return None
        df = pd.DataFrame(res)
        for c in ["open","high","low","close","volume"]:
            if c in df.columns: df[c] = df[c].astype(float)
        if "volume" in df.columns: df = df.rename(columns={"volume":"vol"})
        return df
    except: return None

# ================== ORDERBOOK & WALLS ==================
def get_orderbook_binance(symbol):
    try:
        r = requests.get("https://api.binance.com/api/v3/depth",
                         params={"symbol":symbol.upper(),"limit":50},timeout=10)
        j = r.json()
        bids = [(float(p[0]),float(p[1])) for p in j.get("bids",[])]
        asks = [(float(p[0]),float(p[1])) for p in j.get("asks",[])]
        return bids, asks
    except: return [],[]

def get_orderbook_huobi(symbol):
    try:
        r = requests.get("https://api.huobi.pro/market/depth",
                         params={"symbol":symbol,"type":"step0"},timeout=10)
        j = r.json()
        tick = j.get("tick") or {}
        bids = [(float(p[0]),float(p[1])) for p in tick.get("bids",[])]
        asks = [(float(p[0]),float(p[1])) for p in tick.get("asks",[])]
        return bids, asks
    except: return [],[]

def get_orderbook_bybit(symbol):
    try:
        r = requests.get("https://api.bybit.com/v5/market/orderbook",
                         params={"category":"spot","symbol":symbol.upper(),"limit":50},timeout=10)
        j = r.json()
        if j.get("retCode") !=0: return [],[]
        bids = [(float(p[0]),float(p[1])) for p in j["result"].get("b",[])]
        asks = [(float(p[0]),float(p[1])) for p in j["result"].get("a",[])]
        return bids, asks
    except: return [],[]

def check_large_walls(symbol):
    messages=[]
    exch_list = [("Binance",get_orderbook_binance),("Huobi",get_orderbook_huobi),("Bybit",get_orderbook_bybit)]
    for ex,func in exch_list:
        try:
            bids,asks = func(symbol)
            for price,qty in bids+asks:
                try: notional=price*qty
                except: continue
                if notional>WALL_THRESHOLD:
                    side="买单墙" if (price,qty) in bids else "卖单墙"
                    messages.append(f"💎 {ex} {symbol.upper()} 出现{side} 价格:{format_price(price)} 数量:{qty:.2f} 总额:{notional/1e6:.2f}M")
        except: continue
    return messages

# ================== SIGNAL ==================
def calc_signal(df):
    try:
        df_work = df.iloc[:-1].copy() if len(df)>1 else df.copy()
        close,high,low = df_work["close"],df_work["high"],df_work["low"]

        ema5 = close.ewm(span=5).mean()
        ema10 = close.ewm(span=10).mean()
        ema30 = close.ewm(span=30).mean()
        macd = ta.trend.MACD(close)
        macd_diff = macd.macd_diff()
        rsi = ta.momentum.RSIIndicator(close,window=14).rsi()
        stoch = ta.momentum.StochasticOscillator(high,low,close,window=9,smooth_window=3)
        k,d = stoch.stoch(), stoch.stoch_signal()
        j = 3*k - 2*d

        long_signal = (ema5.iloc[-1]>ema10.iloc[-1]>ema30.iloc[-1]) and (macd_diff.iloc[-1]>0) and (rsi.iloc[-1]<70) and (j.iloc[-1
