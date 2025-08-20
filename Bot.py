# Bot.py
# Complete trading signal & analysis bot for Railway / Termux / VPS
# Features:
# - Multi-exchange Kline (Huobi, Binance, Bybit)
# - Multi-period signals (60min, 4hour, 1day)
# - EMA/MACD/RSI/KDJ rules (kept from original)
# - Support/resistance stop-loss + ATR dynamic target (new)
# - News sentiment aggregation (RSS)
# - Large orderbook wall detection across exchanges
# - 15-minute polling, 1-hour push, 4-hour GPT summary (simulated by default)
# - Avoid duplicate "strong" alerts; push sudden GPT analysis on signal change
# - Friendly price formatting for tiny-value memecoins
# - Telegram push

import os
import time
import requests
import pandas as pd
import numpy as np
import ta
import feedparser
from datetime import datetime, timedelta

# ================== CONFIG (ENV) ==================
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")      # Telegram bot token
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")      # Telegram chat id
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Optional: if present, can enable actual OpenAI calls (not active by default)

# Polling and timing
POLL_INTERVAL = 900            # 15 minutes checking
PUSH_INTERVAL = 3600           # send main pushes every hour
GPT_SUMMARY_INTERVAL = 4 * 3600  # every 4 hours send GPT comprehensive analysis
WALL_THRESHOLD = 5_000_000     # 5M USD large wall threshold (adjustable)
ATR_MULTIPLIER = 1.5           # ATR multiplier to determine dynamic target distance (can tune)

# ================== SYMBOL LISTS ==================
main_coins = ["btcusdt","ethusdt","xrpusdt","bnbusdt","solusdt","dogeusdt","trxusdt","adausdt","ltcusdt","linkusdt"]
meme_coins = ["dogeusdt","shibusdt","pepeusdt","penguusdt","bonkusdt","trumpusdt","spkusdt","flokusdt"]
main_periods = ["60min","4hour","1day"]  # used as labels only

# ================== UTILITIES ==================
def safe_get(df, col):
    try:
        return df[col]
    except Exception:
        return None

def format_price(price: float) -> str:
    """根据币价范围自动决定小数位；对 None 返回 '-'"""
    try:
        if price is None or (isinstance(price, float) and np.isnan(price)):
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
    except Exception:
        return "-"

def compute_atr(df, period=14):
    """计算 ATR，返回最后一根 ATR 值；若失败返回 None"""
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
    except Exception:
        return None

# ================== NEWS SENTIMENT (simple RSS aggregated) ==================
NEWS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://www.theblock.co/rss",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
]

def get_btc_news_sentiment():
    """简单关键词打分：正/负/中性 -> 1/-1/0"""
    news_titles = []
    for feed in NEWS_FEEDS:
        try:
            d = feedparser.parse(feed)
            for e in d.entries[:4]:
                if hasattr(e, "title"):
                    news_titles.append(e.title.lower())
        except Exception:
            continue
    score = 0
    for t in news_titles:
        if any(k in t for k in ["drop", "falls", "fall", "decline", "bear", "sell-off", "delist", "lawsuit", "scam", "liquidation", "regulator"]):
            score -= 1
        if any(k in t for k in ["rise", "rally", "adoption", "record high", "institutional", "bull", "approval"]):
            score += 1
    if score > 0:
        return 1
    elif score < 0:
        return -1
    else:
        return 0

# ================== KLINE FETCHERS ==================
def get_kline_huobi(symbol, period="60min", size=200):
    url = "https://api.huobi.pro/market/history/kline"
    try:
        r = requests.get(url, params={"symbol": symbol, "period": period, "size": size}, timeout=15)
        j = r.json()
        if not j or "data" not in j:
            return None
        df = pd.DataFrame(j["data"]).sort_values("id")
        for col in ["open","high","low","close","vol"]:
            df[col] = df[col].astype(float)
        return df
    except Exception:
        return None

def get_kline_binance(symbol, period="1h", limit=200):
    url = "https://api.binance.com/api/v3/klines"
    try:
        r = requests.get(url, params={"symbol": symbol.upper(), "interval": period, "limit": limit}, timeout=15)
        j = r.json()
        if not isinstance(j, list):
            return None
        df = pd.DataFrame(j, columns=[
            "open_time","open","high","low","close","vol","close_time",
            "qav","num_trades","taker_base_vol","taker_quote_vol","ignore"])
        for col in ["open","high","low","close","vol"]:
            df[col] = df[col].astype(float)
        return df
    except Exception:
        return None

def get_kline_bybit(symbol, period="60", limit=200):
    url = "https://api.bybit.com/v2/public/kline/list"
    try:
        r = requests.get(url, params={"symbol": symbol.upper(), "interval": period, "limit": limit}, timeout=15)
        j = r.json()
        if not j or j.get("ret_code") not in (0, None):
            return None
        res = j.get("result") or j.get("result", [])
        if not res:
            return None
        df = pd.DataFrame(res)
        for c in ["open","high","low","close","volume"]:
            if c in df.columns:
                df[c] = df[c].astype(float)
        if "volume" in df.columns:
            df = df.rename(columns={"volume":"vol"})
        return df
    except Exception:
        return None

# ================== ORDERBOOK / WALLS ==================
def get_orderbook_binance(symbol):
    try:
        r = requests.get("https://api.binance.com/api/v3/depth", params={"symbol": symbol.upper(), "limit": 50}, timeout=10)
        j = r.json()
        bids = [(float(p[0]), float(p[1])) for p in j.get("bids", [])]
        asks = [(float(p[0]), float(p[1])) for p in j.get("asks", [])]
        return bids, asks
    except Exception:
        return [], []

def get_orderbook_huobi(symbol):
    try:
        r = requests.get("https://api.huobi.pro/market/depth", params={"symbol": symbol, "type": "step0"}, timeout=10)
        j = r.json()
        tick = j.get("tick") or {}
        bids = [(float(p[0]), float(p[1])) for p in tick.get("bids", [])]
        asks = [(float(p[0]), float(p[1])) for p in tick.get("asks", [])]
        return bids, asks
    except Exception:
        return [], []

def get_orderbook_bybit(symbol):
    try:
        r = requests.get("https://api.bybit.com/v5/market/orderbook", params={"category":"spot", "symbol":symbol.upper(), "limit":50}, timeout=10)
        j = r.json()
        if j.get("retCode") != 0:
            return [], []
        bids = [(float(p[0]), float(p[1])) for p in j["result"].get("b",[])]
        asks = [(float(p[0]), float(p[1])) for p in j["result"].get("a",[])]
        return bids, asks
    except Exception:
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
                try:
                    notional = price * qty
                except Exception:
                    continue
                if notional > threshold:
                    side = "买单墙" if (price, qty) in bids else "卖单墙"
                    messages.append(f"💎 {ex} {symbol.upper()} 出现{side} 价格:{format_price(price)} 数量:{qty:.2f} 总额:{notional/1e6:.2f}M")
        except Exception:
            continue
    return messages

# ================== SIGNAL CALCULATION (keep your original rules) ==================
def calc_signal(df):
    try:
        if len(df) > 1:
            df_work = df.copy()
            df_work = df_work.iloc[:-1].copy()
        else:
            df_work = df.copy()

        close = df_work["close"]
        high = df_work["high"]
        low = df_work["low"]

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

        long_signal = (ema5.iloc[-1] > ema10.iloc[-1] > ema30.iloc[-1]) and (macd_diff.iloc[-1] > 0) and (rsi.iloc[-1] < 70) and (j.iloc[-1] > d.iloc[-1])
        short_signal = (ema5.iloc[-1] < ema10.iloc[-1] < ema30.iloc[-1]) and (macd_diff.iloc[-1] < 0) and (rsi.iloc[-1] > 30) and (j.iloc[-1] < d.iloc[-1])

        entry = float(close.iloc[-1])

        if long_signal:
            return "做多", entry
        elif short_signal:
            return "做空", entry
        else:
            return None, entry
    except Exception:
        return None, None

def calc_stop_loss(df, signal, entry, lookback=10):
    try:
        support = float(df["low"].tail(lookback).min())
        resistance = float(df["high"].tail(lookback).max())
        if signal and "多" in signal:
            return support
        elif signal and "空" in signal:
            return resistance
        return None
    except Exception:
        return None

def calc_dynamic_target_by_atr(df, signal, entry):
    try:
        atr = compute_atr(df, period=14)
        if atr is None or entry is None:
            return None, None
        if "多" in signal:
            target = entry + ATR_MULTIPLIER * atr
            stop = max(calc_stop_loss(df, signal, entry), entry - ATR_MULTIPLIER * atr)
        else:
            target = entry - ATR_MULTIPLIER * atr
            stop = min(calc_stop_loss(df, signal, entry), entry + ATR_MULTIPLIER * atr)
        return float(target), float(stop)
    except Exception:
        return None, None

# ================== GPT SIMULATED ANALYSIS (detailed) ==================
def gpt_analysis(symbol, df, signal):
    try:
        closes = df["close"].tail(60).astype(float).tolist()
        highs = df["high"].tail(60).astype(float).tolist()
        lows = df["low"].tail(60).astype(float).tolist()
        if len(closes) < 10:
            return f"{symbol.upper()}: 数据不足，无法综合分析。"

        avg = sum(closes) / len(closes)
        support = min(closes[-20:])
        resistance = max(closes[-20:])
        atr = compute_atr(df.tail(30))
        volatility = (np.std(closes[-30:]) / (avg+1e-9)) * 100

        recent = closes[-5:]
        shape = "无明显形态"
        if recent[-1] > recent[-2] > recent[-3]:
            shape = "近期出现连续上涨，趋势偏多（可能形成 W 底）"
        elif recent[-1] < recent[-2] < recent[-3]:
            shape = "近期出现连续下跌，趋势偏空（可能形成 M 头）"

        last_open = df["open"].iloc[-2] if len(df) >= 2 else df["open"].iloc[-1]
        last_close = df["close"].iloc[-2] if len(df) >= 2 else df["close"].iloc[-1]
        last_high = df["high"].iloc[-2] if len(df) >= 2 else df["high"].iloc[-1]
        last_low = df["low"].iloc[-2] if len(df) >= 2 else df["low"].iloc[-1]
        pattern = ""
        if last_close < last_open and (last_open - last_close) > (last_high - last_low) * 0.6:
            pattern = "看跌吞没/长上影线"
        elif last_close > last_open and (last_close - last_open) > (last_high - last_low) * 0.6:
            pattern = "看涨吞没/长下影线"

        news_sent = get_btc_news_sentiment()
        news_txt = "新闻中性"
        if news_sent == 1:
            news_txt = "新闻偏多（可能利好市场）"
        elif news_sent == -1:
            news_txt = "新闻偏空（可能带来下压）"

        rec = f"{symbol.upper()} 综合分析：\n"
        rec += f"- 当前信号：{signal}\n"
        rec += f"- K线形态：{shape} {pattern}\n"
        rec += f"- 支撑：{format_price(support)}, 阻力：{format_price(resistance)}\n"
        rec += f"- 均价（50根）：{format_price(avg)}, ATR(14)：{format_price(atr) if atr else '-'}\n"
        rec += f"- 波动率(近30)：{volatility:.2f}%\n"
        rec += f"- 外部因子：{news_txt}\n"
        rec += "📌 建议：结合成交量与多周期确认；若在超卖区且 pattern 指示反转，可考虑逢低部分建仓；若新闻利空且多周期一致空头，避免逆势追多。\n"
        return rec
    except Exception as e:
        return f"GPT 分析失败: {e}"

# ================== TELEGRAM SENDER ==================
def send_telegram_message(message):
    if not TOKEN or not CHAT_ID:
        print("⚠️ Telegram 配置未设置，无法发送消息。")
        return False
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        data = {"chat_id": CHAT_ID, "text": message}
        r = requests.post(url, data=data, timeout=15)
        if r.status_code != 200:
            print("Telegram 返回:", r.status_code, r.text)
        else:
            print("✅ Telegram 已发送消息")
        return
