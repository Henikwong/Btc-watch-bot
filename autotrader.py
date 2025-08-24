# autotrader.py
# Binance USDM 自动交易：多周期策略 + ATR SL/TP + 跟踪止损 + 4h MACD 弱化部分止盈
import os
import time
from datetime import datetime

import requests
import ccxt
import pandas as pd
import ta
from dotenv import load_dotenv

load_dotenv()

# ========= ENV =========
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT  = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# 同时兼容两种写法
API_KEY    = (os.getenv("API_KEY") or os.getenv("BINANCE_API_KEY") or "").strip()
API_SECRET = (os.getenv("API_SECRET") or os.getenv("BINANCE_API_SECRET") or "").strip()

LEVERAGE       = int(os.getenv("LEVERAGE", "10"))
BASE_USDT      = float(os.getenv("BASE_USDT", "15"))       # 每次下单的名义保证金（USDT）
POLL_INTERVAL  = int(os.getenv("POLL_INTERVAL", "60"))     # 轮询秒数
LIVE_TRADE     = int(os.getenv("LIVE_TRADE", "0"))         # 1=实盘，0=纸面
TIMEFRAMES     = ["1h", "4h"]

TRADE_SYMBOLS   = [s.strip() for s in os.getenv("TRADE_SYMBOLS", "BTC/USDT,ETH/USDT").split(",") if s.strip()]
OBSERVE_SYMBOLS = [s.strip() for s in os.getenv("OBSERVE_SYMBOLS", "").split(",") if s.strip()]
ALL_SYMBOLS     = list(dict.fromkeys(TRADE_SYMBOLS + OBSERVE_SYMBOLS))  # 去重保序

# 风控参数（ATR倍数）
SL_ATR_MULT      = float(os.getenv("SL_ATR_MULT", "2.0"))
TP_ATR_MULT      = float(os.getenv("TP_ATR_MULT", "3.0"))
TRAIL_ATR_MULT   = float(os.getenv("TRAIL_ATR_MULT", "1.5"))
PARTIAL_TP_RATIO = float(os.getenv("PARTIAL_TP_RATIO", "0.3"))  # 部分止盈比例

# ========= utils =========
def nowstr():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"[{nowstr()}] {msg}", flush=True)

def tg_send(text):
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            data={"chat_id": TG_CHAT, "text": text},
            timeout=10,
        )
    except Exception as e:
        log(f"TG发送失败: {e}")

def fmt_price(p: float) -> str:
    p = float(p)
    if p >= 100: return f"{p:.2f}"
    if p >= 1:   return f"{p:.4f}"
    if p >= 0.01:return f"{p:.6f}"
    return f"{p:.8f}"

def df_from_ohlcv(ohlcv):
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","vol"])
    for c in ["open","high","low","close","vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ========= Exchange =========
def build_exchange():
    # 优先使用 binanceusdm；若不可用则回退到 binance+future 选项
    ex = None
    try:
        if hasattr(ccxt, "binanceusdm"):
            ex = ccxt.binanceusdm({
                "apiKey": API_KEY,
                "secret": API_SECRET,
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            })
            ex.load_markets()
            log("Binance USDM futures 初始化成功")
            return ex
    except Exception as e:
        log(f"binanceusdm 初始化失败：{e}")

    try:
        ex = ccxt.binance({
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        ex.load_markets()
        log("使用 ccxt.binance（期货）初始化成功")
        return ex
    except Exception as e:
        log(f"ccxt.binance 初始化失败: {e}")
        raise

def set_leverage_safe(ex, symbol, leverage):
    try:
        # ccxt 统一方法
        ex.set_leverage(int(leverage), symbol)
        log(f"{symbol} 杠杆设置为 {leverage}x（set_leverage）")
        return
    except Exception as e:
        log(f"{symbol} set_leverage 失败: {e}")

    # 如果统一方法失败，尝试其它方案
    try:
        m = ex.market(symbol)
        # 一些 ccxt 版本支持 custom endpoint，通过 ex.fapiPrivate 接口；若没有则会 AttributeError
        if hasattr(ex, "fapiPrivate_post_leverage"):
            ex.fapiPrivate_post_leverage({"symbol": m["id"], "leverage": int(leverage)})
            log(f"{symbol} 杠杆设置为 {leverage}x（fapiPrivate_post_leverage）")
            return
    except Exception as e:
        log(f"{symbol} fapiPrivate_post_leverage 失败: {e}")

    log(f"⚠️ {symbol} 杠杆未能设置成功（忽略继续运行）")

def amount_for_futures(ex, symbol, price):
    # 名义保证金 * 杠杆 / 价格 = 数量
    raw_qty = (BASE_USDT * LEVERAGE) / max(float(price), 1e-12)
    try:
        qty = ex.amount_to_precision(symbol, raw_qty)
    except Exception:
        qty = raw_qty
    try:
        return float(qty)
    except Exception:
        return float(raw_qty)

# ========= 指标 & 决策 =========
def analyze_one_df(df):
    # 使用已收盘的K线（去掉最后一根）
    if df is None or len(df) < 60:
        return None, None
    work = df.iloc[:-1].copy()
    close = work["close"]
    high  = work["high"]
    low   = work["low"]
    vol   = work["vol"]

    # EMA趋势
    ema5  = close.ewm(span=5).mean().iloc[-1]
    ema10 = close.ewm(span=10).mean().iloc[-1]
    ema30 = close.ewm(span=30).mean().iloc[-1]
    ema_trend = "多" if (ema5 > ema10 > ema30) else ("空" if (ema5 < ema10 < ema30) else "中性")
