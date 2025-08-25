# autotrader.py
import os, time, math, traceback
import requests
import ccxt
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# ================== ENV ==================
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT  = os.getenv("TELEGRAM_CHAT_ID", "")

EXCHANGE_NAME = os.getenv("EXCHANGE", "binance").lower()
API_KEY   = os.getenv("API_KEY", "")
API_SECRET= os.getenv("API_SECRET", "")

MARKET_TYPE = os.getenv("MARKET_TYPE", "future").lower()  # spot/future
# 强制使用这 5 个币（你也可以改 .env 的 SYMBOLS，但脚本默认只用这 5）
SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "LTC/USDT", "DOGE/USDT"]

BASE_USDT = float(os.getenv("BASE_USDT", "15"))
LEVERAGE  = int(os.getenv("LEVERAGE", "10"))
ATR_MULT_INFO = float(os.getenv("RISK_ATR_MULT", "1.5"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "2.0"))
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "3.0"))

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))
LIVE_TRADE    = int(os.getenv("LIVE_TRADE", "0"))

REQUIRED_CONFIRMS = int(os.getenv("REQUIRED_CONFIRMS", "3"))
TIMEFRAMES = ["1h", "4h", "1d", "1w"]

# ================== 小工具 ==================
def nowstr(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def log(msg): print(f"[{nowstr()}] {msg}", flush=True)

def tg_send(text):
    """仅用于向 Telegram 发送消息（尽量少发）"""
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TG_CHAT, "text": text, "disable_web_page_preview": True}, timeout=10)
    except Exception as e:
        log(f"TG发送失败: {e}")

def format_price(p):
    try:
        p = float(p)
        if p >= 100: return f"{p:.2f}"
        if p >= 1:   return f"{p:.4f}"
        if p >= 0.01:return f"{p:.6f}"
        return f"{p:.8f}"
    except:
        return "-"

def tier_text(n):
    return "🟢 强(3+/4)" if n>=3 else ("🟡 中(2/4)" if n==2 else ("🔴 弱(1/4)" if n==1 else "⚪ 无(0/4)"))

def floor_to_step(value, step):
    if step is None or step == 0:
        return value
    return math.floor(value / step) * step

def round_to_precision(value, precision):
    if precision is None:
        return value
    fmt = "{:." + str(precision) + "f}"
    return float(fmt.format(value))

# ================== 交易所连接 ==================
def build_exchange():
    if EXCHANGE_NAME != "binance":
        raise ValueError("此脚本当前仅支持 Binance（USDT futures）。")
    ex = ccxt.binance({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {
            "defaultType": "future",
            "adjustForTimeDifference": True,
        }
    })
    return ex

# =================
