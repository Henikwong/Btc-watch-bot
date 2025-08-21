# autotrader_debug.py
import os, time, traceback
import ccxt
import requests
import pandas as pd
import ta
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ========== ENV ==========
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

EXCHANGE_NAME = os.getenv("EXCHANGE", "huobi").lower()
API_KEY   = os.getenv("API_KEY", "")
API_SECRET= os.getenv("API_SECRET", "")

MARKET_TYPE = os.getenv("MARKET_TYPE", "spot").lower()   # spot / swap
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,LTC/USDT").split(",") if s.strip()]

BASE_USDT = float(os.getenv("BASE_USDT", "100"))
ATR_MULT  = float(os.getenv("RISK_ATR_MULT", "1.5"))
LEVERAGE  = int(os.getenv("LEVERAGE", "5"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))
LIVE_TRADE = int(os.getenv("LIVE_TRADE", "0"))

REQUIRED_CONFIRMS = 3
PERIODS = {"1h":"1h","4h":"4h","1d":"1d","1w":"1w"}

def nowstr(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def log(msg): print(f"[{nowstr()}] {msg}", flush=True)

def tg_send(text):
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text})
        except Exception as e:
            log(f"TG发送失败: {e}")

def build_exchange():
    params = {"apiKey": API_KEY, "secret": API_SECRET, "enableRateLimit": True}
    if EXCHANGE_NAME == "huobi": ex = ccxt.huobi(params)
    elif EXCHANGE_NAME == "binance": ex = ccxt.binance(params)
    elif EXCHANGE_NAME == "okx": ex = ccxt.okx(params)
    else: raise ValueError("仅支持 huobi/binance/okx")
    if MARKET_TYPE=="swap":
        try:
            ex.options["defaultType"]="swap"
        except: pass
    return ex

def fetch_ohlcv(ex, symbol, timeframe="1h", limit=200):
    return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

def df_from_ohlcv(ohlcv):
    cols = ["ts","open","high","low","close","vol"]
    df = pd.DataFrame(ohlcv, columns=cols)
    for c in ["open","high","low","close","vol"]: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def compute_atr(df, period=14):
    try:
        high=df["high"]; low=df["low"]; close=df["close"]
        tr1 = high-low
        tr2 = (high-close.shift(1)).abs()
        tr3 = (low-close.shift(1)).abs()
        tr = pd.concat([tr1,tr2,tr3],axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return float(atr.iloc[-1])
    except: return None

def indicators_and_side(df):
    if df is None or len(df)<35: return None, None
    work = df.copy().iloc[:-1]
    close = work["close"].astype(float)
    high = work["high"].astype(float)
    low = work["low"].astype(float)
    vol = work["vol"].astype(float)

    ema5 = close.ewm(span=5).mean().iloc[-1]
    ema10 = close.ewm(span=10).mean().iloc[-1]
    ema30 = close.ewm(span=30).mean().iloc[-1]
    ema_trend = "多" if ema5>ema10>ema30 else ("空" if ema5<ema10<ema30 else "中性")

    macd_hist = ta.trend.MACD(close).macd_diff().iloc[-1]
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    wr = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r().iloc[-1]
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=9, smooth_window=3)
    k_val = stoch.stoch().iloc[-1]; d_val = stoch.stoch_signal().iloc[-1]
    k_trend = "多" if k_val>d_val else ("空" if k_val<d_val else "中性")

    vol_trend = (vol.iloc[-1]-vol.iloc[-2])/(vol.iloc[-2]+1e-12)

    score_bull=0; score_bear=0
    score_bull += 1 if ema_trend=="多" else 0
    score_bear += 1 if ema_trend=="空" else 0
    score_bull += 1 if macd_hist>0 else 0
    score_bear += 1 if macd_hist<0 else 0
    score_bull += 1 if rsi>50 else 0
    score_bear += 1 if rsi<50 else 0
    score_bull += 1 if wr>-50 else 0
    score_bear += 1 if wr<-50 else 0
    score_bull += 1 if k_trend=="多" else 0
    score_bear += 1 if k_trend=="空" else 0
    if vol_trend>0: score_bull+=1
    if vol_trend<0: score_bear+=1

    side=None
    if score_bull>=4 and score_bull>=score_bear+2: side="多"
    elif score_bear>=4 and score_bear>=score_bull+2
