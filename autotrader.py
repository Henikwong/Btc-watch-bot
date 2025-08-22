# autotrader.py
# 三确认（≥3/4 周期）才下单 + 多指标同向确认 + Telegram 推送 + ccxt 实盘/纸面切换
# 支持：huobi / binance / okx （默认现货，MARKET_TYPE=swap 时尝试合约）
# 周期：1h, 4h, 1d, 1w；满足 >=3 个周期同方向 -> 触发下单
# 指标：EMA(5/10/30)趋势、MACD hist、RSI(14)、WR(14)、K/D(金叉/死叉)、VOL变化、ATR止损止盈

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
    if score_bull>=4 and score_bull>=score_bear+2:
        side="多"
    elif score_bear>=4 and score_bear>=score_bull+2:
        side="空"

    details={
        "ema_trend": ema_trend,
        "ema_vals":[float(ema5),float(ema10),float(ema30)],
        "macd":float(macd_hist),
        "rsi":float(rsi),
        "wr":float(wr),
        "k_trend":k_trend,
        "vol_trend":float(vol_trend),
        "entry":float(close.iloc[-1])
    }
    return side, details

def calc_stop_target(df, side, entry):
    atr=compute_atr(df)
    if atr is None: return None, None
    if side=="多": return entry-ATR_MULT*atr, entry+ATR_MULT*atr
    else: return entry+ATR_MULT*atr, entry-ATR_MULT*atr

def format_price(p):
    try:
        p=float(p)
        if p>=100: return f"{p:.2f}"
        if p>=1: return f"{p:.4f}"
        if p>=0.01: return f"{p:.6f}"
        return f"{p:.8f}"
    except: return "-"

def place_order(ex, symbol, side, entry, stop, target):
    qty=max(1e-8, BASE_USDT/max(entry,1e-8))
    order_side="buy" if side=="多" else "sell"
    if LIVE_TRADE!=1:
        log(f"[纸面单] {symbol} {side} 市价数量≈{qty}")
        return {"id":"paper","status":"simulated","side":order_side,"amount":qty}
    try:
        if MARKET_TYPE=="swap":
            try: ex.set_leverage(LEVERAGE, symbol)
            except Exception as e: log(f"{symbol} 设置杠杆失败: {e}")
        o=ex.create_order(symbol, type="market", side=order_side, amount=qty)
        log(f"[下单成功] {o}")
        return o
    except Exception as e:
        log(f"[下单失败] {e}")
        return None

def summarize_details(tf, side, det):
    return (f"{tf} | 方向:{side or '无'}  入场:{format_price(det['entry']) if det else '-'} | "
            f"EMA:{det['ema_trend'] if det else '-'} MACD:{det['macd'] if det else '-'} "
            f"RSI:{det['rsi'] if det else '-'} WR:{det['wr'] if det else '-'} "
            f"KDJ:{det['k_trend'] if det else '-'} VOLΔ:{round(det['vol_trend'],3) if det else '-'}")

def main():
    ex=build_exchange()
    mode="实盘" if LIVE_TRADE==1 else "纸面"
    log(f"启动交易Bot | {EXCHANGE_NAME} {MARKET_TYPE} 轮询{POLL_INTERVAL}s 模式={mode}")
    tg_send(f"🤖 交易Bot已启动：{EXCHANGE_NAME}/{MARKET_TYPE} 轮询{POLL_INTERVAL}s 模式={mode}")

    last_hourly_push_ts=0

    while True:
        loop_start=time.time()
        try:
            ex.load_markets()
            for symbol in SYMBOLS:
                sides=[]; details_map={}
                for tf in ["1h","4h","1d","1w"]:
                    try:
                        ohlcv=fetch_ohlcv
