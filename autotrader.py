import os, time, traceback
from datetime import datetime
import ccxt
import pandas as pd
import numpy as np
import ta
import requests
from dotenv import load_dotenv

load_dotenv()

# ================== ENV ==================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
EXCHANGE_NAME      = os.getenv("EXCHANGE", "huobi").lower()
API_KEY            = os.getenv("API_KEY")
API_SECRET         = os.getenv("API_SECRET")
MARKET_TYPE        = os.getenv("MARKET_TYPE", "spot").lower()
SYMBOLS            = [s.strip() for s in os.getenv("SYMBOLS", "").split(",") if s.strip()]
BASE_USDT          = float(os.getenv("BASE_USDT", "100"))
ATR_MULT           = float(os.getenv("RISK_ATR_MULT", "1.5"))
LEVERAGE           = int(os.getenv("LEVERAGE", "5"))
POLL_INTERVAL      = int(os.getenv("POLL_INTERVAL", "60"))
LIVE_TRADE         = int(os.getenv("LIVE_TRADE", "0"))

REQUIRED_CONFIRMS = 3  # 4个周期中至少3个同向

PERIODS = {
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1w": "1w",
}

# ================== 工具函数 ==================
def nowstr(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def log(msg): print(f"[{nowstr()}] {msg}", flush=True)
def tg_send(msg):
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
        except Exception as e:
            log(f"TG发送失败: {e}")

def build_exchange():
    params = {"apiKey": API_KEY, "secret": API_SECRET, "enableRateLimit": True}
    if EXCHANGE_NAME=="huobi": ex = ccxt.huobi(params)
    elif EXCHANGE_NAME=="binance": ex = ccxt.binance(params)
    elif EXCHANGE_NAME=="okx": ex = ccxt.okx(params)
    else: raise ValueError("只支持 huobi/binance/okx")
    
    if MARKET_TYPE=="swap":
        try:
            ex.options["defaultType"] = "swap"
        except: pass
    return ex

def fetch_df(ex, symbol, tf="1h", limit=200):
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","vol"])
    df[["open","high","low","close","vol"]] = df[["open","high","low","close","vol"]].astype(float)
    return df

def compute_atr(df, period=14):
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]

def indicators_and_side(df):
    if len(df)<35: return None, None
    df_work = df.iloc[:-1]
    close, high, low, vol = df_work["close"], df_work["high"], df_work["low"], df_work["vol"]
    ema5, ema10, ema30 = close.ewm(span=5).mean().iloc[-1], close.ewm(span=10).mean().iloc[-1], close.ewm(span=30).mean().iloc[-1]
    ema_trend = "多" if ema5>ema10>ema30 else ("空" if ema5<ema10<ema30 else "中性")
    macd_hist = ta.trend.MACD(close).macd_diff().iloc[-1]
    rsi = ta.momentum.RSIIndicator(close,14).rsi().iloc[-1]
    wr = ta.momentum.WilliamsRIndicator(high, low, close, 14).williams_r().iloc[-1]
    stoch = ta.momentum.StochasticOscillator(high, low, close,9,3)
    k_trend = "多" if stoch.stoch().iloc[-1] > stoch.stoch_signal().iloc[-1] else "空"
    vol_trend = (vol.iloc[-1]-vol.iloc[-2])/(vol.iloc[-2]+1e-12)
    score_bull = sum([ema_trend=="多", macd_hist>0, rsi>50, wr>-50, k_trend=="多", vol_trend>0])
    score_bear = sum([ema_trend=="空", macd_hist<0, rsi<50, wr<-50, k_trend=="空", vol_trend<0])
    side = "多" if score_bull>=4 and score_bull>=score_bear+2 else ("空" if score_bear>=4 and score_bear>=score_bull+2 else None)
    details = {"ema_trend":ema_trend,"macd":float(macd_hist),"rsi":float(rsi),"wr":float(wr),"k_trend":k_trend,"vol_trend":float(vol_trend),"entry":float(close.iloc[-1])}
    return side, details

def place_order(ex, symbol, side, entry):
    qty = max(1e-8, BASE_USDT / max(entry,1e-8))
    order_side = "buy" if side=="多" else "sell"
    if LIVE_TRADE!=1:
        log(f"[纸面单] {symbol} {side} 数量≈{qty}")
        return {"id":"paper"}
    try:
        if MARKET_TYPE=="spot":
            o = ex.create_order(symbol,"market",order_side,qty)
        else:
            o = ex.create_order(symbol,"market",order_side,qty)
        log(f"[下单成功] {o}")
        return o
    except Exception as e:
        log(f"[下单失败] {e}")
        return None

# ================== 主循环 ==================
def main():
    ex = build_exchange()
    log(f"启动Bot {EXCHANGE_NAME}/{MARKET_TYPE} LIVE={LIVE_TRADE}")
    tg_send(f"🤖 Bot启动 {EXCHANGE_NAME}/{MARKET_TYPE} 模式={'实盘' if LIVE_TRADE==1 else '纸面'}")
    
    while True:
        try:
            ex.load_markets()
            for symbol in SYMBOLS:
                sides=[]
                details_map={}
                for tf in ["1h","4h","1d","1w"]:
                    try:
                        df = fetch_df(ex,symbol,tf)
                        side, det = indicators_and_side(df)
                        details_map[tf]=(side,det)
                        sides.append(side)
                    except Exception as e_tf:
                        log(f"{symbol} {tf} 指标异常: {e_tf}")
                        details_map[tf]=(None,None)
                bull = sides.count("多")
                bear = sides.count("空")
                final_side = None
                if bull>=REQUIRED_CONFIRMS and bull>bear: final_side="多"
                elif bear>=REQUIRED_CONFIRMS and bear>bull: final_side="空"
                
                 if final_side:
                    side1h, det1h = details_map["1h"]
                    entry = det1h["entry"]
                    o = place_order(ex, symbol, final_side, entry)
                    tg_send(f"下单触发 {symbol}if final_side:
    side1h, det1h = details_map["1h"]
    entry = det1h["entry"]
    o = place_order(ex, symbol, final_side, entry)
    tg_send(f"下单触发 {symbol} {final_side} 入场价 {entry} 数量≈{BASE_USDT/entry:.6f}")
