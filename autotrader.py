import os, time, math, traceback
import requests
import ccxt
import pandas as pd
import numpy as np
import ta
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ========= ENV =========
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT  = os.getenv("TELEGRAM_CHAT_ID", "")

EXCHANGE_NAME = os.getenv("EXCHANGE", "binance").lower()  # binance / okx / huobi
API_KEY   = os.getenv("API_KEY", "")
API_SECRET= os.getenv("API_SECRET", "")

MARKET_TYPE = os.getenv("MARKET_TYPE", "future").lower()  # future / spot
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT").split(",") if s.strip()]

BASE_USDT = float(os.getenv("BASE_USDT", "15"))
ATR_MULT  = float(os.getenv("RISK_ATR_MULT", "1.5"))
LEVERAGE  = int(os.getenv("LEVERAGE", "10"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))
LIVE_TRADE    = int(os.getenv("LIVE_TRADE", "0"))

REQUIRED_CONFIRMS = 2  # 多周期共识
TIMEFRAMES = ["1h", "4h", "1d"]

# ========= 工具函数 =========
def nowstr():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"[{nowstr()}] {msg}", flush=True)

def tg_send(text):
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TG_CHAT, "text": text})
    except Exception as e:
        log(f"TG发送失败: {e}")

# ========= CCXT 构建交易所 =========
def build_exchange(name, api_key, api_secret):
    params = {"apiKey": api_key, "secret": api_secret, "enableRateLimit": True}
    if name=="binance":
        ex = ccxt.binance(params)
        ex.options["defaultType"] = "future"
        ex.set_sandbox_mode(False)
    elif name=="okx":
        ex = ccxt.okx(params)
        ex.options["defaultType"] = "future"
    elif name=="huobi":
        ex = ccxt.huobi(params)
        ex.options["defaultType"] = "swap"
    else:
        raise RuntimeError(f"不支持交易所: {name}")
    return ex

# ========= OHLCV 抓取 =========
def fetch_df(ex, symbol, timeframe, limit=200):
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","vol"])
    for c in ["open","high","low","close","vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ========= 指标计算 =========
def compute_atr(df, period=14):
    high = df["high"]; low=df["low"]; close=df["close"]
    tr1 = high-low
    tr2 = (high-close.shift(1)).abs()
    tr3 = (low-close.shift(1)).abs()
    tr = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])

def indicators_and_side(df):
    if df is None or len(df)<35:
        return None, None
    work = df.iloc[:-1].copy()
    close, high, low, vol = work["close"], work["high"], work["low"], work["vol"]

    ema5  = close.ewm(span=5).mean().iloc[-1]
    ema10 = close.ewm(span=10).mean().iloc[-1]
    ema30 = close.ewm(span=30).mean().iloc[-1]
    ema_trend = "多" if (ema5>ema10>ema30) else ("空" if (ema5<ema10<ema30) else "中性")

    macd_hist = ta.trend.MACD(close).macd_diff().iloc[-1]
    rsi = ta.momentum.RSIIndicator(close,14).rsi().iloc[-1]
    wr  = ta.momentum.WilliamsRIndicator(high, low, close,14).williams_r().iloc[-1]
    stoch = ta.momentum.StochasticOscillator(high, low, close,9,3)
    k_val = stoch.stoch().iloc[-1]; d_val = stoch.stoch_signal().iloc[-1]
    k_trend = "多" if k_val>d_val else ("空" if k_val<d_val else "中性")

    vol_trend = (vol.iloc[-1]-vol.iloc[-2])/(vol.iloc[-2]+1e-12)

    score_bull = sum([ema_trend=="多", macd_hist>0, rsi>50, wr>-50, k_trend=="多", vol_trend>0])
    score_bear = sum([ema_trend=="空", macd_hist<0, rsi<50, wr<-50, k_trend=="空", vol_trend<0])

    side = None
    if score_bull>=4 and score_bull>=score_bear+2:
        side="多"
    elif score_bear>=4 and score_bear>=score_bull+2:
        side="空"

    det = {
        "ema_trend": ema_trend,
        "macd": float(macd_hist),
        "rsi": float(rsi),
        "wr": float(wr),
        "k_trend": k_trend,
        "vol_trend": float(vol_trend),
        "entry": float(close.iloc[-1]),
        "atr": compute_atr(df)
    }
    return side, det

# ========= 下单 =========
def futures_qty(entry, leverage, ex, symbol):
    qty = BASE_USDT * leverage / max(entry,1e-8)
    min_qty = ex.markets[symbol]['limits']['amount']['min']
    return max(qty, min_qty)

def place_order(ex, symbol, side, entry):
    qty = futures_qty(entry, LEVERAGE, ex, symbol)
    order_side = "buy" if side=="多" else "sell"
    params = {}
    if LIVE_TRADE!=1:
        log(f"[纸面单] {symbol} {side} 市价 数量≈{qty:.6f}")
        return {"id":"paper","amount":qty,"side":order_side}
    try:
        o = ex.create_order(symbol, type="market", side=order_side, amount=qty, params=params)
        log(f"[下单成功] {o}")
        return o
    except Exception as e:
        log(f"[下单失败] {e}")
        tg_send(f"❌ 下单失败 {symbol} {side}: {e}")
        return None

def format_price(p):
    p = float(p)
    if p>=100: return f"{p:.2f}"
    if p>=1:   return f"{p:.4f}"
    if p>=0.01:return f"{p:.6f}"
    return f"{p:.8f}"

def summarize(tf, side, det):
    return (f"{tf} | 方向:{side or '无'} 入场:{format_price(det['entry']) if det else '-'} | "
            f"EMA:{det['ema_trend'] if det else '-'} MACD:{round(det['macd'],4) if det else '-'} "
            f"RSI:{round(det['rsi'],2) if det else '-'} WR:{round(det['wr'],2) if det else '-'} "
            f"KDJ:{det['k_trend'] if det else '-'} VOLΔ:{round(det['vol_trend'],3) if det else '-'} ATR:{round(det['atr'],2) if det else '-'}")

# ========= 主循环 =========
def main():
    ex = build_exchange(EXCHANGE_NAME, API_KEY, API_SECRET)
    # 设置杠杆
    for sym in SYMBOLS:
        try:
            ex.fapiPrivate_post_leverage({"symbol": sym.replace("/",""), "leverage": LEVERAGE})
        except Exception as e:
            log(f"设置杠杆失败 {sym}: {e}")

    log(f"启动Bot {EXCHANGE_NAME}/{MARKET_TYPE} LIVE={LIVE_TRADE}")
    tg_send(f"🤖 Bot启动 {EXCHANGE_NAME}/{MARKET_TYPE} 模式={'实盘' if LIVE_TRADE==1 else '纸面'}")

    last_push = 0
    while True:
        loop_start = time.time()
        try:
            for symbol in SYMBOLS:
                sides=[]
                details={}
                for tf in TIMEFRAMES:
                    df = fetch_df(ex, symbol, tf, 200)
                    side, det = indicators_and_side(df)
                    sides.append(side)
                    details[tf] = (side, det, df)
                    log(summarize(tf, side, det))

                bull = sum(1 for s in sides if s=="多")
                bear = sum(1 for s in sides if s=="空")
                final_side = None
                if bull>=REQUIRED_CONFIRMS and bull>bear:
                    final_side="多"
                elif bear>=REQUIRED_CONFIRMS and bear>bull:
                    final_side="空"

                # 每小时推送
                now_ts = int(time.time())
                if now_ts - last_push >= 3600:
                    tg_send(f"⏰ [{symbol}] 多周期评级: 多:{bull} 空:{bear}")
                    for tf in TIMEFRAMES:
                        s, det, _ = details[tf]
                        tg_send(summarize(tf, s, det))
                    last_push = now_ts

                # 下单
                if final_side:
                    s1h, d1h, _ = details["1h"]
                    if d1h:
                        entry = d1h["entry"]
                        place_order(ex, symbol, final_side, entry)

        except Exception as e:
            log(f"[主循环异常] {e}\n{traceback.format_exc()}")

        used = time.time() - loop_start
        time.sleep(max(1, POLL_INTERVAL - int(used)))

if __name__=="__main__":
    main()
