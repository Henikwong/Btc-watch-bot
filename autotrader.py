# autotrader_full.py
import os
import time
from datetime import datetime
import requests
import ccxt
import pandas as pd
import ta
from dotenv import load_dotenv

load_dotenv()

# ===== ENV =====
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT  = os.getenv("TELEGRAM_CHAT_ID", "").strip()

EXCHANGE_NAME = os.getenv("EXCHANGE", "binance").lower()
API_KEY   = os.getenv("API_KEY", "").strip()
API_SECRET= os.getenv("API_SECRET", "").strip()

MARKET_TYPE = os.getenv("MARKET_TYPE", "future").lower()
LEVERAGE  = int(os.getenv("LEVERAGE", "10"))
BASE_USDT = float(os.getenv("BASE_USDT", "15"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))
LIVE_TRADE    = int(os.getenv("LIVE_TRADE", "0"))

TRADE_SYMBOLS   = [s.strip() for s in os.getenv("TRADE_SYMBOLS", "BTC/USDT,ETH/USDT").split(",") if s.strip()]
TIMEFRAMES = ["1h", "4h"]

SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "2.0"))
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "3.0"))
TRAIL_ATR_MULT = float(os.getenv("TRAIL_ATR_MULT", "1.5"))
PARTIAL_TP_RATIO = float(os.getenv("PARTIAL_TP_RATIO", "0.3"))

# ===== helpers =====
def nowstr():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"[{nowstr()}] {msg}", flush=True)

def tg_send(text):
    if TG_TOKEN and TG_CHAT:
        try:
            requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                          data={"chat_id": TG_CHAT, "text": text}, timeout=10)
        except Exception as e:
            log(f"TG发送失败: {e}")

# ===== Exchange =====
def build_exchange():
    ex = ccxt.binanceusdm({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    ex.load_markets()
    log("Binance USDM futures 初始化成功")
    return ex

def set_leverage_safe(ex, symbol, leverage):
    try:
        market = ex.market(symbol)
        ex.fapiPrivate_post_leverage({"symbol": market["id"], "leverage": int(leverage)})
        log(f"{symbol} 杠杆已设置为 {leverage}x")
    except Exception as e:
        log(f"设置杠杆失败 {symbol}: {e}")

def df_from_ohlcv(ohlcv):
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","vol"])
    for c in ["open","high","low","close","vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ===== Indicators =====
def analyze_df(df):
    if df is None or len(df) < 50:
        return None, None
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["vol"]

    ema5 = close.ewm(span=5).mean().iloc[-1]
    ema10 = close.ewm(span=10).mean().iloc[-1]
    ema30 = close.ewm(span=30).mean().iloc[-1]
    ema_trend = "多" if ema5>ema10>ema30 else ("空" if ema5<ema10<ema30 else "中性")

    macd = ta.trend.MACD(close)
    macd_hist_series = macd.macd_diff()
    macd_hist = float(macd_hist_series.iloc[-1])

    rsi = float(ta.momentum.RSIIndicator(close, 14).rsi().iloc[-1])
    wr  = float(ta.momentum.WilliamsRIndicator(high, low, close, 14).williams_r().iloc[-1])
    stoch = ta.momentum.StochasticOscillator(high, low, close, 9, 3)
    k_val = float(stoch.stoch().iloc[-1])
    d_val = float(stoch.stoch_signal().iloc[-1])
    k_trend = "多" if k_val>d_val else ("空" if k_val<d_val else "中性")

    vol_trend = float((vol.iloc[-1]-vol.iloc[-2])/(abs(vol.iloc[-2])+1e-12))
    atr = float(ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1])
    entry = float(close.iloc[-1])

    score_bull = sum([ema_trend=="多", macd_hist>0, rsi>55, wr>-50, k_trend=="多", vol_trend>0])
    score_bear = sum([ema_trend=="空", macd_hist<0, rsi<45, wr<-50, k_trend=="空", vol_trend<0])

    side = None
    if score_bull>=4 and score_bull>=score_bear+2:
        side="多"
    elif score_bear>=4 and score_bear>=score_bull+2:
        side="空"

    det = {"ema_trend": ema_trend, "macd": macd_hist, "macd_hist_series": macd_hist_series,
           "rsi": rsi, "wr": wr, "k_trend": k_trend, "vol_trend": vol_trend,
           "atr": atr, "entry": entry}

    return side, det

def fmt_price(p):
    p=float(p)
    if p>=100: return f"{p:.2f}"
    if p>=1: return f"{p:.4f}"
    if p>=0.01: return f"{p:.6f}"
    return f"{p:.8f}"

def amount_for_futures(ex, symbol, price):
    raw_qty = (BASE_USDT * LEVERAGE) / max(price, 1e-12)
    try:
        qty = ex.amount_to_precision(symbol, raw_qty)
    except Exception:
        qty = raw_qty
    return float(qty)

# ===== SL/TP =====
def create_sl_tp_orders(ex, symbol, side, qty, atr, entry):
    try:
        sl_price = entry - SL_ATR_MULT*atr if side=="多" else entry + SL_ATR_MULT*atr
        tp_price = entry + TP_ATR_MULT*atr if side=="多" else entry - TP_ATR_MULT*atr
        params_sl = {"stopPrice": fmt_price(sl_price), "reduceOnly": True, "workingType":"CONTRACT_PRICE"}
        params_tp = {"stopPrice": fmt_price(tp_price), "reduceOnly": True, "workingType":"CONTRACT_PRICE"}
        if side=="多":
            ex.create_order(symbol, "STOP_MARKET", "sell", qty, None, params_sl)
            ex.create_order(symbol, "TAKE_PROFIT_MARKET", "sell", qty, None, params_tp)
        else:
            ex.create_order(symbol, "STOP_MARKET", "buy", qty, None, params_sl)
            ex.create_order(symbol, "TAKE_PROFIT_MARKET", "buy", qty, None, params_tp)
        return True
    except Exception as e:
        log(f"创建SL/TP失败 {symbol}: {e}")
        return False

# ===== Trailing + partial =====
trail_state = {}

def update_trailing_and_partial(ex, symbol, last_price, tf4h_det):
    st = trail_state.get(symbol)
    if not st: return
    side = st["side"]; best = st["best"]; atr = st["atr"]; qty = st["qty"]; entry = st["entry"]
    moved = False

    if side=="多":
        if last_price>best: trail_state[symbol]["best"]=last_price
        if last_price >= best+TRAIL_ATR_MULT*atr:
            new_sl = last_price - SL_ATR_MULT*atr
            try:
                ex.create_order(symbol, "STOP_MARKET", "sell", qty, None, {"stopPrice": fmt_price(new_sl), "reduceOnly": True, "workingType":"CONTRACT_PRICE"})
                moved=True
            except: pass
    else:
        if last_price<best: trail_state[symbol]["best"]=last_price
        if last_price <= best-TRAIL_ATR_MULT*atr:
            new_sl = last_price + SL_ATR_MULT*atr
            try:
                ex.create_order(symbol, "STOP_MARKET", "buy", qty, None, {"stopPrice": fmt_price(new_sl), "reduceOnly": True, "workingType":"CONTRACT_PRICE"})
                moved=True
            except: pass

    if moved:
        tg_send(f"🔧 跟踪止损 {symbol} side={side} new_best={fmt_price(trail_state[symbol]['best'])}")

    if st.get("partial_done") or tf4h_det is None: return
    macd_series = tf4h_det["macd_hist_series"]
    if len(macd_series)<2: return
    macd_prev = float(macd_series.iloc[-2])
    macd_last = float(macd_series.iloc[-1])
    rsi4h = float(tf4h_det["rsi"])
    weak = False
    if side=="多" and macd_last>0 and macd_last<macd_prev and rsi4h>65:
        weak=True
    if side=="空" and macd_last<0 and abs(macd_last)<abs(macd_prev) and rsi4h<35:
        weak=True
    if weak:
        reduce_qty = max(qty*PARTIAL_TP_RATIO,0)
        if reduce_qty<=0: return
        if LIVE_TRADE==1:
            if side=="多":
                ex.create_order(symbol,"MARKET","sell",reduce_qty,None,{"reduceOnly":True})
            else:
                ex.create_order(symbol,"MARKET","buy",reduce_qty,None,{"reduceOnly":True})
            tg_send(f"🟢 部分止盈 {symbol} side={side} qty≈{reduce_qty} (4h MACD弱化)")
        else:
            tg_send(f"🟡 纸面部分止盈 {symbol} side={side} qty≈{reduce_qty} (4h MACD弱化)")
        trail_state[symbol]["partial_done"]=True

# ===== Main =====
def main():
    ex = build_exchange()
    tg_send(f"🤖 Bot启动 {EXCHANGE_NAME}/{MARKET_TYPE} 模式={'实盘' if LIVE_TRADE else '纸面'} 杠杆x{LEVERAGE}")
    for sym in TRADE_SYMBOLS:
        set_leverage_safe(ex, sym, LEVERAGE)

    while True:
        try:
            for symbol in TRADE_SYMBOLS:
                tf_details={}
                side_final=None
                for tf in TIMEFRAMES:
                    ohlcv = ex.fetch_ohlcv(symbol, tf)
                    df = df_from_ohlcv(ohlcv)
                    side, det = analyze_df(df)
                    tf_details[tf]=(side,det)
                    if det:
                        summary=f"{symbol} {tf}: {side or '无'} | EMA:{det['ema_trend']} MACD:{det['macd']:.4f} RSI:{det['rsi']:.2f} WR:{det['wr']:.2f} KDJ:{det['k_trend']} VOLΔ:{det['vol_trend']:.3f} ATR:{det['atr']:.2f}"
                        log(summary)
                        tg_send(summary)

                side1, det1=tf_details["1h"]
                side4, det4=tf_details["4h"]
                if side1==side4 and side1 is not None:
                    side_final=side1
                    last_price=det1["entry"]
                    atr=det1["atr"]
                    qty=amount_for_futures(ex, symbol, last_price)
                    if symbol not in trail_state:
                        trail_state[symbol]={"side":side_final,"best":last_price,"atr":atr,"qty":qty,"entry":last_price,"partial_done":False}
                        tg_send(f"🟢 开仓信号 {symbol} side={side_final} qty={qty} price={fmt_price(last_price)}")
                        if LIVE_TRADE
