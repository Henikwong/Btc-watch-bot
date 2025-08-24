# autotrader_full.py

import os
import time
import requests
import ccxt
import pandas as pd
import ta
from dotenv import load_dotenv
from datetime import datetime

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

TRADE_SYMBOLS = [s.strip() for s in os.getenv("TRADE_SYMBOLS", "BTC/USDT,ETH/USDT").split(",") if s.strip()]
TIMEFRAMES = ["1h", "4h"]

SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "2.0"))
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "3.0"))
TRAIL_ATR_MULT = float(os.getenv("TRAIL_ATR_MULT", "1.5"))
PARTIAL_TP_RATIO = float(os.getenv("PARTIAL_TP_RATIO", "0.3"))

# ===== Helpers =====
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

    # EMA
    ema5 = close.ewm(span=5).mean().iloc[-1]
    ema10 = close.ewm(span=10).mean().iloc[-1]
    ema30 = close.ewm(span=30).mean().iloc[-1]
    ema_trend = "多" if ema5>ema10>ema30 else ("空" if ema5<ema10<ema30 else "中性")

    # MACD
    macd = ta.trend.MACD(close)
    macd_hist_series = macd.macd_diff()
    macd_hist = float(macd_hist_series.iloc[-1])

    # RSI
    rsi = float(ta.momentum.RSIIndicator(close, 14).rsi().iloc[-1])

    # WR
    wr = float(ta.momentum.WilliamsRIndicator(high, low, close, 14).williams_r().iloc[-1])

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(high, low, close, 9, 3)
    k_val = float(stoch.stoch().iloc[-1])
    d_val = float(stoch.stoch_signal().iloc[-1])
    k_trend = "多" if k_val>d_val else ("空" if k_val<d_val else "中性")

    # Volume trend
    vol_trend = float((vol.iloc[-1]-vol.iloc[-2])/(abs(vol.iloc[-2])+1e-12))

    # ATR
    atr = float(ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1])
    entry = float(close.iloc[-1])

    # Score
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

# ===== Price formatting =====
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

# ===== Main =====
trail_state = {}

def main():
    try:
        ex = build_exchange()
    except Exception as e:
        log(f"交易所初始化失败: {e}")
        return

    tg_send(f"🤖 启动Bot {EXCHANGE_NAME}/{MARKET_TYPE} 模式={'实盘' if LIVE_TRADE else '纸面'} 杠杆x{LEVERAGE}")

    # 设置杠杆
    for sym in TRADE_SYMBOLS:
        set_leverage_safe(ex, sym, LEVERAGE)

    while True:
        try:
            for symbol in TRADE_SYMBOLS:
                # 获取1h和4h的OHLCV
                ohlcv_1h = ex.fetch_ohlcv(symbol, "1h", limit=100)
                ohlcv_4h = ex.fetch_ohlcv(symbol, "4h", limit=100)

                df_1h = df_from_ohlcv(ohlcv_1h)
                df_4h = df_from_ohlcv(ohlcv_4h)

                side1, det1 = analyze_df(df_1h)
                side4, det4 = analyze_df(df_4h)

                # 如果1h和4h方向一致，开仓逻辑
                if side1 == side4 and side1 is not None:
                    last_price = det1["entry"]
                    atr = det1["atr"]
                    qty = amount_for_futures(ex, symbol, last_price)

                    # 初始化跟踪止损状态
                    if symbol not in trail_state:
                        trail_state[symbol] = {"side": side1, "best": last_price, "atr": atr, "qty": qty,
                                               "entry": last_price, "partial_done": False}
                        tg_send(f"🟢 开仓信号 {symbol} side={side1} qty={qty} price={fmt_price(last_price)}")

                    if LIVE_TRADE:
                        try:
                            if side1 == "多":
                                ex.create_order(symbol, "MARKET", "buy", qty)
                            else:
                                ex.create_order(symbol, "MARKET", "sell", qty)
                        except Exception as e:
                            log(f"下单失败 {symbol}: {e}")

            time.sleep(POLL_INTERVAL)

        except Exception as e:
            log(f"主循环错误: {e}")
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
