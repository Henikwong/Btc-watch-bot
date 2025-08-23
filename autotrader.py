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

EXCHANGE_NAME = os.getenv("EXCHANGE", "huobi").lower()     # huobi / htx (部分版本)
API_KEY   = os.getenv("API_KEY", "")
API_SECRET= os.getenv("API_SECRET", "")

MARKET_TYPE = os.getenv("MARKET_TYPE", "spot").lower()     # 这里强制只做 spot
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT").split(",") if s.strip()]

BASE_USDT = float(os.getenv("BASE_USDT", "15"))
ATR_MULT  = float(os.getenv("RISK_ATR_MULT", "1.5"))
LEVERAGE  = int(os.getenv("LEVERAGE", "1"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))
LIVE_TRADE    = int(os.getenv("LIVE_TRADE", "0"))

REQUIRED_CONFIRMS = 3  # 4 个周期里 ≥3 同向才触发
TIMEFRAMES = ["1h","4h","1d","1w"]

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

# ---------- 只用现货行情（避免 hbdm 域） ----------
# Huobi 现货K线： https://api.huobi.pro/market/history/kline?symbol=btcusdt&period=60min&size=200
HUOBI_PERIOD_MAP = {
    "1h": "60min",
    "4h": "4hour",
    "1d": "1day",
    "1w": "1week",
}
def htx_spot_ohlcv(symbol: str, timeframe: str, limit: int = 200):
    """直接用现货行情API获取K线，返回 ccxt 兼容的 ohlcv 列表。"""
    base, quote = symbol.replace("/", "").split("USDT")[0], "USDT"   # 粗略处理常见 xxx/USDT
    sym = (symbol.replace("/", "")).lower()  # e.g. BTC/USDT -> btcusdt
    period = HUOBI_PERIOD_MAP.get(timeframe)
    if period is None:
        raise ValueError(f"不支持的周期: {timeframe}")
    url = "https://api.huobi.pro/market/history/kline"
    params = {"symbol": sym, "period": period, "size": min(limit, 2000)}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "ok":
        raise RuntimeError(f"Huobi行情返回异常: {data}")
    # 数据是倒序 or 正序？官方返回按时间倒序（最近在前），我们需要按时间升序
    items = list(reversed(data["data"]))
    ohlcv = []
    for it in items:
        ts = int(it["id"]) * 1000
        o, h, l, c, v = float(it["open"]), float(it["high"]), float(it["low"]), float(it["close"]), float(it["vol"])
        ohlcv.append([ts, o, h, l, c, v])
    return ohlcv

def df_from_ohlcv(ohlcv):
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","vol"])
    for c in ["open","high","low","close","vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def compute_atr(df, period=14):
    high = df["high"]; low=df["low"]; close=df["close"]
    tr1 = high-low
    tr2 = (high-close.shift(1)).abs()
    tr3 = (low -close.shift(1)).abs()
    tr = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])

def indicators_and_side(df):
    """返回 (side, details)；side: '多'/'空'/None"""
    if df is None or len(df) < 35:
        return None, None
    work = df.iloc[:-1].copy()  # 去掉未收盘K
    close, high, low, vol = work["close"], work["high"], work["low"], work["vol"]

    ema5  = close.ewm(span=5).mean().iloc[-1]
    ema10 = close.ewm(span=10).mean().iloc[-1]
    ema30 = close.ewm(span=30).mean().iloc[-1]
    ema_trend = "多" if (ema5>ema10>ema30) else ("空" if (ema5<ema10<ema30) else "中性")

    macd_hist = ta.trend.MACD(close).macd_diff().iloc[-1]
    rsi = ta.momentum.RSIIndicator(close, 14).rsi().iloc[-1]
    wr  = ta.momentum.WilliamsRIndicator(high, low, close, 14).williams_r().iloc[-1]
    stoch = ta.momentum.StochasticOscillator(high, low, close, 9, 3)
    k_val = stoch.stoch().iloc[-1]; d_val = stoch.stoch_signal().iloc[-1]
    k_trend = "多" if k_val>d_val else ("空" if k_val<d_val else "中性")

    vol_trend = (vol.iloc[-1]-vol.iloc[-2])/(vol.iloc[-2]+1e-12)

    score_bull = sum([
        ema_trend=="多", macd_hist>0, rsi>50, wr>-50, k_trend=="多", vol_trend>0
    ])
    score_bear = sum([
        ema_trend=="空", macd_hist<0, rsi<50, wr<-50, k_trend=="空", vol_trend<0
    ])

    side = None
    if score_bull>=4 and score_bull>=score_bear+2:
        side = "多"
    elif score_bear>=4 and score_bear>=score_bull+2:
        side = "空"

    det = {
        "ema_trend": ema_trend,
        "macd": float(macd_hist),
        "rsi": float(rsi),
        "wr": float(wr),
        "k_trend": k_trend,
        "vol_trend": float(vol_trend),
        "entry": float(close.iloc[-1]),
    }
    return side, det

def tier_text(n):
    return "🟢 强(3+/4)" if n>=3 else ("🟡 中(2/4)" if n==2 else ("🔴 弱(1/4)" if n==1 else "⚪ 无(0/4)"))

def format_price(p):
    p = float(p)
    if p>=100: return f"{p:.2f}"
    if p>=1:   return f"{p:.4f}"
    if p>=0.01:return f"{p:.6f}"
    return f"{p:.8f}"

def build_exchange():
    params = {"apiKey": API_KEY, "secret": API_SECRET, "enableRateLimit": True}
    # 兼容不同 ccxt 版本：优先 huobi；没有就用 htx
    klass = None
    if hasattr(ccxt, "huobi"):
        klass = getattr(ccxt, "huobi")
    elif hasattr(ccxt, "htx"):
        klass = getattr(ccxt, "htx")
    else:
        raise RuntimeError("你的 ccxt 版本不包含 huobi/htx，请先 `pip install -U ccxt`")

    ex = klass(params)
    # 强制现货
    try:
        ex.options["defaultType"] = "spot"
    except Exception:
        pass
    return ex

def spot_qty(entry):
    return max(1e-8, BASE_USDT / max(entry, 1e-8))

def place_order(ex, symbol, side, entry):
    qty = spot_qty(entry)
    order_side = "buy" if side=="多" else "sell"
    if LIVE_TRADE != 1:
        log(f"[纸面单] {symbol} {side} 市价 数量≈{qty}")
        return {"id":"paper","amount":qty,"side":order_side}
    try:
        o = ex.create_order(symbol, type="market", side=order_side, amount=qty)
        log(f"[下单成功] {o}")
        return o
    except Exception as e:
        log(f"[下单失败] {e}")
        tg_send(f"❌ 下单失败 {symbol} {side}：{e}")
        return None

def fetch_df_spot(symbol, timeframe, limit=200):
    """Huobi/HTX 现货行情（避免 hbdm）"""
    ohlcv = htx_spot_ohlcv(symbol, timeframe, limit)
    return df_from_ohlcv(ohlcv)

def summarize(tf, side, det):
    return (f"{tf} | 方向:{side or '无'} 入场:{format_price(det['entry']) if det else '-'} | "
            f"EMA:{det['ema_trend'] if det else '-'} MACD:{round(det['macd'],4) if det else '-'} "
            f"RSI:{round(det['rsi'],2) if det else '-'} WR:{round(det['wr'],2) if det else '-'} "
            f"KDJ:{det['k_trend'] if det else '-'} VOLΔ:{round(det['vol_trend'],3) if det else '-'}")

def main():
    ex = build_exchange()
    log(f"启动Bot {EXCHANGE_NAME}/spot LIVE={LIVE_TRADE}")
    tg_send(f"🤖 Bot启动 {EXCHANGE_NAME}/spot 模式={'实盘' if LIVE_TRADE==1 else '纸面'}")

    last_hourly_push = 0
    while True:
        loop_start = time.time()
        try:
            for symbol in SYMBOLS:
                sides=[]
                details={}

                for tf in TIMEFRAMES:
                    try:
                        df = fetch_df_spot(symbol, tf, 200)
                        side, det = indicators_and_side(df)
                        sides.append(side)
                        details[tf]=(side, det, df)
                        log(summarize(tf, side, det))
                    except Exception as e_tf:
                        log(f"❌ 获取/计算失败 {symbol} {tf}: {e_tf}")
                        sides.append(None)
                        details[tf]=(None, None, None)

                bull = sum(1 for s in sides if s=="多")
                bear = sum(1 for s in sides if s=="空")
                final_side = None
                if bull>=REQUIRED_CONFIRMS and bull>bear:
                    final_side="多"
                elif bear>=REQUIRED_CONFIRMS and bear>bull:
                    final_side="空"

                now_ts = int(time.time())
                if now_ts - last_hourly_push >= 3600:
                    grade = tier_text(max(bull, bear))
                    lines = [f"⏰ 每小时[{symbol}] 评级: {grade}（多:{bull} 空:{bear}）"]
                    for tf in TIMEFRAMES:
                        s, det, _ = details[tf]
                        lines.append(summarize(tf, s, det))
                    tg_send("\n".join(lines))
                    last_hourly_push = now_ts

                if final_side:
                    s1h, d1h, _ = details["1h"]
                    if d1h:
                        entry = d1h["entry"]
                        o = place_order(ex, symbol, final_side, entry)
                        tg_send(
                            f"⚡ 三确认触发 {symbol} 做{'多' if final_side=='多' else '空'}\n"
                            f"入场: {format_price(entry)}  数量≈{spot_qty(entry):.6f}\n"
                            f"一致性: {max(bull,bear)}/4"
                        )

        except Exception as e:
            log(f"[主循环异常] {e}\n{traceback.format_exc()}")

        used = time.time() - loop_start
        time.sleep(max(1, POLL_INTERVAL - int(used)))

if __name__ == "__main__":
    main()
