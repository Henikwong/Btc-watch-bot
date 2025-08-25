import os, time, math, traceback
import requests
import ccxt
import pandas as pd
import numpy as np
import ta
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ========== ENV ==========
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT  = os.getenv("TELEGRAM_CHAT_ID", "")

EXCHANGE_NAME = os.getenv("EXCHANGE", "binance").lower()
API_KEY   = os.getenv("API_KEY", "")
API_SECRET= os.getenv("API_SECRET", "")

MARKET_TYPE = os.getenv("MARKET_TYPE", "future").lower()    # spot/future
# Binance futures 符号写法：BTC/USDT
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT").split(",") if s.strip()]

BASE_USDT = float(os.getenv("BASE_USDT", "15"))
LEVERAGE  = int(os.getenv("LEVERAGE", "10"))
ATR_MULT_INFO = float(os.getenv("RISK_ATR_MULT", "1.5"))   # 仅用于提示
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "2.0"))
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "3.0"))

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))
LIVE_TRADE    = int(os.getenv("LIVE_TRADE", "0"))

# 4个周期中至少 3 个同向
REQUIRED_CONFIRMS = 3
TIMEFRAMES = ["1h", "4h", "1d", "1w"]

# ========== 小工具 ==========
def nowstr():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"[{nowstr()}] {msg}", flush=True)

def tg_send(text):
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TG_CHAT, "text": text, "disable_web_page_preview": True})
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

# ========== 交易所 ==========
def build_exchange():
    if EXCHANGE_NAME != "binance":
        raise ValueError("此脚本当前仅支持 Binance。")

    ex = ccxt.binance({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {
            "defaultType": "future" if MARKET_TYPE=="future" else "spot",
            "adjustForTimeDifference": True,
        },
    })
    return ex

# ========== 数据与指标 ==========
def fetch_df(ex, symbol, tf="1h", limit=300):
    """ 用 ccxt 获取 K线，升序返回 DataFrame """
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","vol"])
    for c in ["open","high","low","close","vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def compute_atr(df, period=14):
    high = df["high"]; low = df["low"]; close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return float(atr.iloc[-1])

def indicators_and_side(df):
    """
    返回 (side, details)
    side: "多" / "空" / None
    details: 指标细节
    """
    if df is None or len(df) < 50:
        return None, None

    work = df.iloc[:-1].copy()  # 丢掉最后一根未收盘K
    close = work["close"].astype(float)
    high  = work["high"].astype(float)
    low   = work["low"].astype(float)
    vol   = work["vol"].astype(float)

    ema5  = close.ewm(span=5).mean().iloc[-1]
    ema10 = close.ewm(span=10).mean().iloc[-1]
    ema30 = close.ewm(span=30).mean().iloc[-1]
    ema_trend = "多" if (ema5>ema10>ema30) else ("空" if (ema5<ema10<ema30) else "中性")

    macd_hist = ta.trend.MACD(close).macd_diff().iloc[-1]
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    wr  = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r().iloc[-1]
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=9, smooth_window=3)
    k_val = stoch.stoch().iloc[-1]; d_val = stoch.stoch_signal().iloc[-1]
    k_trend = "多" if k_val>d_val else ("空" if k_val<d_val else "中性")
    vol_trend = (vol.iloc[-1] - vol.iloc[-2]) / (vol.iloc[-2] + 1e-12)

    score_bull = 0; score_bear = 0
    score_bull += (ema_trend=="多"); score_bear += (ema_trend=="空")
    score_bull += (macd_hist>0);   score_bear += (macd_hist<0)
    score_bull += (rsi>50);        score_bear += (rsi<50)
    score_bull += (wr>-50);        score_bear += (wr<-50)
    score_bull += (k_trend=="多"); score_bear += (k_trend=="空")
    if vol_trend>0: score_bull += 1
    if vol_trend<0: score_bear += 1

    side = None
    if score_bull >= 4 and score_bull >= score_bear + 2:
        side = "多"
    elif score_bear >= 4 and score_bear >= score_bull + 2:
        side = "空"

    details = {
        "ema_trend": ema_trend,
        "ema": [float(ema5), float(ema10), float(ema30)],
        "macd": float(macd_hist),
        "rsi": float(rsi),
        "wr": float(wr),
        "k_trend": k_trend,
        "vol_trend": float(vol_trend),
        "entry": float(close.iloc[-1]),
    }
    return side, details

def summarize(tf, side, det):
    return (f"{tf} | 方向:{side or '无'} 入:{format_price(det['entry']) if det else '-'} | "
            f"EMA:{det['ema_trend'] if det else '-'} MACD:{round(det['macd'],4) if det else '-'} "
            f"RSI:{round(det['rsi'],2) if det else '-'} WR:{round(det['wr'],2) if det else '-'} "
            f"KDJ:{det['k_trend'] if det else '-'} VOLΔ:{round(det['vol_trend'],3) if det else '-'}")

# ========== 下单 ==========
def futures_qty(entry):
    # 合约名义张数 = BASE_USDT / 价格
    return max(1e-6, BASE_USDT / max(entry, 1e-9))

def set_symbol_leverage(ex, symbol):
    try:
        # 尝试设置逐仓 + 杠杆（如果失败就忽略）
        params = {"marginMode": "isolated"}
        ex.set_margin_mode("isolated", symbol, params)
    except Exception:
        pass
    try:
        ex.set_leverage(LEVERAGE, symbol, params={"marginMode": "isolated"})
    except Exception:
        pass

def place_order_and_brackets(ex, symbol, side, entry, df_for_atr):
    """
    futures 市价开仓，并尝试挂 reduceOnly 止损 + 止盈
    返回：下单对象或纸面模拟对象
    """
    qty = futures_qty(entry)
    order_side = "buy" if side == "多" else "sell"
    sl, tp = None, None
    try:
        atr = compute_atr(df_for_atr, period=14)
        if side == "多":
            sl = entry - SL_ATR_MULT * atr
            tp = entry + TP_ATR_MULT * atr
        else:
            sl = entry + SL_ATR_MULT * atr
            tp = entry - TP_ATR_MULT * atr
    except Exception:
        pass

    if LIVE_TRADE != 1:
        log(f"[纸面单] {symbol} {side} 市价 数量≈{qty} | SL:{format_price(sl) if sl else '-'} TP:{format_price(tp) if tp else '-'}")
        return {"id": "paper", "amount": qty, "side": order_side, "sl": sl, "tp": tp}

    try:
        # 市价单
        o = ex.create_order(symbol, type="market", side=order_side, amount=qty)
        log(f"[开仓成功] {symbol} {side} 量≈{qty} 订单: {o.get('id','?')}")

        # 尝试挂出 reduceOnly 止损/止盈（不保证成功）
        if sl:
            try:
                ex.create_order(symbol,
                                type="STOP_MARKET",
                                side="sell" if side=="多" else "buy",
                                amount=qty,
                                params={"stopPrice": sl, "reduceOnly": True, "timeInForce":"GTC"})
                log(f"[挂SL] {symbol} @ {format_price(sl)}")
            except Exception as e_sl:
                log(f"[挂SL失败] {e_sl}")

        if tp:
            try:
                ex.create_order(symbol,
                                type="TAKE_PROFIT_MARKET",
                                side="sell" if side=="多" else "buy",
                                amount=qty,
                                params={"stopPrice": tp, "reduceOnly": True, "timeInForce":"GTC"})
                log(f"[挂TP] {symbol} @ {format_price(tp)}")
            except Exception as e_tp:
                log(f"[挂TP失败] {e_tp}")

        return o
    except Exception as e:
        log(f"[下单失败] {e}")
        tg_send(f"❌ 下单失败 {symbol} {side}：{e}")
        return None

# ========== 主循环 ==========
def main():
    ex = build_exchange()
    ex.load_markets()
    mode_txt = "实盘" if LIVE_TRADE==1 else "纸面"
    log(f"启动Bot {EXCHANGE_NAME}/{MARKET_TYPE} 模式={mode_txt}")
    tg_send(f"🤖 Bot启动 {EXCHANGE_NAME}/{MARKET_TYPE} 模式={mode_txt}")

    # 尽量设置每个交易对的杠杆
    if MARKET_TYPE == "future":
        for sym in SYMBOLS:
            s = sym.split(":")[0]  # 容错: 若用户误写了 :USDT，也只取左边
            try:
                set_symbol_leverage(ex, s)
            except Exception:
                pass

    last_hourly_push = 0

    while True:
        loop_start = time.time()
        try:
            for raw_symbol in SYMBOLS:
                symbol = raw_symbol.split(":")[0]  # 防止带 :USDT
                sides = []
                detail_map = {}

                for tf in TIMEFRAMES:
                    try:
                        df = fetch_df(ex, symbol, tf, limit=300)
                        side, det = indicators_and_side(df)
                        detail_map[tf] = (side, det, df)
                        sides.append(side)
                        log(summarize(tf, side, det))
                    except Exception as e_tf:
                        log(f"❌ {symbol} {tf} 指标失败: {e_tf}")
                        detail_map[tf] = (None, None, None)
                        sides.append(None)

                bull = sum(1 for s in sides if s=="多")
                bear = sum(1 for s in sides if s=="空")
                final_side = None
                if bull >= REQUIRED_CONFIRMS and bull > bear:
                    final_side = "多"
                elif bear >= REQUIRED_CONFIRMS and bear > bull:
                    final_side = "空"

                # 每小时总览
                now_ts = int(time.time())
                if now_ts - last_hourly_push >= 3600:
                    grade = tier_text(max(bull, bear))
                    lines = [f"⏰ 每小时[{symbol}] 评级: {grade}（多:{bull} 空:{bear}）"]
                    for tf in TIMEFRAMES:
                        s, det, _ = detail_map[tf]
                        lines.append(summarize(tf, s, det))
                    tg_send("\n".join(lines))
                    last_hourly_push = now_ts

                # 触发下单
                if final_side:
                    s1h, d1h, df1h = detail_map["1h"]
                    if d1h and df1h is not None:
                        entry = d1h["entry"]
                        o = place_order_and_brackets(ex, symbol, final_side, entry, df1h)
                        if o:
                            tg_send(
                                f"⚡ 三确认触发 {symbol} 做{'多' if final_side=='多' else '空'}\n"
                                f"入场: {format_price(entry)}  名义数量≈{futures_qty(entry):.6f}\n"
                                f"SL≈ATR*{SL_ATR_MULT}  TP≈ATR*{TP_ATR_MULT}"
                            )
        except Exception as e:
            log(f"[主循环异常] {e}\n{traceback.format_exc()}")

        # 间隔
        used = time.time() - loop_start
        time.sleep(max(1, POLL_INTERVAL - int(used)))

if __name__ == "__main__":
    main()
