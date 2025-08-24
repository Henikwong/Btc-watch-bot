# autotrader.py
import os
import time
import math
import traceback
from datetime import datetime
from typing import Optional, Tuple

import requests
import ccxt
import pandas as pd
import numpy as np
import ta
from dotenv import load_dotenv

load_dotenv()

# ========= ENV =========
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT  = os.getenv("TELEGRAM_CHAT_ID", "").strip()

EXCHANGE_NAME = os.getenv("EXCHANGE", "binance").lower()  # binance / binanceusdm / okx(不建议此脚本) ...
API_KEY   = os.getenv("API_KEY", "").strip()
API_SECRET= os.getenv("API_SECRET", "").strip()

MARKET_TYPE = os.getenv("MARKET_TYPE", "future").lower()  # future / spot
# 你要的品种（默认已含 BTC/USDT, ETH/USDT, LTC/USDT, BNB/USDT, DOGE/USDT）
SYMBOLS = [s.strip() for s in os.getenv(
    "SYMBOLS",
    "BTC/USDT,ETH/USDT,LTC/USDT,BNB/USDT,DOGE/USDT"
).split(",") if s.strip()]

BASE_USDT = float(os.getenv("BASE_USDT", "15"))
LEVERAGE  = int(os.getenv("LEVERAGE", "10"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))
LIVE_TRADE    = int(os.getenv("LIVE_TRADE", "0"))

# 风控参数（ATR）
SL_ATR_MULT     = float(os.getenv("SL_ATR_MULT", "2.0"))
TP_ATR_MULT     = float(os.getenv("TP_ATR_MULT", "3.0"))
TRAIL_ATR_MULT  = float(os.getenv("TRAIL_ATR_MULT", "1.5"))
PARTIAL_TP_RATIO= float(os.getenv("PARTIAL_TP_RATIO", "0.5"))  # 50% 止盈

# 多周期共识
TIMEFRAMES = ["1h", "4h", "1d"]
REQUIRED_CONFIRMS = int(os.getenv("REQUIRED_CONFIRMS", "2"))

# ========= 工具 =========
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
            timeout=10
        )
    except Exception as e:
        log(f"TG发送失败: {e}")

# ========= 交易所 =========
def build_exchange():
    last_err = None
    # 优先 USDT 永续
    try:
        if hasattr(ccxt, "binanceusdm"):
            ex = getattr(ccxt, "binanceusdm")({
                "apiKey": API_KEY,
                "secret": API_SECRET,
                "enableRateLimit": True,
            })
            ex.load_markets()
            log("使用 ccxt.binanceusdm 初始化（USDT 永续）")
            return ex
    except Exception as e:
        last_err = e
        log(f"binanceusdm 初始化失败: {e}")

    # 退回普通 binance + future
    try:
        ex = ccxt.binance({
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        ex.load_markets()
        log("使用 ccxt.binance 初始化（defaultType=future）")
        return ex
    except Exception as e:
        last_err = e
        log(f"binance 初始化失败: {e}")

    raise RuntimeError(f"交易所初始化失败，请检查 API、网络或白名单。最后错误: {last_err}")

def set_leverage_safe(ex, symbol, lev):
    """
    兼容不同 ccxt 版本/适配层；失败只告警，不终止。
    """
    try:
        market = ex.market(symbol)
    except Exception as e:
        log(f"无法获取 market 信息 {symbol}: {e}")
        return

    tried = []
    # 方法1：统一接口
    try:
        if hasattr(ex, "set_leverage"):
            try:
                ex.set_leverage(int(lev), market["symbol"])
            except Exception:
                ex.set_leverage(int(lev), market["id"])
            log(f"{symbol} 杠杆设置为 {lev}x (set_leverage)")
            return
    except Exception as e:
        tried.append(f"set_leverage:{e}")

    # 方法2：老式 USDM 接口
    try:
        if hasattr(ex, "fapiPrivate_post_leverage"):
            ex.fapiPrivate_post_leverage({"symbol": market["id"], "leverage": int(lev)})
            log(f"{symbol} 杠杆设置为 {lev}x (fapiPrivate_post_leverage)")
            return
    except Exception as e:
        tried.append(f"fapiPrivate_post_leverage:{e}")

    # 方法3：private_post_leverage
    try:
        if hasattr(ex, "private_post_leverage"):
            ex.private_post_leverage({"symbol": market["id"], "leverage": int(lev)})
            log(f"{symbol} 杠杆设置为 {lev}x (private_post_leverage)")
            return
    except Exception as e:
        tried.append(f"private_post_leverage:{e}")

    log(f"⚠️ 杠杆设置失败 {symbol}（尝试: {tried}）。继续运行。")

# ========= 数据/指标 =========
def fetch_df(ex, symbol, timeframe, limit=200):
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","vol"])
    for c in ["open","high","low","close","vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def analyze_df(df) -> Tuple[Optional[str], Optional[dict]]:
    """
    返回 (side, det)；用倒数第2根收盘之前的数据（避免未收K抖动）
    """
    if df is None or len(df) < 50:
        return None, None
    work = df.iloc[:-1].copy()
    close = work["close"]; high = work["high"]; low = work["low"]; vol = work["vol"]

    # EMA 多空
    ema5 = close.ewm(span=5).mean().iloc[-1]
    ema10 = close.ewm(span=10).mean().iloc[-1]
    ema30 = close.ewm(span=30).mean().iloc[-1]
    ema_trend = "多" if (ema5 > ema10 > ema30) else ("空" if (ema5 < ema10 < ema30) else "中性")

    # MACD
    macd = ta.trend.MACD(close)
    macd_hist_series = macd.macd_diff()
    macd_hist = float(macd_hist_series.iloc[-1])

    # RSI / WR / KDJ
    rsi = float(ta.momentum.RSIIndicator(close, 14).rsi().iloc[-1])
    wr  = float(ta.momentum.WilliamsRIndicator(high, low, close, 14).williams_r().iloc[-1])
    stoch = ta.momentum.StochasticOscillator(high, low, close, 9, 3)
    k_val = float(stoch.stoch().iloc[-1])
    d_val = float(stoch.stoch_signal().iloc[-1])
    k_trend = "多" if k_val > d_val else ("空" if k_val < d_val else "中性")

    # 成交量动量
    vol_trend = float((vol.iloc[-1] - vol.iloc[-2]) / (abs(vol.iloc[-2]) + 1e-12))

    # ATR（用完整df算，逼近当前波动）
    atr = float(ta.volatility.AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range().iloc[-1])
    entry = float(close.iloc[-1])

    score_bull = sum([ema_trend=="多", macd_hist>0, rsi>55, wr>-50, k_trend=="多", vol_trend>0])
    score_bear = sum([ema_trend=="空", macd_hist<0, rsi<45, wr<-50, k_trend=="空", vol_trend<0])

    side = None
    if score_bull >= 4 and score_bull >= score_bear + 2:
        side = "多"
    elif score_bear >= 4 and score_bear >= score_bull + 2:
        side = "空"

    det = {
        "ema_trend": ema_trend,
        "macd": macd_hist,
        "macd_hist_series": macd_hist_series,
        "rsi": rsi,
        "wr": wr,
        "k_trend": k_trend,
        "vol_trend": vol_trend,
        "atr": atr,
        "entry": entry,
    }
    return side, det

def fmt_price(p: float) -> str:
    p = float(p)
    if p >= 100: return f"{p:.2f}"
    if p >= 1:   return f"{p:.4f}"
    if p >= 0.01:return f"{p:.6f}"
    return f"{p:.8f}"

# ========= 下单/风控 =========
def calc_qty(ex, symbol, price):
    raw = (BASE_USDT * LEVERAGE) / max(price, 1e-12)
    try:
        return float(ex.amount_to_precision(symbol, raw))
    except Exception:
        return float(raw)

def create_sl_tp_orders(ex, symbol, side, qty, sl_price, tp_price_50):
    """
    下止损(100%) + 50%止盈两张单；reduceOnly；workingType=CONTRACT_PRICE
    """
    try:
        # 价格精度
        try:
            slp = ex.price_to_precision(symbol, sl_price)
            tpp = ex.price_to_precision(symbol, tp_price_50)
        except Exception:
            slp = sl_price
            tpp = tp_price_50

        # 止损：全量
        sl_params = {"reduceOnly": True, "workingType": "CONTRACT_PRICE", "stopPrice": slp}
        # 止盈：50%
        tp_params = {"reduceOnly": True, "workingType": "CONTRACT_PRICE", "stopPrice": tpp}

        if side == "多":
            ex.create_order(symbol, type="STOP_MARKET", side="sell", amount=qty, params=sl_params)
            ex.create_order(symbol, type="TAKE_PROFIT_MARKET", side="sell", amount=qty * PARTIAL_TP_RATIO, params=tp_params)
        else:
            ex.create_order(symbol, type="STOP_MARKET", side="buy", amount=qty, params=sl_params)
            ex.create_order(symbol, type="TAKE_PROFIT_MARKET", side="buy", amount=qty * PARTIAL_TP_RATIO, params=tp_params)
        return True
    except Exception as e:
        log(f"创建 SL/TP 失败 {symbol}: {e}")
        return False

# 跟踪止损状态
trail_state = {}  # symbol -> {side, best, atr, qty, entry}

def update_trailing_stop(ex, symbol, last_price):
    st = trail_state.get(symbol)
    if not st: return
    side = st["side"]; best = st["best"]; atr = st["atr"]; qty = st["qty"]

    moved = False
    if side == "多":
        if last_price > best:
            trail_state[symbol]["best"] = last_price
        if last_price >= best + TRAIL_ATR_MULT * atr:
            new_sl = last_price - SL_ATR_MULT * atr
            try:
                ex.create_order(symbol, type="STOP_MARKET", side="sell", amount=qty,
                                params={"reduceOnly": True, "workingType":"CONTRACT_PRICE",
                                        "stopPrice": ex.price_to_precision(symbol, new_sl)})
                trail_state[symbol]["best"] = last_price
                moved = True
            except Exception as e:
                log(f"更新跟踪止损失败 {symbol}: {e}")
    else:
        if last_price < best:
            trail_state[symbol]["best"] = last_price
        if last_price <= best - TRAIL_ATR_MULT * atr:
            new_sl = last_price + SL_ATR_MULT * atr
            try:
                ex.create_order(symbol, type="STOP_MARKET", side="buy", amount=qty,
                                params={"reduceOnly": True, "workingType":"CONTRACT_PRICE",
                                        "stopPrice": ex.price_to_precision(symbol, new_sl)})
                trail_state[symbol]["best"] = last_price
                moved = True
            except Exception as e:
                log(f"更新跟踪止损失败 {symbol}: {e}")

    if moved:
        tg_send(f"🔧 跟踪止损上调 {symbol} side={side} new_best={fmt_price(trail_state[symbol]['best'])}")

def summarize(tf, side, det):
    return (f"{tf} | 方向:{side or '无'} 入场:{fmt_price(det['entry']) if det else '-'} | "
            f"EMA:{det['ema_trend'] if det else '-'} MACD:{round(det['macd'],4) if det else '-'} "
            f"RSI:{round(det['rsi'],2) if det else '-'} WR:{round(det['wr'],2) if det else '-'} "
            f"KDJ:{det['k_trend'] if det else '-'} VOLΔ:{round(det['vol_trend'],3) if det else '-'} "
            f"ATR:{round(det['atr'],2) if det else '-'}")

# ========= 主循环 =========
def main():
    try:
        ex = build_exchange()
    except Exception as e:
        log(f"交易所初始化失败: {e}")
        return

    # 杠杆
    if MARKET_TYPE == "future":
        for sym in SYMBOLS:
            try:
                set_leverage_safe(ex, sym, LEVERAGE)
            except Exception as e:
                log(f"设置杠杆异常 {sym}: {e}")

    log(f"启动Bot {EXCHANGE_NAME}/{MARKET_TYPE} LIVE={LIVE_TRADE}")
    tg_send(f"🤖 Bot启动 {EXCHANGE_NAME}/{MARKET_TYPE} 模式={'实盘' if LIVE_TRADE==1 else '纸面'} 杠杆x{LEVERAGE}")

    last_hour_push = 0

    while True:
        loop_start = time.time()
        try:
            for symbol in SYMBOLS:
                # 多周期分析
                sides=[]; details={}
                for tf in TIMEFRAMES:
                    try:
                        df = fetch_df(ex, symbol, tf, 200)
                        side, det = analyze_df(df)
                        sides.append(side)
                        details[tf]=(side, det)
                        log(summarize(tf, side, det))
                    except Exception as e:
                        log(f"❌ 拉取/分析失败 {symbol} {tf}: {e}")
                        sides.append(None)
                        details[tf]=(None, None)

                bull = sum(1 for s in sides if s=="多")
                bear = sum(1 for s in sides if s=="空")
                final_side = None
                if bull>=REQUIRED_CONFIRMS and bull>bear:
                    final_side = "多"
                elif bear>=REQUIRED_CONFIRMS and bear>bull:
                    final_side = "空"

                # 每小时TG汇报一次（如果没异常）
                now_ts = int(time.time())
                if now_ts - last_hour_push >= 3600:
                    try:
                        lines = [f"⏰ {symbol} 多周期统计: 多={bull} 空={bear}"]
                        for tf in TIMEFRAMES:
                            s, det = details[tf]
                            lines.append(summarize(tf, s, det))
                        tg_send("\n".join(lines))
                    except Exception:
                        pass

                # 开仓&风控
                if final_side and details.get("1h") and details["1h"][1] is not None:
                    d1h = details["1h"][1]
                    entry = d1h["entry"]
                    atr1h = d1h["atr"]
                    qty = calc_qty(ex, symbol, entry)
                    if qty <= 0:
                        log(f"{symbol} 数量=0，跳过")
                    else:
                        if final_side == "多":
                            sl = entry - SL_ATR_MULT * atr1h
                            tp50 = entry + TP_ATR_MULT * atr1h
                            side_str = "buy"
                        else:
                            sl = entry + SL_ATR_MULT * atr1h
                            tp50 = entry - TP_ATR_MULT * atr1h
                            side_str = "sell"

                        if LIVE_TRADE != 1:
                            log(f"[纸面单] {symbol} {final_side} qty≈{qty} entry≈{fmt_price(entry)} SL≈{fmt_price(sl)} TP50≈{fmt_price(tp50)} ATR1h≈{fmt_price(atr1h)}")
                            tg_send(f"🧾 纸面单 {symbol} {final_side} qty≈{qty} entry≈{fmt_price(entry)} SL≈{fmt_price(sl)} TP50≈{fmt_price(tp50)}")
                        else:
                            try:
                                o = ex.create_order(symbol, type="MARKET", side=side_str, amount=qty)
                                log(f"[下单成功] {symbol} {side_str} qty={qty}")
                                tg_send(f"⚡ 开仓 {symbol} {final_side} qty≈{qty} entry≈{fmt_price(entry)}")
                                ok = create_sl_tp_orders(ex, symbol, final_side, qty, sl, tp50)
                                if not ok:
                                    tg_send(f"⚠️ {symbol} SL/TP 挂单失败，请手动检查")
                            except Exception as e:
                                log(f"下单失败 {symbol}: {e}")
                                tg_send(f"❌ 下单失败 {symbol}: {e}")

                        trail_state[symbol] = {"side": final_side, "best": entry, "atr": atr1h, "qty": qty, "entry": entry}

                # 跟踪止损更新
                try:
                    ticker = ex.fetch_ticker(symbol)
                    last_price = float(ticker.get("last") or ticker.get("close") or 0.0)
                    if last_price and symbol in trail_state:
                        update_trailing_stop(ex, symbol, last_price)
                except Exception as e:
                    log(f"更新跟踪止损失败 {symbol}: {e}")

            if now_ts - last_hour_push >= 3600:
                last_hour_push = now_ts

        except Exception as e:
            log(f"[主循环异常] {e}\n{traceback.format_exc()}")

        used = time.time() - loop_start
        time.sleep(max(1, POLL_INTERVAL - int(used)))

if __name__ == "__main__":
    main()
