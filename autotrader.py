# autotrader.py
# 完整版：Binance futures (USDM) + 多周期策略 + ATR SL/TP + 跟踪止盈 + 1h+4h MACD 过滤
import os
import time
import math
import traceback
from datetime import datetime

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

EXCHANGE_NAME = os.getenv("EXCHANGE", "binance").lower()
API_KEY   = os.getenv("API_KEY", "").strip()
API_SECRET= os.getenv("API_SECRET", "").strip()

MARKET_TYPE = os.getenv("MARKET_TYPE", "future").lower()  # future / spot
LEVERAGE  = int(os.getenv("LEVERAGE", "10"))
BASE_USDT = float(os.getenv("BASE_USDT", "15"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))
LIVE_TRADE    = int(os.getenv("LIVE_TRADE", "0"))  # 0 paper, 1 live

TRADE_SYMBOLS   = [s.strip() for s in os.getenv("TRADE_SYMBOLS", "BTC/USDT,ETH/USDT").split(",") if s.strip()]
OBSERVE_SYMBOLS = [s.strip() for s in os.getenv("OBSERVE_SYMBOLS", "LTC/USDT,BNB/USDT,SOL/USDT,XRP/USDT").split(",") if s.strip()]
ALL_SYMBOLS = TRADE_SYMBOLS + OBSERVE_SYMBOLS

TIMEFRAMES = ["1h", "4h", "1d"]
REQUIRED_CONFIRMS = int(os.getenv("REQUIRED_CONFIRMS", "2"))

SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "2.0"))
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "3.0"))
TRAIL_ATR_MULT = float(os.getenv("TRAIL_ATR_MULT", "1.5"))
PARTIAL_TP_RATIO = float(os.getenv("PARTIAL_TP_RATIO", "0.3"))
MACD_FILTER_TIMEFRAME = os.getenv("MACD_FILTER_TIMEFRAME", "4h")

# Safety limits
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "5"))  # 可选：最多同时开仓数

# ========= helpers =========
def nowstr():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"[{nowstr()}] {msg}", flush=True)

def tg_send(text):
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                      data={"chat_id": TG_CHAT, "text": text}, timeout=10)
    except Exception as e:
        log(f"TG发送失败: {e}")

# ========= exchange builder (兼容 binanceusdm / binance) =========
def build_exchange():
    """
    优先尝试 ccxt.binanceusdm（USDM 永续）。若不可用，使用 ccxt.binance 并设置 options.
    返回 exchange 对象和一个 helper set_leverage(ex, symbol, lev) 函数。
    """
    ex = None
    set_lev_fn = None
    # try binanceusdm
    try:
        if hasattr(ccxt, "binanceusdm"):
            ex = getattr(ccxt, "binanceusdm")({
                "apiKey": API_KEY,
                "secret": API_SECRET,
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            })
            ex.load_markets()
            log("使用 ccxt.binanceusdm 初始化（USDM futures）")
            def set_lev(exch, symbol, lev):
                try:
                    market = exch.market(symbol)
                    # 有些 ccxt 实现用 fapiPrivate_post_leverage
                    try:
                        exch.fapiPrivate_post_leverage({"symbol": market["id"], "leverage": int(lev)})
                    except Exception:
                        # fallback to unified method
                        try:
                            exch.set_leverage(int(lev), market["symbol"])
                        except Exception:
                            log(f"尝试设置杠杆失败（多个方法） {symbol}")
                except Exception as e:
                    log(f"设置杠杆异常: {e}")
            set_lev_fn = set_lev
            return ex, set_lev_fn
    except Exception as e:
        log(f"binanceusdm init failed: {e}")

    # fallback to ccxt.binance
    try:
        ex = getattr(ccxt, "binance")({
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "future"}  # try to use futures endpoints
        })
        ex.load_markets()
        log("使用 ccxt.binance 初始化（fallback，options defaultType=future）")
        def set_lev(exch, symbol, lev):
            try:
                market = exch.market(symbol)
                # unified set_leverage if exists
                if hasattr(exch, "set_leverage"):
                    try:
                        exch.set_leverage(int(lev), market["symbol"])
                        return
                    except Exception:
                        pass
                # try fapiPrivate_post_leverage if available
                try:
                    exch.fapiPrivate_post_leverage({"symbol": market["id"], "leverage": int(lev)})
                    return
                except Exception:
                    pass
                log(f"无法通过已知方法设置杠杆 {symbol}")
            except Exception as e:
                log(f"设置杠杆异常: {e}")
        set_lev_fn = set_lev
        return ex, set_lev_fn
    except Exception as e:
        log(f"ccxt.binance init failed: {e}")
        raise RuntimeError("初始化交易所失败，请检查 ccxt 版本与环境变量")

# ========= OHLCV -> df =========
def df_from_ohlcv(ohlcv):
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","vol"])
    for c in ["open","high","low","close","vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ========= indicators & decision =========
def analyze_one_df(df):
    """
    输入: pandas df (ts,open,high,low,close,vol)
    返回: (side, det) side in {'多','空',None}
    det 包含 macd_hist_series, atr, entry 等
    """
    if df is None or len(df) < 50:
        return None, None
    work = df.iloc[:-1].copy()  # 用已收盘K
    close = work["close"]
    high = work["high"]
    low = work["low"]
    vol = work["vol"]

    # EMA trend
    ema5  = close.ewm(span=5).mean().iloc[-1]
    ema10 = close.ewm(span=10).mean().iloc[-1]
    ema30 = close.ewm(span=30).mean().iloc[-1]
    ema_trend = "多" if (ema5>ema10>ema30) else ("空" if (ema5<ema10<ema30) else "中性")

    # MACD
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
    atr = float(ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range().iloc[-1])
    entry = float(close.iloc[-1])

    score_bull = sum([ema_trend=="多", macd_hist>0, rsi>55, wr>-50, k_trend=="多", vol_trend>0])
    score_bear = sum([ema_trend=="空", macd_hist<0, rsi<45, wr<-50, k_trend=="空", vol_trend<0])

    side = None
    if score_bull>=4 and score_bull>=score_bear+2:
        side="多"
    elif score_bear>=4 and score_bear>=score_bull+2:
        side="空"

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

def get_macd_status(macd_hist_series):
    """
    返回: '增强' / '减弱' / '翻转' / '未知'
    """
    try:
        if macd_hist_series is None or len(macd_hist_series) < 2:
            return "未知"
        prev = float(macd_hist_series.iloc[-2])
        curr = float(macd_hist_series.iloc[-1])
        # 翻转优先
        if (prev <= 0 and curr > 0) or (prev >= 0 and curr < 0):
            return "翻转"
        if curr > prev and curr > 0:
            return "增强"
        elif curr < prev and curr > 0:
            return "减弱"
        elif curr < prev and curr < 0:
            return "增强"
        elif curr > prev and curr < 0:
            return "减弱"
        return "未知"
    except Exception:
        return "未知"

def macd_strength_label(macd_hist_series):
    return get_macd_status(macd_hist_series) or "—"

def summarize(tf, side, det):
    if not det:
        return f"{tf} | 方向:{side or '无'} 入场:-"
    macd_part = f"{round(det['macd'],4)}"
    if tf == "4h":
        macd_part += f" ({macd_strength_label(det.get('macd_hist_series'))})"
    return (f"{tf} | 方向:{side or '无'} 入场:{fmt_price(det['entry'])} | "
            f"EMA:{det['ema_trend']} MACD:{macd_part} "
            f"RSI:{round(det['rsi'],2)} WR:{round(det['wr'],2)} "
            f"KDJ:{det['k_trend']} VOLΔ:{round(det['vol_trend'],3)} ATR:{round(det['atr'],2)}")

# ========= precision / order helpers =========
def fmt_price(p):
    p = float(p)
    if p>=100: return f"{p:.2f}"
    if p>=1:   return f"{p:.4f}"
    if p>=0.01:return f"{p:.6f}"
    return f"{p:.8f}"

def amount_for_futures(ex, symbol, price):
    raw_qty = (BASE_USDT * LEVERAGE) / max(price, 1e-12)
    # try to use exchange precision helper
    try:
        qty = ex.amount_to_precision(symbol, raw_qty)
    except Exception:
        qty = raw_qty
    try:
        return float(qty)
    except Exception:
        return float(raw_qty)

def create_sl_tp_orders(ex, symbol, side, qty, sl_price, tp_price):
    """
    挂止损/止盈 reduceOnly 单（市场止损/止盈市价触发）
    注意：不同 ccxt 版本参数差异较大，已做常见兼容尝试。
    """
    try:
        # prepare params
        params_sl = {"reduceOnly": True, "workingType": "CONTRACT_PRICE"}
        params_tp = {"reduceOnly": True, "workingType": "CONTRACT_PRICE"}
        # price precision
        try:
            params_sl["stopPrice"] = ex.price_to_precision(symbol, sl_price)
            params_tp["stopPrice"] = ex.price_to_precision(symbol, tp_price)
        except Exception:
            params_sl["stopPrice"] = sl_price
            params_tp["stopPrice"] = tp_price

        if side == "多":
            # reduce only sell orders
            ex.create_order(symbol, type="STOP_MARKET", side="sell", amount=qty, params=params_sl)
            ex.create_order(symbol, type="TAKE_PROFIT_MARKET", side="sell", amount=qty, params=params_tp)
        else:
            ex.create_order(symbol, type="STOP_MARKET", side="buy", amount=qty, params=params_sl)
            ex.create_order(symbol, type="TAKE_PROFIT_MARKET", side="buy", amount=qty, params=params_tp)
        return True
    except Exception as e:
        log(f"创建SL/TP失败 {symbol}: {e}")
        return False

# trail state in memory
trail_state = {}  # {symbol: {side,best,atr,qty,entry,partial_done}}

def update_trailing_stop(ex, symbol, last_price):
    st = trail_state.get(symbol)
    if not st: return
    side = st["side"]; best = st["best"]; atr = st["atr"]; qty = st["qty"]
    moved = False
    if side == "多":
        if last_price > best: trail_state[symbol]["best"] = last_price
        if last_price >= best + TRAIL_ATR_MULT * atr:
            new_sl = last_price - SL_ATR_MULT * atr
            try:
                ex.create_order(symbol, type="STOP_MARKET", side="sell", amount=qty, params={
                    "reduceOnly": True, "stopPrice": ex.price_to_precision(symbol, new_sl), "workingType":"CONTRACT_PRICE"
                })
                trail_state[symbol]["best"] = last_price
                moved = True
            except Exception as e:
                log(f"更新跟踪止损失败 {symbol}: {e}")
    else:
        if last_price < best: trail_state[symbol]["best"] = last_price
        if last_price <= best - TRAIL_ATR_MULT * atr:
            new_sl = last_price + SL_ATR_MULT * atr
            try:
                ex.create_order(symbol, type="STOP_MARKET", side="buy", amount=qty, params={
                    "reduceOnly": True, "stopPrice": ex.price_to_precision(symbol, new_sl), "workingType":"CONTRACT_PRICE"
                })
                trail_state[symbol]["best"] = last_price
                moved = True
            except Exception as e:
                log(f"更新跟踪止损失败 {symbol}: {e}")
    if moved:
        tg_send(f"🔧 跟踪止损上调 {symbol} side={side} new_best={fmt_price(trail_state[symbol]['best'])}")

def macd_weakening_and_partial_tp(ex, symbol, last_price, tf4h_details):
    st = trail_state.get(symbol)
    if not st or st.get("partial_done"): return
    side = st["side"]; entry = st["entry"]; atr1h = st["atr"]; qty_total = st["qty"]
    profit_ok = (last_price - entry) >= (1.0 * atr1h) if side=="多" else (entry - last_price) >= (1.0 * atr1h)
    if not profit_ok: return
    s4h, d4h = tf4h_details
    if not d4h or "macd_hist_series" not in d4h: return
    macd_hist_series = d4h["macd_hist_series"]
    if len(macd_hist_series) < 3: return
    hist_prev = float(macd_hist_series.iloc[-2]); hist_last = float(macd_hist_series.iloc[-1]); rsi4h = float(d4h["rsi"])
    macd_weak = False
    if side == "多":
        macd_weak = (hist_last > 0) and (hist_last < hist_prev) and (rsi4h > 65)
    else:
        macd_weak = (hist_last < 0) and (abs(hist_last) < abs(hist_prev)) and (rsi4h < 35)
    if not macd_weak: return
    reduce_qty = max(qty_total * PARTIAL_TP_RATIO, 0.0)
    if reduce_qty <= 0: return
    if LIVE_TRADE != 1:
        log(f"[纸面-提前止盈] {symbol} side={side} 减仓≈{reduce_qty} last≈{fmt_price(last_price)} entry≈{fmt_price(entry)} RSI4h={rsi4h:.2f}")
        trail_state[symbol]["partial_done"] = True
        tg_send(f"🟡 提前止盈(纸面) {symbol} {side} 减仓≈{reduce_qty:.6f} 价≈{fmt_price(last_price)} (4h MACD弱化 + RSI过滤)")
        return
    try:
        if side == "多":
            ex.create_order(symbol, type="MARKET", side="sell", amount=reduce_qty, params={"reduceOnly": True})
        else:
            ex.create_order(symbol, type="MARKET", side="buy", amount=reduce_qty, params={"reduceOnly": True})
        trail_state[symbol]["partial_done"] = True
        tg_send(f"🟢 提前止盈(已执行) {symbol} {side} 减仓≈{reduce_qty:.6f} 价≈{fmt_price(last_price)} (4h MACD弱化 + RSI过滤)")
        log(f"[提前止盈成功] {symbol} side={side} reduce={reduce_qty}")
    except Exception as e:
        log(f"[提前止盈失败] {symbol}: {e}")
        tg_send(f"❌ 提前止盈失败 {symbol}: {e}")

# ========= should_open_trade (1h + 4h 严格检查) =========
def should_open_trade(consensus, tf_details):
    """
    返回 (allow: bool, status1h: str, status4h: str)
    需要 1h 与 4h 都为 '增强' 才允许开仓；翻转/减弱/未知 均视为不允许。
    """
    def status_for(tf):
        tpl = tf_details.get(tf)
        if not tpl or tpl[1] is None:
            return "未知"
        det = tpl[1]
        return get_macd_status(det.get("macd_hist_series"))
    s1 = status_for("1h")
    s4 = status_for("4h")
    if s1 == "翻转" or s4 == "翻转":
        return False, s1, s4
    if s1 == "增强" and s4 == "增强":
        return True, s1, s4
    return False, s1, s4

# ========= 主循环 =========
def main():
    try:
        ex, set_leverage_fn = build_exchange()
    except Exception as e:
        log(f"交易所初始化失败: {e}")
        return

    tg_send(f"🤖 启动Bot {EXCHANGE_NAME}/{MARKET_TYPE} 模式={'实盘' if LIVE_TRADE==1 else '纸面'} 杠杆x{LEVERAGE}")
    log(f"TRADE_SYMBOLS={TRADE_SYMBOLS} OBSERVE_SYMBOLS={OBSERVE_SYMBOLS}")

    # set leverage for trade symbols if future
    if MARKET_TYPE == "future":
        for s in TRADE_SYMBOLS:
            try:
                set_leverage_fn(ex, s, LEVERAGE)
                log(f"{s} 尝试设置杠杆 {LEVERAGE}x")
            except Exception as e:
                log(f"设置杠杆失败 {s}: {e}")

    while True:
        loop_start = time.time()
        try:
            # limiting total open positions optionally
            open_positions_count = len([k for k,v in trail_state.items() if v.get("qty",0)>0])

            for symbol in ALL_SYMBOLS:
                tf_sides = []
                tf_details = {}

                for tf in TIMEFRAMES:
                    try:
                        ohlcv = ex.fetch_ohlcv(symbol, timeframe=tf, limit=200)
                        df = df_from_ohlcv(ohlcv)
                        side, det = analyze_one_df(df)
                        tf_sides.append(side)
                        tf_details[tf] = (side, det)
                        log(summarize(tf, side, det))
                    except Exception as e:
                        log(f"❌ 获取/分析失败 {symbol} {tf}: {e}")
                        tf_sides.append(None)
                        tf_details[tf] = (None, None)

                # consensus
                bull = sum(1 for s in tf_sides if s=="多")
                bear = sum(1 for s in tf_sides if s=="空")
                consensus = None
                if bull>=REQUIRED_CONFIRMS and bull>bear:
                    consensus="多"
                elif bear>=REQUIRED_CONFIRMS and bear>bull:
                    consensus="空"

                # push summary
                lines = [f"{symbol} 当前多周期共识:（多:{bull} 空:{bear}）"]
                for tf in TIMEFRAMES:
                    s, det = tf_details[tf]
                    lines.append(summarize(tf, s, det))
                tg_send("\n".join(lines))

                # trading only for TRADE_SYMBOLS
                if symbol in TRADE_SYMBOLS and consensus in ("多","空"):
                    # check max open positions
                    if open_positions_count >= MAX_OPEN_POSITIONS:
                        log(f"已达最大同时持仓数 {MAX_OPEN_POSITIONS}，跳过新开仓 {symbol}")
                        continue

                    allow, s1h_status, s4h_status = should_open_trade(consensus, tf_details)
                    if not allow:
                        status = f"{s1h_status}/{s4h_status}"
                        log(f"{symbol} {consensus} 被 MACD 动能过滤（1h+4h）— 跳过开仓 [{status}]")
                        tg_send(f"⚠️ {symbol} {consensus} 被 MACD 动能过滤（1h+4h），取消本次开仓 [{status}]")
                        continue

                    s1h, d1h = tf_details.get("1h", (None, None))
                    if not d1h:
                        continue
                    price = d1h["entry"]; atr1h = d1h["atr"]
                    qty = amount_for_futures(ex, symbol, price)
                    if qty <= 0:
                        log(f"{symbol} 数量过小，跳过")
                        continue
                    if consensus == "多":
                        sl = price - SL_ATR_MULT*atr1h
                        tp = price + TP_ATR_MULT*atr1h
                    else:
                        sl = price + SL_ATR_MULT*atr1h
                        tp = price - TP_ATR_MULT*atr1h

                    if LIVE_TRADE != 1:
                        log(f"[纸面单] {symbol} {consensus} 市价 数量≈{qty} 进场≈{fmt_price(price)} SL≈{fmt_price(sl)} TP≈{fmt_price(tp)} ATR1h≈{fmt_price(atr1h)}")
                        tg_send(f"🧾 纸面单 {symbol} {consensus} qty≈{qty} entry≈{fmt_price(price)} SL≈{fmt_price(sl)} TP≈{fmt_price(tp)}")
                    else:
                        try:
                            side_str = "buy" if consensus=="多" else "sell"
                            ex.create_order(symbol, type="MARKET", side=side_str, amount=qty)
                            log(f"[下单成功] {symbol} {side_str} qty={qty} entry≈{fmt_price(price)}")
                            tg_send(f"⚡ 开仓 {symbol} {consensus} 价≈{fmt_price(price)} 数量≈{qty}\nSL:{fmt_price(sl)} TP:{fmt_price(tp)} ATR1h:{fmt_price(atr1h)}")
                            ok = create_sl_tp_orders(ex, symbol, consensus, qty, sl, tp)
                            if not ok:
                                tg_send(f"⚠️ {symbol} SL/TP 挂单失败，请检查")
                        except Exception as e:
                            log(f"[下单失败] {symbol}: {e}")
                            tg_send(f"❌ 下单失败 {symbol}: {e}")
                            continue

                    # register trail state (paper and real both)
                    trail_state[symbol] = {"side": consensus, "best": price, "atr": atr1h, "qty": qty, "entry": price, "partial_done": False}
                    open_positions_count = len([k for k,v in trail_state.items() if v.get("qty",0)>0])

                # update trailing + partial tp check
                try:
                    ticker = ex.fetch_ticker(symbol)
                    last_price = float(ticker.get("last") or ticker.get("close") or 0.0)
                    if last_price:
                        update_trailing_stop(ex, symbol, last_price)
                        if symbol in trail_state and "4h" in tf_details:
                            macd_weakening_and_partial_tp(ex, symbol, last_price, tf_details["4h"])
                except Exception as e:
                    log(f"获取价格/更新止盈失败 {symbol}: {e}")

        except Exception as e:
            log(f"[主循环异常] {e}\n{traceback.format_exc()}")

        used = time.time() - loop_start
        time.sleep(max(1, POLL_INTERVAL - int(used)))

if __name__ == "__main__":
    main()
