# autotrader.py
"""
自动合约交易器（示例）
功能概述：
 - 多周期（1h/4h/1d）技术面分析（EMA, MACD, RSI, Williams %R, Stochastic KDJ, ATR, 成交量变化）
 - 多周期共识开仓（REQUIRED_CONFIRMS）
 - 开仓前严格 1h+4h MACD 动能过滤（同时为“增强”，遇“翻转”拒绝）
 - 使用 1h ATR 计算止损/止盈（SL_ATR_MULT / TP_ATR_MULT）
 - 跟踪止盈（TRAIL_ATR_MULT）
 - 4h MACD 弱化 + RSI 过滤触发部分止盈（PARTIAL_TP_RATIO）
 - 多种方式尝试设置杠杆（兼容不同 ccxt 版本）
 - Telegram 推送（可选）
注意：
 - 强烈建议在 LIVE_TRADE=0（纸面）下充分测试。
 - 若 Binance API 使用 IP 白名单，需把运行机器的出口 IP 加入白名单或临时取消白名单。
 - 下单/挂单的 params 在不同 ccxt 版本上可能需调整；如出现下单错误把错误日志贴给我我帮你修。
"""

import os
import time
import traceback
from datetime import datetime
from typing import Optional, Tuple

import requests
import ccxt
import pandas as pd
import ta
from dotenv import load_dotenv

load_dotenv()

# ========== 配置（从 .env 读取） ==========
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "").strip()

EXCHANGE_NAME = os.getenv("EXCHANGE", "binance").lower()
API_KEY = os.getenv("API_KEY", "").strip()
API_SECRET = os.getenv("API_SECRET", "").strip()

MARKET_TYPE = os.getenv("MARKET_TYPE", "future").lower()  # future / spot
LEVERAGE = int(os.getenv("LEVERAGE", "10"))
BASE_USDT = float(os.getenv("BASE_USDT", "15"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))
LIVE_TRADE = int(os.getenv("LIVE_TRADE", "0"))  # 0=paper, 1=live

# default trade list (you can change in .env: TRADE_SYMBOLS)
TRADE_SYMBOLS = [s.strip() for s in os.getenv("TRADE_SYMBOLS", "BTC/USDT,ETH/USDT,LTC/USDT,BNB/USDT,DOGE/USDT").split(",") if s.strip()]
OBSERVE_SYMBOLS = [s.strip() for s in os.getenv("OBSERVE_SYMBOLS", "SOL/USDT,XRP/USDT").split(",") if s.strip()]
ALL_SYMBOLS = TRADE_SYMBOLS + OBSERVE_SYMBOLS

TIMEFRAMES = ["1h", "4h", "1d"]
REQUIRED_CONFIRMS = int(os.getenv("REQUIRED_CONFIRMS", "2"))

SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "2.0"))
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "3.0"))
TRAIL_ATR_MULT = float(os.getenv("TRAIL_ATR_MULT", "1.5"))
PARTIAL_TP_RATIO = float(os.getenv("PARTIAL_TP_RATIO", "0.5"))  # 部分止盈 50%
MACD_FILTER_TIMEFRAME = os.getenv("MACD_FILTER_TIMEFRAME", "4h")

MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "6"))

# ========== 日志 / 通知 ==========
def nowstr() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str):
    print(f"[{nowstr()}] {msg}", flush=True)

def tg_send(text: str):
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                      data={"chat_id": TG_CHAT, "text": text}, timeout=10)
    except Exception as e:
        log(f"TG 发送失败: {e}")

# ========== 交易所构建 ==========
def build_exchange():
    """优先尝试 binanceusdm（若 ccxt 支持），否则回退到 binance with defaultType=future"""
    last_err = None
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
            return ex
    except Exception as e:
        last_err = e
        log(f"binanceusdm init failed: {e}")

    try:
        ex = getattr(ccxt, "binance")({
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        ex.load_markets()
        log("使用 ccxt.binance 初始化（fallback，options defaultType=future）")
        return ex
    except Exception as e:
        last_err = e
        log(f"ccxt.binance init failed: {e}")

    raise RuntimeError(f"交易所初始化失败，请检查 API_KEY/API_SECRET 与 ccxt。最后错误: {last_err}")

# ========== 杠杆设置（try 多种方法） ==========
def set_leverage_safe(ex, symbol: str, lev: int):
    try:
        market = ex.market(symbol)
    except Exception as e:
        log(f"无法获取 market 信息 {symbol}: {e}")
        return
    tried = []
    # 方法1: fapiPrivate_post_leverage
    try:
        if hasattr(ex, "fapiPrivate_post_leverage"):
            ex.fapiPrivate_post_leverage({"symbol": market["id"], "leverage": int(lev)})
            log(f"{symbol} 杠杆设置为 {lev}x (fapiPrivate_post_leverage)")
            return
    except Exception as e:
        tried.append(f"fapiPrivate_post_leverage:{e}")
    # 方法2: private_post_leverage
    try:
        if hasattr(ex, "private_post_leverage"):
            ex.private_post_leverage({"symbol": market["id"], "leverage": int(lev)})
            log(f"{symbol} 杠杆设置为 {lev}x (private_post_leverage)")
            return
    except Exception as e:
        tried.append(f"private_post_leverage:{e}")
    # 方法3: unified set_leverage
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
    # 最后记录
    log(f"⚠️ 尝试设置杠杆失败（已尝试: {tried}） {symbol}")

# ========== 数据与指标 ==========
def df_from_ohlcv(ohlcv) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "vol"])
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def analyze_one_df(df: pd.DataFrame) -> Tuple[Optional[str], Optional[dict]]:
    if df is None or len(df) < 50:
        return None, None
    work = df.iloc[:-1].copy()
    close = work["close"]
    high = work["high"]
    low = work["low"]
    vol = work["vol"]

    ema5 = close.ewm(span=5).mean().iloc[-1]
    ema10 = close.ewm(span=10).mean().iloc[-1]
    ema30 = close.ewm(span=30).mean().iloc[-1]
    ema_trend = "多" if (ema5 > ema10 > ema30) else ("空" if (ema5 < ema10 < ema30) else "中性")

    macd = ta.trend.MACD(close)
    macd_hist_series = macd.macd_diff()
    macd_hist = float(macd_hist_series.iloc[-1])

    rsi = float(ta.momentum.RSIIndicator(close, 14).rsi().iloc[-1])
    wr = float(ta.momentum.WilliamsRIndicator(high, low, close, 14).williams_r().iloc[-1])

    stoch = ta.momentum.StochasticOscillator(high, low, close, 9, 3)
    k_val = float(stoch.stoch().iloc[-1])
    d_val = float(stoch.stoch_signal().iloc[-1])
    k_trend = "多" if k_val > d_val else ("空" if k_val < d_val else "中性")

    vol_trend = float((vol.iloc[-1] - vol.iloc[-2]) / (abs(vol.iloc[-2]) + 1e-12))
    atr = float(ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range().iloc[-1])
    entry = float(close.iloc[-1])

    score_bull = sum([ema_trend == "多", macd_hist > 0, rsi > 55, wr > -50, k_trend == "多", vol_trend > 0])
    score_bear = sum([ema_trend == "空", macd_hist < 0, rsi < 45, wr < -50, k_trend == "空", vol_trend < 0])

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

def get_macd_status(macd_hist_series) -> str:
    try:
        if macd_hist_series is None or len(macd_hist_series) < 2:
            return "未知"
        prev = float(macd_hist_series.iloc[-2])
        curr = float(macd_hist_series.iloc[-1])
        if (prev <= 0 and curr > 0) or (prev >= 0 and curr < 0):
            return "翻转"
        if curr > prev and curr > 0:
            return "增强"
        if curr < prev and curr > 0:
            return "减弱"
        if curr < prev and curr < 0:
            return "增强"
        if curr > prev and curr < 0:
            return "减弱"
        return "未知"
    except Exception:
        return "未知"

def summarize(tf: str, side: Optional[str], det: Optional[dict]) -> str:
    if not det:
        return f"{tf} | 方向:{side or '无'} 入场:-"
    macd_part = f"{round(det['macd'], 4)}"
    if tf == "4h":
        macd_part += f" ({get_macd_status(det.get('macd_hist_series'))})"
    return (f"{tf} | 方向:{side or '无'} 入场:{fmt_price(det['entry'])} | "
            f"EMA:{det['ema_trend']} MACD:{macd_part} "
            f"RSI:{round(det['rsi'],2)} WR:{round(det['wr'],2)} "
            f"KDJ:{det['k_trend']} VOLΔ:{round(det['vol_trend'],3)} ATR:{round(det['atr'],2)}")

# ========== 下单/格式化 ==========
def fmt_price(p: float) -> str:
    p = float(p)
    if p >= 100:
        return f"{p:.2f}"
    if p >= 1:
        return f"{p:.4f}"
    if p >= 0.01:
        return f"{p:.6f}"
    return f"{p:.8f}"

def amount_for_futures(ex, symbol: str, price: float) -> float:
    raw_qty = (BASE_USDT * LEVERAGE) / max(price, 1e-12)
    try:
        qty = ex.amount_to_precision(symbol, raw_qty)
    except Exception:
        qty = raw_qty
    try:
        return float(qty)
    except Exception:
        return float(raw_qty)

def create_sl_tp_orders(ex, symbol: str, side: str, qty: float, sl_price: float, tp_price: float) -> bool:
    try:
        params_sl = {"reduceOnly": True, "workingType": "CONTRACT_PRICE"}
        params_tp = {"reduceOnly": True, "workingType": "CONTRACT_PRICE"}
        try:
            params_sl["stopPrice"] = ex.price_to_precision(symbol, sl_price)
            params_tp["stopPrice"] = ex.price_to_precision(symbol, tp_price)
        except Exception:
            params_sl["stopPrice"] = sl_price
            params_tp["stopPrice"] = tp_price

        if side == "多":
            ex.create_order(symbol, type="STOP_MARKET", side="sell", amount=qty, params=params_sl)
            ex.create_order(symbol, type="TAKE_PROFIT_MARKET", side="sell", amount=qty, params=params_tp)
        else:
            ex.create_order(symbol, type="STOP_MARKET", side="buy", amount=qty, params=params_sl)
            ex.create_order(symbol, type="TAKE_PROFIT_MARKET", side="buy", amount=qty, params=params_tp)
        return True
    except Exception as e:
        log(f"创建 SL/TP 挂单失败 {symbol}: {e}")
        return False

# ========== 跟踪止盈与部分止盈 ==========
trail_state = {}  # symbol -> {side, best, atr, qty, entry, partial_done}

def update_trailing_stop(ex, symbol: str, last_price: float):
    st = trail_state.get(symbol)
    if not st:
        return
    side = st["side"]
    best = st["best"]
    atr = st["atr"]
    qty = st["qty"]
    moved = False
    if side == "多":
        if last_price > best:
            trail_state[symbol]["best"] = last_price
        if last_price >= best + TRAIL_ATR_MULT * atr:
            new_sl = last_price - SL_ATR_MULT * atr
            try:
                ex.create_order(symbol, type="STOP_MARKET", side="sell", amount=qty, params={
                    "reduceOnly": True, "stopPrice": ex.price_to_precision(symbol, new_sl), "workingType": "CONTRACT_PRICE"
                })
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
                ex.create_order(symbol, type="STOP_MARKET", side="buy", amount=qty, params={
                    "reduceOnly": True, "stopPrice": ex.price_to_precision(symbol, new_sl), "workingType": "CONTRACT_PRICE"
                })
                trail_state[symbol]["best"] = last_price
                moved = True
            except Exception as e:
                log(f"更新跟踪止损失败 {symbol}: {e}")
    if moved:
        tg_send(f"🔧 跟踪止损上调 {symbol} side={side} new_best={fmt_price(trail_state[symbol]['best'])}")

def macd_weakening_and_partial_tp(ex, symbol: str, last_price: float, tf4h_det):
    st = trail_state.get(symbol)
    if not st or st.get("partial_done"):
        return
    side = st["side"]
    entry = st["entry"]
    atr1h = st["atr"]
    qty_total = st["qty"]
    if side == "多":
        profit_ok = (last_price - entry) >= (1.0 * atr1h)
    else:
        profit_ok = (entry - last_price) >= (1.0 * atr1h)
    if not profit_ok:
        return
    if not tf4h_det or "macd_hist_series" not in tf4h_det:
        return
    macd_series = tf4h_det["macd_hist_series"]
    if len(macd_series) < 3:
        return
    hist_prev = float(macd_series.iloc[-2])
    hist_last = float(macd_series.iloc[-1])
    rsi4h = float(tf4h_det["rsi"])
    if side == "多":
        macd_weak = (hist_last > 0) and (hist_last < hist_prev) and (rsi4h > 65)
    else:
        macd_weak = (hist_last < 0) and (abs(hist_last) < abs(hist_prev)) and (rsi4h < 35)
    if not macd_weak:
        return
    reduce_qty = max(qty_total * PARTIAL_TP_RATIO, 0.0)
    if reduce_qty <= 0:
        return
    if LIVE_TRADE != 1:
        log(f"[纸面-提前止盈] {symbol} side={side} 减仓≈{reduce_qty} last≈{fmt_price(last_price)} entry≈{fmt_price(entry)} RSI4h={rsi4h:.2f}")
        trail_state[symbol]["partial_done"] = True
        tg_send(f"🟡 提前止盈(纸面) {symbol} {side} 减仓≈{reduce_qty:.6f} 价≈{fmt_price(last_price)} (4h MACD弱化)")
        return
    try:
        if side == "多":
            ex.create_order(symbol, type="MARKET", side="sell", amount=reduce_qty, params={"reduceOnly": True})
        else:
            ex.create_order(symbol, type="MARKET", side="buy", amount=reduce_qty, params={"reduceOnly": True})
        trail_state[symbol]["partial_done"] = True
        tg_send(f"🟢 提前止盈(已执行) {symbol} {side} 减仓≈{reduce_qty:.6f} 价≈{fmt_price(last_price)}")
        log(f"[提前止盈成功] {symbol} side={side} reduce={reduce_qty}")
    except Exception as e:
        log(f"[提前止盈失败] {symbol}: {e}")
        tg_send(f"❌ 提前止盈失败 {symbol}: {e}")

# ========== 开仓前严格 1h+4h MACD 检查 ==========
def should_open_trade(consensus: str, tf_details: dict) -> Tuple[bool, str, str]:
    def status_for(tf: str) -> str:
        tpl = tf_details.get(tf)
        if not tpl or tpl[1] is None:
            return "未知"
        return get_macd_status(tpl[1].get("macd_hist_series"))
    s1 = status_for("1h")
    s4 = status_for("4h")
    if s1 == "翻转" or s4 == "翻转":
        return False, s1, s4
    if s1 == "增强" and s4 == "增强":
        return True, s1, s4
    return False, s1, s4

# ========== 主循环 ==========
def main():
    try:
        ex = build_exchange()
    except Exception as e:
        log(f"交易所初始化失败: {e}")
        return

    log(f"TRADE_SYMBOLS={TRADE_SYMBOLS} OBSERVE_SYMBOLS={OBSERVE_SYMBOLS}")
    tg_send(f"🤖 启动Bot {EXCHANGE_NAME}/{MARKET_TYPE} 模式={'实盘' if LIVE_TRADE==1 else '纸面'} 杠杆x{LEVERAGE}")

    if MARKET_TYPE == "future":
        for s in TRADE_SYMBOLS:
            try:
                set_leverage_safe(ex, s, LEVERAGE)
            except Exception as e:
                log(f"设置杠杆异常 {s}: {e}")

    while True:
        loop_start = time.time()
        try:
            open_positions_count = len([k for k, v in trail_state.items() if v.get("qty", 0) > 0])

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

                bull = sum(1 for s in tf_sides if s == "多")
                bear = sum(1 for s in tf_sides if s == "空")
                consensus = None
                if bull >= REQUIRED_CONFIRMS and bull > bear:
                    consensus = "多"
                elif bear >= REQUIRED_CONFIRMS and bear > bull:
                    consensus = "空"

                # 每轮摘要推送（可调）
                try:
                    lines = [f"{symbol} 当前多周期共识:（多:{bull} 空:{bear}）"]
                    for tf in TIMEFRAMES:
                        s, det = tf_details[tf]
                        lines.append(summarize(tf, s, det))
                    tg_send("\n".join(lines))
                except Exception:
                    pass

                # 仅对 TRADE_SYMBOLS 做开仓逻辑
                if symbol in TRADE_SYMBOLS and consensus in ("多", "空"):
                    if open_positions_count >= MAX_OPEN_POSITIONS:
                        log(f"达到最大同时持仓 {MAX_OPEN_POSITIONS}，跳过 {symbol}")
                        continue

                    allow, s1_status, s4_status = should_open_trade(consensus, tf_details)
                    if not allow:
                        log(f"{symbol} {consensus} 被 MACD 动能过滤（1h+4h）— 跳过开仓 [{s1_status}/{s4_status}]")
                        tg_send(f"⚠️ {symbol} {consensus} 被 MACD 动能过滤（1h+4h），取消本次开仓 [{s1_status}/{s4_status}]")
                        continue

                    s1h, d1h = tf_details.get("1h", (None, None))
                    if not d1h:
                        continue
                    price = d1h["entry"]
                    atr1h = d1h["atr"]
                    qty = amount_for_futures(ex, symbol, price)
                    if qty <= 0:
                        log(f"{symbol} 数量计算为 0，跳过")
                        continue

                    if consensus == "多":
                        sl = price - SL_ATR_MULT * atr1h
                        tp = price + TP_ATR_MULT * atr1h
                    else:
                        sl = price + SL_ATR_MULT * atr1h
                        tp = price - TP_ATR_MULT * atr1h

                    if LIVE_TRADE != 1:
                        log(f"[纸面单] {symbol} {consensus} qty≈{qty} entry≈{fmt_price(price)} SL≈{fmt_price(sl)} TP≈{fmt_price(tp)} ATR1h≈{fmt_price(atr1h)}")
                        tg_send(f"🧾 纸面单 {symbol} {consensus} qty≈{qty} entry≈{fmt_price(price)} SL≈{fmt_price(sl)} TP≈{fmt_price(tp)}")
                    else:
                        try:
                            side_str = "buy" if consensus == "多" else "sell"
                            ex.create_order(symbol, type="MARKET", side=side_str, amount=qty)
                            log(f"[下单成功] {symbol} {side_str} qty={qty} entry≈{fmt_price(price)}")
                            tg_send(f"⚡ 开仓 {symbol} {consensus} entry≈{fmt_price(price)} qty≈{qty}\nSL:{fmt_price(sl)} TP:{fmt_price(tp)}")
                            ok = create_sl_tp_orders(ex, symbol, consensus, qty, sl, tp)
                            if not ok:
                                tg_send(f"⚠️ {symbol} SL/TP 挂单失败，请手动检查")
                        except Exception as e:
                            log(f"[下单失败] {symbol}: {e}")
                            tg_send(f"❌ 下单失败 {symbol}: {e}")
                            continue

                    # 初始化内存跟踪状态（确保这一行完整）
                    trail_state[symbol] = {
                        "side": consensus,
                        "best": price,
                        "atr": atr1h,
                        "qty": qty,
                        "entry": price,
                        "partial_done": False
                    }
                    open_positions_count = len([k for k, v in trail_state.items() if v.get("qty", 0) > 0])

                # 更新跟踪止盈 + 检查提前止盈
                try:
                    ticker = ex.fetch_ticker(symbol)
                    last_price = float(ticker.get("last") or ticker.get("close") or 0.0)
                    if last_price:
                        update_trailing_stop(ex, symbol, last_price)
                        if symbol in trail_state and "4h" in tf_details:
                            macd_weakening_and_partial_tp(ex, symbol, last_price, tf_
