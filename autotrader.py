# autotrader.py
"""
Hedge Mode 完整版 AutoTrader
- 多周期共振 (1h, 4h, 1d)
- Hedge Mode 强制使用 positionSide (LONG/SHORT)
- ATR 计算 TP/SL，支持分批止盈 PARTIAL_TP_RATIO（可选）
- 每个币每小时汇总 Telegram（避免刷屏）
- 出错（如 margin insufficient）冷却处理
- LIVE_TRADE=0 为模拟（不实际下单）
"""
import os
import time
import math
import ccxt
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import ta

# ---------- 配置（来自 ENV） ----------
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT").split(",") if s.strip()]
BASE_USDT = float(os.getenv("BASE_USDT", "100"))         # 每次单个币基础名义资金（可按需要调整）
LEVERAGE = int(os.getenv("LEVERAGE", "10"))
LIVE_TRADE = os.getenv("LIVE_TRADE", "0") == "1"
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "3.0"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "2.0"))
PARTIAL_TP_RATIO = float(os.getenv("PARTIAL_TP_RATIO", "0.0"))  # 0 = 不分批
OHLCV_LIMIT = int(os.getenv("OHLCV_LIMIT", "200"))

SUMMARY_INTERVAL = int(os.getenv("SUMMARY_INTERVAL", "3600"))  # 每币种多久汇总推送一次（秒）
MARGIN_COOLDOWN = int(os.getenv("MARGIN_COOLDOWN", "3600"))    # 保证金不足冷却时间（秒）

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

# ---------- 交易所初始化 ----------
exchange = ccxt.binance({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "enableRateLimit": True,
    "options": {"defaultType": "future"},
})

# ---------- 工具 ----------
def now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def send_telegram(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram 未配置，消息将只打印:", msg)
        return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                      json={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
    except Exception as e:
        print("❌ Telegram 推送失败:", e)

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

# ---------- 市场与账户设置 ----------
def load_markets_safe():
    try:
        exchange.load_markets()
    except Exception as e:
        print("⚠️ load_markets 失败:", e)

def symbol_id(symbol):
    try:
        return exchange.market(symbol)["id"]
    except Exception:
        return symbol.replace("/", "")

def is_hedge_mode():
    """检测是否为 hedge (dual side) 模式；若检测失败，默认 True（以 hedge 为优先）"""
    try:
        info = exchange.fapiPrivate_get_positionmode()
        return bool(info.get("dualSidePosition") is True)
    except Exception:
        # 部分 ccxt 版本或权限可能失败，假设为 hedge（因为你指定要 hedge 脚本）
        return True

def ensure_leverage_and_margin(symbol):
    sid = symbol_id(symbol)
    # 尝试设置杠杆与保证金（容错）
    try:
        if hasattr(exchange, "set_leverage"):
            exchange.set_leverage(LEVERAGE, symbol)
        else:
            exchange.fapiPrivate_post_leverage({"symbol": sid, "leverage": LEVERAGE})
    except Exception as e:
        print(f"⚠️ 设置杠杆失败 {symbol}: {e}")
    try:
        if hasattr(exchange, "set_margin_mode"):
            exchange.set_margin_mode("ISOLATED", symbol)
        else:
            exchange.fapiPrivate_post_margintype({"symbol": sid, "marginType": "ISOLATED"})
    except Exception as e:
        print(f"⚠️ 设置保证金模式失败 {symbol}: {e}")

# ---------- OHLCV 与指标 ----------
def fetch_ohlcv_df(symbol, timeframe="1h", limit=OHLCV_LIMIT):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    except Exception as e:
        print(f"⚠️ fetch_ohlcv_df 失败 {symbol} {timeframe}: {e}")
        return pd.DataFrame()

def compute_indicators(df: pd.DataFrame):
    if df.empty:
        return df
    df = df.copy()
    df["ema20"] = ta.trend.EMAIndicator(df["close"], window=20, fillna=True).ema_indicator()
    df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50, fillna=True).ema_indicator()
    macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9, fillna=True)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14, fillna=True).rsi()
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14, fillna=True).average_true_range()
    df["vol_ma20"] = df["volume"].rolling(20, min_periods=1).mean()
    return df

def signal_from_indicators(df: pd.DataFrame):
    if df.empty:
        return None, 0, [], None
    last = df.iloc[-1]
    score = 0
    reasons = []
    if last["ema20"] > last["ema50"]:
        score += 2; reasons.append("EMA 多头")
    else:
        score -= 2; reasons.append("EMA 空头")
    if last["macd"] > last["macd_signal"]:
        score += 1; reasons.append("MACD 多头")
    else:
        score -= 1; reasons.append("MACD 空头")
    if last["rsi"] > 60:
        score += 1; reasons.append(f"RSI 偏强 {last['rsi']:.1f}")
    elif last["rsi"] < 40:
        score -= 1; reasons.append(f"RSI 偏弱 {last['rsi']:.1f}")
    if "vol_ma20" in df.columns and last["volume"] > last["vol_ma20"] * 1.5:
        score += 1; reasons.append("成交量放大")
    if score >= 3:
        return "buy", score, reasons, last
    elif score <= -3:
        return "sell", score, reasons, last
    else:
        return None, score, reasons, last

def check_multi_tf(symbol):
    multi_signal = None
    reasons_all = []
    status = {}
    for tf in ["1h", "4h", "1d"]:
        df = compute_indicators(fetch_ohlcv_df(symbol, tf, 100))
        sig, score, reasons, last = signal_from_indicators(df)
        if last is not None:
            status[tf] = {"signal": sig, "score": score, "reasons": reasons, "last_close": safe_float(last["close"]), "atr": safe_float(last["atr"])}
        else:
            status[tf] = {"signal": None}
        if sig:
            reasons_all.extend([f"{tf}:{r}" for r in reasons])
            if multi_signal is None:
                multi_signal = sig
            elif multi_signal != sig:
                multi_signal = None
    return multi_signal, reasons_all, status

# ---------- 仓位管理 ----------
def fetch_all_positions():
    try:
        pos = exchange.fetch_positions()
        return pos if isinstance(pos, list) else []
    except Exception as e:
        print("⚠️ fetch_positions 失败:", e)
        return []

def parse_position_entry(pos):
    try:
        sym = pos.get("symbol") or (pos.get("info") or {}).get("symbol")
        amt = None
        if "positionAmt" in pos: amt = float(pos["positionAmt"])
        elif "contracts" in pos: amt = float(pos["contracts"])
        else:
            # other shapes
            amt = float(pos.get("amount", 0))
        if amt == 0:
            return None
        side = "long" if amt > 0 else "short"
        entry = pos.get("entryPrice") or (pos.get("info") or {}).get("entryPrice") or None
        return (sym, abs(float(amt)), side, safe_float(entry))
    except Exception:
        return None

def get_position(symbol):
    want = symbol.replace("/", "")
    for p in fetch_all_positions():
        parsed = parse_position_entry(p)
        if not parsed:
            continue
        sym, qty, side, entry = parsed
        if not sym:
            continue
        if sym.replace("/", "") == want:
            return {"symbol": symbol, "qty": qty, "side": side, "entry": entry, "raw": p}
    return None

# ---------- 资金/数量计算 ----------
def amount_from_usdt(symbol, price, usdt_amount):
    """
    根据名义(usdt_amount)和价格计算合约数量（币数量/合约手数）
    使用 exchange.amount_to_precision 兼容交易所精度
    """
    if price <= 0:
        return 0
    base_qty = usdt_amount / price
    try:
        qty = float(exchange.amount_to_precision(symbol, base_qty))
    except Exception:
        qty = round(base_qty, 6)
    return qty

# ---------- 下单 / 平仓（Hedge Mode 固定带 positionSide） ----------
def place_market_with_positionSide(symbol, side, qty):
    """
    side: 'buy' 或 'sell'
    在 Hedge Mode 下，必须传 positionSide (LONG/SHORT)
    """
    if qty <= 0:
        return False, "qty_zero"
    pos_side = "LONG" if side == "buy" else "SHORT"
    params = {"positionSide": pos_side}
    try:
        if not LIVE_TRADE:
            print(f"💡 模拟下单 {symbol} {side} qty={qty} positionSide={pos_side}")
            return True, None
        order = exchange.create_order(symbol, "market", side, qty, None, params)
        return True, order
    except Exception as e:
        msg = getattr(e, "args", [str(e)])[0]
        return False, msg

def close_position_market_with_positionSide(symbol, position):
    """
    在 Hedge Mode 下，平仓时也需要传 positionSide 与持仓一致
    """
    if not position or not position.get("qty"):
        return True
    pos_side = position.get("side", "").lower()
    # 平空 -> buy (positionSide=SHORT), 平多 -> sell (positionSide=LONG)
    action = "buy" if pos_side == "short" else "sell"
    params = {}
    params["positionSide"] = "SHORT" if pos_side == "short" else "LONG"
    qty = position["qty"]
    try:
        if not LIVE_TRADE:
            print(f"💡 模拟平仓 {symbol} {pos_side} qty={qty} positionSide={params['positionSide']}")
            return True
        order = exchange.create_order(symbol, "market", action, qty, None, params)
        send_telegram(f"✅ 已市价平仓 {symbol} {pos_side} 数量={qty}")
        return True
    except Exception as e:
        send_telegram(f"❌ 平仓失败 {symbol}：{e}")
        return False

# ---------- 挂 TP/SL（用条件市价单） ----------
def place_tp_sl_orders(symbol, side, qty, tp_price, sl_price):
    """
    用 TAKE_PROFIT_MARKET / STOP_MARKET 下条件单，配合 positionSide
    side: 'buy' 或 'sell' 是开仓方向
    tp_price/sl_price: 触发价
    """
    pos_side = "LONG" if side == "buy" else "SHORT"
    close_side = "sell" if side == "buy" else "buy"
    results = []
    # TAKE_PROFIT_MARKET
    try:
        params_tp = {"positionSide": pos_side, "stopPrice": tp_price}
        if LIVE_TRADE:
            exchange.create_order(symbol, "TAKE_PROFIT_MARKET", close_side, qty, None, params_tp)
        else:
            print(f"💡 模拟挂 TP {symbol} qty={qty} tp={tp_price} positionSide={pos_side}")
        results.append(("tp", True))
    except Exception as e:
        results.append(("tp", str(e)))
    # STOP_MARKET (SL)
    try:
        params_sl = {"positionSide": pos_side, "stopPrice": sl_price}
        if LIVE_TRADE:
            exchange.create_order(symbol, "STOP_MARKET", close_side, qty, None, params_sl)
        else:
            print(f"💡 模拟挂 SL {symbol} qty={qty} sl={sl_price} positionSide={pos_side}")
        results.append(("sl", True))
    except Exception as e:
        results.append(("sl", str(e)))
    return results

# ---------- 状态与节流 ----------
last_summary_time = {}   # 每币种上次汇总时间 (datetime)
last_executed_signal = {}  # 每币种上次已执行方向 'buy'/'sell'/None
cooldown_until = {}        # 每币种冷却到期 (datetime)

# ---------- 主循环 ----------
def main_loop():
    load_markets_safe()
    # 尝试设置杠杆与保证金模式（容错）
    for s in SYMBOLS:
        ensure_leverage_and_margin(s)

    hedge = is_hedge_mode()
    send_telegram(f"🤖 Bot 启动 - Hedge Mode={hedge} LIVE_TRADE={LIVE_TRADE} SYMBOLS={','.join(SYMBOLS)}")

    while True:
        try:
            now = datetime.now(timezone.utc)

            for symbol in SYMBOLS:
                # 冷却处理
                if symbol in cooldown_until and now < cooldown_until[symbol]:
                    continue

                # 多周期共振
                signal, reasons, status = check_multi_tf(symbol)
                price = status.get("1h", {}).get("last_close") or 0.0
                atr = status.get("1h", {}).get("atr") or None

                # 每小时汇总推送（每币种）
                last_sum = last_summary_time.get(symbol)
                if last_sum is None or (now - last_sum).total_seconds() >= SUMMARY_INTERVAL:
                    pr = f"{price:.2f}" if price else "0"
                    reason_str = ";".join(reasons) if reasons else "无"
                    send_telegram(f"{now_str()} {symbol} 信号:{signal or '无'} 原因:{reason_str} 价格:{pr}")
                    last_summary_time[symbol] = now

                # 仅在信号发生改变时尝试执行（防刷屏/防重复下单）
                prev = last_executed_signal.get(symbol)
                if signal not in ("buy", "sell"):
                    continue

                if signal == prev:
                    # same signal already executed -> skip
                    continue

                # 获取当前仓位
                pos = get_position(symbol)
                need_close_and_reverse = pos and ((signal == "buy" and pos["side"] == "short") or (signal == "sell" and pos["side"] == "long"))

                # 计算实际用于本次开仓的资金：优先使用 BASE_USDT，但若账户资金少可以按比例减小（不做自动增杠杆）
                # 简化策略：用 BASE_USDT（用户需根据余额与杠杆保证该数值不会触发 margin insufficient）
                if price <= 0 or math.isnan(price):
                    continue
                qty = amount_from_usdt(symbol, price, BASE_USDT)

                # 如果需要先平仓（反向仓存在），先平仓
                if need_close_and_reverse:
                    ok = close_position_market_with_positionSide(symbol, pos)
                    if not ok:
                        # 平仓失败，跳过并不再尝试立即开仓
                        continue
                    # 睡一小会儿让位置更新
                    time.sleep(1)

                # 再次确认是否已有同向仓（可能平仓后已无仓）
                pos2 = get_position(symbol)
                has_same = pos2 and ((signal == "buy" and pos2["side"] == "long") or (signal == "sell" and pos2["side"] == "short"))
                if has_same:
                    last_executed_signal[symbol] = signal
                    continue

                # 下市价开仓（Hedge Mode 下带 positionSide）
                ok, err = place_market_with_positionSide(symbol, signal, qty)
                if ok:
                    # 下单成功后挂 TP/SL（条件单），并支持 PARTIAL_TP_RATIO
                    if atr is None or np.isnan(atr):
                        atr = price * 0.005
                    if signal == "buy":
                        tp_price = price + TP_ATR_MULT * atr
                        sl_price = price - SL_ATR_MULT * atr
                    else:
                        tp_price = price - TP_ATR_MULT * atr
                        sl_price = price + SL_ATR_MULT * atr

                    # 若 PARTIAL_TP_RATIO>0 则先挂一笔部分 TP，再挂剩余 TP（这里只做示意：直接用同价位分两笔）
                    if PARTIAL_TP_RATIO > 0 and PARTIAL_TP_RATIO < 1:
                        qty_first = round(qty * PARTIAL_TP_RATIO, 6)
                        qty_rest = round(qty - qty_first, 6)
                        # 挂第一批 TP
                        place_tp_sl_orders(symbol, signal, qty_first, tp_price, sl_price)
                        # 挂剩余 TP（或同价位）
                        if qty_rest > 0:
                            place_tp_sl_orders(symbol, signal, qty_rest, tp_price, sl_price)
                    else:
                        # 全仓挂 TP/SL
                        place_tp_sl_orders(symbol, signal, qty, tp_price, sl_price)

                    send_telegram(f"✅ {symbol} 开仓 {signal} qty={qty} @ {price:.2f} TP≈{tp_price:.2f} SL≈{sl_price:.2f}")
                    last_executed_signal[symbol] = signal
                else:
                    # 错误处理（解析常见错误）
                    errstr = str(err)
                    send_telegram(f"❌ 下单失败 {symbol} {signal}：{errstr}")
                    # margin insufficient -> 冷却
                    if "-2019" in errstr or "Margin is insufficient" in errstr:
                        cooldown_until[symbol] = now + timedelta(seconds=MARGIN_COOLDOWN)
                        send_telegram(f"⏸ {symbol} 因保证金不足进入冷却到 {cooldown_until[symbol].strftime('%Y-%m-%d %H:%M:%S UTC')}")
                    # position side mismatch -> 提示检查
                    if "-4061" in errstr:
                        send_telegram(f"⚠️ {symbol} 报 -4061( position side mismatch )，请确认账户确实为 Hedge Mode 并且 API 有权限。")
            # main for end

            time.sleep(POLL_INTERVAL)

        except Exception as e:
            print("⚠️ 主循环异常:", e)
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main_loop()
