# autotrader.py
"""
Merged Hedge Mode AutoTrader - 完整版 (已合并 RISK_RATIO / ATR / PARTIAL_TP / 修复 fetch)
功能：
- 多周期共振 (1h, 4h, 1d)
- Hedge Mode 强制使用 positionSide (LONG/SHORT)，若账户为单向则自动不传
- ATR 计算 TP/SL，支持分批止盈 PARTIAL_TP_RATIO（可选）
- 每个币每小时汇总 Telegram（避免刷屏）
- 出错（如 margin insufficient）冷却处理
- 动态仓位：使用 RISK_RATIO * 可用 USDT（默认 15%）
- 跳过小于交易所最小下单量的下单
- LIVE_TRADE 支持 env 写 "1" 或 "true"
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

# ================== 配置（ENV） ==================
# SYMBOLS env 例子: SYMBOLS=BTC/USDT,ETH/USDT,LTC/USDT,DOGE/USDT,BNB/USDT
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,LTC/USDT,DOGE/USDT,BNB/USDT").split(",") if s.strip()]

# 资金与仓位
BASE_USDT = float(os.getenv("BASE_USDT", "20"))         # 备选：每次单个币名义资金（若不使用 RISK_RATIO）
RISK_RATIO = float(os.getenv("RISK_RATIO", os.getenv("RISK_RATIO", "0.15")))  # 每次用可用 USDT 的比例，默认 15%
LEVERAGE = int(os.getenv("LEVERAGE", "10"))

# 运行与策略参数
# 支持 LIVE_TRADE=1 或 LIVE_TRADE=True 两种写法
LIVE_TRADE = os.getenv("LIVE_TRADE", "0").lower() in ("1", "true", "yes")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "3.0"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "2.0"))
PARTIAL_TP_RATIO = float(os.getenv("PARTIAL_TP_RATIO", "0.0"))  # 0 = 不分批
OHLCV_LIMIT = int(os.getenv("OHLCV_LIMIT", "200"))
SUMMARY_INTERVAL = int(os.getenv("SUMMARY_INTERVAL", "3600"))  # 每币种多久汇总推送一次（秒）
MARGIN_COOLDOWN = int(os.getenv("MARGIN_COOLDOWN", "3600"))    # 保证金不足冷却时间（秒）

# Telegram & API
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
EXCHANGE_ID = os.getenv("EXCHANGE", "binance")
MARKET_TYPE = os.getenv("MARKET_TYPE", "future")

# ================== 交易所初始化 ==================
exchange = getattr(ccxt, EXCHANGE_ID)({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "enableRateLimit": True,
    "options": {"defaultType": MARKET_TYPE},
})

# ================== 工具函数 ==================
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

# ================== 市场/账户设置 ==================
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
    """检测是否为 hedge 模式；若检测失败，返回 True（以 hedge 为优先）"""
    try:
        info = exchange.fapiPrivate_get_positionmode()
        return bool(info.get("dualSidePosition") is True)
    except Exception:
        return True

def ensure_leverage_and_margin(symbol):
    sid = symbol_id(symbol)
    # 尝试设置杠杆
    try:
        if hasattr(exchange, "set_leverage"):
            try:
                exchange.set_leverage(LEVERAGE, symbol)
                print(f"✅ {symbol} 杠杆设置成功 {LEVERAGE}x")
            except Exception as e:
                # 有时 ccxt 的 set_leverage 接口会报错，继续尝试备用接口
                print(f"⚠️ set_leverage 报错 {symbol}: {e}")
                try:
                    exchange.fapiPrivate_post_leverage({"symbol": sid, "leverage": LEVERAGE})
                    print(f"✅ {symbol} 杠杆设置成功 (备用接口) {LEVERAGE}x")
                except Exception as e2:
                    print(f"⚠️ 设置杠杆失败 {symbol}: {e2}")
        else:
            exchange.fapiPrivate_post_leverage({"symbol": sid, "leverage": LEVERAGE})
            print(f"✅ {symbol} 杠杆设置成功 (post) {LEVERAGE}x")
    except Exception as e:
        print(f"⚠️ 设置杠杆失败 {symbol}: {e}")

    # 尝试设置保证金模式（逐仓），若失败则提示并继续
    try:
        if hasattr(exchange, "set_margin_mode"):
            try:
                exchange.set_margin_mode("ISOLATED", symbol)
                print(f"✅ {symbol} 保证金模式设置成功 ISOLATED")
            except Exception as e:
                print(f"⚠️ set_margin_mode 报错 {symbol}: {e}")
                # 备用调用
                try:
                    exchange.fapiPrivate_post_margintype({"symbol": sid, "marginType": "ISOLATED"})
                    print(f"✅ {symbol} 保证金模式设置成功 (备用) ISOLATED")
                except Exception as e2:
                    print(f"⚠️ 设置保证金模式失败 {symbol}: {e2}")
        else:
            exchange.fapiPrivate_post_margintype({"symbol": sid, "marginType": "ISOLATED"})
            print(f"✅ {symbol} 保证金模式设置成功 (post) ISOLATED")
    except Exception as e:
        # 常见错误：已有仓位无法切换、Multi-Assets 模式不允许等，提示但不中断
        print(f"⚠️ 设置保证金模式失败 {symbol}: {e}")

# ================== OHLCV 与指标 ==================
def fetch_ohlcv_df(symbol, timeframe="1h", limit=OHLCV_LIMIT):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not ohlcv:
            print(f"⚠️ 没有 K 线数据 {symbol} {timeframe}")
            return pd.DataFrame()
        df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    except Exception as e:
        print(f"⚠️ fetch_ohlcv_df 失败 {symbol} {timeframe}: {e}")
        return pd.DataFrame()

def compute_indicators(df: pd.DataFrame):
    if df is None or df.empty:
        return df
    df = df.copy()
    df["ema20"] = ta.trend.EMAIndicator(df["close"], window=20, fillna=True).ema_indicator()
    df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50, fillna=True).ema_indicator()
    macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9, fillna=True)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14, fillna=True).rsi()
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14, fillna=True).average_true_range()
    df["vol_ma20"] = df["volume"].rolling(20, min_periods=1).mean()
    return df

def signal_from_indicators(df: pd.DataFrame):
    if df is None or df.empty:
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
        df = fetch_ohlcv_df(symbol, tf, 100)
        df = compute_indicators(df)
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

# ================== 仓位管理 ==================
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
        if "positionAmt" in pos:
            amt = float(pos["positionAmt"])
        elif "contracts" in pos:
            amt = float(pos["contracts"])
        else:
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

# ================== 数量/金额计算（改为支持 RISK_RATIO） ==================
def amount_from_usdt(symbol, price, usdt_amount=None):
    """
    计算合约数量：
    - 如果 usdt_amount 为 None，则使用 RISK_RATIO * 可用 USDT 余额（优先）
    - 否则使用提供的 usdt_amount（等价于 BASE_USDT）
    """
    try:
        if price <= 0:
            return 0
        # 优先使用 RISK_RATIO 基于账户可用余额
        if usdt_amount is None:
            # 读取账户可用 USDT (free)
            try:
                bal = exchange.fetch_balance()
                usdt_free = float(bal.get("free", {}).get("USDT", bal.get("total", {}).get("USDT", 0) or 0))
            except Exception:
                usdt_free = BASE_USDT
            use_usdt = usdt_free * RISK_RATIO
        else:
            use_usdt = usdt_amount

        # 名义资金乘以杠杆 -> 实际合约名义
        nominal = use_usdt * LEVERAGE
        base_qty = nominal / price

        # 取市场精度限制
        try:
            precision = exchange.markets.get(symbol, {}).get("precision", {}).get("amount")
            if precision is not None:
                qty = round(base_qty, precision)
            else:
                qty = round(base_qty, 6)
            # 再用交易所精度函数
            try:
                qty = float(exchange.amount_to_precision(symbol, qty))
            except Exception:
                pass
            return qty
        except Exception:
            return round(base_qty, 6)
    except Exception as e:
        print(f"⚠️ amount_from_usdt 错误 {symbol}: {e}")
        return 0

def get_min_amount(symbol):
    try:
        return float(exchange.markets.get(symbol, {}).get("limits", {}).get("amount", {}).get("min", 0))
    except Exception:
        return 0

# ================== 下单 / 平仓（Hedge Mode 强制 positionSide） ==================
def place_market_with_positionSide(symbol, side, qty):
    if qty <= 0:
        return False, "qty_zero"
    pos_side = "LONG" if side == "buy" else "SHORT"
    params = {}
    # 优先检测账户是否为 hedge；若是 hedge 则传 positionSide，否则不传
    hedge = is_hedge_mode()
    if hedge:
        params["positionSide"] = pos_side
    else:
        # 单向模式：告知但不传 positionSide（避免 -4061）
        print(f"⚠️ {symbol} 账户是单向模式；已自动不传 positionSide。如仍报错请在币安合约设置里确认模式。")

    # 最小下单量校验
    min_amount = get_min_amount(symbol)
    if min_amount and qty < min_amount:
        msg = f"amount {qty} < min_amount {min_amount}"
        print(f"⚠️ {symbol} 下单量过小，跳过: {msg}")
        return False, msg
    try:
        if not LIVE_TRADE:
            print(f"💡 模拟下单 {symbol} {side} qty={qty} positionSide={params.get('positionSide')}")
            return True, None
        order = exchange.create_order(symbol, "market", side, qty, None, params)
        return True, order
    except Exception as e:
        msg = getattr(e, "args", [str(e)])[0]
        return False, msg

def close_position_market_with_positionSide(symbol, position):
    if not position or not position.get("qty"):
        return True
    pos_side = position.get("side", "").lower()
    action = "buy" if pos_side == "short" else "sell"
    params = {}
    hedge = is_hedge_mode()
    if hedge:
        params["positionSide"] = "SHORT" if pos_side == "short" else "LONG"
    try:
        qty = position["qty"]
        if not LIVE_TRADE:
            print(f"💡 模拟平仓 {symbol} {pos_side} qty={qty} positionSide={params.get('positionSide')}")
            return True
        order = exchange.create_order(symbol, "market", action, qty, None, params)
        send_telegram(f"✅ 已市价平仓 {symbol} {pos_side} 数量={qty}")
        return True
    except Exception as e:
        send_telegram(f"❌ 平仓失败 {symbol}：{e}")
        return False

# ================== 挂 TP/SL（条件市价） + 支持部分止盈 ==================
def place_tp_sl_orders(symbol, side, qty, tp_price, sl_price):
    """
    side 是开仓方向 'buy' 或 'sell'（用于确定 close_side）
    qty: 剩余/部分数量（按合约单位）
    tp_price/sl_price: 触发价（市价触发）
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

# ================== 状态缓存 ==================
last_summary_time = {}   # 每币种上次汇总时间 (datetime)
last_executed_signal = {}  # 每币种上次已执行方向 'buy'/'sell'/None
cooldown_until = {}        # 每币种冷却到期 (datetime)

# ================== 主循环（改写版） ==================
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
            all_status = {}  # 保存每个币种当前状态

            for symbol in SYMBOLS:
                # 冷却处理
                if symbol in cooldown_until and now < cooldown_until[symbol]:
                    continue

                # 多周期共振信号计算
                signal, reasons, status = check_multi_tf(symbol)
                all_status[symbol] = {"signal": signal, "reasons": reasons, "status": status}
                price = status.get("1h", {}).get("last_close") or 0.0
                atr = status.get("1h", {}).get("atr") or None

                # 下单逻辑
                prev_signal = last_executed_signal.get(symbol)
                if signal in ("buy", "sell") and signal != prev_signal:
                    pos = get_position(symbol)
                    need_close_and_reverse = pos and ((signal == "buy" and pos["side"] == "short") or (signal == "sell" and pos["side"] == "long"))

                    # ATR/price 校验
                    if price <= 0 or math.isnan(price) or (atr is None or math.isnan(atr)):
                        print(f"⚠️ {symbol} 当前价格或 ATR 无效，跳过本轮下单")
                        continue

                    # 计算下单数量
                    try:
                        qty = amount_from_usdt(symbol, price, usdt_amount=None)
                    except Exception as e:
                        print(f"⚠️ 计算 {symbol} qty 失败:", e)
                        qty = 0

                    min_amount = get_min_amount(symbol)
                    if qty < min_amount:
                        msg = f"{symbol} 下单量 {qty} < 最小量 {min_amount}，跳过"
                        print("⚠️", msg)
                        send_telegram(msg)
                        last_executed_signal[symbol] = None
                        continue

                    # 平反向仓
                    if need_close_and_reverse:
                        ok = close_position_market_with_positionSide(symbol, pos)
                        if not ok:
                            continue
                        time.sleep(1)

                    # 再确认是否已有同向仓
                    pos2 = get_position(symbol)
                    has_same = pos2 and ((signal == "buy" and pos2["side"] == "long") or (signal == "sell" and pos2["side"] == "short"))
                    if has_same:
                        last_executed_signal[symbol] = signal
                        continue

                    # 开仓
                    ok, err = place_market_with_positionSide(symbol, signal, qty)
                    if ok:
                        # 挂 TP/SL
                        if signal == "buy":
                            tp_price = price + TP_ATR_MULT * atr
                            sl_price = price - SL_ATR_MULT * atr
                        else:
                            tp_price = price - TP_ATR_MULT * atr
                            sl_price = price + SL_ATR_MULT * atr

                        # 分批 TP
                        if PARTIAL_TP_RATIO > 0 and PARTIAL_TP_RATIO < 1:
                            qty_first = round(qty * PARTIAL_TP_RATIO, 6)
                            qty_rest = round(qty - qty_first, 6)
                            if qty_first > 0:
                                place_tp_sl_orders(symbol, signal, qty_first, tp_price, sl_price)
                            if qty_rest > 0:
                                place_tp_sl_orders(symbol, signal, qty_rest, tp_price, sl_price)
                        else:
                            place_tp_sl_orders(symbol, signal, qty, tp_price, sl_price)

                        send_telegram(f"✅ {symbol} 开仓 {signal} qty={qty} @ {price:.2f} TP≈{tp_price:.2f} SL≈{sl_price:.2f}")
                        last_executed_signal[symbol] = signal
                    else:
                        errstr = str(err)
                        send_telegram(f"❌ 下单失败 {symbol} {signal}: {errstr}")
                        if "-2019" in errstr or "Margin is insufficient" in errstr:
                            cooldown_until[symbol] = now + timedelta(seconds=MARGIN_COOLDOWN)
                            send_telegram(f"⏸ {symbol} 保证金不足冷却至 {cooldown_until[symbol].strftime('%Y-%m-%d %H:%M:%S UTC')}")
                        if "-4061" in errstr:
                            send_telegram(f"⚠️ {symbol} -4061 (position side mismatch)")

            # 每小时统一推送五个币种状态
            last_hour = last_summary_time.get("all", datetime.min)
            if (now - last_hour).total_seconds() >= SUMMARY_INTERVAL:
                msgs = []
                for s in SYMBOLS:
                    st = all_status.get(s, {})
                    sig = st.get("signal") or "无"
                    reasons = st.get("reasons") or []
                    reason_str = ";".join(reasons) if reasons else "无"
                    price = st.get("status", {}).get("1h", {}).get("last_close") or 0.0
                    msgs.append(f"{s} 信号:{sig} 原因:{reason_str} 价格:{price:.2f}")
                send_telegram(f"{now_str()}\n" + "\n".join(msgs))
                last_summary_time["all"] = now

            time.sleep(POLL_INTERVAL)

        except Exception as e:
            print("⚠️ 主循环异常:", e)
            time.sleep(5)
                if price <= 0 or math.isnan(price):
                    continue

                # 计算下单数量（优先基于账户余额 RISK_RATIO；若想用固定 BASE_USDT 可传 

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

                    # 若 PARTIAL_TP_RATIO>0 则先挂一笔部分 TP，再挂剩余 TP/SL
                    if PARTIAL_TP_RATIO > 0 and PARTIAL_TP_RATIO < 1:
                        qty_first = round(qty * PARTIAL_TP_RATIO, 6)
                        qty_rest = round(qty - qty_first, 6)
                        if qty_first > 0:
                            # 部分 TP: 用市价触发TAKE_PROFIT_MARKET/ LIMIT（这里用 TAKE_PROFIT_MARKET）
                            place_tp_sl_orders(symbol, signal, qty_first, tp_price, sl_price)
                        if qty_rest > 0:
                            place_tp_sl_orders(symbol, signal, qty_rest, tp_price, sl_price)
                    else:
                        # 全仓 TP/SL
                        place_tp_sl_orders(symbol, signal, qty, tp_price, sl_price)

                    send_telegram(f"✅ {symbol} 开仓 {signal} qty={qty} @ {price:.2f} TP≈{tp_price:.2f} SL≈{sl_price:.2f}")
                    last_executed_signal[symbol] = signal
                else:
                    errstr = str(err)
                    send_telegram(f"❌ 下单失败 {symbol} {signal}：{errstr}")
                    # margin insufficient -> 冷却
                    if "-2019" in errstr or "Margin is insufficient" in errstr:
                        cooldown_until[symbol] = now + timedelta(seconds=MARGIN_COOLDOWN)
                        send_telegram(f"⏸ {symbol} 因保证金不足进入冷却到 {cooldown_until[symbol].strftime('%Y-%m-%d %H:%M:%S UTC')}")
                    if "-4061" in errstr:
                        send_telegram(f"⚠️ {symbol} 报 -4061 (position side mismatch)，请确认账户为 Hedge Mode 并且 API 权限完整")
            # main for end
            time.sleep(POLL_INTERVAL)

        except Exception as e:
            print("⚠️ 主循环异常:", e)
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main_loop()
