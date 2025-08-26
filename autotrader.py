# autotrader.py
import os
import time
import ccxt
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import ta

# =============== 配置 ===============
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT").split(",") if s.strip()]
BASE_USDT = float(os.getenv("BASE_USDT", "100"))
LEVERAGE = int(os.getenv("LEVERAGE", "10"))
LIVE_TRADE = os.getenv("LIVE_TRADE", "0") == "1"
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))  # 秒
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "3.0"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "2.0"))
OHLCV_LIMIT = int(os.getenv("OHLCV_LIMIT", "200"))

# 推送节流（每币种每小时汇总一次）
SUMMARY_INTERVAL = 3600  # 秒
# 下单失败（保证金不足）冷却
MARGIN_COOLDOWN = 3600  # 秒

# =============== 工具 ===============
def now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def send_telegram(msg: str):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("⚠️ 未配置 Telegram 环境变量，跳过推送:", msg)
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": msg})
    except Exception as e:
        print("❌ Telegram 推送失败:", e)

# =============== 交易所初始化 ===============
exchange = ccxt.binance({
    "apiKey": os.getenv("API_KEY"),
    "secret": os.getenv("API_SECRET"),
    "enableRateLimit": True,
    "options": {"defaultType": "future"}  # USDT-M 永续
})

def load_markets_safe():
    try:
        exchange.load_markets()
    except Exception as e:
        print("⚠️ load_markets 失败:", e)

def symbol_id(symbol):
    # BTC/USDT -> BTCUSDT
    try:
        m = exchange.market(symbol)
        return m["id"]
    except Exception:
        return symbol.replace("/", "")

def get_position_mode_is_hedge() -> bool:
    """返回账户是否为双向(hedge)模式。失败时默认 False（单向）。"""
    try:
        info = exchange.fapiPrivate_get_positionmode()
        return bool(info.get("dualSidePosition") is True)
    except Exception:
        # 某些 ccxt 版本可从账户设置/选项推断，保守返回 False
        return False

def setup_account(symbol):
    """设置杠杆与保证金模式（容错、可多次调用）。"""
    sid = symbol_id(symbol)
    try:
        # 先尝试统一方法
        try:
            if hasattr(exchange, "set_leverage"):
                exchange.set_leverage(LEVERAGE, symbol)
            else:
                raise AttributeError("no set_leverage")
        except Exception:
            # 回退到私有端点
            try:
                exchange.fapiPrivate_post_leverage({"symbol": sid, "leverage": LEVERAGE})
            except Exception as e:
                print(f"⚠️ 设置杠杆失败 {symbol}:", e)

        try:
            if hasattr(exchange, "set_margin_mode"):
                exchange.set_margin_mode("ISOLATED", symbol)
            else:
                raise AttributeError("no set_margin_mode")
        except Exception:
            # 回退到私有端点
            try:
                exchange.fapiPrivate_post_margintype({"symbol": sid, "marginType": "ISOLATED"})
            except Exception as e:
                print(f"⚠️ 设置保证金模式失败 {symbol}:", e)
    except Exception as e:
        print("⚠️ setup_account 失败:", e)

# =============== 数据与指标 ===============
def fetch_ohlcv_df(symbol, timeframe="1h", limit=OHLCV_LIMIT):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
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
    if last.get("vol_ma20") and last.get("volume") and last["volume"] > last["vol_ma20"] * 1.5:
        score += 1; reasons.append("成交量放大")
    if score >= 3:
        return "buy", score, reasons, last
    elif score <= -3:
        return "sell", score, reasons, last
    else:
        return None, score, reasons, last

def check_multi_tf(symbol):
    """返回 (共振方向/None, 汇总原因, 各周期状态dict)"""
    multi_tf_signal = None
    reasons_all = []
    status = {}
    for tf in ["1h","4h","1d"]:
        try:
            df = compute_indicators(fetch_ohlcv_df(symbol, tf, 100))
            signal, score, reasons, last = signal_from_indicators(df)
            status[tf] = {
                "signal": signal,
                "score": score,
                "reasons": reasons,
                "last_close": float(last["close"]),
                "atr": float(last["atr"]) if not np.isnan(last["atr"]) else None
            }
            if signal:
                reasons_all.extend([f"{tf}:{r}" for r in reasons])
                if multi_tf_signal is None:
                    multi_tf_signal = signal
                elif multi_tf_signal != signal:
                    multi_tf_signal = None  # 不共振
        except Exception as e:
            status[tf] = {"error": str(e)}
    return multi_tf_signal, reasons_all, status

# =============== 仓位相关 ===============
def fetch_all_positions():
    try:
        pos = exchange.fetch_positions()
        return pos if isinstance(pos, list) else []
    except Exception as e:
        print("⚠️ fetch_positions 不可用:", e)
        return []

def parse_position_entry(pos):
    try:
        symbol = pos.get("symbol") or (pos.get("info") or {}).get("symbol")
        # 数量解析
        contracts = None
        if "contracts" in pos: contracts = float(pos["contracts"])
        elif "positionAmt" in pos: contracts = float(pos["positionAmt"])
        if contracts is None or contracts == 0:
            return None
        # 方向解析
        side = None
        if "side" in pos and pos["side"]:
            side = pos["side"]
        else:
            if "positionAmt" in pos:
                amt = float(pos["positionAmt"])
                side = "long" if amt > 0 else "short"
        entry = pos.get("entryPrice") or (pos.get("info") or {}).get("entryPrice") or None
        return (symbol, abs(contracts), side, float(entry) if entry else None)
    except Exception:
        return None

def get_position(symbol):
    positions = fetch_all_positions()
    want = symbol.replace("/", "")
    for p in positions:
        parsed = parse_position_entry(p)
        if not parsed:
            continue
        sym, qty, side, entry = parsed
        if not sym:
            continue
        sym_norm = sym.replace("/", "")
        if sym_norm == want:
            return {"symbol": symbol, "qty": qty, "side": side, "entry": entry, "raw": p}
    return None

# =============== 下单/平仓 ===============
def amount_from_usdt(symbol, price):
    qty = BASE_USDT * LEVERAGE / price
    try:
        qty = float(exchange.amount_to_precision(symbol, qty))
    except Exception:
        qty = round(qty, 6)
    return qty

def place_market(symbol, side, qty, hedge_mode):
    """side: buy/sell；hedge_mode: 是否双向模式"""
    if not LIVE_TRADE:
        print(f"💡 模拟下单 {symbol} {side} {qty}")
        return True, None
    try:
        params = {}
        if hedge_mode:
            params["positionSide"] = "LONG" if side == "buy" else "SHORT"
        order = exchange.create_order(symbol, "market", side, qty, None, params)
        return True, order
    except ccxt.BaseError as e:
        # 解析 Binance 错误码
        msg = getattr(e, "message", str(e))
        send_telegram(f"❌ 下单失败 {symbol} {side}：{msg}")
        return False, e
    except Exception as e:
        send_telegram(f"❌ 下单失败 {symbol} {side}：{e}")
        return False, e

def close_position_market(symbol, position, hedge_mode):
    """根据现有仓位方向，市价平仓"""
    if not position or not position.get("qty"):
        return True
    pos_side = position.get("side", "").lower()
    side = "buy" if pos_side == "short" else "sell"  # 平空买入，平多卖出
    qty = position["qty"]
    if not LIVE_TRADE:
        print(f"💡 模拟平仓 {symbol} {pos_side} {qty}")
        return True
    try:
        params = {}
        if hedge_mode:
            params["positionSide"] = "SHORT" if side == "sell" else "LONG"  # 平仓时 positionSide 要与持仓一致
        order = exchange.create_order(symbol, "market", side, qty, None, params)
        send_telegram(f"✅ 已市价平仓 {symbol} {pos_side} 数量={qty}")
        return True
    except Exception as e:
        send_telegram(f"❌ 平仓失败 {symbol}：{e}")
        return False

# =============== 状态与节流 ===============
last_summary_time = {}      # 每币种上次汇总时间
last_signal_dir = {}        # 每币种上次已执行/广播的方向: "buy"/"sell"/None
cooldown_until = {}         # 每币种冷却到期时间（margin 不足等）

# =============== 主循环 ===============
def main_loop():
    load_markets_safe()
    # 启动时尽量设置一次账户参数（失败也不致命）
    for s in SYMBOLS:
        setup_account(s)

    while True:
        try:
            hedge_mode = get_position_mode_is_hedge()
            now = datetime.now(timezone.utc)

            for symbol in SYMBOLS:
                # 冷却中跳过
                if symbol in cooldown_until and now < cooldown_until[symbol]:
                    continue

                # 多周期共振
                signal, reasons, status = check_multi_tf(symbol)
                price = status.get("1h", {}).get("last_close") or 0.0
                atr = status.get("1h", {}).get("atr") or None

                # 每小时汇总（每币种）
                last_sum = last_summary_time.get(symbol)
                if (last_sum is None) or ((now - last_sum).total_seconds() >= SUMMARY_INTERVAL):
                    price_str = f"{price:.2f}" if price else "0"
                    reason_str = ";".join(reasons) if reasons else "无"
                    send_telegram(f"{now_str()} {symbol} 信号:{signal or '无'} 原因:{reason_str} 价格:{price_str}")
                    last_summary_time[symbol] = now

                # 仅在信号发生改变时考虑下单，避免每分钟重复尝试
                prev_sig = last_signal_dir.get(symbol)
                if signal != prev_sig and signal in ("buy", "sell"):
                    pos = get_position(symbol)

                    # 如果已有反向仓，先平仓
                    need_close = pos and ((signal == "buy" and pos["side"] == "short") or (signal == "sell" and pos["side"] == "long"))
                    if need_close:
                        ok = close_position_market(symbol, pos, hedge_mode)
                        if not ok:
                            continue  # 平仓失败则不再尝试开新仓

                    # 如果没有同向仓，则开仓
                    pos = get_position(symbol)  # 再查一次
                    has_same = pos and ((signal == "buy" and pos["side"] == "long") or (signal == "sell" and pos["side"] == "short"))
                    if not has_same:
                        if price <= 0:
                            continue
                        qty = amount_from_usdt(symbol, price)
                        ok, err = place_market(symbol, signal, qty, hedge_mode)
                        if ok:
                            # 计算 TP/SL 仅用于消息提示（市价单无法直接挂TP/SL，后续可改为条件单）
                            if atr is None or np.isnan(atr):
                                atr = price * 0.005
                            if signal == "buy":
                                tp = price + TP_ATR_MULT * atr
                                sl = price - SL_ATR_MULT * atr
                            else:
                                tp = price - TP_ATR_MULT * atr
                                sl = price + SL_ATR_MULT * atr
                            send_telegram(f"✅ 已下单 {symbol} {signal} 数量={qty} @ {price:.2f} TP≈{tp:.2f} SL≈{sl:.2f}")
                            last_signal_dir[symbol] = signal
                        else:
                            # 处理常见错误码：-2019 保证金不足 => 冷却 1 小时
                            msg = str(err)
                            if "-2019" in msg or "Margin is insufficient" in msg:
                                cooldown_until[symbol] = now + timedelta(seconds=MARGIN_COOLDOWN)
                                send_telegram(f"⏸ {symbol} 因保证金不足进入冷却，至 {cooldown_until[symbol].strftime('%Y-%m-%d %H:%M:%S UTC')}")
                            # -4061 position side 不匹配：大概率是单向模式下错误传了 positionSide。我们已在单向模式不加 positionSide，如仍出现，则跳过。
                            if "-4061" in msg:
                                send_telegram(f"⚠️ {symbol} 账户是单向模式；已自动不传 positionSide。如仍报错请在币安合约设置里确认模式。")
                    else:
                        # 已有同向仓，不重复开单，只更新已知信号
                        last_signal_dir[symbol] = signal

            time.sleep(POLL_INTERVAL)
        except Exception as e:
            print("⚠️ 主循环异常:", e)
            time.sleep(POLL_INTERVAL)

# =============== 启动 ===============
if __name__ == "__main__":
    print(f"🚀 AutoTrader 启动 {SYMBOLS}，LIVE_TRADE={LIVE_TRADE}")
    main_loop()
