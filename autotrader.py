# autotrader.py
"""
Merged Hedge Mode AutoTrader - 完整优化版
- 多周期共振 (1h, 4h, 1d)
- Hedge Mode 强制使用 positionSide (LONG/SHORT)
- ATR 计算 TP/SL，支持分批止盈 PARTIAL_TP_RATIO
- 每小时汇总 Telegram 推送
- 出错冷却处理
- 动态仓位 RISK_RATIO
- 跳过小于交易所最小下单量
- LIVE_TRADE 支持 env 写 "1" 或 "true"
- 优化了部分函数，使其更具通用性和健壮性。
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
import logging

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# ================== 配置 ==================
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,LTC/USDT,DOGE/USDT,BNB/USDT").split(",") if s.strip()]
# 修复: 直接在脚本中硬编码 BASE_USDT 的值，以避免环境读取问题。
# 如果你想改变这个值，请直接修改下面这一行代码。
BASE_USDT = float("20")
RISK_RATIO = float(os.getenv("RISK_RATIO", "0.15"))
LEVERAGE = int(os.getenv("LEVERAGE", "10"))
LIVE_TRADE = os.getenv("LIVE_TRADE", "0").lower() in ("1", "true", "yes")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "3.0"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "2.0"))
PARTIAL_TP_RATIO = float(os.getenv("PARTIAL_TP_RATIO", "0.0"))
OHLCV_LIMIT = int(os.getenv("OHLCV_LIMIT", "200"))
SUMMARY_INTERVAL = int(os.getenv("SUMMARY_INTERVAL", "3600"))
MARGIN_COOLDOWN = int(os.getenv("MARGIN_COOLDOWN", "3600"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BIANANCE_API_KEY = os.getenv("BIANANCE_API_KEY")
BIANANCE_API_SECRET = os.getenv("BIANANCE_API_SECRET")
EXCHANGE_ID = os.getenv("EXCHANGE", "binance")
MARKET_TYPE = os.getenv("MARKET_TYPE", "future")

# 检查关键环境变量是否已设置
if not BIANANCE_API_KEY or not BIANANCE_API_SECRET:
    logging.error("❌ 致命错误：BIANANCE_API_KEY 或 BIANANCE_API_SECRET 未配置。请在环境中设置这两个变量。")
    exit(1)

# CCXT 实例
exchange = getattr(ccxt, EXCHANGE_ID)({
    "apiKey": BIANANCE_API_KEY,
    "secret": BIANANCE_API_SECRET,
    "enableRateLimit": True,
    "options": {"defaultType": MARKET_TYPE},
})

# ================== 工具函数 ==================
def now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def send_telegram(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("⚠️ Telegram 未配置，消息打印: %s", msg)
        return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                      json={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
    except requests.exceptions.RequestException as e:
        logging.error("❌ Telegram 推送失败: %s", e)

def safe_float(x, default=0.0):
    try:
        return float(x)
    except (ValueError, TypeError):
        return default

# ================== 市场/账户 ==================
def load_markets_safe():
    try:
        exchange.load_markets()
    except ccxt.NetworkError as e:
        logging.warning("⚠️ load_markets 网络错误: %s", e)
    except ccxt.ExchangeError as e:
        logging.warning("⚠️ load_markets 交易所错误: %s", e)

def symbol_id(symbol):
    try:
        return exchange.market(symbol)["id"]
    except (ccxt.ExchangeError, KeyError):
        return symbol.replace("/", "")

# Hedge 模式缓存
HEDGE_MODE_CACHE = None
def is_hedge_mode():
    global HEDGE_MODE_CACHE
    if HEDGE_MODE_CACHE is not None:
        return HEDGE_MODE_CACHE
    
    if EXCHANGE_ID == 'binance':
        try:
            info = exchange.fapiPrivate_get_positionmode()
            HEDGE_MODE_CACHE = bool(info.get("dualSidePosition") is True)
        except (ccxt.ExchangeError, ccxt.NetworkError, AttributeError):
            logging.warning("⚠️ Binance 对冲模式检测失败，默认为单向模式。")
            HEDGE_MODE_CACHE = False
    else:
        logging.info("ℹ️ 交易所 %s 的对冲模式检测暂未实现，默认为单向模式。", EXCHANGE_ID)
        HEDGE_MODE_CACHE = False

    return HEDGE_MODE_CACHE

def _try_set_param(method, symbol, params, action_name):
    try:
        if hasattr(exchange, method):
            getattr(exchange, method)(**params)
        else:
            if EXCHANGE_ID == 'binance':
                if action_name == '杠杆':
                    exchange.fapiPrivate_post_leverage({"symbol": symbol_id(symbol), "leverage": LEVERAGE})
                elif action_name == '保证金模式':
                    exchange.fapiPrivate_post_margintype({"symbol": symbol_id(symbol), "marginType": "ISOLATED"})
    except ccxt.ExchangeError as e:
        logging.warning("⚠️ %s %s 设置失败: %s", symbol, action_name, e)
        return False
    except Exception as e:
        logging.warning("⚠️ %s %s 设置时发生未知错误: %s", symbol, action_name, e)
        return False
    return True

def ensure_leverage_and_margin(symbol):
    leverage_params = {"leverage": LEVERAGE, "symbol": symbol}
    _try_set_param("set_leverage", symbol, leverage_params, "杠杆")
    
    margin_params = {"marginMode": "ISOLATED", "symbol": symbol}
    _try_set_param("set_margin_mode", symbol, margin_params, "保证金模式")

# ================== OHLCV / 指标 ==================
def fetch_ohlcv_df(symbol, timeframe="1h", limit=OHLCV_LIMIT):
    for _ in range(3):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not ohlcv:
                continue
            df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            for c in ["open","high","low","close","volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df
        except (ccxt.NetworkError, ccxt.ExchangeError):
            time.sleep(1)
    return pd.DataFrame()

def compute_indicators(df: pd.DataFrame):
    if df is None or df.empty:
        return df
    df = df.copy()
    df["ema20"] = ta.trend.EMAIndicator(df["close"], window=20, fillna=True).ema_indicator()
    df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50, fillna=True).ema_indicator()
    macd = ta.trend.MACD(df["close"], 26, 12, 9, fillna=True)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], 14, fillna=True).rsi()
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], 14, fillna=True).average_true_range()
    df["vol_ma20"] = df["volume"].rolling(20, min_periods=1).mean()
    return df

def signal_from_indicators(df: pd.DataFrame):
    if df is None or df.empty:
        return None, 0, [], None
    last = df.iloc[-1]
    score = 0; reasons=[]
    if last["ema20"] > last["ema50"]: score+=2; reasons.append("EMA 多头")
    else: score-=2; reasons.append("EMA 空头")
    if last["macd"] > last["macd_signal"]: score+=1; reasons.append("MACD 多头")
    else: score-=1; reasons.append("MACD 空头")
    if last["rsi"] > 60: score+=1; reasons.append(f"RSI 偏强 {last['rsi']:.1f}")
    elif last["rsi"] < 40: score-=1; reasons.append(f"RSI 偏弱 {last['rsi']:.1f}")
    if "vol_ma20" in df.columns and last["volume"] > last["vol_ma20"]*1.5: score+=1; reasons.append("成交量放大")
    if score>=3: return "buy", score, reasons, last
    elif score<=-3: return "sell", score, reasons, last
    else: return None, score, reasons, last

def check_multi_tf(symbol):
    multi_signal=None; reasons_all=[]; status={}
    for tf in ["1h","4h","1d"]:
        df=compute_indicators(fetch_ohlcv_df(symbol, tf, 100))
        sig,score,reasons,last = signal_from_indicators(df)
        status[tf] = {"signal": sig, "score": score, "reasons": reasons, "last_close": safe_float(last["close"]) if last is not None else 0, "atr": safe_float(last["atr"]) if last is not None else None}
        if sig: reasons_all.extend([f"{tf}:{r}" for r in reasons])
        if sig:
            if multi_signal is None: multi_signal=sig
            elif multi_signal!=sig: multi_signal=None
    return multi_signal, reasons_all, status

# ================== 仓位管理 ==================
def parse_position_entry(pos):
    try:
        if pos is None: return None
        sym = pos.get("symbol") or (pos.get("info") or {}).get("symbol")
        amt = safe_float(pos.get("positionAmt") or pos.get("contracts") or pos.get("amount") or 0)
        if amt == 0: return None
        side = "long" if amt > 0 else "short"
        entry = safe_float(pos.get("entryPrice") or (pos.get("info") or {}).get("entryPrice"))
        return sym, abs(amt), side, entry
    except Exception: return None

def fetch_all_positions():
    try:
        return exchange.fetch_positions()
    except (ccxt.NetworkError, ccxt.ExchangeError):
        logging.error("❌ 获取所有仓位失败，网络或交易所错误。")
        return []

def get_position(symbol):
    want = symbol.replace("/", "")
    for p in fetch_all_positions():
        parsed = parse_position_entry(p)
        if not parsed: continue
        sym, qty, side, entry = parsed
        if sym.replace("/", "") == want:
            return {"symbol": symbol, "qty": qty, "side": side, "entry": entry, "raw": p}
    return None

# ================== 数量计算 ==================
def amount_from_usdt(symbol, price, usdt_amount=None):
    try:
        if price <= 0: return 0
        if usdt_amount is None:
            # 修复: 直接使用硬编码的 BASE_USDT
            use_usdt = BASE_USDT * RISK_RATIO
        else: use_usdt = usdt_amount
        nominal = use_usdt * LEVERAGE
        qty = nominal / price
        try: qty = float(exchange.amount_to_precision(symbol, qty))
        except: pass
        return qty
    except Exception: return 0

def get_min_amount(symbol):
    try:
        return float(exchange.markets.get(symbol, {}).get("limits", {}).get("amount", {}).get("min", 0))
    except (KeyError, ValueError): return 0

# ================== 下单 / 平仓 ==================
def place_market_with_positionSide(symbol, side, qty):
    if qty <= 0:
        return False, "qty_zero"
    
    pos_side = "LONG" if side=="buy" else "SHORT"
    params = {}
    hedge = is_hedge_mode()
    if hedge:
        params["positionSide"] = pos_side
    
    min_amount = get_min_amount(symbol)
    if qty < min_amount:
        msg = f"下单量 {qty} < 最小量 {min_amount}"
        logging.warning("⚠️ %s %s", symbol, msg)
        return False, msg

    try:
        if not LIVE_TRADE:
            logging.info("💡 模拟下单 %s %s qty=%s positionSide=%s", symbol, side, qty, params.get('positionSide'))
            return True, None
        order = exchange.create_order(symbol, "market", side, qty, None, params)
        return True, order
    except ccxt.ExchangeError as e:
        errstr = str(e)
        if "-2019" in errstr or "Margin is insufficient" in errstr:
            return False, "保证金不足"
        return False, errstr
    except Exception as e:
        return False, str(e)

def close_position_market_with_positionSide(symbol, position):
    if not position or not position.get("qty"):
        return True
    
    pos_side = position.get("side","").lower()
    action = "buy" if pos_side == "short" else "sell"
    params = {}
    hedge = is_hedge_mode()
    if hedge:
        params["positionSide"] = "SHORT" if pos_side == "short" else "LONG"
    
    try:
        qty = position["qty"]
        if not LIVE_TRADE:
            logging.info("💡 模拟平仓 %s %s qty=%s positionSide=%s", symbol, pos_side, qty, params.get('positionSide'))
            return True
        exchange.create_order(symbol, "market", action, qty, None, params)
        send_telegram(f"✅ 已市价平仓 {symbol} {pos_side} 数量={qty}")
        return True
    except ccxt.ExchangeError as e:
        send_telegram(f"❌ 平仓失败 {symbol}: {e}")
        return False
    except Exception as e:
        logging.error("❌ 平仓时发生未知错误 %s: %s", symbol, e)
        return False

# ================== 挂 TP/SL ==================
def place_tp_sl_orders(symbol, side, qty, tp_price, sl_price):
    pos_side = "LONG" if side=="buy" else "SHORT"
    close_side = "sell" if side=="buy" else "buy"
    
    tp_qty = qty * PARTIAL_TP_RATIO if PARTIAL_TP_RATIO > 0 else qty
    tp_params = {"positionSide": pos_side, "stopPrice": tp_price}
    try:
        if LIVE_TRADE:
            exchange.create_order(symbol, "TAKE_PROFIT_MARKET", close_side, tp_qty, None, tp_params)
        else:
            logging.info("💡 模拟挂 TP %s qty=%s tp=%s positionSide=%s", symbol, tp_qty, tp_price, pos_side)
    except ccxt.ExchangeError as e:
        logging.error("❌ 挂 TP 失败 %s: %s", e)
    
    sl_params = {"positionSide": pos_side, "stopPrice": sl_price}
    try:
        if LIVE_TRADE:
            exchange.create_order(symbol, "STOP_MARKET", close_side, qty, None, sl_params)
        else:
            logging.info("💡 模拟挂 SL %s qty=%s sl=%s positionSide=%s", symbol, qty, sl_price, pos_side)
    except ccxt.ExchangeError as e:
        logging.error("❌ 挂 SL 失败 %s: %s", e)

# ================== 状态缓存 ==================
last_summary_time = {}
last_executed_signal = {}
cooldown_until = {}

# ================== 主循环 ==================
def main_loop():
    load_markets_safe()
    for s in SYMBOLS:
        ensure_leverage_and_margin(s)
    
    send_telegram(f"🤖 Bot 启动 - Hedge Mode={is_hedge_mode()} LIVE_TRADE={LIVE_TRADE} SYMBOLS={','.join(SYMBOLS)}")

    while True:
        try:
            now = datetime.now(timezone.utc)
            all_status = {}

            for symbol in SYMBOLS:
                if symbol in cooldown_until and now < cooldown_until[symbol]:
                    continue
                elif symbol in cooldown_until and now >= cooldown_until[symbol]:
                    cooldown_until.pop(symbol)
                    last_executed_signal[symbol] = None

                signal, reasons, status = check_multi_tf(symbol)
                all_status[symbol] = {"signal": signal, "reasons": reasons, "status": status}

                prev_signal = last_executed_signal.get(symbol)
                price = status.get("1h", {}).get("last_close") or 0
                atr = status.get("1h", {}).get("atr") or None

                if signal in ("buy", "sell") and signal != prev_signal:
                    pos = get_position(symbol)
                    need_close_and_reverse = pos and ((signal=="buy" and pos["side"]=="short") or (signal=="sell" and pos["side"]=="long"))

                    if price <= 0 or atr is None or math.isnan(price) or math.isnan(atr):
                        logging.warning("⚠️ %s 当前价格或 ATR 无效", symbol)
                        continue

                    if need_close_and_reverse:
                        if not close_position_market_with_positionSide(symbol, pos):
                            continue
                        time.sleep(1)

                    pos2 = get_position(symbol)
                    has_same = pos2 and ((signal=="buy" and pos2["side"]=="long") or (signal=="sell" and pos2["side"]=="short"))
                    if has_same:
                        last_executed_signal[symbol] = signal
                        continue

                    qty = amount_from_usdt(symbol, price)
                    ok, err = place_market_with_positionSide(symbol, signal, qty)
                    
                    if ok:
                        if signal == "buy":
                            tp_price = price + TP_ATR_MULT * atr
                            sl_price = price - SL_ATR_MULT * atr
                        else:
                            tp_price = price - TP_ATR_MULT * atr
                            sl_price = price + SL_ATR_MULT * atr

                        place_tp_sl_orders(symbol, signal, qty, tp_price, sl_price)
                        msg = f"✅ {symbol} 开仓 {signal} qty={qty:.4f} @ {price:.2f} TP≈{tp_price:.2f} SL≈{sl_price:.2f}"
                        logging.info(msg)
                        send_telegram(msg)
                        last_executed_signal[symbol] = signal
                    else:
                        send_telegram(f"❌ 下单失败 {symbol} {signal}: {err}")
                        if "保证金不足" in err:
                            cooldown_until[symbol] = now + timedelta(seconds=MARGIN_COOLDOWN)
                            send_telegram(f"⏸ {symbol} 保证金不足冷却至 {cooldown_until[symbol]}")

            summary_key = "global_summary"
            last_summary = last_summary_time.get(summary_key, datetime.min.replace(tzinfo=timezone.utc))
            if (now - last_summary).total_seconds() >= SUMMARY_INTERVAL:
                msgs = []
                for symbol in SYMBOLS:
                    info = all_status.get(symbol, {})
                    sig = info.get("signal") or "无信号"
                    reasons = info.get("reasons") or []
                    status = info.get("status") or {}
                    last_close = status.get("1h", {}).get("last_close") or 0
                    atr = status.get("1h", {}).get("atr") or 0
                    msg_line = f"{symbol}: 信号={sig}, 价格={last_close:.2f}, ATR={atr:.2f}, 理由={'|'.join(reasons)}"
                    msgs.append(msg_line)
                summary_text = "🕐 每小时汇总:\n" + "\n".join(msgs)
                send_telegram(summary_text)
                last_summary_time[summary_key] = now

            time.sleep(POLL_INTERVAL)

        except Exception as e:
            logging.error("❌ 主循环异常: %s", e)
            send_telegram(f"❌ 主循环发生致命错误: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main_loop()

