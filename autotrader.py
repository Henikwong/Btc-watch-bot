# autotrader.py
import os
import time
import ccxt
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import ta

# ===========================
# 配置（可通过环境变量覆盖）
# ===========================
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT").split(",") if s.strip()]
BASE_USDT = float(os.getenv("BASE_USDT", "100"))
LEVERAGE = int(os.getenv("LEVERAGE", "10"))
LIVE_TRADE = os.getenv("LIVE_TRADE", "0") == "1"
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))  # 每分钟默认 60 秒
ONLY_RESONANCE = os.getenv("ONLY_RESONANCE", "1") == "1"  # 是否仅在多周期共振时下单

TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "3.0"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "2.0"))

OHLCV_LIMIT = int(os.getenv("OHLCV_LIMIT", "200"))

# ===========================
# 工具函数
# ===========================
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

# ===========================
# 初始化交易所（Binance Futures）
# ===========================
exchange = ccxt.binance({
    "apiKey": os.getenv("API_KEY"),
    "secret": os.getenv("API_SECRET"),
    "enableRateLimit": True,
    "options": {"defaultType": "future"}
})

def setup_account(symbol):
    try:
        m = exchange.market(symbol)
        ex_symbol = m["id"]
        try:
            exchange.fapiPrivate_post_leverage({"symbol": ex_symbol, "leverage": LEVERAGE})
            exchange.fapiPrivate_post_margintype({"symbol": ex_symbol, "marginType": "ISOLATED"})
            print(f"✅ 已设置 {symbol} 杠杆与保证金模式")
        except Exception as e:
            print("⚠️ 设置杠杆/保证金失败:", e)
    except Exception as e:
        print("⚠️ setup_account 失败:", e)

# ===========================
# OHLCV 与指标
# ===========================
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
    prev = df.iloc[-2] if len(df) >= 2 else last
    score = 0
    reasons = []

    # EMA
    if last["ema20"] > last["ema50"]:
        score += 2; reasons.append("EMA 多头")
    else:
        score -= 2; reasons.append("EMA 空头")
    # MACD
    if last["macd"] > last["macd_signal"]:
        score += 1; reasons.append("MACD 多头")
    else:
        score -= 1; reasons.append("MACD 空头")
    # RSI
    if last["rsi"] > 60:
        score += 1; reasons.append(f"RSI 偏强 {last['rsi']:.1f}")
    elif last["rsi"] < 40:
        score -= 1; reasons.append(f"RSI 偏弱 {last['rsi']:.1f}")
    # Volume spike
    if "vol_ma20" in df.columns and last["volume"] > last["vol_ma20"] * 1.5:
        score += 1; reasons.append("成交量放大")

    if score >= 3:
        return "买入", score, reasons, last
    elif score <= -3:
        return "卖出", score, reasons, last
    else:
        return None, score, reasons, last

# ===========================
# 仓位管理
# ===========================
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
        contracts = None
        if "contracts" in pos: contracts = float(pos["contracts"])
        elif "positionAmt" in pos: contracts = float(pos["positionAmt"])
        elif "amount" in pos: contracts = float(pos["amount"])
        if contracts is None or contracts==0: return None
        side = None
        if "side" in pos and pos["side"]: side = pos["side"]
        else:
            if "positionAmt" in pos:
                amt = float(pos["positionAmt"])
                side = "long" if amt > 0 else "short"
            elif contracts > 0:
                side = pos.get("side") or (pos.get("info") or {}).get("positionSide") or "long"
        entry = pos.get("entryPrice") or (pos.get("info") or {}).get("entryPrice") or None
        return (symbol, abs(contracts), side, float(entry) if entry else None)
    except Exception as e:
        print("⚠️ parse_position_entry 失败:", e)
        return None

def get_position(symbol):
    positions = fetch_all_positions()
    for p in positions:
        parsed = parse_position_entry(p)
        if not parsed: continue
        sym, qty, side, entry = parsed
        if not sym: continue
        if sym.replace("/", "") == symbol.replace("/", "") or sym == symbol:
            return {"symbol": symbol, "qty": qty, "side": side, "entry": entry, "raw": p}
    return None

def close_position(symbol, position):
    try:
        qty = position.get("qty")
        if qty is None or qty==0:
            send_telegram(f"❌ 平仓失败 {symbol}：无法解析仓位数量")
            return False
        pos_side = position.get("side","").lower()
        side = "buy" if pos_side.startswith("short") else "sell"
        if LIVE_TRADE:
            try: qty_precise = float(exchange.amount_to_precision(symbol, qty))
            except: qty_precise = round(qty,6)
            exchange.create_market_order(symbol, side, qty_precise)
            send_telegram(f"✅ 已市价平仓 {symbol} {pos_side} 数量={qty_precise}")
        else:
            send_telegram(f"📌 模拟平仓 {symbol} {pos_side} 数量={qty}")
        return True
    except Exception as e:
        send_telegram(f"❌ 平仓失败 {symbol}，原因: {e}")
        return False

# ===========================
# 下单（兼容单向和双向模式）
# ===========================
def place_order(symbol, side_text, price, atr):
    """
    side_text: '买入' 或 '卖出'
    自动处理单向/双向模式下的仓位问题
    """
    side = "buy" if side_text == "买入" else "sell"

    # 计算下单数量
    try:
        qty = BASE_USDT * LEVERAGE / price
        try:
            qty = float(exchange.amount_to_precision(symbol, qty))
        except Exception:
            qty = round(qty, 6)
    except Exception as e:
        send_telegram(f"❌ 计算下单数量失败 {symbol}：{e}")
        return

    if not LIVE_TRADE:
        send_telegram(f"📌 模拟下单 {symbol} {side_text} 数量={qty} @ {price:.2f}")
        return

    try:
        # 检查账户是否是双向模式
        is_hedge = False
        try:
            info = exchange.fapiPrivate_get_positionmode()
            is_hedge = info.get("dualSidePosition") == True
        except Exception:
            pass

        params = {}
        if is_hedge:
            params["positionSide"] = "LONG" if side=="buy" else "SHORT"

        # 开仓市价单
        exchange.create_market_order(symbol, side, qty, params=params)

        # 止损/止盈计算
        if atr is None or np.isnan(atr):
            atr = price * 0.005

        if side == "buy":
            stop_loss = price - SL_ATR_MULT * atr
            take_profit = price + TP_ATR_MULT * atr
            close_side = "sell"
        else:
            stop_loss = price + SL_ATR_MULT * atr
            take_profit = price - TP_ATR_MULT * atr
            close_side = "buy"

        # 下止损/止盈挂单
        try:
            sl_params = params.copy()
            sl_params["stopPrice"] = stop_loss
            tp_params = params.copy()
            tp_params["stopPrice"] = take_profit

            exchange.create_order(symbol, "STOP_MARKET", close_side, qty, None, sl_params)
            exchange.create_order(symbol, "TAKE_PROFIT_MARKET", close_side, qty, None, tp_params)

            send_telegram(
                f"✅ 已下单 {symbol} {side_text} 数量={qty} @ {price:.2f}\n🎯 止盈: {take_profit:.2f}\n🛡 止损: {stop_loss:.2f}"
            )
        except Exception as e:
            send_telegram(f"✅ 已下单 {symbol} {side_text} 数量={qty} @ {price:.2f}\n⚠️ 挂止盈/止损失败: {e}")

    except Exception as e:
        send_telegram(f"❌ 下单失败 {symbol}，原因: {e}")


# ===========================
# 平仓函数
# ===========================
def close_position(symbol, position):
    """
    市价平掉给定仓位
    自动处理单向/双向模式
    """
    try:
        qty = position.get("qty")
        if qty is None or qty == 0:
            send_telegram(f"❌ 平仓失败 {symbol}：无法解析仓位数量")
            return False

        pos_side = position.get("side", "").lower()
        side = "buy" if pos_side.startswith("short") else "sell"

        # 检查账户是否是双向模式
        is_hedge = False
        try:
            info = exchange.fapiPrivate_get_positionmode()
            is_hedge = info.get("dualSidePosition") == True
        except Exception:
            pass

        params = {}
        if is_hedge:
            params["positionSide"] = "SHORT" if side=="buy" else "LONG"

        if LIVE_TRADE:
            try:
                qty_precise = float(exchange.amount_to_precision(symbol, qty))
            except Exception:
                qty_precise = round(qty, 6)

            exchange.create_market_order(symbol, side, qty_precise, params=params)
            send_telegram(f"✅ 已市价平仓 {symbol} {pos_side} 数量={qty_precise}")
        else:
            send_telegram(f"📌 模拟平仓 {symbol} {pos_side} 数量={qty}")

        return True
    except Exception as e:
        send_telegram(f"❌ 平仓失败 {symbol}，原因: {e}")
        return False

# ===========================
# 趋势检测
# ===========================
def check_trend_once(symbol):
    alerts = []
    status = {}
    multi_tf_signal = None
    reasons_all = []

    for tf in ["1h","4h","1d"]:  # 多周期共振
        try:
            df = compute_indicators(fetch_ohlcv_df(symbol, tf, 100))
            signal, score, reasons, last = signal_from_indicators(df)
            status[tf] = {"signal": signal, "score": score, "reasons": reasons, "last_close": last["close"]}
            if signal:
                reasons_all.extend([f"{tf}:{r}" for r in reasons])
                if multi_tf_signal is None:
                    multi_tf_signal = signal
                elif multi_tf_signal != signal:
                    multi_tf_signal = None  # 不共振
        except Exception as e:
            status[tf] = {"error": str(e)}
    
    if multi_tf_signal:
        alerts.append(f"{now_str()} {symbol} 多周期共振信号: {multi_tf_signal} 原因: {';'.join(reasons_all)}")
    
    return alerts, status, multi_tf_signal

# ===========================
# 主循环
# ===========================
def main_loop():
    for symbol in SYMBOLS:
        setup_account(symbol)
    
    while True:
        try:
            for symbol in SYMBOLS:
                alerts, status, signal = check_trend_once(symbol)
                
                for alert in alerts:
                    print(alert)
                    send_telegram(alert)
                
                if signal:
                    df = compute_indicators(fetch_ohlcv_df(symbol, "1h", 100))
                    last_close = df.iloc[-1]["close"]
                    last_atr = df.iloc[-1]["atr"]
                    pos = get_position(symbol)
                    
                    if pos:
                        # 如果已有仓位且方向不同，先平仓
                        if (signal=="买入" and pos["side"]=="short") or (signal=="卖出" and pos["side"]=="long"):
                            close_position(symbol, pos)
                            time.sleep(1)  # 等待平仓
                            place_order(symbol, signal, last_close, last_atr)
                    else:
                        place_order(symbol, signal, last_close, last_atr)
            time.sleep(POLL_INTERVAL)
        except Exception as e:
            print("⚠️ 主循环异常:", e)
            time.sleep(POLL_INTERVAL)

# ===========================
# 启动
# ===========================
if __name__ == "__main__":
    print(f"🚀 AutoTrader 启动 {SYMBOLS}，LIVE_TRADE={LIVE_TRADE}")
    main_loop()
