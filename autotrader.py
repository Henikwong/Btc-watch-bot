# autotrader.py
import os
import time
import ccxt
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import ta

# ===========================
# 配置（可通过环境变量覆盖）
# ===========================
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT").split(",") if s.strip()]
BASE_USDT = float(os.getenv("BASE_USDT", "100"))
LEVERAGE = int(os.getenv("LEVERAGE", "10"))
LIVE_TRADE = os.getenv("LIVE_TRADE", "0") == "1"
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))
ONLY_RESONANCE = os.getenv("ONLY_RESONANCE", "1") == "1"

TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "3.0"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "2.0"))
OHLCV_LIMIT = int(os.getenv("OHLCV_LIMIT", "200"))

# 用于缓存最后一次信号，避免重复推送
last_signal = {}

# ===========================
# 工具函数
# ===========================
def now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def send_telegram(msg: str):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("⚠️ 未配置 Telegram，跳过:", msg)
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

# ===========================
# 仓位和账户
# ===========================
def setup_account(symbol):
    # Binance CCXT 新版本已没有 fapiPrivate_post_leverage，需要用 set_leverage
    try:
        market = exchange.market(symbol)
        exchange.fapiPrivate_post_margintype({"symbol": market["id"], "marginType": "ISOLATED"})
        exchange.fapiPrivate_post_leverage({"symbol": market["id"], "leverage": LEVERAGE})
        print(f"✅ 已设置 {symbol} 杠杆与保证金模式")
    except Exception as e:
        print("⚠️ setup_account 失败:", e)

def fetch_all_positions():
    try:
        return exchange.fetch_positions()
    except Exception as e:
        print("⚠️ fetch_positions 失败:", e)
        return []

def parse_position_entry(pos):
    try:
        symbol = pos.get("symbol") or (pos.get("info") or {}).get("symbol")
        contracts = pos.get("contracts") or pos.get("positionAmt") or pos.get("amount")
        if contracts is None or float(contracts) == 0:
            return None
        contracts = abs(float(contracts))
        side = "long" if float(pos.get("positionAmt",0)) > 0 else "short"
        entry = pos.get("entryPrice") or (pos.get("info") or {}).get("entryPrice") or None
        return (symbol, contracts, side, float(entry) if entry else None)
    except:
        return None

def get_position(symbol):
    positions = fetch_all_positions()
    for p in positions:
        parsed = parse_position_entry(p)
        if not parsed: continue
        sym, qty, side, entry = parsed
        if sym.replace("/", "") == symbol.replace("/", ""):
            return {"symbol": symbol, "qty": qty, "side": side, "entry": entry, "raw": p}
    return None

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

def compute_indicators(df):
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

def signal_from_indicators(df):
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
    if last["rsi"] > 60: score += 1
    elif last["rsi"] < 40: score -= 1
    if last["volume"] > last["vol_ma20"] * 1.5: score +=1; reasons.append("成交量放大")
    if score >= 3: return "买入", score, reasons, last
    elif score <= -3: return "卖出", score, reasons, last
    return None, score, reasons, last

# ===========================
# 初始化已有仓位信号
# ===========================
def init_last_signal():
    for symbol in SYMBOLS:
        pos = get_position(symbol)
        if pos:
            side_text = "买入" if pos["side"]=="long" else "卖出"
            last_signal[symbol] = f"{now_str()} {symbol} 多周期共振信号: {side_text} (启动已有仓位)"
            print(f"📌 启动已有仓位 {symbol}: {side_text}, 已缓存 last_signal")

# ===========================
# 下单
# ===========================
def place_order(symbol, side_text, price, atr):
    side = "buy" if side_text=="买入" else "sell"
    try:
        qty = BASE_USDT * LEVERAGE / price
        qty = float(exchange.amount_to_precision(symbol, qty))
    except:
        qty = round(qty,6)
    if not LIVE_TRADE:
        send_telegram(f"📌 模拟下单 {symbol} {side_text} 数量={qty} @ {price:.2f}")
        return
    try:
        # 处理双向模式
        params = {}
        try:
            res = exchange.fapiPrivate_get_positionmode()
            dual_side = res.get("dualSidePosition", True)
            if dual_side:
                params["positionSide"] = "LONG" if side_text=="买入" else "SHORT"
        except:
            params["positionSide"] = "LONG" if side_text=="买入" else "SHORT"

        exchange.create_market_order(symbol, side, qty, params=params)

        # 止盈止损
        if atr is None or np.isnan(atr):
            atr = price * 0.005
        if side_text=="买入":
            stop_loss = price - SL_ATR_MULT*atr
            take_profit = price + TP_ATR_MULT*atr
            close_side = "sell"; close_pos_side="LONG"
        else:
            stop_loss = price + SL_ATR_MULT*atr
            take_profit = price - TP_ATR_MULT*atr
            close_side="buy"; close_pos_side="SHORT"

        try:
            exchange.create_order(symbol, "STOP_MARKET", close_side, qty, None,
                                  {"stopPrice": stop_loss, "positionSide": close_pos_side})
            exchange.create_order(symbol, "TAKE_PROFIT_MARKET", close_side, qty, None,
                                  {"stopPrice": take_profit, "positionSide": close_pos_side})
            send_telegram(f"✅ 已下单 {symbol} {side_text} 数量={qty} @ {price:.2f}\n🎯 止盈: {take_profit:.2f}\n🛡 止损: {stop_loss:.2f}")
        except Exception as e:
            send_telegram(f"✅ 已下单 {symbol} {side_text} 数量={qty} @ {price:.2f}\n⚠️ 挂止盈/止损失败: {e}")
    except Exception as e:
        send_telegram(f"❌ 下单失败 {symbol}，原因: {e}")

# ===========================
# 平仓
# ===========================
def close_position(symbol, position):
    try:
        qty = position.get("qty")
        if qty is None or qty==0: return False
        pos_side = position.get("side","").lower()
        side = "buy" if pos_side.startswith("short") else "sell"
        params={}
        try:
            info = exchange.fapiPrivate_get_positionmode()
            if info.get("dualSidePosition")==True:
                params["positionSide"]="SHORT" if side=="buy" else "LONG"
        except: pass
        if LIVE_TRADE:
            qty_precise=float(exchange.amount_to_precision(symbol, qty))
            exchange.create_market_order(symbol, side, qty_precise, params=params)
            send_telegram(f"✅ 已市价平仓 {symbol} {pos_side} 数量={qty_precise}")
        else:
            send_telegram(f"📌 模拟平仓 {symbol} {pos_side} 数量={qty}")
        return True
    except Exception as e:
        send_telegram(f"❌ 平仓失败 {symbol}，原因: {e}")
        return False

# ===========================
# 多周期共振检测
# ===========================
def check_trend_once(symbol):
    alerts=[]
    status={}
    multi_tf_signal=None
    reasons_all=[]
    for tf in ["1h","4h","1d"]:
        try:
            df = compute_indicators(fetch_ohlcv_df(symbol, tf, 100))
            signal, score, reasons, last = signal_from_indicators(df)
            status[tf]={"signal":signal,"score":score,"reasons":reasons,"last_close":last["close"]}
            if signal:
                reasons_all.extend([f"{tf}:{r}" for r in reasons])
                if multi_tf_signal is None:
                    multi_tf_signal = signal
                elif multi_tf_signal != signal:
                    multi_tf_signal = None
        except Exception as e:
            status[tf]={"error":str(e)}
    if multi_tf_signal:
        msg = f"{now_str()} {symbol} 多周期共振信号: {multi_tf_signal} 原因: {';'.join(reasons_all)}"
        if last_signal.get(symbol)!=msg:
            last_signal[symbol]=msg
            alerts.append(msg)
    return alerts, status, multi_tf_signal

# ===========================
# 主循环
# ===========================
def main_loop():
    for symbol in SYMBOLS:
        setup_account(symbol)
        pos = get_position(symbol)
        print(f"📌 启动时 {symbol} 仓位: {pos}")

    init_last_signal()

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
                        if (signal=="买入" and pos["side"]=="short") or (signal=="卖出" and pos["side"]=="long"):
                            close_position(symbol, pos)
                            time.sleep(1)
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
