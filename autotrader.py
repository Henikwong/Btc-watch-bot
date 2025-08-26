# app/autotrader.py
import os
import time
import ccxt
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import ta

# ===========================
# 配置（环境变量覆盖）
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

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ===========================
# 帮助函数
# ===========================
def now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def send_telegram(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ 未配置 Telegram 环境变量，跳过推送:", msg)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
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
    df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
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
        return "买入", score, reasons, last
    elif score <= -3:
        return "卖出", score, reasons, last
    else:
        return None, score, reasons, last

# ===========================
# 仓位相关
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
        contracts = float(pos.get("contracts") or pos.get("positionAmt") or pos.get("amount") or 0)
        if contracts == 0: return None
        side = pos.get("side") or ("long" if contracts > 0 else "short")
        entry = pos.get("entryPrice") or (pos.get("info") or {}).get("entryPrice")
        return (symbol, abs(contracts), side, float(entry) if entry else None)
    except: return None

def get_position(symbol):
    positions = fetch_all_positions()
    for p in positions:
        parsed = parse_position_entry(p)
        if parsed and parsed[0] and parsed[0].replace("/","") == symbol.replace("/",""):
            return {"symbol": symbol, "qty": parsed[1], "side": parsed[2], "entry": parsed[3], "raw": p}
    return None

def close_position(symbol, position):
    try:
        qty = position.get("qty")
        if qty is None or qty==0:
            send_telegram(f"❌ 平仓失败 {symbol}：无法解析仓位数量"); return False
        pos_side = position.get("side","")
        side = "buy" if pos_side.lower().startswith("short") else "sell"
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
# 下单
# ===========================
def place_order(symbol, side_text, price, atr):
    side = "buy" if side_text=="买入" else "sell"
    try:
        qty = BASE_USDT*LEVERAGE/price
        try: qty = float(exchange.amount_to_precision(symbol, qty))
        except: qty=round(qty,6)
    except Exception as e:
        send_telegram(f"❌ 计算下单数量失败 {symbol}：{e}"); return
    if not LIVE_TRADE:
        send_telegram(f"📌 模拟下单 {symbol} {side_text} 数量={qty} @ {price:.2f}")
        return
    try:
        exchange.create_market_order(symbol, side, qty)
        if atr is None or np.isnan(atr): atr = price*0.005
        if side=="buy":
            stop_loss = price-SL_ATR_MULT*atr
            take_profit = price+TP_ATR_MULT*atr
            close_side="sell"
        else:
            stop_loss=price+SL_ATR_MULT*atr
            take_profit=price-TP_ATR_MULT*atr
            close_side="buy"
        try:
            exchange.create_order(symbol,"STOP_MARKET",close_side,qty,None,{"stopPrice":stop_loss})
            exchange.create_order(symbol,"TAKE_PROFIT_MARKET",close_side,qty,None,{"stopPrice":take_profit})
            send_telegram(f"✅ 已下单 {symbol} {side_text} 数量={qty} @ {price:.2f}\n🎯 止盈: {take_profit:.2f}\n🛡 止损: {stop_loss:.2f}")
        except Exception as e:
            send_telegram(f"✅ 已下单 {symbol} {side_text} 数量={qty} @ {price:.2f}\n⚠️ 挂止盈/止损失败: {e}")
    except Exception as e:
        send_telegram(f"❌ 下单失败 {symbol}，原因: {e}")

# ===========================
# 趋势检测
# ===========================
def check_trend_once(symbol):
    result={"alerts":[],"status":{}}
    for tf in ["4h","1d"]:
        try:
            df = compute_indicators(fetch_ohlcv_df(symbol,timeframe=tf))
            last=df.iloc[-1]; prev=df.iloc[-2]
            status="多头" if last["ema20"]>last["ema50"] else "空头"
            result["status"][tf]=status
            if last["ema20"]>last["ema50"] and prev["ema20"]<=prev["ema50"]:
                result["alerts"].append(f"⚡ 趋势提醒: {symbol} {tf} 出现金叉 → 趋势看多")
            elif last["ema20"]<last["ema50"] and prev["ema20"]>=prev["ema50"]:
                result["alerts"].append(f"⚡ 趋势提醒: {symbol} {tf} 出现死叉 → 趋势转空")
        except Exception as e:
            result["alerts"].append(f"❌ 趋势检测失败 {symbol} {tf}: {e}")
    if result["status"].get("4h") and result["status"].get("1d") and result["status"]["4h"]==result["status"]["1d"]:
        result["alerts"].append(f"🔥 趋势共振: {symbol} ({result['status']['4h']})")
    return result

def startup_trend_report():
    report=["📌 启动时趋势检测:"]
    for symbol in SYMBOLS:
        r=check_trend_once(symbol)
        st4=r["status"].get("4h","未知"); st1=r["status"].get("1d","未知")
        report.append(f"{symbol} 4h:{st4} | 1d:{st1}")
