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
# 初始化交易所
# ===========================
exchange = ccxt.binance({
    "apiKey": os.getenv("API_KEY"),
    "secret": os.getenv("API_SECRET"),
    "enableRateLimit": True,
    "options": {"defaultType": "future"}
})

def setup_account(symbol):
    try:
        market = exchange.market(symbol)
        ex_symbol = market["id"]
        try:
            exchange.fapiPrivate_post_margintype({"symbol": ex_symbol, "marginType": "ISOLATED"})
        except Exception as e:
            print("⚠️ 设置保证金模式失败:", e)
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
        if contracts is None or contracts==0: return None
        side = None
        if "side" in pos and pos["side"]: side = pos["side"]
        else:
            if "positionAmt" in pos:
                amt = float(pos["positionAmt"])
                side = "long" if amt > 0 else "short"
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

# ===========================
# 下单函数
# ===========================
def place_order(symbol, side, amount, price=None):
    if not LIVE_TRADE:
        print(f"💡 模拟下单 {symbol} {side} {amount} @ {price}")
        return None
    try:
        order_type = "MARKET"
        params = {"reduceOnly": False}
        if side == "buy":
            order = exchange.create_market_buy_order(symbol, amount, params)
        else:
            order = exchange.create_market_sell_order(symbol, amount, params)
        print(f"✅ 下单成功: {symbol} {side} {amount}")
        return order
    except Exception as e:
        print(f"❌ 下单失败 {symbol} {side}: {e}")
        return None

# ===========================
# 多周期共振
# ===========================
def check_multi_tf(symbol):
    multi_tf_signal = None
    reasons_all = []
    status = {}
    for tf in ["1h","4h","1d"]:
        try:
            df = compute_indicators(fetch_ohlcv_df(symbol, tf, 100))
            signal, score, reasons, last = signal_from_indicators(df)
            status[tf] = {"signal": signal, "score": score, "reasons": reasons, "last_close": last["close"], "atr": last["atr"]}
            if signal:
                reasons_all.extend([f"{tf}:{r}" for r in reasons])
                if multi_tf_signal is None:
                    multi_tf_signal = signal
                elif multi_tf_signal != signal:
                    multi_tf_signal = None
        except Exception as e:
            status[tf] = {"error": str(e)}
    return multi_tf_signal, reasons_all, status

# ===========================
# 主循环
# ===========================
def main_loop():
    last_report_time = datetime.now(timezone.utc) - timedelta(hours=1)
    while True:
        try:
            report_msgs = []
            for symbol in SYMBOLS:
                setup_account(symbol)
                signal, reasons, status = check_multi_tf(symbol)
                pos = get_position(symbol)
                current_price = status.get("1h", {}).get("last_close") or 0
                atr = status.get("1h", {}).get("atr") or 0

                # 每小时 Telegram 汇总
                now_time = datetime.now(timezone.utc)
                if (now_time - last_report_time) >= timedelta(hours=1):
                    msg = f"{now_str()} {symbol} 信号:{signal or '无'} 原因:{';'.join(reasons) if reasons else '无'} 价格:{current_price:.2f if current_price else 0}"
                    report_msgs.append(msg)

                # 开仓逻辑
                if signal and not pos:
                    # 计算仓位数量
                    amount = round(BASE_USDT * LEVERAGE / current_price, 5)
                    place_order(symbol, signal, amount)
                # 平仓逻辑
                elif signal and pos:
                    if (signal=="buy" and pos["side"]=="short") or (signal=="sell" and pos["side"]=="long"):
                        # 先平仓
                        place_order(symbol, "buy" if pos["side"]=="short" else "sell", pos["qty"])
                        # 再开新仓
                        amount = round(BASE_USDT * LEVERAGE / current_price, 5)
                        place_order(symbol, signal, amount)

            if report_msgs:
                for m in report_msgs:
                    send_telegram(m)
                last_report_time = datetime.now(timezone.utc)

            time.sleep(POLL_INTERVAL)
        except Exception as e:
            print("⚠️ 主循环异常:", e)
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    print(f"🚀 AutoTrader 启动 {SYMBOLS}，LIVE_TRADE={LIVE_TRADE}")
    main_loop()
