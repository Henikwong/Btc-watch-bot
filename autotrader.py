# app/autotrader.py

import os
import time
import ccxt
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import ta

# ===========================
# 工具函数
# ===========================
def now():
    return datetime.now(timezone.utc).isoformat()

def send_telegram(msg: str):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            requests.post(url, json={"chat_id": chat_id, "text": msg})
        except Exception as e:
            print("❌ Telegram 发送失败:", e)

# ===========================
# 初始化交易所
# ===========================
exchange = ccxt.binance({
    "apiKey": os.getenv("API_KEY"),
    "secret": os.getenv("API_SECRET"),
    "enableRateLimit": True,
    "options": {"defaultType": "future"}  # ✅ futures
})

# ===========================
# 环境参数
# ===========================
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT").split(",") if s.strip()]
BASE_USDT = float(os.getenv("BASE_USDT", "100"))
LEVERAGE = int(os.getenv("LEVERAGE", "10"))
LIVE_TRADE = os.getenv("LIVE_TRADE", "0") == "1"
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))

# 止盈止损 ATR 倍数
RISK_ATR_MULT = float(os.getenv("RISK_ATR_MULT", "1.5"))
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "3.0"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "2.0"))

# 指标窗口
OHLCV_LIMIT = int(os.getenv("OHLCV_LIMIT", "200"))
TIMEFRAME = os.getenv("TIMEFRAME", "1h")

# ===========================
# 指标计算与信号逻辑
# ===========================
def fetch_ohlcv_df(symbol, timeframe=TIMEFRAME, limit=OHLCV_LIMIT):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # MA / EMA
    df["ma20"] = df["close"].rolling(20, min_periods=1).mean()
    df["ma50"] = df["close"].rolling(50, min_periods=1).mean()
    df["ema20"] = ta.trend.EMAIndicator(df["close"], window=20, fillna=True).ema_indicator()
    df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50, fillna=True).ema_indicator()

    # MACD
    macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9, fillna=True)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14, fillna=True).rsi()

    # ATR
    df["atr"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14, fillna=True
    ).average_true_range()

    # Williams %R
    df["wr"] = ta.momentum.WilliamsRIndicator(
        df["high"], df["low"], df["close"], lbp=14, fillna=True
    ).williams_r()

    # Stochastic (K, D, J)
    stoch = ta.momentum.StochasticOscillator(
        df["high"], df["low"], df["close"], window=14, smooth_window=3, fillna=True
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    df["kdj_j"] = 3 * df["stoch_k"] - 2 * df["stoch_d"]

    # Volume MA
    df["vol_ma20"] = df["volume"].rolling(20, min_periods=1).mean()

    return df

def signal_from_indicators(df: pd.DataFrame):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0
    reasons = []

    # EMA
    if last["ema20"] > last["ema50"]:
        score += 2
        reasons.append("EMA20>EMA50")
    else:
        score -= 2
        reasons.append("EMA20<EMA50")

    # MACD
    if last["macd"] > last["macd_signal"] and last["macd_hist"] > 0:
        score += 2
        reasons.append("MACD bullish")
    elif last["macd"] < last["macd_signal"] and last["macd_hist"] < 0:
        score -= 2
        reasons.append("MACD bearish")

    # RSI
    if last["rsi"] > 60:
        score += 1
        reasons.append(f"RSI high {last['rsi']:.1f}")
    elif last["rsi"] < 40:
        score -= 1
        reasons.append(f"RSI low {last['rsi']:.1f}")

    # Williams %R
    if last["wr"] <= -80:
        score += 1
        reasons.append(f"WR oversold {last['wr']:.1f}")
    elif last["wr"] >= -20:
        score -= 1
        reasons.append(f"WR overbought {last['wr']:.1f}")

    # KDJ
    if last["stoch_k"] > last["stoch_d"] and last["stoch_k"] > prev["stoch_k"]:
        score += 1
        reasons.append("KDJ bullish")
    elif last["stoch_k"] < last["stoch_d"] and last["stoch_k"] < prev["stoch_k"]:
        score -= 1
        reasons.append("KDJ bearish")

    # Volume
    if last["volume"] > last["vol_ma20"] * 1.5:
        score += 1
        reasons.append("Volume spike")

    threshold_long = 3
    threshold_short = -3

    if score >= threshold_long:
        return "buy", score, reasons, last
    elif score <= threshold_short:
        return "sell", score, reasons, last
    else:
        return None, score, reasons, last

# ===========================
# 下单函数
# ===========================
def place_order(symbol, side, price, atr):
    market = exchange.market(symbol)
    qty = BASE_USDT * LEVERAGE / price
    try:
        qty = float(exchange.amount_to_precision(symbol, qty))
    except Exception:
        qty = float(round(qty, 6))

    if not LIVE_TRADE:
        msg = f"📌 模拟下单 {symbol} {side} {qty:.6f} @ {price:.2f} (ATR={atr:.4f})"
        print(msg)
        send_telegram(msg)
        return

    try:
        order = exchange.create_market_order(symbol, side, qty)
        msg = f"✅ 入场 {symbol} {side} {qty:.6f} @ {price:.2f}"
        print(msg)
        send_telegram(msg)

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

        try:
            exchange.create_order(symbol, "STOP_MARKET", close_side, qty, None, {"stopPrice": stop_loss})
            exchange.create_order(symbol, "TAKE_PROFIT_MARKET", close_side, qty, None, {"stopPrice": take_profit})
            send_telegram(f"🎯 止盈: {take_profit:.4f}\n🛡 止损: {stop_loss:.4f}")
        except Exception as e:
            print("⚠️ 止盈/止损 挂单失败:", e)
            send_telegram(f"⚠️ 止盈/止损 挂单失败: {e}")

    except Exception as e:
        print("❌ 下单失败:", e)
        send_telegram(f"❌ 下单失败 {symbol}: {e}")

# ===========================
# 主循环
# ===========================
def main():
    send_telegram("🚀 AutoTrader (ta version) 启动...")
    print("启动: ", now())

    while True:
        for symbol in SYMBOLS:
            try:
                df = fetch_ohlcv_df(symbol)
                df = compute_indicators(df)
                signal, score, reasons, last = signal_from_indicators(df)

                ts = last["time"]
                price = float(last["close"])
                atr = float(last["atr"])

                log = f"{ts} {symbol} price={price:.2f} score={score} reasons={reasons}"
                print(log)

                if signal:
                    place_order(symbol, signal, price, atr)
                else:
                    print(f"[{symbol}] no clear signal. score={score}")

            except Exception as e:
                print(f"❌ {symbol} 出错:", e)
                send_telegram(f"❌ {symbol} 出错: {e}")

        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
