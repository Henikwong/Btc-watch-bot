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

# ================== 配置 ==================
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

LEVERAGE = int(os.getenv("LEVERAGE", "10"))
BASE_USDT = float(os.getenv("BASE_USDT", "20"))  # 默认 20 USDT，可环境变量配置
RISK_RATIO = float(os.getenv("RISK_RATIO", "0.15"))

TP_ATR_MULT = 3.0
SL_ATR_MULT = 2.0

# ================== 初始化 ==================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
exchange = ccxt.binance({
    "apiKey": BINANCE_API_KEY,
    "secret": BINANCE_API_SECRET,
    "enableRateLimit": True,
    "options": {"defaultType": "future"}  # 使用 USDT-M 合约
})

# ================== 工具函数 ==================
def send_telegram(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram 未配置，消息未发送")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
        requests.post(url, data=data, timeout=5)
    except Exception as e:
        logging.error(f"Telegram 发送失败: {e}")

def log_and_notify(level, msg):
    getattr(logging, level)(msg)
    send_telegram(msg)

def amount_from_usdt(symbol, usdt_amount, price):
    """根据 USDT 金额换算下单数量"""
    try:
        qty = usdt_amount / price
        return float(exchange.amount_to_precision(symbol, qty))
    except Exception:
        return 0.0

# ================== 策略函数 (示例) ==================
def compute_indicators(df: pd.DataFrame):
    df["ema_fast"] = ta.trend.ema_indicator(df["close"], 12)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], 26)
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], 14)
    return df

def signal_from_indicators(df: pd.DataFrame):
    if df["ema_fast"].iloc[-1] > df["ema_slow"].iloc[-1]:
        return "buy", None, None
    elif df["ema_fast"].iloc[-1] < df["ema_slow"].iloc[-1]:
        return "sell", None, None
    return None, None, None

# ================== 下单逻辑 ==================
def place_order(symbol, side, qty, price, atr):
    """开仓并自动挂止盈止损"""
    if qty <= 0:
        log_and_notify("error", f"❌ 下单失败 {symbol} 数量为 0")
        return

    try:
        # 开仓单
        order = exchange.create_order(
            symbol=symbol,
            type="MARKET",
            side=side.upper(),
            amount=qty
        )

        # TP/SL 价格
        if side == "buy":
            tp_price = price + TP_ATR_MULT * atr
            sl_price = price - SL_ATR_MULT * atr
            pos_side = "LONG"
        else:
            tp_price = price - TP_ATR_MULT * atr
            sl_price = price + SL_ATR_MULT * atr
            pos_side = "SHORT"

        # 止盈单
        exchange.create_order(
            symbol=symbol,
            type="TAKE_PROFIT_MARKET",
            side="SELL" if side == "buy" else "BUY",
            amount=qty,
            params={"stopPrice": tp_price, "reduceOnly": True, "positionSide": pos_side}
        )

        # 止损单
        exchange.create_order(
            symbol=symbol,
            type="STOP_MARKET",
            side="SELL" if side == "buy" else "BUY",
            amount=qty,
            params={"stopPrice": sl_price, "reduceOnly": True, "positionSide": pos_side}
        )

        log_and_notify("info", f"✅ 开仓 {side.upper()} {symbol} qty={qty} @ {price:.2f} | TP={tp_price:.2f} SL={sl_price:.2f}")

    except Exception as e:
        log_and_notify("error", f"❌ 下单失败 {symbol}: {e}")

# ================== 主循环 ==================
def main_loop():
    symbol = "BTC/USDT"
    timeframe = "1h"

    while True:
        try:
            # 获取历史数据
            ohlcvs = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=200)
            df = pd.DataFrame(ohlcvs, columns=["time", "open", "high", "low", "close", "volume"])
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df.set_index("time", inplace=True)

            df = compute_indicators(df)
            signal, _, _ = signal_from_indicators(df)

            price = df["close"].iloc[-1]
            atr = df["atr"].iloc[-1]
            qty = amount_from_usdt(symbol, BASE_USDT * RISK_RATIO * LEVERAGE, price)

            if signal in ["buy", "sell"]:
                place_order(symbol, signal, qty, price, atr)

        except Exception as e:
            log_and_notify("error", f"主循环异常: {e}")

        time.sleep(60)  # 等待一分钟再跑

if __name__ == "__main__":
    main_loop()
