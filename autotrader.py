import os
import math
import time
import ccxt
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# ============= 配置 =============
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
LEVERAGE = 10
RISK_RATIO = 0.15
TP_ATR_MULT = 3.0
SL_ATR_MULT = 2.0
COOLDOWN = 60  # 信号冷却时间秒
SUMMARY_INTERVAL = 1800  # 每 30 分钟汇总一次

# 从环境变量读取资金基数
BASE_USDT = float(os.getenv("BASE_USDT", "20"))

# Telegram 配置
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Binance
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

LIVE_TRADE = os.getenv("LIVE_TRADE", "false").lower() == "true"

# ============= 工具函数 =============
def telegram_send(msg):
    """发送 Telegram 消息"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
    except Exception as e:
        print(f"⚠️ Telegram 发送失败: {e}")

def fetch_ohlcv(symbol, timeframe, limit=200):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        df.set_index("time", inplace=True)
        return df
    except Exception as e:
        print(f"❌ 获取K线失败: {e}")
        return pd.DataFrame()

def compute_indicators(df):
    df["ema_fast"] = df["close"].ewm(span=12).mean()
    df["ema_slow"] = df["close"].ewm(span=26).mean()
    df["macd"] = df["ema_fast"] - df["ema_slow"]
    df["signal"] = df["macd"].ewm(span=9).mean()
    df["atr"] = df["high"] - df["low"]
    return df

def signal_from_indicators(df):
    score = 0
    if df["ema_fast"].iloc[-1] > df["ema_slow"].iloc[-1]:
        score += 1
    else:
        score -= 1
    if df["macd"].iloc[-1] > df["signal"].iloc[-1]:
        score += 1
    else:
        score -= 1
    if score >= 2:
        return "buy", score
    elif score <= -2:
        return "sell", score
    return None, score

def round_step(value, step):
    return math.floor(value / step) * step

def get_symbol_info(symbol):
    markets = exchange.load_markets()
    market = markets[symbol]
    lot = market["limits"]["amount"]
    step = lot["min"]
    return {"minQty": lot["min"], "stepSize": step}

# ============= 下单逻辑 =============
def place_order(symbol, side, qty, price, atr):
    if qty <= 0:
        return False
    print(f"📥 下单: {side} {qty:.6f} {symbol} @ {price}")
    telegram_send(f"📥 下单: {side} {qty:.6f} {symbol} @ {price}")
    return True

# ============= 主循环 =============
def main_loop():
    last_signal_time = datetime.min.replace(tzinfo=timezone.utc)
    last_summary_time = datetime.now(timezone.utc)

    while True:
        try:
            df = fetch_ohlcv(SYMBOL, TIMEFRAME, limit=200)
            if df.empty: 
                time.sleep(10)
                continue
            df = compute_indicators(df)
            signal, score = signal_from_indicators(df)
            price = df["close"].iloc[-1]
            atr = df["atr"].iloc[-1]

            if signal:
                now = datetime.now(timezone.utc)
                if (now - last_signal_time).total_seconds() > COOLDOWN:
                    balance = BASE_USDT
                    qty = (balance * RISK_RATIO * LEVERAGE) / price
                    info = get_symbol_info(SYMBOL)
                    qty = round_step(qty, info["stepSize"])
                    if qty < info["minQty"]:
                        print(f"⚠️ {SYMBOL} 下单量 {qty} < 最小量 {info['minQty']}")
                        continue
                    place_order(SYMBOL, signal, qty, price, atr)
                    last_signal_time = now

            # 定期汇总
            if (datetime.now(timezone.utc) - last_summary_time).total_seconds() > SUMMARY_INTERVAL:
                print(f"📊 {datetime.now(timezone.utc)} | 最新价 {price:.2f} | 信号 {signal} (score={score})")
                last_summary_time = datetime.now(timezone.utc)

            time.sleep(10)
        except Exception as e:
            print(f"❌ 主循环异常: {e}")
            time.sleep(5)

if __name__ == "__main__":
    exchange = ccxt.binance()
    main_loop()1
