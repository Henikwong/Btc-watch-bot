import os
import ccxt
import pandas as pd
import numpy as np
import ta
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# ============ 环境变量 ============
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
EXCHANGE_NAME = os.getenv("EXCHANGE", "binance").lower()
SYMBOLS = os.getenv("SYMBOLS", "BTC/USDT").split(",")
MARKET_TYPE = os.getenv("MARKET_TYPE", "future")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 60))

# ============ 初始化交易所 ============
def build_exchange(name, api_key, api_secret):
    if name == "binance":
        exchange = ccxt.binance({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": MARKET_TYPE}
        })
    else:
        raise RuntimeError(f"不支持交易所: {name}")
    return exchange

# ============ 技术指标分析 ============
def analyze(df: pd.DataFrame):
    # 计算 EMA
    df["ema"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    # MACD
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd_diff()
    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    # Williams %R
    df["wr"] = ta.momentum.WilliamsRIndicator(df["high"], df["low"], df["close"], lbp=14).williams_r()
    # KDJ (用随机指标近似)
    stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    df["kdj"] = stoch.stoch()
    # 成交量变化率
    df["vol_delta"] = df["volume"].pct_change()
    # 🔥 新增 ATR
    df["atr"] = ta.volatility.AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14
    ).average_true_range()

    latest = df.iloc[-1]
    return {
        "ema": "多" if latest["close"] > latest["ema"] else "空" if latest["close"] < latest["ema"] else "中性",
        "macd": latest["macd"],
        "rsi": latest["rsi"],
        "wr": latest["wr"],
        "kdj": "多" if latest["kdj"] > 50 else "空",
        "vol_delta": latest["vol_delta"],
        "atr": latest["atr"],   # 🔥 加上 ATR
    }

# ============ Telegram 推送 ============
def send_telegram(msg: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    try:
        requests.post(url, data=data)
    except Exception as e:
        print("发送Telegram失败:", e)

# ============ 主循环 ============
def main():
    ex = build_exchange(EXCHANGE_NAME, API_KEY, API_SECRET)
    timeframes = ["1h", "4h", "1d"]

    while True:
        all_msgs = []
        for sym in SYMBOLS:
            sym = sym.strip()
            sym_msgs = [f"{sym} 当前多周期共识:"]
            for tf in timeframes:
                ohlcv = ex.fetch_ohlcv(sym, tf, limit=200)
                df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
                info = analyze(df)

                direction = "多" if info["macd"] > 0 and info["rsi"] > 55 else "空" if info["macd"] < 0 and info["rsi"] < 45 else "无"
                price = df["close"].iloc[-1]

                line = (
                    f"{tf} | 方向:{direction} 入场:{price:.2f} | "
                    f"EMA:{info['ema']} MACD:{info['macd']:.4f} RSI:{info['rsi']:.2f} "
                    f"WR:{info['wr']:.2f} KDJ:{info['kdj']} VOLΔ:{info['vol_delta']:.3f} ATR:{info['atr']:.2f}"
                )
                sym_msgs.append(line)

            all_msgs.append("\n".join(sym_msgs))

        send_telegram("加密bot:\n" + "\n\n".join(all_msgs))
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
