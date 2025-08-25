import os, time, traceback
import requests
import pandas as pd
import ccxt
import numpy as np
from datetime import datetime, timezone

# ========= 环境变量 =========
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
EXCHANGE = os.getenv("EXCHANGE", "binance")
MARKET_TYPE = os.getenv("MARKET_TYPE", "future")
SYMBOLS = os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,LTC/USDT,BNB/USDT,DOGE/USDT").split(",")
BASE_USDT = float(os.getenv("BASE_USDT", "100"))
LEVERAGE = int(os.getenv("LEVERAGE", "10"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))
LIVE_TRADE = int(os.getenv("LIVE_TRADE", "0"))

SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "2.0"))
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "3.0"))

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


# ========= 工具函数 =========
def now():
    return datetime.now(timezone.utc).isoformat()


def log(msg):
    print(f"[{now()}] {msg}", flush=True)


def tg_send(msg: str):
    """发消息到 Telegram"""
    if not TG_TOKEN or not TG_CHAT_ID:
        print("⚠️ TELEGRAM 未配置")
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        data = {"chat_id": TG_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
        r = requests.post(url, data=data, timeout=10)
        if r.status_code != 200:
            print(f"❌ TG发送失败: {r.text}")
    except Exception as e:
        print(f"❌ TG异常: {e}")


# ========= 数据 & 指标 =========
def fetch_ohlcv_df(ex, symbol, timeframe="15m", limit=200):
    try:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "vol"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        return df
    except Exception as e:
        log(f"❌ fetch_ohlcv {symbol} {e}")
        return None


def compute_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr.iloc[-1]


def indicators_and_side(df):
    """简单指标示例：均线金叉/死叉"""
    df["ma_fast"] = df["close"].rolling(9).mean()
    df["ma_slow"] = df["close"].rolling(21).mean()
    if df["ma_fast"].iloc[-1] > df["ma_slow"].iloc[-1]:
        return "多", {"entry": df["close"].iloc[-1]}
    elif df["ma_fast"].iloc[-1] < df["ma_slow"].iloc[-1]:
        return "空", {"entry": df["close"].iloc[-1]}
    return None, None


def format_price(p):
    try:
        return f"{p:.4f}"
    except:
        return str(p)


# ========= 下单逻辑 =========
def place_order_and_brackets(ex, symbol, side, entry, df):
    """这里只演示，不直接下真实单"""
    log(f"📌 模拟下单 {symbol} {side} @ {entry}")
    return {"symbol": symbol, "side": side, "price": entry}


def safe_place_order(ex, symbol, side, entry, df):
    try:
        atr = compute_atr(df, period=14)
        if side == "多":
            sl = entry - SL_ATR_MULT * atr
            tp = entry + TP_ATR_MULT * atr
        else:
            sl = entry + SL_ATR_MULT * atr
            tp = entry - TP_ATR_MULT * atr

        order = place_order_and_brackets(ex, symbol, side, entry, df)

        msg = (
            f"🚀 *入场信号*\n"
            f"交易对: `{symbol}`\n"
            f"方向: {side}\n"
            f"入场价: {format_price(entry)}\n"
            f"止盈: {format_price(tp)}\n"
            f"止损: {format_price(sl)}"
        )
        tg_send(msg)
        return order
    except Exception as e:
        log(f"❌ 下单失败 {symbol}: {e}")
        tg_send(f"❌ 下单失败 {symbol}: {e}")
        return None


# ========= 主循环 =========
def main():
    ex = getattr(ccxt, EXCHANGE)({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
    })
    ex.options["defaultType"] = MARKET_TYPE

    log("🚀 AutoTrader 启动...")
    tg_send("🤖 AutoTrader 已启动，开始监控行情...")

    last_hourly_push = 0
    positions = []

    while True:
        loop_start = time.time()
        try:
            report_lines = []
            for symbol in SYMBOLS:
                df = fetch_ohlcv_df(ex, symbol, timeframe="15m", limit=200)
                if df is None or len(df) < 50:
                    continue

                side, detail = indicators_and_side(df)
                if side:
                    entry = detail["entry"]
                    order = safe_place_order(ex, symbol, side, entry, df)
                    if order:
                        positions.append(order)
                else:
                    log(f"⏸ {symbol} 暂无信号")

                report_lines.append(f"{symbol}: {side or '无信号'}")

                time.sleep(1)

            # 每小时报告
            now_ts = int(time.time())
            if now_ts - last_hourly_push >= 3600:
                report_msg = "📊 每小时汇总报告\n" + "\n".join(report_lines)
                tg_send(report_msg)
                last_hourly_push = now_ts

        except Exception as e:
            log(f"❌ 主循环错误: {e}\n{traceback.format_exc()}")
            tg_send(f"⚠️ 主循环错误: {e}")

        used = time.time() - loop_start
        time.sleep(max(1, POLL_INTERVAL - int(used)))


if __name__ == "__main__":
    main()
