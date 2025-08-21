import time
import logging
import requests
import talib
import numpy as np
from datetime import datetime

# ================= 日志配置 =================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ================= 交易对 & 周期 =================
symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "LTCUSDT"]
timeframes = {"1h": "60min", "4h": "240min", "1d": "1day", "1w": "1week"}

# ================= 交易所 API =================
def fetch_klines(exchange, symbol, interval, limit=200):
    try:
        if exchange == "binance":
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
            res = requests.get(url).json()
            return [[float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in res]

        elif exchange == "okx":
            url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}&bar={interval}&limit={limit}"
            res = requests.get(url).json()["data"]
            return [[float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in res][::-1]

        elif exchange == "huobi":
            url = f"https://api.huobi.pro/market/history/kline?symbol={symbol.lower()}&period={interval}&size={limit}"
            res = requests.get(url).json()["data"]
            return [[float(x["open"]), float(x["high"]), float(x["low"]), float(x["close"]), float(x["vol"])] for x in res][::-1]
    except Exception as e:
        logging.error(f"[{exchange.upper()} ERROR] {symbol} {e}")
        return []

# ================= 指标计算 =================
def calc_indicators(data):
    closes = np.array([x[3] for x in data], dtype=float)
    highs = np.array([x[1] for x in data], dtype=float)
    lows = np.array([x[2] for x in data], dtype=float)
    vols = np.array([x[4] for x in data], dtype=float)

    ema_short = talib.EMA(closes, 12)
    ema_long = talib.EMA(closes, 26)
    macd, macd_signal, _ = talib.MACD(closes)
    rsi = talib.RSI(closes, 14)
    wr = talib.WILLR(highs, lows, closes, 14)
    vol_delta = (vols[-1] - np.mean(vols[-10:])) / np.mean(vols[-10:])

    return {
        "EMA_trend": "多" if ema_short[-1] > ema_long[-1] else "空",
        "MACD": macd[-1] - macd_signal[-1],
        "RSI": rsi[-1],
        "WR": wr[-1],
        "VOLΔ": vol_delta,
        "close": closes[-1],
    }

# ================= 信号判断 =================
def check_signal(indicators_all):
    trends = [i["EMA_trend"] for i in indicators_all]
    if all(t == "多" for t in trends):
        return "多"
    elif all(t == "空" for t in trends):
        return "空"
    return None

# ================= Telegram 推送 =================
def send_telegram(msg):
    token = "你的TOKEN"
    chat_id = "你的CHAT_ID"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, data={"chat_id": chat_id, "text": msg})

# ================= 主循环 =================
def main():
    logging.info("启动 Bot - 多交易所多周期监控")
    last_hour = -1  # 控制普通信息每小时一次

    while True:
        now = datetime.now()
        hour = now.hour

        all_results = {}

        for symbol in symbols:
            all_results[symbol] = {}

            for tf_name, tf_api in timeframes.items():
                results = []
                for ex in ["binance", "okx", "huobi"]:
                    data = fetch_klines(ex, symbol, tf_api, 200)
                    if len(data) > 50:
                        indicators = calc_indicators(data)
                        results.append(indicators)
                        all_results[symbol][tf_name] = indicators
                        logging.info(f"{symbol} {tf_name} {ex}: {indicators}")

                # 检查突发信号
                if len(results) == 3:
                    signal = check_signal(results)
                    if signal:
                        price = results[0]["close"]
                        entry = price
                        tp = price * (1.02 if signal == "多" else 0.98)
                        sl = price * (0.98 if signal == "多" else 1.02)

                        msg = f"🔥🔥🔥 强烈高度动向捕捉到（满足所有条件）\n⚠️ {symbol} {signal} 信号\n入场: {entry:.2f}\n目标: {tp:.2f}\n止损: {sl:.2f}"
                        logging.info(msg)
                        send_telegram(msg)

        # 每小时推送普通信息
        if hour != last_hour:
            last_hour = hour
            report = "📢 每小时普通信息\n"
            for symbol, tf_data in all_results.items():
                report += f"\n{symbol}:\n"
                for tf, ind in tf_data.items():
                    report += f" ⏱ {tf} | 趋势={ind['EMA_trend']} MACD={ind['MACD']:.4f} RSI={ind['RSI']:.2f} WR={ind['WR']:.2f} VOLΔ={ind['VOLΔ']:.2f}\n"
            send_telegram(report)

        logging.info("等待 60 秒后继续...")
        time.sleep(60)

if __name__ == "__main__":
    main()
