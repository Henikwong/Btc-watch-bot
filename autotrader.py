import os
import time
import ccxt
import requests
import numpy as np
from collections import deque
from datetime import datetime
from dotenv import load_dotenv

# 读取 .env 配置
load_dotenv()
TG_TOKEN = os.getenv("TG_TOKEN")
TG_CHAT_ID = os.getenv("TG_CHAT_ID")

# 初始化交易所 (HTX / 火币)
exchange = ccxt.huobi({
    "enableRateLimit": True,
})

# 价格缓存（最多存 20 个点）
prices = deque(maxlen=20)

def tg_send(msg: str):
    """发送Telegram通知"""
    if not TG_TOKEN or not TG_CHAT_ID:
        print(f"[WARN] TG未配置, 消息: {msg}")
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TG_CHAT_ID, "text": msg})
    except Exception as e:
        print(f"[ERR] TG发送失败: {e}")

def get_price(symbol="BTC/USDT"):
    """获取最新价格"""
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker["last"]
    except Exception as e:
        print(f"[ERR] 获取价格失败: {e}")
        return None

def strategy():
    """简单均线策略"""
    if len(prices) < 10:
        return None

    short_ma = np.mean(list(prices)[-3:])   # 3分钟均线
    long_ma = np.mean(list(prices)[-10:])  # 10分钟均线

    if short_ma > long_ma:
        return "BUY"
    elif short_ma < long_ma:
        return "SELL"
    return None

def main():
    tg_send("🤖 Bot启动, 策略=均线交叉, 模式=纸面测试")
    while True:
        price = get_price()
        if price:
            prices.append(price)
            signal = strategy()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] BTC/USDT={price:.2f} 信号={signal}")
            if signal:
                tg_send(f"📊 {now} 信号: {signal} @ {price:.2f}")

        time.sleep(60)  # 每分钟循环一次

if __name__ == "__main__":
    main()
