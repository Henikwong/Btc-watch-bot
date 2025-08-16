import requests
import time
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# 获取 Binance BTC/USDT 价格
def get_price():
    url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
    r = requests.get(url)
    return float(r.json()["price"])

# GPT 分析
def analyze_signal(price):
    prompt = f"当前BTC价格为 {price}，基于趋势判断是否买入、卖出或观望，并用简短建议回答。"
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "model": "gpt-4o-mini",  # 你也可以换成 gpt-5
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        }
    )
    return r.json()["choices"][0]["message"]["content"]

# 发送 Telegram 消息
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message})

# 主循环
while True:
    price = get_price()
    signal = analyze_signal(price)
    print(f"{price} -> {signal}")
    send_telegram(f"BTC价格 {price}，GPT建议：{signal}")
    time.sleep(60)  # 每 60 秒检测一次
