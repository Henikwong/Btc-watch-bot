import requests
import os

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Telegram 配置缺失，请检查 Railway Variables")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}

    print("📤 准备发送到 Telegram:")
    print("   Chat ID:", TELEGRAM_CHAT_ID)
    print("   Message:", message)

    try:
        r = requests.post(url, data=payload, timeout=10)
        print("📨 Telegram 返回:", r.status_code, r.text)
    except Exception as e:
        print("❌ Telegram 请求失败:", e)
