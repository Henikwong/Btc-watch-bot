import os, requests

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

print("🔧 TOKEN:", TOKEN)
print("🔧 CHAT_ID:", CHAT_ID)

url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
payload = {"chat_id": CHAT_ID, "text": "✅ Telegram 最小化测试"}
r = requests.post(url, json=payload)

print("📨 状态:", r.status_code)
print("📨 返回:", r.text)
