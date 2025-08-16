import os
import time
import requests

print("Bot started!")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
if TOKEN and CHAT_ID:
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": "Bot is running!"}
    requests.post(url, data=data)

while True:
    print("Bot running...")
    time.sleep(10)
