import os, requests

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

resp = requests.get(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                    params={"chat_id": CHAT_ID, "text": "测试消息 from Railway"})
print(resp.json())
