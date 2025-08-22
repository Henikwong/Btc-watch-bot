import os, requests

token = os.getenv("TELEGRAM_BOT_TOKEN", "你的token")
chat_id = os.getenv("TELEGRAM_CHAT_ID", "你的chatid")

resp = requests.get(
    f"https://api.telegram.org/bot{token}/sendMessage",
    params={"chat_id": chat_id, "text": "测试消息 from Railway"}
)
print(resp.json())
