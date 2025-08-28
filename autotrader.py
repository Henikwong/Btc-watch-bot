import requests

TOKEN = "你的BOT_TOKEN"
CHAT_ID = "你的CHAT_ID"

msg = "测试消息，检查BOT是否可用"

url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
res = requests.post(url, json={"chat_id": CHAT_ID, "text": msg})

print(res.status_code, res.json())
