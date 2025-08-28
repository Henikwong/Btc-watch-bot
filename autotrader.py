import os
import requests

# 从环境变量读取
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    raise ValueError("请检查 TELEGRAM_BOT_TOKEN 和 TELEGRAM_CHAT_ID 是否正确设置")

# 测试发送消息
message = "✅ Bot 已启动并测试成功！"
url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
payload = {
    "chat_id": CHAT_ID,
    "text": message
}

response = requests.post(url, data=payload)

if response.status_code == 200:
    print("消息发送成功！")
else:
    print(f"发送失败: {response.status_code} {response.text}")
