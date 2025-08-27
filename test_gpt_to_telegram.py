import os, requests
from openai import OpenAI

# GPT client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Telegram 配置
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(msg: str):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    r = requests.post(url, json={"chat_id": CHAT_ID, "text": msg})
    print("📨 Telegram 返回:", r.status_code, r.text)

if __name__ == "__main__":
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "你好 GPT，你能介绍一下自己吗？"}],
        )
        reply = resp.choices[0].message.content.strip()
    except Exception as e:
        reply = f"❌ GPT 调用失败: {e}"

    send_telegram(f"🤖 GPT回应:\n{reply}")
