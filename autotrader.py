import os, time, requests

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_tg(msg):
    if not TOKEN or not CHAT_ID:
        print("⚠️ 没有设置 Telegram 环境变量")
        return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    r = requests.get(url, params={"chat_id": CHAT_ID, "text": msg})
    print("Telegram response:", r.json())

if __name__ == "__main__":
    send_tg("🤖 Bot启动 huobi/spot 模式=纸面")
    while True:
        print("⏳ 正在运行...")
        time.sleep(60)
