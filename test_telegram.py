def send_telegram(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram 未配置，消息打印:", msg)
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}
        )
        print("📨 Telegram 返回:", r.status_code, r.text)  # 加这一行
    except Exception as e:
        print("❌ Telegram 推送失败:", e)
