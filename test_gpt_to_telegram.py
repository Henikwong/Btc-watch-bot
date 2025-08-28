# test_gpt_to_telegram.py
import os, requests, openai

openai.api_key = os.getenv("OPENAI_API_KEY")
TG_BOT = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID")

def send_tg(text):
    r = requests.post(
        f"https://api.telegram.org/bot{TG_BOT}/sendMessage",
        json={"chat_id": TG_CHAT, "text": text}
    )
    print("TG:", r.status_code, r.text)

def main():
    if not (openai.api_key and TG_BOT and TG_CHAT):
        print("❌ 环境变量不完整")
        return
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":"Say hi in one short sentence."}],
            max_tokens=50,
        )
        msg = resp.choices[0].message.content.strip()
        print("GPT:", msg)
        send_tg("GPT 回应：" + msg)
    except Exception as e:
        send_tg(f"❌ GPT 调用失败：{e}")
        print("❌", e)

if __name__ == "__main__":
    main()
