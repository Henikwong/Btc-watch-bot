import openai
import os

# 读取 API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# GPT 生成回应
def gpt_response(prompt):
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ GPT 调用失败: {e}"


# === 关键：把 GPT 回应发到 Telegram ===
def gpt_to_telegram(prompt):
    reply = gpt_response(prompt)
    send_telegram(f"🤖 GPT 回复:\n{reply}")
