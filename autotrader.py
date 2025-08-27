import os
import openai

# 读取 API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# 测试函数
def test_gpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=100
    )
    return response['choices'][0]['message']['content']

# 测试调用
prompt = "你好 GPT，请给我一句鼓励的话"
reply = test_gpt(prompt)
print("GPT 回复:", reply)
