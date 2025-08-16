import requests

TELEGRAM_BOT_TOKEN = "在这里填你BotFather给的token"

url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
r = requests.get(url)
print(r.text)
