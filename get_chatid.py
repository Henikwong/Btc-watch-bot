import requests

TELEGRAM_BOT_TOKEN =8201207952:AAGxeasWHR6u-l3SiNyf5h4U7Pa4oQTsEAM

url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
r = requests.get(url)
print(r.text)
