import os, time, ccxt, requests

# === 从环境变量读取配置 ===
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
SYMBOLS = os.getenv("SYMBOLS", "BTC/USDT").split(",")
BASE_USDT = float(os.getenv("BASE_USDT", 15))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 60))
LIVE_TRADE = int(os.getenv("LIVE_TRADE", 0))

# === 初始化交易所 (Binance 现货) ===
exchange = ccxt.binance({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "enableRateLimit": True
})

# === Telegram 发送函数 ===
def send_tg(msg: str):
    if not TOKEN or not CHAT_ID:
        print("⚠️ 没有设置 Telegram 环境变量")
        return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        r = requests.get(url, params={"chat_id": CHAT_ID, "text": msg})
        print("Telegram response:", r.json())
    except Exception as e:
        print("Telegram 发送失败:", e)

# === 主逻辑 ===
def run_bot():
    mode = "实盘" if LIVE_TRADE else "纸面"
    send_tg(f"🤖 Bot启动 {exchange.id}/spot 模式={mode}")

    while True:
        for symbol in SYMBOLS:
            try:
                ticker = exchange.fetch_ticker(symbol)
                price = ticker["last"]
                msg = f"📈 {symbol} 最新价: {price}"
                print(msg)
                send_tg(msg)
            except Exception as e:
                print(f"❌ 获取行情失败 {symbol}:", e)
                send_tg(f"❌ 获取行情失败 {symbol}: {e}")
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    run_bot()
