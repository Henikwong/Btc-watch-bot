import os, time, ccxt, requests, pandas as pd, numpy as np
from datetime import datetime

# ========= 读取环境变量 =========
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

EXCHANGE = os.getenv("EXCHANGE", "huobi")
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
MARKET_TYPE = os.getenv("MARKET_TYPE", "spot")
SYMBOLS = os.getenv("SYMBOLS", "BTC/USDT").split(",")
BASE_USDT = float(os.getenv("BASE_USDT", 15))
RISK_ATR_MULT = float(os.getenv("RISK_ATR_MULT", 1.5))
LEVERAGE = int(os.getenv("LEVERAGE", 1))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 60))
LIVE_TRADE = int(os.getenv("LIVE_TRADE", 0))

# ========= Telegram 通知 =========
def send_tg(msg: str):
    if not TOKEN or not CHAT_ID:
        print("⚠️ 没有设置 Telegram 环境变量")
        return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        r = requests.get(url, params={"chat_id": CHAT_ID, "text": msg})
        print("Telegram:", r.json())
    except Exception as e:
        print("Telegram 发送失败:", e)

# ========= 初始化交易所 =========
def init_exchange():
    if EXCHANGE == "huobi":
        huobi = ccxt.huobi({
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "enableRateLimit": True,
        })
        if MARKET_TYPE == "swap":
            huobi = ccxt.huobifutures({
                "apiKey": API_KEY,
                "secret": API_SECRET,
                "enableRateLimit": True,
            })
        return huobi
    else:
        raise ValueError("暂时只支持 huobi/huobifutures")

exchange = init_exchange()

# ========= 简单策略示例（均线交叉） =========
def get_signal(symbol):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe="5m", limit=50)
        df = pd.DataFrame(ohlcv, columns=["t","o","h","l","c","v"])
        df["ma7"] = df["c"].rolling(7).mean()
        df["ma25"] = df["c"].rolling(25).mean()

        if df["ma7"].iloc[-2] < df["ma25"].iloc[-2] and df["ma7"].iloc[-1] > df["ma25"].iloc[-1]:
            return "buy"
        elif df["ma7"].iloc[-2] > df["ma25"].iloc[-2] and df["ma7"].iloc[-1] < df["ma25"].iloc[-1]:
            return "sell"
        else:
            return None
    except Exception as e:
        print(f"❌ 获取行情失败 {symbol}: {e}")
        return None

# ========= 下单逻辑 =========
def place_order(symbol, side, amount):
    try:
        if LIVE_TRADE == 1:
            order = exchange.create_market_order(symbol, side, amount)
            send_tg(f"✅ 实盘下单: {symbol} {side} {amount}\n{order}")
        else:
            send_tg(f"📊 模拟下单: {symbol} {side} {amount}")
    except Exception as e:
        send_tg(f"❌ 下单失败 {symbol}: {e}")

# ========= 主循环 =========
if __name__ == "__main__":
    mode = "实盘" if LIVE_TRADE == 1 else "纸面"
    send_tg(f"🤖 Bot启动 {EXCHANGE}/{MARKET_TYPE} 模式={mode}")

    while True:
        for symbol in SYMBOLS:
            sig = get_signal(symbol)
            if sig:
                # 固定 USDT 下单（市价）
                price = exchange.fetch_ticker(symbol)["last"]
                amount = BASE_USDT / price
                place_order(symbol, sig, amount)
        time.sleep(POLL_INTERVAL)
