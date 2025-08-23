import os, time, requests, ccxt, pandas as pd, numpy as np

# ========== Telegram ==========
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_tg(msg: str):
    if not TOKEN or not CHAT_ID:
        print("⚠️ 没有设置 Telegram 环境变量")
        return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        r = requests.get(url, params={"chat_id": CHAT_ID, "text": msg})
        print("Telegram:", r.json())
    except Exception as e:
        print("❌ Telegram 发送失败:", e)

# ========== 环境变量 ==========
EXCHANGE_NAME = os.getenv("EXCHANGE", "huobi")
MARKET_TYPE   = os.getenv("MARKET_TYPE", "spot")   # spot 或 swap
API_KEY       = os.getenv("API_KEY")
API_SECRET    = os.getenv("API_SECRET")

SYMBOLS       = os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT").split(",")
BASE_USDT     = float(os.getenv("BASE_USDT", 15))
RISK_ATR_MULT = float(os.getenv("RISK_ATR_MULT", 1.5))
LEVERAGE      = int(os.getenv("LEVERAGE", 1))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 60))
LIVE_TRADE    = int(os.getenv("LIVE_TRADE", 0))   # 0=模拟 1=实盘

# ========== 初始化交易所 ==========
if MARKET_TYPE == "spot":
    exchange = ccxt.huobipro({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True
    })
elif MARKET_TYPE == "swap":
    exchange = ccxt.huobiswap({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True
    })
else:
    raise ValueError(f"未知 MARKET_TYPE: {MARKET_TYPE}")

send_tg(f"🤖 Bot启动 {EXCHANGE_NAME}/{MARKET_TYPE} 模式={'实盘' if LIVE_TRADE else '纸面'}")

# ========== 简单行情函数 ==========
def get_ohlcv(symbol, timeframe="1m", limit=100):
    try:
        return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        print(f"❌ 获取行情失败 {symbol}: {e}")
        return None

# ========== 示例策略 ==========
def simple_strategy(symbol):
    ohlcv = get_ohlcv(symbol)
    if not ohlcv:
        return None
    df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
    df["ma_fast"] = df["close"].rolling(5).mean()
    df["ma_slow"] = df["close"].rolling(20).mean()
    if df["ma_fast"].iloc[-1] > df["ma_slow"].iloc[-1]:
        return "buy"
    elif df["ma_fast"].iloc[-1] < df["ma_slow"].iloc[-1]:
        return "sell"
    return None

# ========== 下单函数 ==========
def place_order(symbol, side, amount):
    try:
        if LIVE_TRADE:
            order = exchange.create_market_order(symbol, side, amount)
            send_tg(f"✅ 实盘下单: {symbol} {side} {amount}\n{order}")
        else:
            send_tg(f"📊 模拟下单: {symbol} {side} {amount}")
    except Exception as e:
        send_tg(f"❌ 下单失败 {symbol}: {e}")

# ========== 主循环 ==========
while True:
    for sym in SYMBOLS:
        signal = simple_strategy(sym)
        if signal:
            balance = exchange.fetch_balance()
            usdt = balance["total"].get("USDT", BASE_USDT)
            amount = BASE_USDT / exchange.fetch_ticker(sym)["last"]
            place_order(sym, signal, amount)
    time.sleep(POLL_INTERVAL)
