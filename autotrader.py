# app/autotrader.py
import os, time, ccxt, requests
import numpy as np
import talib
from datetime import datetime, timezone

# ===========================
# 工具函数
# ===========================
def now(): 
    return datetime.now(timezone.utc).isoformat()

def send_telegram(msg: str):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            requests.post(url, json={"chat_id": chat_id, "text": msg})
        except Exception as e:
            print("❌ Telegram 发送失败:", e)

# ===========================
# 初始化交易所
# ===========================
exchange = ccxt.binance({
    "apiKey": os.getenv("API_KEY"),
    "secret": os.getenv("API_SECRET"),
    "enableRateLimit": True,
    "options": {"defaultType": "future"}  # ✅ futures
})

# ===========================
# 环境参数
# ===========================
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT").split(",") if s.strip()]
BASE_USDT = float(os.getenv("BASE_USDT", "100"))
LEVERAGE = int(os.getenv("LEVERAGE", "10"))
LIVE_TRADE = os.getenv("LIVE_TRADE", "0") == "1"
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))

# 止盈止损 ATR 倍数
RISK_ATR_MULT = float(os.getenv("RISK_ATR_MULT", "1.5"))
TP_ATR_MULT   = float(os.getenv("TP_ATR_MULT", "3.0"))
SL_ATR_MULT   = float(os.getenv("SL_ATR_MULT", "2.0"))

# ===========================
# 技术指标信号
# ===========================
def check_signal(symbol):
    ohlcv = exchange.fetch_ohlcv(symbol, "1h", limit=150)
    closes = np.array([c[4] for c in ohlcv], dtype=float)
    highs  = np.array([c[2] for c in ohlcv], dtype=float)
    lows   = np.array([c[3] for c in ohlcv], dtype=float)

    ema20 = talib.EMA(closes, 20)
    ema50 = talib.EMA(closes, 50)
    macd, macdsignal, _ = talib.MACD(closes, 12, 26, 9)
    rsi = talib.RSI(closes, 14)
    atr = talib.ATR(highs, lows, closes, timeperiod=14)

    if ema20[-1] > ema50[-1] and macd[-1] > macdsignal[-1] and rsi[-1] > 50:
        return "buy", closes[-1], atr[-1]
    elif ema20[-1] < ema50[-1] and macd[-1] < macdsignal[-1] and rsi[-1] < 50:
        return "sell", closes[-1], atr[-1]
    else:
        return None, closes[-1], atr[-1]

# ===========================
# 下单函数
# ===========================
def place_order(symbol, side, price, atr):
    market = exchange.market(symbol)
    qty = BASE_USDT * LEVERAGE / price
    qty = float(exchange.amount_to_precision(symbol, qty))

    if not LIVE_TRADE:
        print(f"📌 模拟下单 {symbol} {side} {qty} @ {price}")
        send_telegram(f"📌 模拟下单 {symbol} {side} {qty} @ {price}")
        return

    try:
        # 开仓市价单
        order = exchange.create_market_order(symbol, side, qty)
        msg = f"✅ 入场 {symbol} {side} {qty} @ {price}"
        print(msg); send_telegram(msg)

        # 止盈止损
        if side == "buy":
            stop_loss = price - SL_ATR_MULT * atr
            take_profit = price + TP_ATR_MULT * atr
        else:
            stop_loss = price + SL_ATR_MULT * atr
            take_profit = price - TP_ATR_MULT * atr

        # Binance 期货 OCO 不直接支持 → 分别挂单
        exchange.create_order(symbol, "STOP_MARKET", "sell" if side=="buy" else "buy", qty, None, {"stopPrice": stop_loss})
        exchange.create_order(symbol, "TAKE_PROFIT_MARKET", "sell" if side=="buy" else "buy", qty, None, {"stopPrice": take_profit})

        send_telegram(f"🎯 止盈挂单: {take_profit}\n🛡 止损挂单: {stop_loss}")

    except Exception as e:
        print("❌ 下单失败:", e)
        send_telegram(f"❌ 下单失败 {symbol}: {e}")

# ===========================
# 主循环
# ===========================
def main():
    send_telegram("🚀 AutoTrader 启动...")
    while True:
        for symbol in SYMBOLS:
            try:
                signal, price, atr = check_signal(symbol)
                if signal:
                    place_order(symbol, signal, price, atr)
            except Exception as e:
                print(f"❌ {symbol} 出错:", e)
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
