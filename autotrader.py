autotrader_usdt_futures.py

import os import time import ccxt import pandas as pd import ta import requests from dotenv import load_dotenv from datetime import datetime

load_dotenv()

===== ENV =====

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip() TG_CHAT  = os.getenv("TELEGRAM_CHAT_ID", "").strip() API_KEY   = os.getenv("API_KEY", "").strip() API_SECRET= os.getenv("API_SECRET", "").strip() BASE_USDT = float(os.getenv("BASE_USDT", "15")) LEVERAGE  = int(os.getenv("LEVERAGE", "10")) POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60")) LIVE_TRADE    = int(os.getenv("LIVE_TRADE", "0"))

TRADE_SYMBOLS = ["BTC/USDT", "ETH/USDT", "LTC/USDT", "BNB/USDT", "DOGE/USDT"] TIMEFRAME = "1h" SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "2.0")) TP_RATIO = float(os.getenv("TP_RATIO", "0.5"))  # 止盈50%

===== Helpers =====

def nowstr(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg): print(f"[{nowstr()}] {msg}", flush=True)

def tg_send(text): if TG_TOKEN and TG_CHAT: try: requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", data={"chat_id": TG_CHAT, "text": text}, timeout=10) except Exception as e: log(f"TG发送失败: {e}")

===== Exchange =====

def build_exchange(): ex = ccxt.binanceusdm({ "apiKey": API_KEY, "secret": API_SECRET, "enableRateLimit": True, "options": {"defaultType": "future"}, }) ex.load_markets() log("Binance USDT futures 初始化成功") return ex

def set_leverage_safe(ex, symbol, leverage): try: market = ex.market(symbol) ex.fapiPrivate_post_leverage({"symbol": market["id"], "leverage": int(leverage)}) log(f"{symbol} 杠杆已设置为 {leverage}x") except Exception as e: log(f"设置杠杆失败 {symbol}: {e}")

===== DataFrame =====

def df_from_ohlcv(ohlcv): df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","vol"]) for c in ["open","high","low","close","vol"]: df[c] = pd.to_numeric(df[c], errors="coerce") return df

===== Indicators & Signals =====

def analyze_df(df): close = df["close"] high = df["high"] low = df["low"]

ema5 = close.ewm(span=5).mean().iloc[-1]
ema10 = close.ewm(span=10).mean().iloc[-1]
ema_trend = "多" if ema5>ema10 else ("空" if ema5<ema10 else None)

macd = ta.trend.MACD(close)
macd_hist = float(macd.macd_diff().iloc[-1])

side = None
if ema_trend == "多" and macd_hist > 0:
    side = "多"
elif ema_trend == "空" and macd_hist < 0:
    side = "空"

atr = float(ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1])
entry = float(close.iloc[-1])

return side, entry, atr

===== Position Size =====

def amount_for_futures(ex, symbol, price): raw_qty = (BASE_USDT * LEVERAGE) / max(price, 1e-12) try: qty = ex.amount_to_precision(symbol, raw_qty) except Exception: qty = raw_qty return float(qty)

===== Main =====

positions = {} last_telegram_send = 0

def main(): global last_telegram_send ex = build_exchange()

# 设置杠杆
for sym in TRADE_SYMBOLS:
    set_leverage_safe(ex, sym, LEVERAGE)

tg_send(f"🤖 Bot 启动 USDT 合约 模式={'实盘' if LIVE_TRADE else '纸面'} 杠杆x{LEVERAGE}")

while True:
    try:
        for symbol in TRADE_SYMBOLS:
            ohlcv = ex.fetch_ohlcv(symbol, TIMEFRAME, limit=100)
            df = df_from_ohlcv(ohlcv)
            side, entry, atr = analyze_df(df)
            qty = amount_for_futures(ex, symbol, entry)

            pos = positions.get(symbol, {})

            # 开仓
            if side and (not pos or pos.get('side') != side):
                if LIVE_TRADE:
                    try:
                        if side == "多":
                            ex.create_order(symbol, "MARKET", "buy", qty)
                        else:
                            ex.create_order(symbol, "MARKET", "sell", qty)
                    except Exception as e:
                        log(f"下单失败 {symbol}: {e}")
                        continue
                positions[symbol] = {"side": side, "entry": entry, "atr": atr, "qty": qty, "partial_done": False}
                if time.time() - last_telegram_send > 3600:
                    tg_send(f"🟢 开仓信号 {symbol} side={side} qty={qty} price={entry:.2f}")
                    last_telegram_send = time.time()

            # 止损/止盈
            if pos:
                current_price = float(df["close"].iloc[-1])
                side = pos["side"]
                entry_price = pos["entry"]
                atr_val = pos["atr"]
                qty_val = pos["qty"]

                stop_loss = entry_price - SL_ATR_MULT*atr_val if side=="多" else entry_price + SL_ATR_MULT*atr_val
                take_profit = entry_price*(1+TP_RATIO) if side=="多" else entry_price*(1-TP_RATIO)

                exit_flag = False
                if side=="多" and (current_price <= stop_loss or current_price >= take_profit):
                    exit_flag = True
                elif side=="空" and (current_price >= stop_loss or current_price <= take_profit):
                    exit_flag = True

                if exit_flag:
                    if LIVE_TRADE:
                        try:
                            ex.create_order(symbol, "MARKET", "sell" if side=="多" else "buy", qty_val)
                        except Exception as e:
                            log(f"平仓失败 {symbol}: {e}")
                            continue
                    positions[symbol] = {}
                    if time.time() - last_telegram_send > 3600:
                        tg_send(f"🔴 平仓信号 {symbol} side={side} qty={qty_val} price={current_price:.2f}")
                        last_telegram_send = time.time()

        time.sleep(POLL_INTERVAL)

    except Exception as e:
        log(f"主循环错误: {e}")
        time.sleep(POLL_INTERVAL)

if name == "main": main()

