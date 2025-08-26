import os, time, ccxt, requests, numpy as np, pandas as pd
from datetime import datetime, timezone
import ta

# ====== 配置 ======
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS","BTC/USDT").split(",") if s.strip()]
BASE_USDT = float(os.getenv("BASE_USDT","100"))
LEVERAGE = int(os.getenv("LEVERAGE","10"))
LIVE_TRADE = os.getenv("LIVE_TRADE","0")=="1"
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL","60"))
ONLY_RESONANCE = os.getenv("ONLY_RESONANCE","1")=="1"
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT","2.0"))
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT","3.0"))
OHLCV_LIMIT = 200

TELEGRAM_BOT_TOKEN=os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID=os.getenv("TELEGRAM_CHAT_ID")

# ====== 工具 ======
def now_str(): return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
def send_telegram(msg):
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try: requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                           json={"chat_id":TELEGRAM_CHAT_ID,"text":msg})
        except: print("❌ Telegram 推送失败", msg)
    else: print(msg)

# ====== 交易所 ======
exchange = ccxt.binance({"apiKey":os.getenv("API_KEY"),"secret":os.getenv("API_SECRET"),"enableRateLimit":True,"options":{"defaultType":"future"}})

# ====== 数据 & 指标 ======
def fetch_ohlcv_df(symbol, timeframe="1h", limit=OHLCV_LIMIT):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
    df["time"]=pd.to_datetime(df["time"],unit="ms")
    for c in ["open","high","low","close","volume"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    df["ema20"]=ta.trend.EMAIndicator(df["close"],20).ema_indicator()
    df["ema50"]=ta.trend.EMAIndicator(df["close"],50).ema_indicator()
    macd=ta.trend.MACD(df["close"],26,12,9)
    df["macd"]=macd.macd(); df["macd_signal"]=macd.macd_signal()
    df["rsi"]=ta.momentum.RSIIndicator(df["close"],14).rsi()
    df["atr"]=ta.volatility.AverageTrueRange(df["high"],df["low"],df["close"],14).average_true_range()
    df["vol_ma20"]=df["volume"].rolling(20,min_periods=1).mean()
    return df

def signal_from_indicators(df):
    last=df.iloc[-1]; score=0; reasons=[]
    if last["ema20"]>last["ema50"]: score+=2; reasons.append("EMA 多头")
    else: score-=2; reasons.append("EMA 空头")
    if last["macd"]>last["macd_signal"]: score+=1; reasons.append("MACD 多头")
    else: score-=1; reasons.append("MACD 空头")
    if last["rsi"]>60: score+=1; reasons.append(f"RSI 强 {last['rsi']:.1f}")
    elif last["rsi"]<40: score-=1; reasons.append(f"RSI 弱 {last['rsi']:.1f}")
    if last["volume"]>last["vol_ma20"]*1.5: score+=1; reasons.append("量放大")
    if score>=3: return "买入", reasons, last
    elif score<=-3: return "卖出", reasons, last
    return None, reasons, last

# ====== 仓位 ======
def get_position(symbol):
    try:
        pos = exchange.fetch_positions()
        for p in pos:
            if p['symbol'].replace("/","")==symbol.replace("/","") and float(p.get("positionAmt",0))!=0:
                amt=float(p["positionAmt"]); side="long" if amt>0 else "short"
                return {"symbol":symbol,"qty":abs(amt),"side":side,"entry":float(p.get("entryPrice",0))}
    except: pass
    return None

def close_position(position):
    symbol=position["symbol"]; qty=position["qty"]; side=position["side"]
    order_side="buy" if side=="short" else "sell"
    try:
        if LIVE_TRADE:
            qty_precise=float(exchange.amount_to_precision(symbol, qty))
            exchange.create_market_order(symbol, order_side, qty_precise)
        send_telegram(f"✅ 已平仓 {symbol} {side} 数量={qty}")
    except Exception as e:
        send_telegram(f"❌ 平仓失败 {symbol} {e}")

# ====== 下单 ======
def place_order(symbol, side_text, price, atr):
    side="buy" if side_text=="买入" else "sell"
    qty=BASE_USDT*LEVERAGE/price
    try: qty=float(exchange.amount_to_precision(symbol,qty))
    except: qty=round(qty,6)
    if not LIVE_TRADE:
        send_telegram(f"📌 模拟下单 {symbol} {side_text} 数量={qty} @ {price:.2f}"); return
    try:
        exchange.create_market_order(symbol, side, qty)
        stop_loss=price-SL_ATR_MULT*atr if side_text=="买入" else price+SL_ATR_MULT*atr
        take_profit=price+TP_ATR_MULT*atr if side_text=="买入" else price-TP_ATR_MULT*atr
        send_telegram(f"✅ 下单 {symbol} {side_text} 数量={qty} @ {price:.2f}\n🎯 TP:{take_profit:.2f} 🛡 SL:{stop_loss:.2f}")
    except Exception as e: send_telegram(f"❌ 下单失败 {symbol} {e}")

# ====== 多周期共振 ======
def check_resonance(symbol):
    multi_signal=None; reasons_all=[]
    last_close=None; last_atr=None
    for tf in ["1h","4h","1d"]:
        try:
            df=fetch_ohlcv_df(symbol,tf)
            sig,reasons,last=signal_from_indicators(df)
            last_close=last["close"]; last_atr=last["atr"]
            if sig:
                reasons_all.extend([f"{tf}:{r}" for r in reasons])
                if multi_signal is None: multi_signal=sig
                elif multi_signal!=sig: multi_signal=None
        except: continue
    return multi_signal, reasons_all, last_close, last_atr

# ====== 主循环 ======
def main_loop():
    last_report={}  # 每小时记录推送时间
    while True:
        try:
            for symbol in SYMBOLS:
                if ONLY_RESONANCE:
                    signal,reasons,last_close,last_atr=check_resonance(symbol)
                else:
                    df=fetch_ohlcv_df(symbol,"1h")
                    signal,reasons,last=signal_from_indicators(df)
                    last_close=last["close"]; last_atr=last["atr"]

                pos=get_position(symbol)
                # 如果已有仓位方向不同，先平仓再开新仓
                if signal and pos and ((signal=="买入" and pos["side"]=="short") or (signal=="卖出" and pos["side"]=="long")):
                    close_position(pos)
                    time.sleep(1)
                    place_order(symbol, signal, last_close, last_atr)
                elif signal and not pos:
                    place_order(symbol, signal, last_close, last_atr)

                # 每小时推送一次信号
                now_hour=datetime.now().hour
                if last_report.get(symbol)!=now_hour:
                    msg=f"{now_str()} {symbol} 信号:{signal or '无'} 原因:{';'.join(reasons)} 价格:{last_close:.2f}"
                    send_telegram(msg)
                    last_report[symbol]=now_hour
            time.sleep(POLL_INTERVAL)
        except Exception as e: print("⚠️ 主循环异常:", e); time.sleep(POLL_INTERVAL)

if __name__=="__main__":
    print(f"🚀 AutoTrader 启动 {SYMBOLS}，LIVE_TRADE={LIVE_TRADE}")
    main_loop()
