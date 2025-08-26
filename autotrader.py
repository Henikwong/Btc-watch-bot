# ===========================
# AutoTrader 完整版（带分批止盈+防刷屏）
# ===========================
import os, time, ccxt, requests, numpy as np, pandas as pd
from datetime import datetime, timezone
import ta

# ===========================
# 配置
# ===========================
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS","BTC/USDT").split(",") if s.strip()]
BASE_USDT = float(os.getenv("BASE_USDT","100"))
LEVERAGE = int(os.getenv("LEVERAGE","10"))
LIVE_TRADE = os.getenv("LIVE_TRADE","0")=="1"
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL","60"))
ONLY_RESONANCE = os.getenv("ONLY_RESONANCE","1")=="1"
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT","2.0"))
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT","3.0"))
PARTIAL_TP_RATIO = float(os.getenv("PARTIAL_TP_RATIO","0.3"))

# ===========================
# 工具函数
# ===========================
def now_str(): return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def send_telegram(msg):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id: return
    try: requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id":chat_id,"text":msg})
    except: pass

# ===========================
# 交易所初始化
# ===========================
exchange = ccxt.binance({
    "apiKey": os.getenv("API_KEY"),
    "secret": os.getenv("API_SECRET"),
    "enableRateLimit": True,
    "options":{"defaultType":"future"}
})

# ===========================
# OHLCV与指标
# ===========================
def fetch_ohlcv_df(symbol, timeframe="1h", limit=100):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["time"],unit="ms")
    for c in ["open","high","low","close","volume"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    return df

def compute_indicators(df):
    df=df.copy()
    df["ema20"]=ta.trend.EMAIndicator(df["close"],20,fillna=True).ema_indicator()
    df["ema50"]=ta.trend.EMAIndicator(df["close"],50,fillna=True).ema_indicator()
    macd=ta.trend.MACD(df["close"],26,12,9,fillna=True)
    df["macd"]=macd.macd()
    df["macd_signal"]=macd.macd_signal()
    df["macd_hist"]=macd.macd_diff()
    df["rsi"]=ta.momentum.RSIIndicator(df["close"],14,fillna=True).rsi()
    df["atr"]=ta.volatility.AverageTrueRange(df["high"],df["low"],df["close"],14,fillna=True).average_true_range()
    df["vol_ma20"]=df["volume"].rolling(20,min_periods=1).mean()
    return df

def signal_from_indicators(df):
    last=df.iloc[-1]
    score=0; reasons=[]
    if last["ema20"]>last["ema50"]: score+=2; reasons.append("EMA多头")
    else: score-=2; reasons.append("EMA空头")
    if last["macd"]>last["macd_signal"]: score+=1; reasons.append("MACD多头")
    else: score-=1; reasons.append("MACD空头")
    if last["rsi"]>60: score+=1; reasons.append(f"RSI偏强 {last['rsi']:.1f}")
    elif last["rsi"]<40: score-=1; reasons.append(f"RSI偏弱 {last['rsi']:.1f}")
    if "vol_ma20" in df.columns and last["volume"]>df["vol_ma20"].iloc[-1]*1.5: score+=1; reasons.append("成交量放大")
    if score>=3: return "买入",score,reasons,last
    elif score<=-3: return "卖出",score,reasons,last
    else: return None,score,reasons,last

# ===========================
# 仓位管理
# ===========================
def fetch_all_positions():
    try: pos=exchange.fetch_positions(); return pos if isinstance(pos,list) else []
    except: return []

def parse_position_entry(pos):
    try:
        symbol = pos.get("symbol") or (pos.get("info") or {}).get("symbol")
        contracts = float(pos.get("contracts") or pos.get("positionAmt") or pos.get("amount") or 0)
        if contracts==0: return None
        side = "long" if contracts>0 else "short"
        entry = float(pos.get("entryPrice") or (pos.get("info") or {}).get("entryPrice") or 0)
        return symbol, abs(contracts), side, entry
    except: return None

def get_position(symbol):
    for p in fetch_all_positions():
        parsed=parse_position_entry(p)
        if not parsed: continue
        sym, qty, side, entry = parsed
        if sym.replace("/","")==symbol.replace("/",""): return {"symbol":symbol,"qty":qty,"side":side,"entry":entry,"raw":p}
    return None

# ===========================
# 下单（支持部分止盈）
# ===========================
def place_order(symbol, side_text, price, atr):
    side = "buy" if side_text=="买入" else "sell"
    qty_total = BASE_USDT*LEVERAGE/price
    try: qty_total=float(exchange.amount_to_precision(symbol,qty_total))
    except: qty_total=round(qty_total,6)
    if not LIVE_TRADE: return send_telegram(f"📌 模拟下单 {symbol} {side_text} 数量={qty_total} @ {price:.2f}")

    # 双向模式
    params={}
    try:
        res=exchange.fapiPrivate_get_positionmode()
        if res.get("dualSidePosition",True): params["positionSide"]="LONG" if side_text=="买入" else "SHORT"
    except: params["positionSide"]="LONG" if side_text=="买入" else "SHORT"

    # 开仓
    exchange.create_market_order(symbol, side, qty_total, params=params)

    if atr is None or np.isnan(atr): atr=price*0.005
    if side_text=="买入":
        stop_loss=price-SL_ATR_MULT*atr; take_profit=price+TP_ATR_MULT*atr
        close_side="sell"; close_pos_side="LONG"
    else:
        stop_loss=price+SL_ATR_MULT*atr; take_profit=price-TP_ATR_MULT*atr
        close_side="buy"; close_pos_side="SHORT"

    qty_first_tp = qty_total*PARTIAL_TP_RATIO
    qty_rest = qty_total - qty_first_tp

    try:
        # 第一批止盈
        exchange.create_order(symbol,"TAKE_PROFIT_MARKET",close_side,qty_first_tp,None,
                              {"stopPrice":take_profit,"positionSide":close_pos_side})
        # 剩余仓位止盈
        exchange.create_order(symbol,"TAKE_PROFIT_MARKET",close_side,qty_rest,None,
                              {"stopPrice":take_profit,"positionSide":close_pos_side})
        # 止损
        exchange.create_order(symbol,"STOP_MARKET",close_side,qty_total,None,
                              {"stopPrice":stop_loss,"positionSide":close_pos_side})
    except Exception as e: send_telegram(f"⚠️ 挂止盈/止损失败: {e}")
    send_telegram(f"✅ 已下单 {symbol} {side_text} 数量={qty_total} @ {price:.2f}\n🎯 TP={take_profit:.2f} 🛡 SL={stop_loss:.2f}")

# ===========================
# 平仓函数
# ===========================
def close_position(symbol, position):
    qty = position.get("qty")
    if not qty: return
    side="buy" if position.get("side","").startswith("short") else "sell"
    params={}
    try: info=exchange.fapiPrivate_get_positionmode(); is_hedge=info.get("dualSidePosition",False)
    except: is_hedge=False
    if is_hedge: params["positionSide"]="SHORT" if side=="buy" else "LONG"
    if LIVE_TRADE:
        try: qty_precise=float(exchange.amount_to_precision(symbol,qty))
        except: qty_precise=round(qty,6)
        exchange.create_market_order(symbol,side,qty_precise,params=params)
        send_telegram(f"✅ 已平仓 {symbol} {position.get('side')} 数量={qty_precise}")
    else: send_telegram(f"📌 模拟平仓 {symbol} {position.get('side')} 数量={qty}")

# ===========================
# 多周期共振信号
# ===========================
last_signal_cache = {}
def check_trend_once(symbol):
    alerts=[]; multi_tf_signal=None; reasons_all=[]
    for tf in ["1h","4h","1d"]:
        try:
            df=compute_indicators(fetch_ohlcv_df(symbol,tf,100))
            signal,score,reasons,last=signal_from_indicators(df)
            if signal:
                reasons_all.extend([f"{tf}:{r}" for r in reasons])
                if multi_tf_signal is None: multi_tf_signal=signal
                elif multi_tf_signal!=signal: multi_tf_signal=None
        except: continue
    if multi_tf_signal:
        # 防刷屏
        if last_signal_cache.get(symbol)!=multi_tf_signal:
            msg=f"{now_str()} {symbol} 多周期共振信号: {multi_tf_signal} 原因: {';'.join(reasons_all)}"
            alerts.append(msg)
            last_signal_cache[symbol]=multi_tf_signal
    return alerts, multi_tf_signal

# ===========================
# 主循环
# ===========================
def main_loop():
    while True:
        try:
            for symbol in SYMBOLS:
                alerts, signal = check_trend_once(symbol)
                for a in alerts: print(a); send_telegram(a)

                df=compute_indicators(fetch_ohlcv_df(symbol,"1h",100))
                last_close=df.iloc[-1]["close"]
                last_atr=df.iloc[-1]["atr"]
                pos=get_position(symbol)
                if signal:
                    if pos:
                        # 不同方向先平仓
                        if (signal=="买入" and pos["side"]=="short") or (signal=="卖出" and pos["side"]=="long"):
                            close_position(symbol,pos)
                            time.sleep(1)
                            place_order(symbol,signal,last_close,last_atr)
                    else:
                        place_order(symbol,signal,last_close,last_atr)
            time.sleep(POLL_INTERVAL)
        except Exception as e:
            print("⚠️ 主循环异常:",e)
            time.sleep(POLL_INTERVAL)

# ===========================
# 启动
# ===========================
if __name__=="__main__":
    print(f"🚀 AutoTrader 启动 {SYMBOLS}，LIVE_TRADE={LIVE_TRADE}")
    main_loop()
