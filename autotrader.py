# autotrader_run.py
import os, time, traceback
import ccxt, requests, pandas as pd, ta
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
EXCHANGE_NAME = os.getenv("EXCHANGE", "huobi").lower()
API_KEY   = os.getenv("API_KEY", "")
API_SECRET= os.getenv("API_SECRET", "")
MARKET_TYPE = os.getenv("MARKET_TYPE","spot").lower()
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS","BTC/USDT,ETH/USDT,LTC/USDT").split(",")]
BASE_USDT = float(os.getenv("BASE_USDT","15"))
ATR_MULT = float(os.getenv("RISK_ATR_MULT","1.5"))
LEVERAGE = int(os.getenv("LEVERAGE","5"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL","60"))
LIVE_TRADE = int(os.getenv("LIVE_TRADE","0"))
REQUIRED_CONFIRMS = 3
PERIODS = {"1h":"1h","4h":"4h","1d":"1d","1w":"1w"}

def nowstr(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def log(msg): print(f"[{nowstr()}] {msg}", flush=True)
def tg_send(text):
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                          data={"chat_id": TELEGRAM_CHAT_ID, "text": text})
        except: pass

def build_exchange():
    params = {"apiKey":API_KEY,"secret":API_SECRET,"enableRateLimit":True}
    if EXCHANGE_NAME=="huobi": ex=ccxt.huobi(params)
    elif EXCHANGE_NAME=="binance": ex=ccxt.binance(params)
    elif EXCHANGE_NAME=="okx": ex=ccxt.okx(params)
    else: raise ValueError("仅支持 huobi/binance/okx")
    if MARKET_TYPE=="swap":
        try: ex.options["defaultType"]="swap"
        except: pass
    return ex

def fetch_df(ex,symbol,tf,limit=200):
    ohlcv = ex.fetch_ohlcv(symbol,timeframe=tf,limit=limit)
    df = pd.DataFrame(ohlcv,columns=["ts","open","high","low","close","vol"])
    for c in ["open","high","low","close","vol"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    return df

def compute_atr(df,period=14):
    h,l,c = df["high"],df["low"],df["close"]
    tr = pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]

def indicators(df):
    close,high,low,vol = df["close"],df["high"],df["low"],df["vol"]
    ema5=close.ewm(span=5).mean().iloc[-1]; ema10=close.ewm(span=10).mean().iloc[-1]; ema30=close.ewm(span=30).mean().iloc[-1]
    ema_trend = "多" if ema5>ema10>ema30 else ("空" if ema5<ema10<ema30 else "中性")
    macd = ta.trend.MACD(close).macd_diff().iloc[-1]; rsi = ta.momentum.RSIIndicator(close,14).rsi().iloc[-1]
    wr = ta.momentum.WilliamsRIndicator(high,low,close,14).williams_r().iloc[-1]
    stoch = ta.momentum.StochasticOscillator(high,low,close,9,3)
    k,d = stoch.stoch().iloc[-1],stoch.stoch_signal().iloc[-1]; k_trend="多" if k>d else ("空" if k<d else "中性")
    vol_trend = (vol.iloc[-1]-vol.iloc[-2])/(vol.iloc[-2]+1e-12)
    # 简单评分
    score_bull=0; score_bear=0
    score_bull += ema_trend=="多"; score_bear += ema_trend=="空"
    score_bull += macd>0; score_bear += macd<0
    score_bull += rsi>50; score_bear += rsi<50
    score_bull += wr>-50; score_bear += wr<-50
    score_bull += k_trend=="多"; score_bear += k_trend=="空"
    score_bull += vol_trend>0; score_bear += vol_trend<0
    side = "多" if score_bull>=4 and score_bull>=score_bear+2 else ("空" if score_bear>=4 and score_bear>=score_bull+2 else None)
    return side, {"ema_trend":ema_trend,"macd":macd,"rsi":rsi,"wr":wr,"k_trend":k_trend,"vol_trend":vol_trend,"entry":close.iloc[-1]}

def calc_stop(entry,side,atr):
    if side=="多": return entry-ATR_MULT*atr, entry+ATR_MULT*atr
    else: return entry+ATR_MULT*atr, entry-ATR_MULT*atr

def place(ex,symbol,side,entry):
    qty=max(1e-8,BASE_USDT/max(entry,1e-8))
    oside="buy" if side=="多" else "sell"
    if LIVE_TRADE!=1:
        log(f"[纸面单] {symbol} {side} qty≈{qty} entry={entry}")
        return {"id":"paper","status":"simulated","side":oside,"amount":qty}
    try:
        if MARKET_TYPE=="swap": ex.set_leverage(LEVERAGE,symbol)
        o=ex.create_order(symbol,"market",oside,qty)
        log(f"[下单成功] {o}")
        return o
    except Exception as e:
        log(f"[下单失败] {e}")
        return None

def main():
    ex=build_exchange()
    mode="实盘" if LIVE_TRADE==1 else "纸面"
    log(f"启动Bot {EXCHANGE_NAME}/{MARKET_TYPE} 模式={mode}")
    tg_send(f"🤖 Bot启动 {EXCHANGE_NAME}/{MARKET_TYPE} 模式={mode}")

    while True:
        try:
            ex.load_markets()
            for symbol in SYMBOLS:
                sides=[]; details_map={}
                for tf in PERIODS:
                    try:
                        df = fetch_df(ex,symbol,PERIODS[tf])
                        side,det = indicators(df)
                        sides.append(side)
                        details_map[tf]=(side,det,df)
                        log(f"{symbol} {tf} {side} entry={det['entry'] if det else '-'}")
                    except Exception as e: log(f"{symbol} {tf} 获取失败: {e}")
                bull = sides.count("多"); bear=sides.count("空")
                final_side = "多" if bull>=REQUIRED_CONFIRMS and bull>bear else ("空" if bear>=REQUIRED_CONFIRMS and bear>bull else None)
                if final_side:
                    s1h,d1h,df1h = details_map["1h"]
                    if d1h:
                        entry = d1h["entry"]
                        atr = compute_atr(df1h)
                        stop,target = calc_stop(entry,final_side,atr)
                        place(ex,symbol,final_side,entry)
                        msg=f"{symbol} {final_side} entry={entry:.4f} stop={stop:.4f} target={target:.4f}"
                        tg_send(msg)
        except Exception as e:
            log(f"循环异常: {e}\n{traceback.format_exc()}")
        time.sleep(POLL_INTERVAL)

if __name__=="__main__":
    main()
