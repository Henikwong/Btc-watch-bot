# bot_final_full.py
import os
import time
import requests
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta

# ====== 配置 ======
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
POLL_INTERVAL = 60  # 每分钟抓取一次，测试可用，实际可改大
ATR_MULT = 1.5
RSI_THRESHOLD = 5
WR_THRESHOLD = 5

main_coins = ["btcusdt","ethusdt","xrpusdt","bnbusdt","solusdt","dogeusdt","trxusdt","adausdt","ltcusdt","linkusdt"]
meme_coins = ["shibusdt","pepeusdt","penguusdt","bonkusdt","trumpusdt","spkusdt","flokusdt"]

period_map = {
    "60min": {"binance":"1h","okx":"1H","huobi":"60min"},
    "4hour": {"binance":"4h","okx":"4H","huobi":"4hour"},
    "1day": {"binance":"1d","okx":"1D","huobi":"1day"},
    "1week": {"binance":"1w","okx":"1W","huobi":"1week"}
}

# ====== 工具函数 ======
def log(msg: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")

def format_price(p):
    try:
        if p is None or np.isnan(p):
            return "-"
        p = float(p)
        if p>=100: return f"{p:.2f}"
        if p>=1: return f"{p:.4f}"
        if p>=0.01: return f"{p:.6f}"
        return f"{p:.8f}"
    except: return "-"

def send_telegram_message(text: str):
    if not TOKEN or not CHAT_ID:
        log("⚠️ Telegram 未配置")
        return
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=12)
    except Exception as e:
        log(f"❌ Telegram 发送失败: {e}")

# ====== K线抓取 ======
def get_kline_huobi(symbol: str, period="60min", size=200):
    try:
        r = requests.get("https://api.huobi.pro/market/history/kline",
                         params={"symbol": symbol,"period": period,"size": size}, timeout=12)
        j = r.json()
        if "data" not in j: return None
        df = pd.DataFrame(j["data"]).sort_values("id")
        for c in ["open","high","low","close","vol"]:
            df[c] = df[c].astype(float)
        return df
    except Exception as e:
        log(f"[Huobi ERROR] {symbol} {e}")
        return None

def get_kline_binance(symbol: str, interval="1h", limit=200):
    try:
        r = requests.get("https://api.binance.com/api/v3/klines",
                         params={"symbol": symbol.upper(),"interval": interval,"limit": limit}, timeout=12)
        j = r.json()
        if not isinstance(j, list): return None
        df = pd.DataFrame(j, columns=[
            "open_time","open","high","low","close","vol","close_time",
            "qav","num_trades","taker_base_vol","taker_quote_vol","ignore"])
        for c in ["open","high","low","close","vol"]:
            df[c] = df[c].astype(float)
        return df
    except Exception as e:
        log(f"[Binance ERROR] {symbol} {e}")
        return None

def get_kline_okx(symbol: str, bar="1H", limit=200):
    try:
        instId = symbol.upper().replace("USDT","-USDT-SWAP")
        r = requests.get("https://www.okx.com/api/v5/market/candles",
                         params={"instId":instId,"bar":bar,"limit":limit}, timeout=12)
        j = r.json()
        if not isinstance(j, dict) or j.get("code")!="0" or "data" not in j: return None
        df = pd.DataFrame([row[:6] for row in j["data"]], columns=["ts","open","high","low","close","vol"])
        for c in ["open","high","low","close","vol"]:
            df[c] = df[c].astype(float)
        df = df.iloc[::-1].reset_index(drop=True)
        df["id"] = pd.to_numeric(df["ts"], errors="coerce")
        return df
    except Exception as e:
        log(f"[OKX ERROR] {symbol} {e}")
        return None

# ====== 指标计算 ======
def calc_indicators(df: pd.DataFrame):
    if df is None or len(df)<35: return None
    try:
        work = df.copy().iloc[:-1]
        close = work["close"].astype(float)
        high = work["high"].astype(float)
        low = work["low"].astype(float)
        vol = work["vol"].astype(float)

        ema5 = close.ewm(span=5).mean().iloc[-1]
        ema10 = close.ewm(span=10).mean().iloc[-1]
        ema30 = close.ewm(span=30).mean().iloc[-1]
        ema_trend = "多" if ema5>ema10>ema30 else "空" if ema5<ema10<ema30 else "中性"

        macd_diff = ta.trend.MACD(close).macd_diff().iloc[-1]
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
        wr = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r().iloc[-1]

        stoch = ta.momentum.StochasticOscillator(high, low, close, window=9, smooth_window=3)
        k_val = stoch.stoch().iloc[-1]
        d_val = stoch.stoch_signal().iloc[-1]
        j_val = 3*k_val-2*d_val
        k_trend = "多" if k_val>d_val else "空" if k_val<d_val else "中性"
        vol_trend = (vol.iloc[-1]-vol.iloc[-2])/(abs(vol.iloc[-2])+1e-12)
        entry = float(close.iloc[-1])
        return {"ema_trend":ema_trend,"ema_vals":[ema5,ema10,ema30],"macd":macd_diff,"rsi":rsi,
                "wr":wr,"k":k_val,"d":d_val,"j":j_val,"k_trend":k_trend,"vol_trend":vol_trend,"entry":entry}
    except Exception as e:
        log(f"[IND ERROR] {e}")
        return None

# ====== ATR止损止盈 ======
def compute_stop_target_from_df(df: pd.DataFrame, side: str, entry: float):
    try:
        high, low, close = df["high"], df["low"], df["close"]
        tr = pd.concat([(high-low),(high-close.shift(1)).abs(),(low-close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        if side=="多":
            stop = entry-ATR_MULT*atr
            target = entry+ATR_MULT*atr
        else:
            stop = entry+ATR_MULT*atr
            target = entry-ATR_MULT*atr
        return stop, target
    except:
        return None, None

# ====== 一致性检测 ======
def indicators_agree(list_of_inds):
    inds = [x for x in list_of_inds if x is not None]
    if len(inds)<2: return False,"数据不足"
    try:
        if len(set([i["ema_trend"] for i in inds]))>1: return False,"EMA不一致"
        if len(set([1 if i["macd"]>0 else -1 for i in inds]))>1: return False,"MACD符号不同"
        if len(set([1 if i["k"]>i["d"] else -1 for i in inds]))>1: return False,"KDJ不同"
        if max([i["rsi"] for i in inds])-min([i["rsi"] for i in inds])>RSI_THRESHOLD: return False,"RSI差过大"
        if max([i["wr"] for i in inds])-min([i["wr"] for i in inds])>WR_THRESHOLD: return False,"WR差过大"
        if len(set([1 if i["vol_trend"]>0 else -1 for i in inds]))>1: return False,"VOL趋势不同"
        return True,"一致"
    except Exception as e:
        return False,f"异常:{e}"

def build_consistency_block(coin_upper, side, entry, target, stop, consistent_count):
    entry_s=format_price(entry)
    target_s=format_price(target)
    stop_s=format_price(stop)
    if consistent_count==3:
        return f"⚠️ {coin_upper} {'做多信号' if side=='多' else '做空信号'}\n入场:{entry_s}\n目标:{target_s}\n止损:{stop_s}\n\n⚡ 一致性: 3/3 周期"
    elif consistent_count==2:
        return f"⚠️ {coin_upper} {'做多信号' if side=='多' else '做空信号'}\n入场:{entry_s}\n目标:{target_s}\n止损:{stop_s}\n⚡ 一致性: 2/3 周期"
    else:
        return f"⚠️ {coin_upper} {'做多信号' if side=='多' else '做空信号'}\n入场:{entry_s}\n目标:{target_s}\n止损:{stop_s}\n一致性: 1/3 周期"

# ====== 主循环 ======
prev_high_signal={}
last_hour_msg=None
log(f"启动 Bot 多交易所多周期监控 POLL_INTERVAL={POLL_INTERVAL}s")

while True:
    try:
        coins = main_coins+meme_coins
        now = datetime.now()
        hourly_report_lines = ["📢 每小时普通信息（含 1h /4h /1d /1w 指标&一致性）"]
        strong_alerts=[]
        for coin in coins:
            coin_upper=coin.upper()
            per_period_results={}
            for period_label in ["60min","4hour","1day","1week"]:
                huobi_df=get_kline_huobi(coin,period=period_label)
                binance_df=get_kline_binance(coin,interval=period_map[period_label]["binance"])
                okx_df=get_kline_okx(coin,bar=period_map[period_label]["okx"])
                h_ind=calc_indicators(huobi_df)
                b_ind=calc_indicators(binance_df)
                o_ind=calc_indicators(okx_df)
                per_period_results[period_label]={"huobi_df":huobi_df,"binance_df":binance_df,"okx_df":okx_df,
                                                  "huobi":h_ind,"binance":b_ind,"okx":o_ind}
            consistent_counts=0
            per_period_consistent={}
            for p in ["60min","4hour","1day"]:
                inds=[per_period_results[p].get("huobi"),per_period_results[p].get("binance"),per_period_results[p].
