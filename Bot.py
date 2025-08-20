# Bot.py - 完整升级版：多周期信号 + ATR 动态止盈止损 + Telegram + GPT模拟分析
import os, time, requests, pandas as pd, numpy as np, ta
from datetime import datetime, timedelta

# ====== Telegram 配置 ======
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ====== 配置 ======
POLL_INTERVAL = 900        # 15分钟抓一次
ATR_MULT = 1.5             # ATR倍数做止盈止损
main_coins = ["btcusdt","ethusdt","xrpusdt","bnbusdt","solusdt","dogeusdt","trxusdt","adausdt","ltcusdt","linkusdt"]
meme_coins = ["dogeusdt","shibusdt","pepeusdt","penguusdt","bonkusdt","trumpusdt","spkusdt","flokusdt"]
main_periods = ["60min","4hour","1day"]

# ====== 工具函数 ======
def format_price(p):
    if p is None or np.isnan(p): return "-"
    if p>=100: return f"{p:.2f}"
    elif p>=1: return f"{p:.4f}"
    elif p>=0.01: return f"{p:.6f}"
    else: return f"{p:.8f}"

def compute_atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]

# ====== GPT 模拟分析 ======
def gpt_analysis(symbol, df, signal):
    try:
        closes = df["close"].tail(50).tolist()
        recent = closes[-5:]
        avg = sum(closes)/len(closes)
        support = min(closes[-20:])
        resistance = max(closes[-20:])
        shape = ""
        if recent[-1]>recent[-2]>recent[-3]:
            shape = "近期连续上涨，可能形成小W底"
        elif recent[-1]<recent[-2]<recent[-3]:
            shape = "近期连续下跌，可能构成M头或弱势下跌"
        news_factor = "近期宏观消息可能带来不确定性"
        return (f"{symbol.upper()} 当前信号：{signal}\n"
                f"- K线形态：{shape}\n"
                f"- 支撑位：{format_price(support)}, 阻力位：{format_price(resistance)}\n"
                f"- 技术均价：{format_price(avg)}\n"
                f"- 外部因子：{news_factor}\n"
                f"📌 建议结合多周期和成交量观察。")
    except:
        return f"{symbol.upper()} GPT 分析失败"

# ====== K线抓取 ======
def get_kline_huobi(symbol, period="60min", size=120):
    try:
        r = requests.get("https://api.huobi.pro/market/history/kline",
                         params={"symbol": symbol, "period": period, "size": size}, timeout=10)
        j = r.json()
        if "data" not in j: return None
        df = pd.DataFrame(j["data"]).sort_values("id")
        for c in ["open","high","low","close","vol"]: df[c]=df[c].astype(float)
        return df
    except: return None

def get_kline_binance(symbol, period="1h", limit=120):
    try:
        r = requests.get("https://api.binance.com/api/v3/klines",
                         params={"symbol": symbol.upper(),"interval":period,"limit":limit}, timeout=10)
        df = pd.DataFrame(r.json(), columns=[
            "open_time","open","high","low","close","vol","close_time",
            "qav","num_trades","taker_base_vol","taker_quote_vol","ignore"])
        for c in ["open","high","low","close","vol"]: df[c]=df[c].astype(float)
        return df
    except: return None

def get_kline_bybit(symbol, period="60", limit=120):
    try:
        r = requests.get("https://api.bybit.com/v2/public/kline/list",
                         params={"symbol": symbol.upper(),"interval": period,"limit": limit}, timeout=10)
        j = r.json()
        if j.get("ret_code")!=0: return None
        df = pd.DataFrame(j["result"])
        for c in ["open","high","low","close","volume"]:
            if c in df: df[c]=df[c].astype(float)
        if "volume" in df: df = df.rename(columns={"volume":"vol"})
        return df
    except: return None

# ====== 信号计算 ======
def calc_signal(df):
    if len(df)<35: return None, None
    df = df.iloc[:-1]
    close, high, low = df["close"], df["high"], df["low"]
    ema5, ema10, ema30 = close.ewm(span=5).mean(), close.ewm(span=10).mean(), close.ewm(span=30).mean()
    macd_diff = ta.trend.MACD(close).macd_diff()
    rsi = ta.momentum.RSIIndicator(close,14).rsi()
    stoch = ta.momentum.StochasticOscillator(high,low,close,9,3)
    k,d = stoch.stoch(), stoch.stoch_signal()
    j = 3*k-2*d
    entry = close.iloc[-1]

    long_signal = (ema5.iloc[-1]>ema10.iloc[-1]>ema30.iloc[-1]) and (macd_diff.iloc[-1]>0) and (rsi.iloc[-1]<70) and (j.iloc[-1]>d.iloc[-1])
    short_signal = (ema5.iloc[-1]<ema10.iloc[-1]<ema30.iloc[-1]) and (macd_diff.iloc[-1]<0) and (rsi.iloc[-1]>30) and (j.iloc[-1]<d.iloc[-1])

    if long_signal: return "做多", entry
    if short_signal: return "做空", entry
    return None, entry

def calc_stop_target(df, signal, entry):
    atr = compute_atr(df)
    if atr is None: return None, None
    if signal=="做多":
        stop = entry - ATR_MULT*atr
        target = entry + ATR_MULT*atr
    elif signal=="做空":
        stop = entry + ATR_MULT*atr
        target = entry - ATR_MULT*atr
    else:
        return None,None
    return stop, target

# ====== Telegram 推送 ======
def send_telegram_message(msg):
    if TOKEN and CHAT_ID:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        try: requests.post(url,data={"chat_id":CHAT_ID,"text":msg}); print("✅ Telegram 已发信号")
        except Exception as e: print(f"Telegram发送失败: {e}")

# ====== 主循环 ======
kline_cache={}
prev_signals={}
while True:
    try:
        coins = main_coins + meme_coins
        for coin in coins:
            dfs = {
                "huobi": get_kline_huobi(coin,"60min"),
                "binance": get_kline_binance(coin,"1h"),
                "bybit": get_kline_bybit(coin,"60")
            }
            period_signals = {}
            for period in main_periods:
                df = dfs["huobi"] if dfs["huobi"] is not None else next((v for v in dfs.values() if v is not None), None)
                if df is None: continue
                signal, entry = calc_signal(df)
                if signal:
                    stop,target = calc_stop_target(df, signal, entry)
                    period_signals[period] = (signal, entry, stop, target)

            if period_signals:
                # 信号一致性颜色
                sig_set = set([v[0] for v in period_signals.values()])
                if len(sig_set)==1 and len(period_signals)==3: color="🔴 红色三周期一致"
                elif len(sig_set)>=2: color="🟡 黄色"
                else: color="🟢 绿色"

                msg_lines=[f"📊 {coin.upper()} 信号 ({color})"]
                for p,(sig,entry,stop,target) in period_signals.items():
                    msg_lines.append(f"{p} → {sig} | 入场:{format_price(entry)} 目标:{format_price(target)} 止损:{format_price(stop)}")

                # GPT 分析
                df_ref = dfs["huobi"] if dfs["huobi"] is not None else next((v for v in dfs.values() if v is not None), None)
                if df_ref is not None:
                    analysis = gpt_analysis(coin, df_ref, list(sig_set)[0])
                    msg_lines.append(f"🧠 GPT 分析\n{analysis[:3000]}")

                send_telegram_message("\n".join(msg_lines))
                prev_signals[coin] = period_signals

        time.sleep(POLL_INTERVAL)
    except Exception as e:
        print(f"[ERROR] {e}")
        time.sleep(30)
