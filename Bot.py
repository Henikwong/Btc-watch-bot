
import os
import time
import requests
import pandas as pd
import ta
from datetime import datetime, timedelta

# ================== Telegram ==================
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ================== GPT 模拟分析 ==================
def gpt_analysis(symbol, df, signal):
    try:
        closes = df["close"].tail(50).tolist()
        recent = closes[-5:]
        avg = sum(closes) / len(closes)
        support = min(closes[-20:])
        resistance = max(closes[-20:])

        shape = ""
        if recent[-1] > recent[-2] > recent[-3]:
            shape = "近期连续上涨，可能形成小W底"
        elif recent[-1] < recent[-2] < recent[-3]:
            shape = "近期连续下跌，可能构成M头或弱势下跌"

        news_factor = "近期宏观市场消息或交易所动态可能带来不确定性"

        return (f"{symbol.upper()} 当前信号：{signal}\n"
                f"- K线形态：{shape}\n"
                f"- 支撑位：{support:.2f}, 阻力位：{resistance:.2f}\n"
                f"- 技术均价：{avg:.2f}\n"
                f"- 外部因子：{news_factor}\n"
                f"📌 建议结合多周期和成交量观察。")
    except Exception as e:
        return f"GPT 分析失败: {e}"

# ================== 币种 ==================
main_coins = ["btcusdt","ethusdt","xrpusdt","bnbusdt","solusdt","dogeusdt","trxusdt","adausdt","ltcusdt","linkusdt"]
meme_coins = ["dogeusdt","shibusdt","pepeusdt","penguusdt","bonkusdt","trumpusdt","spkusdt","flokusdt"]
main_periods = ["60min","4hour","1day"]

# ================== 工具函数 ==================
def get_kline_huobi(symbol, period="60min", size=120):
    url = "https://api.huobi.pro/market/history/kline"
    try:
        r = requests.get(url, params={"symbol": symbol, "period": period, "size": size})
        data = r.json()
        if "data" not in data:
            return None
        df = pd.DataFrame(data["data"]).sort_values("id")
        for col in ["open","high","low","close","vol"]:
            df[col] = df[col].astype(float)
        return df
    except:
        return None

def get_kline_binance(symbol, period="1h", limit=120):
    url = "https://api.binance.com/api/v3/klines"
    try:
        r = requests.get(url, params={"symbol": symbol.upper(), "interval": period, "limit": limit})
        data = r.json()
        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","vol","close_time",
            "qav","num_trades","taker_base_vol","taker_quote_vol","ignore"])
        for col in ["open","high","low","close","vol"]:
            df[col] = df[col].astype(float)
        return df
    except:
        return None

def get_kline_bybit(symbol, period="60", limit=120):
    url = "https://api.bybit.com/v2/public/kline/list"
    try:
        r = requests.get(url, params={"symbol": symbol.upper(), "interval": period, "limit": limit})
        j = r.json()
        if j.get("ret_code") != 0:
            return None
        df = pd.DataFrame(j["result"])
        for col in ["open","high","low","close","volume"]:
            if col in df:
                df[col] = df[col].astype(float)
        return df.rename(columns={"volume":"vol"})
    except:
        return None

def calc_signal(df):
    if len(df) > 0:  # 丢掉最后一根未收盘K
        df = df.iloc[:-1].copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]

    ema5 = close.ewm(span=5).mean()
    ema10 = close.ewm(span=10).mean()
    ema30 = close.ewm(span=30).mean()
    macd = ta.trend.MACD(close)
    macd_diff = macd.macd_diff()
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=9, smooth_window=3)
    k = stoch.stoch()
    d = stoch.stoch_signal()
    j = 3*k - 2*d

    long_signal = (ema5.iloc[-1] > ema10.iloc[-1] > ema30.iloc[-1]) and (macd_diff.iloc[-1] > 0) and (rsi.iloc[-1] < 70) and (j.iloc[-1] > d.iloc[-1])
    short_signal = (ema5.iloc[-1] < ema10.iloc[-1] < ema30.iloc[-1]) and (macd_diff.iloc[-1] < 0) and (rsi.iloc[-1] > 30) and (j.iloc[-1] < d.iloc[-1])

    entry = close.iloc[-1]

    if long_signal:
        return "做多", entry
    elif short_signal:
        return "做空", entry
    else:
        return None, entry

def calc_stop_loss(df, signal, entry, lookback=10):
    support = df["low"].tail(lookback).min()
    resistance = df["high"].tail(lookback).max()
    if signal and "多" in signal:
        return support
    elif signal and "空" in signal:
        return resistance
    return None

def send_telegram_message(message):
    if TOKEN and CHAT_ID:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        try:
            requests.post(url, data={"chat_id": CHAT_ID, "text": message})
            print("✅ 已发送信号到 Telegram")
        except Exception as e:
            print(f"消息发送失败: {e}")

# ================== 主循环 ==================
kline_cache = {}
last_send = datetime.utcnow() - timedelta(hours=1)
prev_signals = {}

while True:
    now = datetime.utcnow()
    try:
        coins = main_coins + meme_coins
        kline_cache = {c: {"60min":{}, "4hour":{}, "1day":{}} for c in coins}

        for coin in coins:
            # Huobi
            kline_cache[coin]["60min"]["huobi"] = get_kline_huobi(coin, "60min")
            kline_cache[coin]["4hour"]["huobi"] = get_kline_huobi(coin, "4hour")
            kline_cache[coin]["1day"]["huobi"]  = get_kline_huobi(coin, "1day")
            # Binance
            kline_cache[coin]["60min"]["binance"] = get_kline_binance(coin, "1h")
            kline_cache[coin]["4hour"]["binance"] = get_kline_binance(coin, "4h")
            kline_cache[coin]["1day"]["binance"]  = get_kline_binance(coin, "1d")
            # Bybit
            kline_cache[coin]["60min"]["bybit"] = get_kline_bybit(coin, "60")
            kline_cache[coin]["4hour"]["bybit"] = get_kline_bybit(coin, "240")
            kline_cache[coin]["1day"]["bybit"]  = get_kline_bybit(coin, "D")

        if (now - last_send).total_seconds() >= 3600:
            messages = []
            for coin in coins:
                period_signals, period_entries = {}, {}

                for period in main_periods:
                    signals, entries = [], []
                    dfs = kline_cache[coin].get(period, {})
                    for ex, df in dfs.items():
                        if df is not None and len(df) > 35:
                            sig, entry = calc_signal(df)
                            if sig:
                                signals.append(sig)
                                entries.append(entry)

                    if signals:
                        final_sig = max(set(signals), key=signals.count)
                        period_signals[period] = final_sig
                        period_entries[period] = sum(entries)/len(entries)

                if period_signals:
                    sig_values = list(period_signals.values())
                    unique_count = len(set(sig_values))
                    color = "🟢 绿色"
                    if unique_count == 1 and len(sig_values) == 3:
                        color = "🔴 红色"
                    elif len(sig_values) >= 2:
                        color = "🟡 黄色"

                    msg_lines = [f"📊 {coin.upper()} 信号 ({color})"]
                    for p in main_periods:
                        if p in period_signals:
                            entry = period_entries[p]
                            stop_loss = calc_stop_loss(dfs["huobi"], period_signals[p], entry)
                            target = entry*(1.01 if "多" in period_signals[p] else 0.99)
                            line = f"{p} → {period_signals[p]} | 入场:{entry:.2f} 目标:{target:.2f} 止损:{stop_loss:.2f}"
                            prev_sig = prev_signals.get(coin, {}).get(p)
                            if prev_sig and prev_sig != period_signals[p]:
                                line += " ⚡ 信号变化"
                                # 突发时也推 GPT 分析
                                analysis = gpt_analysis(coin, dfs["huobi"], period_signals[p])
                                send_telegram_message(f"🧠 突发 GPT 分析\n{analysis[:3000]}")
                            msg_lines.append(line)

                    if len(set(sig_values)) == 1 and len(sig_values) == 3:
                        msg_lines.append("🌟 强信号！三周期一致")

                    messages.append("\n".join(msg_lines))
                    prev_signals[coin] = period_signals

                    # 每轮都推 GPT 综合分析
                    try:
                        df_ref = dfs.get("huobi") or list(dfs.values())[0]
                        analysis = gpt_analysis(coin, df_ref, period_signals)
                        send_telegram_message(f"🧠 GPT 综合分析\n{analysis[:3000]}")
                    except Exception as e:
                        print(f"[GPT ERROR] {e}")

            if messages:
                send_telegram_message("\n\n".join(messages))
            last_send = now

    except Exception as e:
        print(f"循环错误: {e}")

    time.sleep(900)
