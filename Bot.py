import os
import time
import requests
import pandas as pd
import ta
from datetime import datetime, timedelta

# ================== Telegram ==================
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

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
prev_signals = {}  # 保存上一次信号，用于变化提醒

while True:
    now = datetime.utcnow()
    try:
        for coin in main_coins + meme_coins:
            kline_cache[coin] = {}
            # 抓三家交易所
            kline_cache[coin]["huobi"] = get_kline_huobi(coin, "60min")
            kline_cache[coin]["binance"] = get_kline_binance(coin, "1h")
            kline_cache[coin]["bybit"] = get_kline_bybit(coin, "60")

        if (now - last_send).total_seconds() >= 3600:
            messages = []
            for coin, dfs in kline_cache.items():
                period_signals = {}
                period_entries = {}
                # 获取1h, 4h, 1d信号
                for period in main_periods:
                    signals = []
                    entries = []
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
                    # 判断颜色
                    sig_values = list(period_signals.values())
                    unique_count = len(set(sig_values))
                    color = "🟢 绿色"  # 默认1个周期
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
                            # 信号变化提醒
                            prev_sig = prev_signals.get(coin, {}).get(p)
                            if prev_sig and prev_sig != period_signals[p]:
                                line += " ⚡ 信号变化"
                            msg_lines.append(line)
                            # 接近止盈止损提醒 ±0.5%
                            last_close = dfs["huobi"]["close"].iloc[-1]
                            if abs(last_close - target)/target <= 0.005:
                                msg_lines.append(f"⚠️ {p} 接近目标价格")
                            if abs(last_close - stop_loss)/stop_loss <= 0.005:
                                msg_lines.append(f"⚠️ {p} 接近止损价格")

                    # 三交易所一致特发信息
                    if len(set(sig_values)) == 1 and len(sig_values) == 3:
                        msg_lines.append("🌟 强信号！三交易所一致")

                    messages.append("\n".join(msg_lines))
                    prev_signals[coin] = period_signals  # 保存本次信号

            if messages:
                send_telegram_message("\n\n".join(messages))
            last_send = now

    except Exception as e:
        print(f"循环错误: {e}")

    time.sleep(900)
