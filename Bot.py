# bot.py
# 高度动向多周期多交易所监控（Huobi + Binance + OKX）
# 要求：环境变量 TELEGRAM_BOT_TOKEN 与 TELEGRAM_CHAT_ID 已设置

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

POLL_INTERVAL = 60            # 测试用，每分钟抓一次，可改回 900（15分钟）
ATR_MULT = 1.5                # ATR 止盈/止损倍数
RSI_THRESHOLD = 5
WR_THRESHOLD = 5
VOL_REL_THRESHOLD = 0.20

# 币种
main_coins = ["btcusdt","ethusdt","xrpusdt","bnbusdt","solusdt","dogeusdt","trxusdt","adausdt","ltcusdt","linkusdt"]
meme_coins = ["shibusdt","pepeusdt","penguusdt","bonkusdt","trumpusdt","spkuspt","flokusdt"]

# 周期
main_periods = ["60min","4hour","1day"]
all_periods = ["60min","4hour","1day","1week"]

# ====== 工具函数 ======
def log(msg: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")

def format_price(p):
    try:
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return "-"
        p = float(p)
        if p >= 100:
            return f"{p:.2f}"
        if p >= 1:
            return f"{p:.4f}"
        if p >= 0.01:
            return f"{p:.6f}"
        return f"{p:.8f}"
    except:
        return "-"

def compute_atr(df: pd.DataFrame, period=14):
    try:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return float(atr.iloc[-1])
    except:
        return None

# ====== K线抓取函数 ======
period_map = {
    "60min": {"binance":"1h","okx":"1H","huobi":"60min"},
    "4hour": {"binance":"4h","okx":"4H","huobi":"4hour"},
    "1day": {"binance":"1d","okx":"1D","huobi":"1day"},
    "1week": {"binance":"1w","okx":"1W","huobi":"1week"}
}

def get_kline_huobi(symbol: str, period="60min", size=200):
    try:
        r = requests.get("https://api.huobi.pro/market/history/kline",
                         params={"symbol": symbol, "period": period, "size": size}, timeout=12)
        j = r.json()
        if not j or "data" not in j or (j.get("status") and j.get("status")!="ok"):
            return None
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
        if not isinstance(j, list):
            return None
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
        instId = symbol.upper().replace("USDT", "-USDT-SWAP")
        r = requests.get("https://www.okx.com/api/v5/market/candles",
                         params={"instId": instId, "bar": bar, "limit": limit}, timeout=12)
        j = r.json()
        if not isinstance(j, dict) or j.get("code") != "0" or "data" not in j:
            return None
        df = pd.DataFrame(j["data"])
        if isinstance(df.iloc[0,0], list) or isinstance(df.iloc[0,0], str):
            df = pd.DataFrame([row for row in j["data"]])
        df = df.iloc[:, :6]
        df.columns = ["ts","open","high","low","close","vol"]
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
    if df is None or len(df) < 35:
        return None
    try:
        work = df.copy().iloc[:-1]
        close = work["close"].astype(float)
        high = work["high"].astype(float)
        low = work["low"].astype(float)
        vol = work["vol"].astype(float)

        ema5 = close.ewm(span=5).mean().iloc[-1]
        ema10 = close.ewm(span=10).mean().iloc[-1]
        ema30 = close.ewm(span=30).mean().iloc[-1]

        macd_obj = ta.trend.MACD(close)
        macd_diff = macd_obj.macd_diff().iloc[-1] if hasattr(macd_obj, "macd_diff") else float('nan')

        rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
        wr = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r().iloc[-1]

        stoch = ta.momentum.StochasticOscillator(high, low, close, window=9, smooth_window=3)
        k_val = stoch.stoch().iloc[-1]
        d_val = stoch.stoch_signal().iloc[-1]
        j_val = 3*k_val - 2*d_val
        k_trend = "多" if k_val > d_val else "空" if k_val < d_val else "中性"

        ema_trend = "多" if (ema5 > ema10 and ema10 > ema30) else ("空" if (ema5 < ema10 and ema10 < ema30) else "中性")
        vol_trend = (vol.iloc[-1] - vol.iloc[-2]) / (abs(vol.iloc[-2]) + 1e-12)

        return {
            "ema_trend": ema_trend,
            "ema_vals": np.array([float(ema5), float(ema10), float(ema30)], dtype=float),
            "macd": float(macd_diff),
            "rsi": float(rsi),
            "wr": float(wr),
            "k": float(k_val),
            "d": float(d_val),
            "j": float(j_val),
            "k_trend": k_trend,
            "vol_trend": float(vol_trend),
            "entry": float(close.iloc[-1])
        }
    except Exception as e:
        log(f"[IND ERROR] {e}")
        return None

# ====== ATR止盈止损 ======
def compute_stop_target_from_df(df: pd.DataFrame, side: str, entry: float):
    atr = compute_atr(df)
    if atr is None:
        return None, None
    if side == "多":
        stop = entry - ATR_MULT * atr
        target = entry + ATR_MULT * atr
    else:
        stop = entry + ATR_MULT * atr
        target = entry - ATR_MULT * atr
    return stop, target

# ====== Telegram 发送 ======
def send_telegram_message(text: str):
    if not TOKEN or not CHAT_ID:
        log("⚠️ Telegram 未配置 TOKEN/CHAT_ID，跳过发送")
        return False
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=12)
        if r.status_code == 200:
            log("✅ Telegram 已发送")
            return True
        else:
            log(f"⚠️ Telegram 返回 {r.status_code}: {r.text}")
            return False
    except Exception as e:
        log(f"❌ Telegram 发送异常: {e}")
        return False

# ====== 一致性检测 ======
def indicators_agree(list_of_inds):
    try:
        inds = [x for x in list_of_inds if x is not None]
        if len(inds) < 2:
            return False, "数据不足"
        ema_trends = [ind["ema_trend"] for ind in inds]
        if len(set(ema_trends)) != 1:
            return False, f"EMA 不一致 {ema_trends}"
        macd_signs = [1 if ind["macd"] > 0 else -1 for ind in inds]
        if max(macd_signs) - min(macd_signs) != 0:
            return False, f"MACD 符号不同 {macd_signs}"
        kd_trends = [1 if ind["k"] > ind["d"] else -1 for ind in inds]
        if max(kd_trends) - min(kd_trends) != 0:
            return False, f"KDJ 不一致 {kd_trends}"
        rsi_vals = [ind["rsi"] for ind in inds]
        if max(rsi_vals) - min(rsi_vals) > RSI_THRESHOLD:
            return False, f"RSI 差异超出阈值 {max(rsi_vals)-min(rsi_vals):.2f}"
        wr_vals = [ind["wr"] for ind in inds]
        if max(wr_vals) - min(wr_vals) > WR_THRESHOLD:
            return False, f"WR 差异超出阈值 {max(wr_vals)-min(wr_vals):.2f}"
        vol_signs = [1 if ind["vol_trend"] > 0 else -1 for ind in inds]
        if max(vol_signs) - min(vol_signs) != 0:
            return False, f"VOL趋势不同 {vol_signs}"
        return True, "一致"
    except Exception as e:
        return False, f"异常: {e}"

# ====== 构建一致性 block ======
def build_consistency_block(coin_upper, side, entry, target, stop, consistent_count):
    entry_s = format_price(entry)
    target_s = format_price(target)
    stop_s = format_price(stop)
    if consistent_count == 3:
        return (f"⚠️ {coin_upper} { '做多信号' if side=='多' else '做空信号' }\n"
                f"入场: {entry_s}\n"
                f"目标: {target_s}\n"
                f"止损: {stop_s}\n\n"
                f"⚡ 一致性: 3/3 周期")
    elif consistent_count == 2:
        return (f"⚠️ {coin_upper} { '做多信号' if side=='多' else '做空信号' }\n"
                f"入场: {entry_s}\n"
                f"目标: {target_s}\n"
                f"止损: {stop_s}\n"
                f"⚡ 一致性: 2/3 周期")
    else:
        return (f"⚠️ {coin_upper} { '做多信号' if side=='多' else '做空信号' }\n"
                f"入场: {entry_s}\n"
                f"目标: {target_s}\n"
                f"止损: {stop_s}\n"
                f"一致性: 1/3 周期")

# ====== 每小时指标快照生成 ======
def build_indicator_report(coins, per_coin_results):
    lines = ["📢 每小时普通信息（含 1h / 4h / 24h / 1w 指标 & 一致性状态）"]
    for coin in coins:
        coin_upper = coin.upper()
        per_period_results = per_coin_results[coin]
        summary_parts = []
        for p in all_periods:
            ind_h = per_period_results[p]["huobi"]
            ind_b = per_period_results[p]["binance"]
            ind_o = per_period_results[p]["okx"]
            ind_display = ind_h or ind_b or ind_o
            if ind_display:
                summary_parts.append(f"{p} → EMA:{ind_display['ema_trend']} MACD:{ind_display['macd']:.3f} RSI:{ind_display['rsi']:.2f} WR:{ind_display['wr']:.2f} VOLΔ:{ind_display['vol_trend']:.3f}")
            else:
                summary_parts.append(f"{p} → 无数据")
        lines.append(f"{coin_upper} 监控快照:\n" + "\n".join(summary_parts))
    return "\n\n".join(lines)

# ====== 主循环 ======
prev_high_signal = {}
last_hour_msg = None

log("启动 Bot，多交易所多周期监控（POLL_INTERVAL = {}s）".format(POLL_INTERVAL))

while True:
