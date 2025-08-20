# bot.py
# 高度动向多周期多交易所监控（Huobi + Binance + Bybit v5）
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

POLL_INTERVAL = 60            # 测试用：每分钟抓取一次。正式可改为 900 (15min)
ATR_MULT = 1.5                # ATR 止盈/止损倍数
RSI_THRESHOLD = 5             # RSI 最大允许差（跨交易所）
WR_THRESHOLD = 5              # WR 最大允许差
VOL_REL_THRESHOLD = 0.20      # 成交量增减比例允许差（跨交易所）
main_coins = ["btcusdt","ethusdt","xrpusdt","bnbusdt","solusdt","dogeusdt","trxusdt","adausdt","ltcusdt","linkusdt"]
meme_coins = ["shibusdt","pepeusdt","penguusdt","bonkusdt","trumpusdt","spkusdt","flokusdt"]
main_periods = ["60min","4hour","1day"]  # 保持和 Huobi period 命名一致

# ====== 工具函数 ======
def log(msg: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")

def format_price(p):
    try:
        if p is None or (isinstance(p, float) and np.isnan(p)): return "-"
        p = float(p)
        if p >= 100: return f"{p:.2f}"
        if p >= 1: return f"{p:.4f}"
        if p >= 0.01: return f"{p:.6f}"
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
    except Exception:
        return None

# ====== K线抓取函数（兼容名字/大小写） ======
def get_kline_huobi(symbol: str, period="60min", size=200):
    try:
        r = requests.get("https://api.huobi.pro/market/history/kline",
                         params={"symbol": symbol, "period": period, "size": size}, timeout=10)
        j = r.json()
        if not j or "data" not in j:
            return None
        df = pd.DataFrame(j["data"]).sort_values("id")
        for c in ["open","high","low","close","vol"]:
            if c in df.columns:
                df[c] = df[c].astype(float)
        return df
    except Exception as e:
        log(f"[Huobi ERROR] {symbol} {e}")
        return None

def get_kline_binance(symbol: str, period="1h", limit=200):
    try:
        r = requests.get("https://api.binance.com/api/v3/klines",
                         params={"symbol": symbol.upper(),"interval": period, "limit": limit}, timeout=10)
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

def get_kline_bybit(symbol: str, period="60", limit=200):
    """
    使用 Bybit v5 接口，symbol 必须大写 (ETHUSDT)
    period 参数：1,3,5,15,30,60,120,240,360,720,D,W,M
    """
    try:
        sym = symbol.upper()
        url = "https://api.bybit.com/v5/market/kline"
        r = requests.get(url, params={"symbol": sym, "interval": period, "limit": limit}, timeout=10)
        j = r.json()
        if not isinstance(j, dict) or j.get("retCode") != 0:
            log(f"[Bybit FAIL] {sym} retCode={j.get('retCode') if isinstance(j,dict) else 'nojson'}")
            return None
        data = j["result"]["list"]
        df = pd.DataFrame(data)
        # v5 fields: t (ts), o, h, l, c, v
        df = df.rename(columns={"t":"id","o":"open","h":"high","l":"low","c":"close","v":"vol"})
        for c in ["open","high","low","close","vol"]:
            df[c] = df[c].astype(float)
        df = df.sort_values("id")
        return df
    except Exception as e:
        log(f"[Bybit ERROR] {symbol} {e}")
        return None

# ====== 指标计算（返回标准化指标） ======
def calc_indicators(df: pd.DataFrame):
    """
    要求 df 包含 open,high,low,close,vol
    返回 dict:
      { 'ema_trend': '多'/'空'/'中性',
        'macd': float,
        'rsi': float,
        'wr': float,
        'k_trend': '多'/'空'/'中性',
        'vol_trend': float (近两根成交量比值-1),
        'entry': latest_close,
        'ema_vals': [ema5,ema10,ema30] }
    """
    if df is None or len(df) < 35:
        return None
    try:
        work = df.copy().iloc[:-1]  # 丢掉未收盘的最新
        close = work["close"].astype(float)
        high = work["high"].astype(float)
        low = work["low"].astype(float)
        vol = work["vol"].astype(float)

        ema5 = close.ewm(span=5).mean().iloc[-1]
        ema10 = close.ewm(span=10).mean().iloc[-1]
        ema30 = close.ewm(span=30).mean().iloc[-1]

        macd_diff = ta.trend.MACD(close).macd_diff().iloc[-1]
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
        wr = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r().iloc[-1]
        stoch = ta.momentum.StochasticOscillator(high, low, close, window=9, smooth_window=3)
        k_val = stoch.stoch().iloc[-1]
        d_val = stoch.stoch_signal().iloc[-1]
        k_trend = "多" if k_val > d_val else "空" if k_val < d_val else "中性"

        ema_trend = "多" if (ema5 > ema10 and ema10 > ema30) else ("空" if (ema5 < ema10 and ema10 < ema30) else "中性")

        # vol_trend: percentage change from previous candle
        vol_trend = (vol.iloc[-1] - vol.iloc[-2]) / (vol.iloc[-2] + 1e-12)

        return {
            "ema_trend": ema_trend,
            "ema_vals": np.array([float(ema5), float(ema10), float(ema30)], dtype=float),
            "macd": float(macd_diff),
            "rsi": float(rsi),
            "wr": float(wr),
            "k_trend": k_trend,
            "vol_trend": float(vol_trend),
            "entry": float(close.iloc[-1])
        }
    except Exception as e:
        log(f"[IND ERROR] {e}")
        return None

# ====== 停损止盈 (ATR) ======
def compute_stop_target(df: pd.DataFrame, side: str, entry: float):
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
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=10)
        if r.status_code == 200:
            log("✅ Telegram 已发送")
            return True
        else:
            log(f"⚠️ Telegram 返回 {r.status_code}: {r.text}")
            return False
    except Exception as e:
        log(f"❌ Telegram 发送异常: {e}")
        return False

# ====== 主循环 ======
prev_high_signal = {}
last_hour_msg = None

# mapping period strings for different exchanges:
# Huobi uses "60min","4hour","1day" in this code (we already used those)
# Binance intervals: "1h","4h","1d"
# Bybit v5: "60","240","D"
period_map = {
    "60min": {"binance":"1h","bybit":"60"},
    "4hour": {"binance":"4h","bybit":"240"},
    "1day": {"binance":"1d","bybit":"D"}
}

log("启动 Bot，多交易所多周期监控（测试模式 - POLL_INTERVAL = {}s）".format(POLL_INTERVAL))

while True:
    try:
        coins = main_coins + meme_coins
        now = datetime.now()

        # 每个币循环
        for coin in coins:
            coin_upper = coin.upper()
            period_results = {}  # store per period indicators across exchanges

            # 对每个周期抓取 3 家交易所的数据
            for period in main_periods:
                # Huobi period matches our label
                huobi_df = get_kline_huobi(coin, period=period)
                binance_interval = period_map[period]["binance"]
                bybit_interval = period_map[period]["bybit"]

                binance_df = get_kline_binance(coin, period=binance_interval)
                bybit_df = get_kline_bybit(coin_upper, period=bybit_interval)

                # log fetch status
                log(f"{coin_upper} {period} 抓取状态: Huobi={'OK' if huobi_df is not None else 'FAIL'}, "
                    f"Binance={'OK' if binance_df is not None else 'FAIL'}, Bybit={'OK' if bybit_df is not None else 'FAIL'}")

                # ensure we have 3 valid dfs; if any missing, skip this period
                if huobi_df is None or binance_df is None or bybit_df is None:
                    continue

                # compute indicators for each exchange
                h_ind = calc_indicators(huobi_df)
                b_ind = calc_indicators(binance_df)
                by_ind = calc_indicators(bybit_df)

                # if any indicator calc failed skip
                if h_ind is None or b_ind is None or by_ind is None:
                    continue

                period_results[period] = {
                    "huobi": h_ind,
                    "binance": b_ind,
                    "bybit": by_ind,
                }

                # 日志显示每交易所指标（只显示关键数值，避免太长）
                def short_ind_text(name, ind):
                    return (f"{name}: EMA_trend={ind['ema_trend']} EMA_vals={[round(x,4) for x in ind['ema_vals']]} "
                            f"MACD={ind['macd']:.4f} RSI={ind['rsi']:.2f} WR={ind['wr']:.2f} VOLΔ={ind['vol_trend']:.3f}")
                log(f"{coin_upper} {period} 指标 => {short_ind_text('Huobi',h_ind)} | {short_ind_text('Binance',b_ind)} | {short_ind_text('Bybit',by_ind)}")

            # 如果没有任何周期可用，跳过该币
            if not period_results:
                # 在 hourly message we will show "无一致信号 (监控中)" later
                continue

            # 判断每个周期是否满足 三家交易所一致性与指标一致
            period_consistent = {}
            for period, exch_inds in period_results.items():
                # collect trends/values
                ema_trends = [exch_inds[e]['ema_trend'] for e in exch_inds]
                k_trends = [("多" if exch_inds[e]['ema_vals'][0] > exch_inds[e]['ema_vals'][1] else "空") for e in exch_inds]  # approximate
                macd_signs = [1 if exch_inds[e]['macd'] > 0 else -1 for e in exch_inds]
                rsi_vals = [exch_inds[e]['rsi'] for e in exch_inds]
                wr_vals = [exch_inds[e]['wr'] for e in exch_inds]
                vol_vals = [exch_inds[e]['vol_trend'] for e in exch_inds]

                # basic consistency checks:
                ema_consistent = (len(set(ema_trends)) == 1)  # all same '多'/'空'/'中性'
                macd_consistent = (max(macd_signs) - min(macd_signs) == 0)
                rsi_consistent = (max(rsi_vals) - min(rsi_vals) <= RSI_THRESHOLD)
                wr_consistent = (max(wr_vals) - min(wr_vals) <= WR_THRESHOLD)
                # vol: all vol_trend within relative threshold
                vol_consistent = (max(vol_vals) - min(vol_vals) <= VOL_REL_THRESHOLD)

                is_consistent = ema_consistent and macd_consistent and rsi_consistent and wr_consistent and vol_consistent

                period_consistent[period] = {
                    "consistent": is_consistent,
                    "ema_trend": ema_trends[0] if ema_consistent else "混合",
                    "exch_inds": exch_inds
                }

                log(f"{coin_upper} {period} 一致性: EMA_consistent={ema_consistent} MACD_consistent={macd_consistent} RSI_consistent={rsi_consistent} WR_consistent={wr_consistent} VOL_consistent={vol_consistent} => FINAL={is_consistent}")

            # 最后：要求所有 main_periods 都存在并且一致，才构成高度动向信号
            good_periods = [p for p,v in period_consistent.items() if v["consistent"]]
            if len(good_periods) == len(main_periods):
                # all periods consistent
                # final side: take ema_trend of 60min period (they're same across periods by logic)
                final_side = period_consistent["60min"]["ema_trend"]
                # entry from huobi 60min
                entry = period_consistent["60min"]["exch_inds"]["huobi"]["entry"]
                stop, target = compute_stop_target(period_results["60min"]["huobi"], final_side, entry)
                prev = prev_high_signal.get(coin)
                if prev != final_side:
                    prev_high_signal[coin] = final_side
                    # build message
                    lines = [f"🚨🚨 高度动向信号：{coin_upper} → {final_side}"]
                    for p in main_periods:
                        ind_ref = period_consistent[p]["exch_inds"]["huobi"]
                        lines.append(f"{p} | 入场:{format_price(ind_ref['entry'])} 目标:{format_price(target)} 止损:{format_price(stop)} | EMA_trend:{period_consistent[p]['ema_trend']}")
                        # add per-exchange short lines
                        for ex in ("huobi","binance","bybit"):
                            ind = period_consistent[p]["exch_inds"][ex]
                            lines.append(f"  {ex.upper()} EMA:{[round(x,4) for x in ind['ema_vals']]} MACD:{ind['macd']:.4f} RSI:{ind['rsi']:.2f} WR:{ind['wr']:.2f} VOLΔ:{ind['vol_trend']:.3f}")
                    # GPT quick analysis from huobi 60min
                    df_ref = period_results["60min"]["huobi"]
                    try:
                        from_text = ""
                        if df_ref is not None:
                            # try include a short GPT style analysis snippet
                            closes = df_ref["close"].tail(50).astype(float).tolist()
                            avg = sum(closes)/len(closes)
                            from_text = f"\n🧠 快速提示: 均价(50):{format_price(avg)}"
                        msg = "\n".join(lines) + from_text
                    except:
                        msg = "\n".join(lines)
                    send_telegram_message(msg)
                    log(f"🔥 {coin_upper} 高度动向信息已发送: {final_side}")
                else:
                    log(f"{coin_upper} 已有相同高度动向({final_side})，跳过重复发送")

            else:
                # 非强信号 — hourly message will show "无一致信号 (监控中)"
                log(f"{coin_upper} 未满足所有周期一致性（满足周期数 {len(good_periods)}/{len(main_periods)})")

        # 普通信息每小时发送一次（更清楚地说明是“无一致信号”）
        if last_hour_msg is None or (now - last_hour_msg) >= timedelta(hours=1):
            msg_lines = ["📢 每小时普通信息（仅显示是否有强一致信号）"]
            for coin in main_coins:
                status = "无一致信号 (监控中)"
                if prev_high_signal.get(coin):
                    status = f"上次高度动向: {prev_high_signal[coin]}"
                msg_lines.append(f"{coin.upper()}  {status}")
            # meme coins as well
            for coin in meme_coins:
                status = prev_high_signal.get(coin) or "无一致信号 (监控中)"
                msg_lines.append(f"{coin.upper()}  {status}")
            send_telegram_message("\n".join(msg_lines))
            last_hour_msg = now
            log("📢 每小时普通信息 已发送")

        time.sleep(POLL_INTERVAL)

    except Exception as e:
        log(f"[LOOP ERROR] {e}")
        time.sleep(10)
