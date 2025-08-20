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

POLL_INTERVAL = 60            # 测试用：每分钟抓取一次；你可以改回 900（15分钟）或其他
ATR_MULT = 1.5                # ATR 止盈/止损倍数
RSI_THRESHOLD = 5             # RSI 最大允许差（跨交易所）
WR_THRESHOLD = 5              # WR 最大允许差
VOL_REL_THRESHOLD = 0.20      # 成交量增减比例允许差（跨交易所）

# 币种（你原本的列表）
main_coins = ["btcusdt","ethusdt","xrpusdt","bnbusdt","solusdt","dogeusdt","trxusdt","adausdt","ltcusdt","linkusdt"]
meme_coins = ["shibusdt","pepeusdt","penguusdt","bonkusdt","trumpusdt","spkuspt","flokusdt"]  # 注意：保留你使用的列表（可能需校正拼写）
# 我会监控 3 个主周期：1小时、4小时、1日；另外发送每小时 summary 时会包含 1周 数据
main_periods = ["60min","4hour","1day"]

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
    except Exception:
        return None

# ====== K线抓取函数（Huobi / Binance / OKX） ======
# period mapping: Huobi style: "60min","4hour","1day","1week"
# Binance intervals: "1h","4h","1d","1w"
# OKX bars: "1H","4H","1D","1W"

period_map = {
    "60min": {"binance":"1h","okx":"1H","huobi":"60min","binance_raw":"1h"},
    "4hour": {"binance":"4h","okx":"4H","huobi":"4hour","binance_raw":"4h"},
    "1day": {"binance":"1d","okx":"1D","huobi":"1day","binance_raw":"1d"},
    "1week": {"binance":"1w","okx":"1W","huobi":"1week","binance_raw":"1w"}
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
            if c in df.columns:
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
        # OKX instId uses e.g. "BTC-USDT-SWAP" for perpetual; keep swap because many memecoins are on swap
        instId = symbol.upper().replace("USDT", "-USDT-SWAP")
        r = requests.get("https://www.okx.com/api/v5/market/candles",
                         params={"instId": instId, "bar": bar, "limit": limit}, timeout=12)
        j = r.json()
        if not isinstance(j, dict) or j.get("code") != "0" or "data" not in j:
            return None
        # OKX returns reverse chronological; columns: ts, o, h, l, c, vol, ...
        df = pd.DataFrame(j["data"])
        # ensure columns exist; some OKX responses are nested arrays - handle generically
        # data rows are like [ts, o, h, l, c, vol, ...]
        if isinstance(df.iloc[0,0], list) or isinstance(df.iloc[0,0], str):
            # If data are lists, convert
            df = pd.DataFrame([row for row in j["data"]])
        # Use first five columns as ts,o,h,l,c,vol
        df = df.iloc[:, :6]
        df.columns = ["ts","open","high","low","close","vol"]
        for c in ["open","high","low","close","vol"]:
            df[c] = df[c].astype(float)
        df = df.iloc[::-1].reset_index(drop=True)  # ascending
        df["id"] = pd.to_numeric(df["ts"], errors="coerce")
        return df
    except Exception as e:
        log(f"[OKX ERROR] {symbol} {e}")
        return None

# ====== 指标计算（返回标准化指标） ======
def calc_indicators(df: pd.DataFrame):
    """
    输入：DataFrame 包含 open,high,low,close,vol
    返回：
      {
        "ema_trend": '多'/'空'/'中性',
        "ema_vals": [ema5, ema10, ema30],
        "macd": float (macd_diff),
        "rsi": float,
        "wr": float,
        "k": float,
        "d": float,
        "j": float,
        "k_trend": '多'/'空'/'中性',
        "vol_trend": float (近两根成交量比值-1),
        "entry": float (latest close)
      }
    """
    if df is None or len(df) < 35:
        return None
    try:
        work = df.copy().iloc[:-1]  # 丢掉未收盘的最新K
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

# ====== 停损止盈 (ATR) ======
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

# ====== 一致性检测函数 ======
def indicators_agree(list_of_inds):
    """
    list_of_inds: list of indicator dicts from different exchanges for the same period.
    返回 (agree_boolean, reason_details)
    规则：
      - EMA 趋势必须相同（多/空/中性）
      - MACD 符号一致（都>0 或 都<0）
      - KDJ: k>d 同向
      - RSI 差 <= RSI_THRESHOLD
      - WR 差 <= WR_THRESHOLD
      - VOL trend 同向（增长或下降）
    """
    try:
        # remove Nones
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

# ====== 格式化你要求的 3/3 2/3 1/3 模板（原封不动内容） ======
def build_consistency_block(coin_upper, side, entry, target, stop, consistent_count):
    """
    consistent_count: 3 -> green template
                      2 -> yellow template
                      1 -> red template
    Must include exact text fragments you provided.
    """
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

# ====== 主循环 ======
prev_high_signal = {}    # 保存最近一次已发的高度动向，避免重复
last_hour_msg = None

log("启动 Bot，多交易所多周期监控（POLL_INTERVAL = {}s）".format(POLL_INTERVAL))

while True:
    try:
        coins = main_coins + meme_coins
        now = datetime.now()

        # 为每个币计算每个周期每个交易所的指标（1h,4h,1d）
        hourly_report_lines = ["📢 每小时普通信息（含 1h / 4h / 24h / 1w 指标 & 一致性状态）"]
        strong_alerts = []  # 突发信息列表（即时发送）

        for coin in coins:
            coin_upper = coin.upper()
            per_period_results = {}  # { period: {exch: indicators dict} }

            # we will fetch 1h,4h,1d and also 1week for the hourly summary
            for period_label in ["60min","4hour","1day","1week"]:
                # map to API params
                huobi_period = period_label if period_label in ["60min","4hour","1day","1week"] else "60min"
                binance_interval = period_map[period_label]["binance"] if period_label in period_map else period_map["60min"]["binance"]
                okx_bar = period_map[period_label]["okx"] if period_label in period_map else period_map["60min"]["okx"]

                # fetch per exchange
                huobi_df = get_kline_huobi(coin, period=huobi_period)
                binance_df = get_kline_binance(coin, interval=binance_interval)
                okx_df = get_kline_okx(coin, bar=okx_bar)

                # log fetch state for 1h/4h/1d only to avoid spamming for week
                if period_label in ["60min","4hour","1day"]:
                    log(f"{coin_upper} {period_label} 抓取状态: Huobi={'OK' if huobi_df is not None else 'FAIL'}, "
                        f"Binance={'OK' if binance_df is not None else 'FAIL'}, OKX={'OK' if okx_df is not None else 'FAIL'}")

                h_ind = calc_indicators(huobi_df) if huobi_df is not None else None
                b_ind = calc_indicators(binance_df) if binance_df is not None else None
                o_ind = calc_indicators(okx_df) if okx_df is not None else None

                # store whatever we have (for summary)
                per_period_results[period_label] = {
                    "huobi_df": huobi_df, "binance_df": binance_df, "okx_df": okx_df,
                    "huobi": h_ind, "binance": b_ind, "okx": o_ind
                }

            # ---- 判断突发/高度动向信号规则 ----
            # 对 60min,4hour,1day 每个周期要求 Huobi+Binance+OKX 都存在并且内部一致
            consistent_counts = 0
            per_period_consistent = {}
            for p in ["60min","4hour","1day"]:
                inds = [per_period_results[p].get("huobi"), per_period_results[p].get("binance"), per_period_results[p].get("okx")]
                ok, reason = indicators_agree(inds)
                per_period_consistent[p] = {"ok": ok, "reason": reason, "inds": inds}
                if ok:
                    consistent_counts += 1
                log(f"{coin_upper} {p} 指标一致性: {ok} ({reason})")

            # 如果 3/3 则强突发（🔥🔥🔥）
            if consistent_counts == 3:
                # 根据 60min 的 huobi entry 作为入场参考
                ind_ref = per_period_results["60min"]["huobi"]
                if ind_ref is not None:
                    side = ind_ref["ema_trend"]  # '多' 或 '空'
                    entry = ind_ref["entry"]
                    # target/stop 使用 huobi 60min ATR
                    huobi_60_df = per_period_results["60min"]["huobi_df"]
                    stop, target = compute_stop_target_from_df(huobi_60_df, side, entry)
                    # build exact block with triple 🔥 note
                    block = build_consistency_block(coin_upper, side, entry, target, stop, 3)
                    strong_alerts.append(("strong", coin, block, 3, side, entry, target, stop))
                    log(f"🔥 {coin_upper} 触发 3/3 高度动向 ({side})")
            # 2/3 yellow condition: we will still include in hourly summary but as yellow
            elif consistent_counts == 2:
                # prepare a yellow block for hourly send (not emergency)
                # choose the period that is consistent - take entry from the first consistent period
                chosen_period = next((p for p,v in per_period_consistent.items() if v["ok"]), None)
                if chosen_period:
                    ind_ref = per_period_results[chosen_period]["huobi"]
                    if ind_ref is not None:
                        side = ind_ref["ema_trend"]
                        entry = ind_ref["entry"]
                        huobi_df_for_atr = per_period_results[chosen_period]["huobi_df"]
                        stop, target = compute_stop_target_from_df(huobi_df_for_atr, side, entry)
                        block = build_consistency_block(coin_upper, side, entry, target, stop, 2)
                        # add to hourly report lines (not immediate emergency)
                        hourly_report_lines.append(block)
            elif consistent_counts == 1:
                chosen_period = next((p for p,v in per_period_consistent.items() if v["ok"]), None)
                if chosen_period:
                    ind_ref = per_period_results[chosen_period]["huobi"]
                    if ind_ref is not None:
                        side = ind_ref["ema_trend"]
                        entry = ind_ref["entry"]
                        huobi_df_for_atr = per_period_results[chosen_period]["huobi_df"]
                        stop, target = compute_stop_target_from_df(huobi_df_for_atr, side, entry)
                        block = build_consistency_block(coin_upper, side, entry, target, stop, 1)
                        hourly_report_lines.append(block)

            # ---- 为 hourly summary 生成指标快照（1h,4h,24h,1w） ----
            # 如果你需要每个币都看到指标（哪怕没有一致信号），这里会把关键数字放到 hourly_report_lines
            # 仅放 summary 的关键字段，避免消息过长
            summary_parts = []
            for p in ["60min","4hour","1day","1week"]:
                ind_h = per_period_results[p]["huobi"]
                ind_b = per_period_results[p]["binance"]
                ind_o = per_period_results[p]["okx"]
                # 取 huobi 为主显示（若无则选其他）
                ind_display = ind_h or ind_b or ind_o
                if ind_display:
                    summary_parts.append(f"{p} → EMA:{ind_display['ema_trend']} MACD:{ind_display['macd']:.3f} RSI:{ind_display['rsi']:.2f} WR:{ind_display['wr']:.2f} VOLΔ:{ind_display['vol_trend']:.3f}")
                else:
                    summary_parts.append(f"{p} → 无数据")
            hourly_report_lines.append(f"{coin_upper} 监控快照:\n" + "\n".join(summary_parts))

        # === 突发信号优先：即时发送（如果有强信号） ===
        # we send strong alerts immediately: triple-fire only when first detected (avoid duplicates)
        for level, coin, block, count, side, entry, target, stop in strong_alerts:
            prev = prev_high_signal.get(coin)
            if prev != (side, count):
                # add triple 🔥 indicator in message header
                msg = "🔥🔥🔥 强烈高度动向捕捉到（满足所有条件）\n" + block
                send_telegram_message(msg)
                prev_high_signal[coin] = (side, count)
                log(f"🚨 已发送强突发消息: {coin.upper()} {side} {count}/3")

        # === 每小时普通信息 ===
        if last_hour_msg is None or (now - last_hour_msg) >= timedelta(hours=1):
            # compose hourly message: first the blocks for 3/2/1 that we appended earlier (they follow the templates)
            # plus the snapshots
            # ensure not to exceed Telegram message size: join and send
            msg_text = "\n\n".join(hourly_report_lines)
            send_telegram_message(msg_text)
            last_hour_msg = now
            log("📢 每小时普通信息 已发送")

        # 间隔
        time.sleep(POLL_INTERVAL)

    except Exception as e:
        log(f"[LOOP ERROR] {e}")
        time.sleep(10)
