# autotrader.py
# 三确认（≥3/4 周期）才下单 + 多指标同向确认 + Telegram 推送 + ccxt 实盘/纸面切换
# 支持：huobi / binance / okx （默认现货，MARKET_TYPE=swap 时尝试合约）
# 周期：1h, 4h, 1d, 1w；满足 >=3 个周期同方向 -> 触发下单
# 指标：EMA(5/10/30)趋势、MACD hist、RSI(14)、WR(14)、K/D(金叉/死叉)、VOL变化、ATR止损止盈

import os, time, math, traceback
import ccxt
import requests
import pandas as pd
import numpy as np
import ta
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ========== ENV ==========
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

EXCHANGE_NAME = os.getenv("EXCHANGE", "huobi").lower()   # huobi/binance/okx
API_KEY   = os.getenv("API_KEY", "")
API_SECRET= os.getenv("API_SECRET", "")

MARKET_TYPE = os.getenv("MARKET_TYPE", "spot").lower()   # spot / swap
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,LTC/USDT").split(",") if s.strip()]

BASE_USDT = float(os.getenv("BASE_USDT", "100"))
ATR_MULT  = float(os.getenv("RISK_ATR_MULT", "1.5"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))
LIVE_TRADE = int(os.getenv("LIVE_TRADE", "0"))

# 周期与 ccxt interval 映射
PERIODS = {
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1w": "1w",
}
REQUIRED_CONFIRMS = 3  # 4 个周期中至少 3 个同向

def nowstr():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"[{nowstr()}] {msg}", flush=True)

def tg_send(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text})
    except Exception as e:
        log(f"TG发送失败: {e}")

def build_exchange():
    # 统一用 enableRateLimit，走 API KEY
    params = {"apiKey": API_KEY, "secret": API_SECRET, "enableRateLimit": True}
    if EXCHANGE_NAME == "huobi":
        ex = ccxt.huobi(params)
    elif EXCHANGE_NAME == "binance":
        ex = ccxt.binance(params)
    elif EXCHANGE_NAME == "okx":
        ex = ccxt.okx(params)
    else:
        raise ValueError("EXCHANGE 仅支持 huobi/binance/okx")
    # 选择衍生品/现货市场
    if MARKET_TYPE == "swap" and hasattr(ex, "options"):
        # 对支持的交易所启用合约市场（不同交易所有不同配置；ccxt 会自动切换）
        try:
            ex.options["defaultType"] = "swap"
            if hasattr(ex, "set_sandbox_mode"):
                # 你可以根据需要开启沙盒
                pass
        except Exception:
            pass
    return ex

def fetch_ohlcv(ex, symbol, timeframe="1h", limit=200):
    return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

def df_from_ohlcv(ohlcv):
    # ohlcv: [ts, open, high, low, close, volume]
    cols = ["ts","open","high","low","close","vol"]
    df = pd.DataFrame(ohlcv, columns=cols)
    for c in ["open","high","low","close","vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def compute_atr(df, period=14):
    try:
        high = df["high"]; low = df["low"]; close = df["close"]
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return float(atr.iloc[-1])
    except Exception:
        return None

def indicators_and_side(df):
    """
    返回 (side, details)
    side: "多"/"空"/None
    details: dict(ema_trend, macd, rsi, wr, k_trend, vol_trend, entry)
    """
    if df is None or len(df) < 35:
        return None, None
    work = df.copy().iloc[:-1]  # 丢掉未收盘
    close = work["close"].astype(float)
    high  = work["high"].astype(float)
    low   = work["low"].astype(float)
    vol   = work["vol"].astype(float)

    ema5  = close.ewm(span=5).mean().iloc[-1]
    ema10 = close.ewm(span=10).mean().iloc[-1]
    ema30 = close.ewm(span=30).mean().iloc[-1]
    ema_trend = "多" if (ema5>ema10 and ema10>ema30) else ("空" if (ema5<ema10 and ema10<ema30) else "中性")

    macd_hist = ta.trend.MACD(close).macd_diff().iloc[-1]     # >0 多，<0 空
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]  # >50 多，<50 空
    wr  = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r().iloc[-1]  # >-50 偏多，<-50 偏空
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=9, smooth_window=3)
    k_val = stoch.stoch().iloc[-1]; d_val = stoch.stoch_signal().iloc[-1]
    k_trend = "多" if k_val>d_val else ("空" if k_val<d_val else "中性")

    vol_trend = (vol.iloc[-1] - vol.iloc[-2]) / (vol.iloc[-2] + 1e-12)  # 成交量相对变化

    # 条件同向计分（严格一点提升“准度”）
    score_bull = 0
    score_bear = 0
    score_bull += 1 if ema_trend=="多" else 0
    score_bear += 1 if ema_trend=="空" else 0
    score_bull += 1 if macd_hist>0 else 0
    score_bear += 1 if macd_hist<0 else 0
    score_bull += 1 if rsi>50 else 0
    score_bear += 1 if rsi<50 else 0
    score_bull += 1 if wr>-50 else 0
    score_bear += 1 if wr<-50 else 0
    score_bull += 1 if k_trend=="多" else 0
    score_bear += 1 if k_trend=="空" else 0
    # 成交量：上涨更偏多，下降更偏空（不强制，但作为加分项）
    if vol_trend>0: score_bull += 1
    if vol_trend<0: score_bear += 1

    # 需要至少 4 项同向才判定该周期方向（避免噪音）
    side = None
    if score_bull >= 4 and score_bull >= score_bear+2:
        side = "多"
    elif score_bear >= 4 and score_bear >= score_bull+2:
        side = "空"

    details = {
        "ema_trend": ema_trend,
        "ema_vals": [float(ema5), float(ema10), float(ema30)],
        "macd": float(macd_hist),
        "rsi": float(rsi),
        "wr": float(wr),
        "k_trend": k_trend,
        "vol_trend": float(vol_trend),
        "entry": float(close.iloc[-1]),
    }
    return side, details

def calc_stop_target(df, side, entry):
    atr = compute_atr(df)
    if atr is None: return None, None
    if side=="多":
        return entry - ATR_MULT*atr, entry + ATR_MULT*atr
    else:
        return entry + ATR_MULT*atr, entry - ATR_MULT*atr

def format_price(p):
    try:
        p = float(p)
        if p >= 100: return f"{p:.2f}"
        if p >= 1:   return f"{p:.4f}"
        if p >= 0.01:return f"{p:.6f}"
        return f"{p:.8f}"
    except:
        return "-"

def tier_color_text(cons):
    # cons: 一致个数
    if cons >= 3:
        return "🟢 强（3+/4）"
    elif cons == 2:
        return "🟡 中（2/4）"
    elif cons == 1:
        return "🔴 弱（1/4）"
    else:
        return "⚪ 无（0/4）"

def place_order(ex, symbol, side, entry, stop, target):
    """
    以 BASE_USDT / entry 计算数量，市价单。
    LIVE_TRADE=0 时仅打印，不下单。
    """
    qty = max(1e-8, BASE_USDT / max(entry, 1e-8))
    order_side = "buy" if side=="多" else "sell"
    if LIVE_TRADE != 1:
        log(f"[纸面单] {symbol} {side} 市价数量≈{qty}")
        return {"id":"paper", "status":"simulated", "side":order_side, "amount":qty}

    # 真下单
    try:
        # 现货市价
        if MARKET_TYPE == "spot":
            o = ex.create_order(symbol, type="market", side=order_side, amount=qty)
        else:
            # 合约：不同交易所可能需要额外参数；这里用最通用的 create_order
            # 有些交易所需 symbol 形如 BTC/USDT:USDT 或不同命名，若报错请调整 SYMBOLS 或用 ex.load_markets() 查看
            o = ex.create_order(symbol, type="market", side=order_side, amount=qty)
        log(f"[下单成功] {o}")
        return o
    except Exception as e:
        log(f"[下单失败] {e}")
        return None

def summarize_details(tf, side, det):
    return (f"{tf} | 方向:{side or '无'}  入场:{format_price(det['entry']) if det else '-'} | "
            f"EMA趋势:{det['ema_trend'] if det else '-'}  MACD:{det['macd'] if det else '-'}  "
            f"RSI:{det['rsi'] if det else '-'}  WR:{det['wr'] if det else '-'}  "
            f"KDJ:{det['k_trend'] if det else '-'}  VOLΔ:{round(det['vol_trend'],3) if det else '-'}")

def main():
    ex = build_exchange()
    log(f"启动交易Bot | EXCHANGE={EXCHANGE_NAME} MARKET_TYPE={MARKET_TYPE} LIVE_TRADE={LIVE_TRADE} POLL={POLL_INTERVAL}s")
    tg_send(f"🤖 交易Bot已启动：{EXCHANGE_NAME} / {MARKET_TYPE} / 轮询{POLL_INTERVAL}s / 纸面={1 if LIVE_TRADE!=1 else 0}")

    last_hourly_push_ts = 0

    while True:
        loop_start = time.time()
        try:
            ex.load_markets()

            for symbol in SYMBOLS:
                sides = []
                details_map = {}

                # 逐周期拉K并判定
                for tf in ["1h","4h","1d","1w"]:
                    try:
                        ohlcv = fetch_ohlcv(ex, symbol, timeframe=PERIODS[tf], limit=200)
                        df = df_from_ohlcv(ohlcv)
                        side, det = indicators_and_side(df)
                        details_map[tf] = (side, det, df)
                        sides.append(side)
                        log(summarize_details(tf, side, det))
                    except Exception as e_tf:
                        log(f"{symbol} {tf} 获取/指标失败: {e_tf}")
                        details_map[tf] = (None, None, None)

                # 统计方向一致性
                bull = sum(1 for s in sides if s=="多")
                bear = sum(1 for s in sides if s=="空")
                final_side = None
                confirms = 0
                if bull >= REQUIRED_CONFIRMS and bull > bear:
                    final_side = "多"; confirms = bull
                elif bear >= REQUIRED_CONFIRMS and bear > bull:
                    final_side = "空"; confirms = bear

                # 每小时汇总推送（含强/中/弱评级）
                now_ts = int(time.time())
                if now_ts - last_hourly_push_ts >= 3600:
                    grade = tier_color_text(max(bull, bear))
                    lines = [f"⏰ 每小时汇总 [{symbol}] 评级: {grade}（多:{bull} 空:{bear}）"]
                    for tf in ["1h","4h","1d","1w"]:
                        s, det, _ = details_map[tf]
                        lines.append(summarize_details(tf, s, det))
                    tg_send("\n".join(lines))
                    last_hourly_push_ts = now_ts

                # 三确认才下单（突发）
                if final_side is not None:
                    # 用 1h 的 df 做 ATR（更灵敏），并拿 1h 的 entry
                    s1h, d1h, df1h = details_map["1h"]
                    if d1h and df1h is not None:
                        entry = d1h["entry"]
                        stop, target = calc_stop_target(df1h, final_side, entry)
                        if stop is None or target is None:
                            log(f"{symbol} 无法计算ATR止盈止损，跳过下单")
                            continue

                        # 额外的“所有条件满足”强劲突发（KDJ/EMA/MACD/RSI/WR/VOL 同向）
                        strong = False
                        if s1h == final_side:
                            det = d1h
                            conds_ok = [
                                (det["ema_trend"] == ("多" if final_side=="多" else "空")),
                                ((det["macd"]>0) if final_side=="多" else (det["macd"]<0)),
                                ((det["rsi"]>50) if final_side=="多" else (det["rsi"]<50)),
                                ((det["wr"]>-50) if final_side=="多" else (det["wr"]<-50)),
                                ((det["k_trend"]=="多") if final_side=="多" else (det["k_trend"]=="空")),
                                ((det["vol_trend"]>0) if final_side=="多" else (det["vol_trend"]<0)),
                            ]
                            strong = all(conds_ok)

                        # 下单
                        o = place_order(ex, symbol, final_side, entry, stop, target)

                        msg = []
                        if strong:
                            msg.append("🔥🔥🔥 强烈高度动向捕捉到（满足所有条件）")
                        else:
                            msg.append("⚡ 三确认触发下单")
                        msg.append(f"{symbol} 做{'多' if final_side=='多' else '空'}")
                        msg.append(f"入场: {format_price(entry)}  目标: {format_price(target)}  止损: {format_price(stop)}")
                        msg.append(f"一致性: {max(bull,bear)}/4 周期")
                        tg_send("\n".join(msg))

        except Exception as e:
            log(f"[主循环异常] {e}\n{traceback.format_exc()}")

        # 快速轮询
        used = time.time() - loop_start
        sleep_s = max(1, POLL_INTERVAL - int(used))
        time.sleep(sleep_s)

if __name__ == "__main__":
    main()
