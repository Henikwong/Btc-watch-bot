import os, time, math, traceback
import requests
import ccxt
import pandas as pd
import numpy as np
import ta
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ========== 环境变量 ==========
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT  = os.getenv("TELEGRAM_CHAT_ID", "")

EXCHANGE_NAME = os.getenv("EXCHANGE", "binance").lower()      # 仅支持 binance
API_KEY   = os.getenv("API_KEY", "")
API_SECRET= os.getenv("API_SECRET", "")

MARKET_TYPE = os.getenv("MARKET_TYPE", "future").lower()      # future / spot
LEVERAGE  = int(os.getenv("LEVERAGE", "10"))
BASE_USDT = float(os.getenv("BASE_USDT", "15"))               # 每次下单的保证金(USDT)
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))         # 秒
LIVE_TRADE    = int(os.getenv("LIVE_TRADE", "0"))             # 0=纸面, 1=实盘

# 交易对配置（也可用环境变量传入：用英文逗号分隔）
TRADE_SYMBOLS   = [s.strip() for s in os.getenv("TRADE_SYMBOLS", "BTC/USDT,ETH/USDT").split(",") if s.strip()]
OBSERVE_SYMBOLS = [s.strip() for s in os.getenv("OBSERVE_SYMBOLS", "LTC/USDT,BNB/USDT,SOL/USDT,XRP/USDT").split(",") if s.strip()]
ALL_SYMBOLS = TRADE_SYMBOLS + OBSERVE_SYMBOLS

# 策略参数
TIMEFRAMES = ["1h", "4h", "1d"]
REQUIRED_CONFIRMS = 2                         # 多周期至少同向数量
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "2.0"))   # 止损=2*ATR
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "3.0"))   # 止盈=3*ATR
TRAIL_ATR_MULT = float(os.getenv("TRAIL_ATR_MULT", "1.5"))  # 跟踪止损抬升阈值
PARTIAL_TP_RATIO = float(os.getenv("PARTIAL_TP_RATIO", "0.3"))  # 提前止盈减仓比例（默认30%）

# ========== 工具 ==========
def nowstr():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"[{nowstr()}] {msg}", flush=True)

def tg_send(text):
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TG_CHAT, "text": text})
    except Exception as e:
        log(f"TG发送失败: {e}")

# ========== 交易所 ==========
def build_exchange():
    if EXCHANGE_NAME != "binance":
        raise RuntimeError(f"仅示例 binance，当前: {EXCHANGE_NAME}")
    ex = ccxt.binance({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": MARKET_TYPE},  # futures
    })
    ex.load_markets()
    return ex

def binance_set_leverage(ex, symbol, lev):
    if MARKET_TYPE != "future":
        return
    try:
        market = ex.market(symbol)
        ex.fapiPrivate_post_leverage({
            "symbol": market["id"],
            "leverage": lev
        })
        ex.fapiPrivate_post_margintype({
            "symbol": market["id"],
            "marginType": "CROSSED"  # 如需逐仓可改为 ISOLATED
        })
    except Exception as e:
        log(f"设置杠杆/保证金失败 {symbol}: {e}")

# ========== 指标 & 分析 ==========
def df_from_ohlcv(ohlcv):
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","vol"])
    for c in ["open","high","low","close","vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def analyze_one_df(df):
    """返回 side('多'/'空'/None), details(dict)；使用已收盘K（去掉最后一根）"""
    if df is None or len(df) < 50:
        return None, None
    work = df.iloc[:-1].copy()
    close, high, low, vol = work["close"], work["high"], work["low"], work["vol"]

    # EMA 金叉/死叉趋势
    ema5  = close.ewm(span=5).mean().iloc[-1]
    ema10 = close.ewm(span=10).mean().iloc[-1]
    ema30 = close.ewm(span=30).mean().iloc[-1]
    ema_trend = "多" if (ema5>ema10>ema30) else ("空" if (ema5<ema10<ema30) else "中性")

    macd = ta.trend.MACD(close)
    macd_hist_series = macd.macd_diff()
    macd_hist = float(macd_hist_series.iloc[-1])
    rsi = float(ta.momentum.RSIIndicator(close, 14).rsi().iloc[-1])
    wr  = float(ta.momentum.WilliamsRIndicator(high, low, close, 14).williams_r().iloc[-1])
    stoch = ta.momentum.StochasticOscillator(high, low, close, 9, 3)
    k_val = float(stoch.stoch().iloc[-1]); d_val = float(stoch.stoch_signal().iloc[-1])
    k_trend = "多" if k_val>d_val else ("空" if k_val<d_val else "中性")

    vol_trend = float((vol.iloc[-1]-vol.iloc[-2])/(abs(vol.iloc[-2])+1e-12))

    atr = float(ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range().iloc[-1])
    entry = float(close.iloc[-1])

    score_bull = sum([
        ema_trend=="多", macd_hist>0, rsi>55, wr>-50, k_trend=="多", vol_trend>0
    ])
    score_bear = sum([
        ema_trend=="空", macd_hist<0, rsi<45, wr<-50, k_trend=="空", vol_trend<0
    ])
    side = None
    if score_bull>=4 and score_bull>=score_bear+2:
        side="多"
    elif score_bear>=4 and score_bear>=score_bull+2:
        side="空"

    det = {
        "ema_trend": ema_trend,
        "macd": macd_hist,
        "macd_hist_series": macd_hist_series,  # 给4h弱化判断复用
        "rsi": rsi,
        "wr": wr,
        "k_trend": k_trend,
        "vol_trend": vol_trend,
        "atr": atr,
        "entry": entry,
    }
    return side, det

def summarize(tf, side, det):
    return (f"{tf} | 方向:{side or '无'} 入场:{fmt_price(det['entry']) if det else '-'} | "
            f"EMA:{det['ema_trend'] if det else '-'} MACD:{round(det['macd'],4) if det else '-'} "
            f"RSI:{round(det['rsi'],2) if det else '-'} WR:{round(det['wr'],2) if det else '-'} "
            f"KDJ:{det['k_trend'] if det else '-'} VOLΔ:{round(det['vol_trend'],3) if det else '-'} ATR:{round(det['atr'],2) if det else '-'}")

# ========== 精度 & 下单工具 ==========
def fmt_price(p):
    p = float(p)
    if p>=100: return f"{p:.2f}"
    if p>=1:   return f"{p:.4f}"
    if p>=0.01:return f"{p:.6f}"
    return f"{p:.8f}"

def amount_for_futures(ex, symbol, price):
    # 合约名义： 用 BASE_USDT * LEVERAGE / price
    raw_qty = BASE_USDT * LEVERAGE / max(price, 1e-12)
    try:
        qty = ex.amount_to_precision(symbol, raw_qty)
    except Exception:
        qty = raw_qty
    return max(float(qty), 0.0)

def create_sl_tp_orders(ex, symbol, side, qty, sl_price, tp_price):
    """在合约上创建止损/止盈减仓单（reduceOnly True）"""
    params_sl = {
        "reduceOnly": True,
        "stopPrice": ex.price_to_precision(symbol, sl_price),
        "timeInForce": "GTC",
        "workingType": "CONTRACT_PRICE",  # 或 MARK_PRICE，看你偏好
    }
    params_tp = {
        "reduceOnly": True,
        "stopPrice": ex.price_to_precision(symbol, tp_price),
        "timeInForce": "GTC",
        "workingType": "CONTRACT_PRICE",
    }
    try:
        if side == "多":
            ex.create_order(symbol, type="STOP_MARKET", side="sell", amount=qty, params=params_sl)
            ex.create_order(symbol, type="TAKE_PROFIT_MARKET", side="sell", amount=qty, params=params_tp)
        else:
            ex.create_order(symbol, type="STOP_MARKET", side="buy", amount=qty, params=params_sl)
            ex.create_order(symbol, type="TAKE_PROFIT_MARKET", side="buy", amount=qty, params=params_tp)
        return True
    except Exception as e:
        log(f"创建SL/TP失败 {symbol}: {e}")
        return False

# 跟踪止盈状态（内存）
trail_state = {}  # { symbol: {"side": "多"/"空", "best": float, "atr": float, "qty": float, "entry": float, "partial_done": bool} }

def update_trailing_stop(ex, symbol, last_price):
    """价格向有利方向移动 >= TRAIL_ATR_MULT * ATR 时，上调止损（简化示例：直接再挂更优STOP_MARKET）"""
    st = trail_state.get(symbol)
    if not st: 
        return
    side = st["side"]; best = st["best"]; atr = st["atr"]; qty = st["qty"]

    moved = False
    if side == "多":
        if last_price > best:
            trail_state[symbol]["best"] = last_price
        if last_price >= best + TRAIL_ATR_MULT * atr:
            new_sl = last_price - SL_ATR_MULT * atr
            try:
                ex.create_order(symbol, type="STOP_MARKET", side="sell", amount=qty, params={
                    "reduceOnly": True,
                    "stopPrice": ex.price_to_precision(symbol, new_sl),
                    "workingType": "CONTRACT_PRICE",
                })
                trail_state[symbol]["best"] = last_price
                moved = True
            except Exception as e:
                log(f"更新跟踪止损失败 {symbol}: {e}")
    else:  # 空
        if last_price < best:
            trail_state[symbol]["best"] = last_price
        if last_price <= best - TRAIL_ATR_MULT * atr:
            new_sl = last_price + SL_ATR_MULT * atr
            try:
                ex.create_order(symbol, type="STOP_MARKET", side="buy", amount=qty, params={
                    "reduceOnly": True,
                    "stopPrice": ex.price_to_precision(symbol, new_sl),
                    "workingType": "CONTRACT_PRICE",
                })
                trail_state[symbol]["best"] = last_price
                moved = True
            except Exception as e:
                log(f"更新跟踪止损失败 {symbol}: {e}")
    if moved:
        tg_send(f"🔧 跟踪止损上调 {symbol} side={side} new_best={fmt_price(trail_state[symbol]['best'])}")

def macd_weakening_and_partial_tp(ex, symbol, last_price, tf4h_details):
    """
    提前止盈逻辑（仅做一次）：
    - 持仓已盈利 >= 1×ATR(1h)
    - 4h MACD 柱子弱化：多单 正值且变小；空单 负值且绝对值变小（向0靠近）
    - RSI 过滤：多>65，空<35
    触发：reduceOnly 市价减仓 PARTIAL_TP_RATIO
    """
    st = trail_state.get(symbol)
    if not st or st.get("partial_done"):
        return
    side = st["side"]; entry = st["entry"]; atr1h = st["atr"]; qty_total = st["qty"]

    # 盈利判断
    profit_ok = False
    if side == "多":
        profit_ok = (last_price - entry) >= (1.0 * atr1h)
    else:
        profit_ok = (entry - last_price) >= (1.0 * atr1h)
    if not profit_ok:
        return

    # 4h 指标
    s4h, d4h = tf4h_details
    if not d4h or "macd_hist_series" not in d4h:
        return

    macd_hist_series = d4h["macd_hist_series"]
    if len(macd_hist_series) < 3:
        return
    # 用已收盘柱：-2 和 -1（因为 analyze_one_df 已去掉最后一根，这里 -1 是最近收盘柱）
    hist_prev = float(macd_hist_series.iloc[-2])
    hist_last = float(macd_hist_series.iloc[-1])
    rsi4h = float(d4h["rsi"])

    macd_weak = False
    if side == "多":
        # 正值且变小（走弱）
        macd_weak = (hist_last > 0) and (hist_last < hist_prev) and (rsi4h > 65)
    else:
        # 负值且绝对值变小（走弱）
        macd_weak = (hist_last < 0) and (abs(hist_last) < abs(hist_prev)) and (rsi4h < 35)

    if not macd_weak:
        return

    reduce_qty = max(qty_total * PARTIAL_TP_RATIO, 0.0)
    if reduce_qty <= 0:
        return

    if LIVE_TRADE != 1:
        log(f"[纸面-提前止盈] {symbol} side={side} 减仓≈{reduce_qty} last≈{fmt_price(last_price)} entry≈{fmt_price(entry)} RSI4h={rsi4h:.2f}")
        trail_state[symbol]["partial_done"] = True
        tg_send(f"🟡 提前止盈(纸面) {symbol} {side} 减仓≈{reduce_qty:.6f} 价≈{fmt_price(last_price)} (4h MACD弱化 + RSI过滤)")
        return

    try:
        if side == "多":
            ex.create_order(symbol, type="MARKET", side="sell", amount=reduce_qty, params={"reduceOnly": True})
        else:
            ex.create_order(symbol, type="MARKET", side="buy", amount=reduce_qty, params={"reduceOnly": True})
        trail_state[symbol]["partial_done"] = True
        tg_send(f"🟢 提前止盈(已执行) {symbol} {side} 减仓≈{reduce_qty:.6f} 价≈{fmt_price(last_price)} (4h MACD弱化 + RSI过滤)")
        log(f"[提前止盈成功] {symbol} side={side} reduce={reduce_qty}")
    except Exception as e:
        log(f"[提前止盈失败] {symbol}: {e}")
        tg_send(f"❌ 提前止盈失败 {symbol}: {e}")

# ========== 主循环 ==========
def main():
    ex = build_exchange()
    tg_send(f"🤖 启动Bot {EXCHANGE_NAME}/{MARKET_TYPE} 模式={'实盘' if LIVE_TRADE==1 else '纸面'} 杠杆x{LEVERAGE}")

    # 设置各交易对杠杆
    if MARKET_TYPE == "future":
        for sym in TRADE_SYMBOLS:
            try:
                binance_set_leverage(ex, sym, LEVERAGE)
            except Exception as e:
                log(f"设置杠杆失败 {sym}: {e}")

    while True:
        loop_start = time.time()
        try:
            for symbol in ALL_SYMBOLS:
                # 多周期分析
                tf_sides = []
                tf_details = {}
                for tf in TIMEFRAMES:
                    try:
                        ohlcv = ex.fetch_ohlcv(symbol, tf, limit=200)
                        df = df_from_ohlcv(ohlcv)
                        side, det = analyze_one_df(df)
                        tf_sides.append(side)
                        tf_details[tf] = (side, det)
                        log(summarize(tf, side, det))
                    except Exception as e:
                        log(f"❌ 获取/分析失败 {symbol} {tf}: {e}")
                        tf_sides.append(None)
                        tf_details[tf] = (None, None)

                # 共识方向
                bull = sum(1 for s in tf_sides if s=="多")
                bear = sum(1 for s in tf_sides if s=="空")
                consensus = None
                if bull>=REQUIRED_CONFIRMS and bull>bear:
                    consensus="多"
                elif bear>=REQUIRED_CONFIRMS and bear>bull:
                    consensus="空"

                # 推送
                lines = [f"{symbol} 当前多周期共识:（多:{bull} 空:{bear}）"]
                for tf in TIMEFRAMES:
                    s, det = tf_details[tf]
                    lines.append(summarize(tf, s, det))
                tg_send("\n".join(lines))

                # ======= 交易逻辑：仅对 TRADE_SYMBOLS 真下单 =======
                if symbol in TRADE_SYMBOLS and consensus in ("多","空"):
                    # 以 1h ATR 为基准风控
                    s1h, d1h = tf_details.get("1h", (None, None))
                    if not d1h:
                        continue
                    entry = d1h["entry"]
                    atr1h = d1h["atr"]
                    price = entry

                    # 数量
                    qty = amount_for_futures(ex, symbol, price)
                    if qty <= 0:
                        log(f"{symbol} 数量过小，跳过")
                        continue

                    # SL / TP
                    if consensus == "多":
                        sl = price - SL_ATR_MULT*atr1h
                        tp = price + TP_ATR_MULT*atr1h
                    else:
                        sl = price + SL_ATR_MULT*atr1h
                        tp = price - TP_ATR_MULT*atr1h

                    if LIVE_TRADE != 1:
                        log(f"[纸面单] {symbol} {consensus} 市价 数量≈{qty} 进场≈{fmt_price(price)} SL≈{fmt_price(sl)} TP≈{fmt_price(tp)} ATR1h≈{fmt_price(atr1h)}")
                    else:
                        try:
                            order_side = "buy" if consensus=="多" else "sell"
                            ex.create_order(symbol, type="MARKET", side=order_side, amount=qty)
                            log(f"[下单成功] {symbol} {order_side} qty={qty} price≈{fmt_price(price)}")
                            tg_send(f"⚡ 开仓 {symbol} {consensus} 价≈{fmt_price(price)} 数量≈{qty}\nSL:{fmt_price(sl)} TP:{fmt_price(tp)} ATR1h:{fmt_price(atr1h)}")
                            ok = create_sl_tp_orders(ex, symbol, consensus, qty, sl, tp)
                            if not ok:
                                tg_send(f"⚠️ {symbol} SL/TP 挂单失败，请检查")
                        except Exception as e:
                            log(f"[下单失败] {symbol}: {e}")
                            tg_send(f"❌ 下单失败 {symbol}: {e}")
                            continue

                    # 初始化跟踪/提前止盈状态（纸面/实盘都维护，便于观察）
                    trail_state[symbol] = {
                        "side": consensus,
                        "best": price,
                        "atr": atr1h,      # 用1h ATR 作为跟踪阈值与盈利阈值
                        "qty": qty,
                        "entry": price,
                        "partial_done": False,
                    }

                # ======= 每轮：更新跟踪止盈 + 检查4h MACD弱化提前止盈 =======
                try:
                    ticker = ex.fetch_ticker(symbol)
                    last_price = float(ticker.get("last") or ticker.get("close") or 0.0)
                    if last_price:
                        update_trailing_stop(ex, symbol, last_price)
                        # 只有持仓中的 symbol 才检查提前止盈
                        if symbol in trail_state and "4h" in tf_details:
                            macd_weakening_and_partial_tp(ex, symbol, last_price, tf_details["4h"])
                except Exception as e:
                    log(f"获取价格/更新止盈失败 {symbol}: {e}")

        except Exception as e:
            log(f"[主循环异常] {e}\n{traceback.format_exc()}")

        used = time.time() - loop_start
        time.sleep(max(1, POLL_INTERVAL - int(used)))

if __name__ == "__main__":
    main()
