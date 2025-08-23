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
        # 保证 reduceOnly 可用
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
    macd_hist = float(macd.macd_diff().iloc[-1])
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
    market = ex.market(symbol)
    # 将数量对齐到 step
    step = market.get("limits", {}).get("amount", {}).get("min", None)
    precision = market.get("precision", {}).get("amount", None)
    try:
        qty = ex.amount_to_precision(symbol, raw_qty)
    except Exception:
        qty = raw_qty
        if precision is not None:
            qty = float(f"{qty:.{precision}f}")
    # 避免太小
    if step and qty < step:
        qty = step
    return max(qty, 0.0)

def create_sl_tp_orders(ex, symbol, side, qty, sl_price, tp_price):
    """在合约上创建止损/止盈减仓单（reduceOnly True）"""
    market = ex.market(symbol)
    params_sl = {
        "reduceOnly": True,
        "stopPrice": ex.price_to_precision(symbol, sl_price),
        "timeInForce": "GTC",
        "workingType": "CONTRACT_PRICE",  # or MARK_PRICE
    }
    params_tp = {
        "reduceOnly": True,
        "stopPrice": ex.price_to_precision(symbol, tp_price),
        "timeInForce": "GTC",
        "workingType": "CONTRACT_PRICE",
    }
    try:
        if side == "多":
            # 多单：止损卖出，止盈卖出
            ex.create_order(symbol, type="STOP_MARKET", side="sell", amount=qty, params=params_sl)
            ex.create_order(symbol, type="TAKE_PROFIT_MARKET", side="sell", amount=qty, params=params_tp)
        else:
            # 空单：止损买入，止盈买入
            ex.create_order(symbol, type="STOP_MARKET", side="buy", amount=qty, params=params_sl)
            ex.create_order(symbol, type="TAKE_PROFIT_MARKET", side="buy", amount=qty, params=params_tp)
        return True
    except Exception as e:
        log(f"创建SL/TP失败 {symbol}: {e}")
        return False

# 跟踪止盈状态（内存）
trail_state = {}  # { symbol: {"side": "多"/"空", "best": float, "atr": float, "qty": float} }

def update_trailing_stop(ex, symbol, last_price):
    """价格向有利方向移动 >= TRAIL_ATR_MULT * ATR 时，上调止损（简单实现：取消旧SL后重挂更优SL）"""
    st = trail_state.get(symbol)
    if not st: 
        return
    side = st["side"]; best = st["best"]; atr = st["atr"]; qty = st["qty"]

    moved = False
    if side == "多":
        if last_price > best:
            trail_state[symbol]["best"] = last_price
        # 只在价格超出 (best + 1*ATR) 这种级别后上调一次（避免太频繁）
        if last_price >= best + TRAIL_ATR_MULT * atr:
            # 新SL抬高到 (last_price - SL_ATR_MULT*atr)
            new_sl = last_price - SL_ATR_MULT * atr
            try:
                # 简化：直接挂一个新的 STOP_MARKET（真实场景应先取消旧SL；这里示例化）
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

    last_push_ts = 0

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

                # 每分钟也推一次（或你按小时推）
                lines = [f"{symbol} 当前多周期共识:（多:{bull} 空:{bear}）"]
                for tf in TIMEFRAMES:
                    s, det = tf_details[tf]
                    lines.append(summarize(tf, s, det))
                tg_send("\n".join(lines))

                # ======= 交易逻辑：仅对 TRADE_SYMBOLS 真下单 =======
                if symbol in TRADE_SYMBOLS and consensus in ("多","空"):
                    # 以 1h ATR 为基准风控（也可改成4h/加权）
                    s1h, d1h = tf_details.get("1h", (None, None))
                    if not d1h:
                        continue
                    entry = d1h["entry"]
                    atr   = d1h["atr"]
                    price = entry

                    # 计算数量
                    qty = amount_for_futures(ex, symbol, price)
                    if qty <= 0:
                        log(f"{symbol} 数量过小，跳过")
                        continue

                    # 价格精度
                    sl = None; tp=None
                    if consensus == "多":
                        sl = price - SL_ATR_MULT*atr
                        tp = price + TP_ATR_MULT*atr
                    else:
                        sl = price + SL_ATR_MULT*atr
                        tp = price - TP_ATR_MULT*atr

                    if LIVE_TRADE != 1:
                        log(f"[纸面单] {symbol} {consensus} 市价 数量≈{qty} 进场≈{fmt_price(price)} SL≈{fmt_price(sl)} TP≈{fmt_price(tp)} ATR≈{fmt_price(atr)}")
                        continue

                    try:
                        # 开仓
                        order_side = "buy" if consensus=="多" else "sell"
                        o = ex.create_order(symbol, type="MARKET", side=order_side, amount=qty)
                        log(f"[下单成功] {symbol} {order_side} qty={qty} price≈{fmt_price(price)}")
                        tg_send(f"⚡ 开仓 {symbol} {consensus} 价≈{fmt_price(price)} 数量≈{qty}\nSL:{fmt_price(sl)} TP:{fmt_price(tp)} ATR:{fmt_price(atr)}")

                        # 挂 SL / TP 减仓单
                        ok = create_sl_tp_orders(ex, symbol, consensus, qty, sl, tp)
                        if ok:
                            # 初始化跟踪止盈状态
                            trail_state[symbol] = {
                                "side": consensus,
                                "best": price,   # 多：最高价；空：最低价（简单用进场价初始化）
                                "atr": atr,
                                "qty": qty
                            }
                    except Exception as e:
                        log(f"[下单失败] {symbol}: {e}")
                        tg_send(f"❌ 下单失败 {symbol}: {e}")

                # ======= 跟踪止盈：每轮更新 =======
                try:
                    ticker = ex.fetch_ticker(symbol)
                    last_price = float(ticker.get("last") or ticker.get("close") or 0.0)
                    if last_price:
                        update_trailing_stop(ex, symbol, last_price)
                except Exception as e:
                    log(f"获取价格/更新跟踪失败 {symbol}: {e}")

        except Exception as e:
            log(f"[主循环异常] {e}\n{traceback.format_exc()}")

        used = time.time() - loop_start
        time.sleep(max(1, POLL_INTERVAL - int(used)))

if __name__ == "__main__":
    main()
