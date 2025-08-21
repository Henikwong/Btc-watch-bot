# autotrader.py
import os, time, math, json, requests, joblib, warnings
import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from collections import deque
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# ===== 配置 =====
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")         # 可留空，本地仅打印
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))   # 轮询间隔(秒)
ATR_MULT = float(os.getenv("ATR_MULT", "1.5"))
RSI_THRESHOLD = 5
WR_THRESHOLD  = 5
VOL_REL_THRESHOLD = 0.20
MAX_POS_USDT = float(os.getenv("MAX_POS_USDT","300"))   # 单币最大名义
COOLDOWN_MIN = int(os.getenv("COOLDOWN_MIN","15"))      # 触发后冷却(分钟)
MODEL_PATH   = os.getenv("MODEL_PATH","model.pkl")      # 可选 AI 模型

SYMBOLS = ["BTCUSDT","ETHUSDT","LTCUSDT"]
# 周期映射（用 binance 免费行情做示例；换交易所只改获取函数）
PERIODS = {"1h":"1h", "4h":"4h", "1d":"1d", "1w":"1w"}   # 监控 1h/4h/1d/1w

# ===== 简易工具 =====
def log(msg):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")

def fmt(p):
    try:
        p = float(p)
        if p >= 100: return f"{p:.2f}"
        if p >= 1:   return f"{p:.4f}"
        if p >= 0.01:return f"{p:.6f}"
        return f"{p:.8f}"
    except: return "-"

def tg(text):
    if not TOKEN or not CHAT_ID:
        log("TG未配置，打印代替：\n"+text)
        return
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": text})
        if r.status_code == 200: log("✅ TG已发")
        else: log(f"⚠️ TG错误 {r.status_code}: {r.text}")
    except Exception as e:
        log(f"❌ TG异常: {e}")

# ===== 行情（示例：Binance 公共K线）=====
def get_binance_klines(symbol, interval="1h", limit=300):
    url = "https://api.binance.com/api/v3/klines"
    sym = symbol.upper()
    try:
        r = requests.get(url, params={"symbol": sym, "interval": interval, "limit": limit}, timeout=10)
        j = r.json()
        if not isinstance(j, list): return None
        df = pd.DataFrame(j, columns=[
            "open_time","open","high","low","close","volume","close_time","q1","q2","q3","q4","q5"
        ])
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)
        df = df[["open","high","low","close","volume"]]
        return df
    except Exception as e:
        log(f"[行情ERR]{symbol} {interval} {e}")
        return None

# ===== 指标 + 特征 =====
def compute_indicators(df: pd.DataFrame):
    if df is None or len(df) < 80: return None
    work = df.copy().iloc[:-1]  # 舍弃未收K
    close, high, low, vol = work["close"], work["high"], work["low"], work["volume"]

    ema5  = ta.ema(close, length=5).iloc[-1]
    ema10 = ta.ema(close, length=10).iloc[-1]
    ema30 = ta.ema(close, length=30).iloc[-1]
    ma20  = ta.sma(close, length=20).iloc[-1]

    macd = ta.macd(close)
    macd_hist = macd["MACDh_12_26_9"].iloc[-1]

    rsi = ta.rsi(close, length=14).iloc[-1]
    wr  = ta.willr(high, low, close, length=14).iloc[-1]

    stoch = ta.stoch(high, low, close, k=9, d=3, smooth_k=3)
    k_val = stoch["STOCHk_9_3_3"].iloc[-1]
    d_val = stoch["STOCHd_9_3_3"].iloc[-1]

    vol_trend = (vol.iloc[-1] - vol.iloc[-2]) / (vol.iloc[-2] + 1e-12)

    ema_trend = "多" if (ema5>ema10>ema30) else ("空" if (ema5<ema10<ema30) else "中性")
    k_trend = "多" if k_val>d_val else ("空" if k_val<d_val else "中性")

    atr = ta.atr(high, low, close, length=14).iloc[-1]
    entry = close.iloc[-1]

    feats = {
        "ema5":float(ema5),"ema10":float(ema10),"ema30":float(ema30),"ma20":float(ma20),
        "macd_hist":float(macd_hist),"rsi":float(rsi),"wr":float(wr),
        "k":float(k_val),"d":float(d_val),"vol_chg":float(vol_trend),
        "ema_trend":ema_trend,"k_trend":k_trend,"atr":float(atr),"entry":float(entry)
    }
    return feats

def atr_stop_target(side, entry, atr, mult=ATR_MULT):
    if atr is None or np.isnan(atr): return None, None
    if side=="多":
        return entry - mult*atr, entry + mult*atr
    else:
        return entry + mult*atr, entry - mult*atr

# ======= AI/规则判定 =======
def rule_all_strong_long(f):
    ok = []
    ok.append(f["ema_trend"]=="多")
    ok.append(f["ma20"]<f["entry"] and f["ema10"]>f["ema30"])
    ok.append(f["macd_hist"]>0)
    ok.append(f["vol_chg"]>0)             # 成交量放大
    ok.append(f["k"]>f["d"])              # KDJ 同向
    ok.append(f["rsi"]>=50)               # RSI 偏多
    ok.append(f["wr"]>-50)                # WR 上半区
    return all(ok)

def rule_all_strong_short(f):
    ok = []
    ok.append(f["ema_trend"]=="空")
    ok.append(f["ma20"]>f["entry"] and f["ema10"]<f["ema30"])
    ok.append(f["macd_hist"]<0)
    ok.append(f["vol_chg"]<0)
    ok.append(f["k"]<f["d"])
    ok.append(f["rsi"]<=50)
    ok.append(f["wr"]<-50)
    return all(ok)

# 可选：AI 模型（如果 model.pkl 存在，用它做概率打分；否则用规则）
clf = None
if os.path.exists(MODEL_PATH):
    try:
        clf = joblib.load(MODEL_PATH)
        log("🎯 已加载AI模型")
    except Exception as e:
        log(f"AI模型加载失败：{e}")
        clf = None

def ai_direction_score(feature_dict):
    # 返回 (p_long, p_short)
    if clf is None:
        # 规则回退：给个简单分
        p_long  = 1.0 if rule_all_strong_long(feature_dict) else 0.3
        p_short = 1.0 if rule_all_strong_short(feature_dict) else 0.3
        return p_long, p_short
    cols = ["ema5","ema10","ema30","ma20","macd_hist","rsi","wr","k","d","vol_chg"]
    x = np.array([[feature_dict[c] for c in cols]], dtype=float)
    proba = clf.predict_proba(x)[0]  # 假设 0=空仓/1=做多/2=做空（训练时自己定义）
    # 如果你训练的是二分类，按你的顺序调整
    p_long  = float(proba[1]) if len(proba)>=2 else 0.5
    p_short = float(proba[2]) if len(proba)>=3 else 0.5
    return p_long, p_short

# ===== 执行层：先用 PaperBroker，接口与实盘一致 =====
class PaperBroker:
    def __init__(self):
        self.pos = {}  # symbol -> {"side":"多/空", "qty": float, "entry": float}
        self.last_ts = {}  # 冷却
    def can_trade(self, symbol):
        t = self.last_ts.get(symbol)
        return (t is None) or (datetime.now()-t >= timedelta(minutes=COOLDOWN_MIN))
    def size_by_usdt(self, symbol, price):
        # 极简：名义=MAX_POS_USDT，合约张数按名义/price
        qty = MAX_POS_USDT / max(price,1e-9)
        return round(qty, 4)
    def open(self, symbol, side, price, stop, target):
        if not self.can_trade(symbol):
            return False, "冷却中"
        qty = self.size_by_usdt(symbol, price)
        self.pos[symbol] = {"side":side, "qty":qty, "entry":price, "stop":stop, "target":target}
        self.last_ts[symbol] = datetime.now()
        return True, f"OPEN {symbol} {side} qty={qty} entry={fmt(price)} SL={fmt(stop)} TP={fmt(target)}"
    def close(self, symbol, reason):
        if symbol in self.pos:
            p = self.pos.pop(symbol)
            return True, f"CLOSE {symbol} {p['side']} qty={p['qty']} ({reason})"
        return False, "NO POS"
    # 火币实盘时：实现 place_order/cancel/get_position 等方法，保持同名签名即可

BROKER = PaperBroker()

# ===== 监控主循环 =====
last_hour_report = None
strong_last_sent = {}  # 记录上次突发侧别，避免刷屏
rolling_cache = {s:{k:deque(maxlen=1) for k in PERIODS} for s in SYMBOLS}

def color_tag(cons_ok_count):
    if cons_ok_count>=3: return "🟩(强)"
    if cons_ok_count==2: return "🟨(中)"
    return "🟥(弱)"

def period_check(symbol, period):
    df = get_binance_klines(symbol, interval=PERIODS[period], limit=240)
    feats = compute_indicators(df)
    return feats

def format_signal_block(symbol, side, entry, stop, target, ok_cnt):
    head = "⚠️ {} 做{}信号".format(symbol, "多" if side=="多" else "空")
    return (f"{head}\n入场: {fmt(entry)}\n目标: {fmt(target)}\n止损: {fmt(stop)}\n\n"
            f"⚡ 一致性: {ok_cnt}/3 周期 {color_tag(ok_cnt)}")

def try_fire_breaking(symbol, multi_feats, final_side):
    # 全部条件满足 → 突发
    f1, f4, fD = multi_feats["1h"], multi_feats["4h"], multi_feats["1d"]
    checks = []
    if final_side=="多":
        checks += [rule_all_strong_long(f1), rule_all_strong_long(f4), rule_all_strong_long(fD)]
    else:
        checks += [rule_all_strong_short(f1), rule_all_strong_short(f4), rule_all_strong_short(fD)]

    if all(checks):
        if strong_last_sent.get(symbol)!=final_side:
            strong_last_sent[symbol]=final_side
            stop, target = atr_stop_target(final_side, f1["entry"], f1["atr"], ATR_MULT)
            msg = (f"🔥🔥🔥 强烈高度动向捕捉到（满足所有条件）\n"
                   f"⚠️ {symbol} 做{final_side}信号\n入场: {fmt(f1['entry'])}\n目标: {fmt(target)}\n止损: {fmt(stop)}\n"
                   f"⚡ 一致性: 3/3 周期")
            tg(msg)

def hourly_report(all_results):
    lines = ["📢 每小时回报（按 1h/4h/1d/1w 指标 & 一致性分级）"]
    for sym, res in all_results.items():
        ok_cnt = res["ok_cnt"]
        side   = res["side"]
        entry  = res["entry"]; stop=res["stop"]; target=res["target"]
        lines.append(format_signal_block(sym, side, entry, stop, target, ok_cnt))
        # 把关键指标也列一下（你要的 EMA/MACD/RSI/WR/VOL）
        for p in ["1h","4h","1d","1w"]:
            f = res["feats"][p]
            lines.append(f"{sym} {p} | EMA:{fmt(f['ema5'])}/{fmt(f['ema10'])}/{fmt(f['ema30'])} "
                        f"MACDhist:{fmt(f['macd_hist'])} RSI:{fmt(f['rsi'])} WR:{fmt(f['wr'])} VOLΔ:{fmt(f['vol_chg'])}")
        lines.append("-"*20)
    tg("\n".join(lines))

log(f"启动 AI 自动合约监控（每{POLL_INTERVAL}s 轮询；每小时汇总）")
while True:
    try:
        now = datetime.now()
        summary = {}

        for sym in SYMBOLS:
            feats_multi = {}
            # 多周期抓取
            for p in ["1h","4h","1d","1w"]:
                f = period_check(sym, p)
                if f is None: break
                feats_multi[p] = f
            if len(feats_multi)!=4:
                log(f"{sym} 周期数据不足，跳过")
                continue

            # 以 1h 为入场，4h/1d 做一致性，1w 参考权重
            f1,f4,fD,fW = feats_multi["1h"], feats_multi["4h"], feats_multi["1d"], feats_multi["1w"]

            # AI 概率（可选）
            pL1,pS1 = ai_direction_score(f1)
            # 简单一致性：三周期同向计数
            dir_1h = "多" if pL1>pS1 else "空"
            dir_4h = "多" if f4["ema_trend"]=="多" else ("空" if f4["ema_trend"]=="空" else "中性")
            dir_1d = "多" if fD["ema_trend"]=="多" else ("空" if fD["ema_trend"]=="空" else "中性")
            consensus = [dir_1h, dir_4h, dir_1d]
            side = "多" if consensus.count("多")>=2 else ("空" if consensus.count("空")>=2 else ("多" if dir_1h=="多" else "空"))
            ok_cnt = max(consensus.count("多"), consensus.count("空"))

            stop, target = atr_stop_target(side, f1["entry"], f1["atr"], ATR_MULT)

            # 突发（所有条件都满足）
            try_fire_breaking(sym, feats_multi, side)

            # 如需自动开仓（纸交易）
            if BROKER.can_trade(sym):
                opened, info = BROKER.open(sym, side, f1["entry"], stop, target)
                log(info)

            summary[sym] = {
                "side": side, "ok_cnt": ok_cnt, "entry": f1["entry"],
                "stop": stop, "target": target, "feats": feats_multi
            }

        # 每小时汇总（你要的“每小时才回报”）
        if (last_hour_report is None) or (now - last_hour_report >= timedelta(hours=1)):
            if summary:
                hourly_report(summary)
            last_hour_report = now

        time.sleep(POLL_INTERVAL)

    except Exception as e:
        log(f"[LOOP ERR] {e}")
        time.sleep(5)
