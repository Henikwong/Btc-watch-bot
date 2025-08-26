# app/autotrader.py
import os
import time
import ccxt
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import ta

# ===========================
# 配置（可通过环境变量覆盖）
# ===========================
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT").split(",") if s.strip()]
BASE_USDT = float(os.getenv("BASE_USDT", "100"))
LEVERAGE = int(os.getenv("LEVERAGE", "10"))
LIVE_TRADE = os.getenv("LIVE_TRADE", "0") == "1"
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))  # 每分钟默认 60 秒
ONLY_RESONANCE = os.getenv("ONLY_RESONANCE", "1") == "1"  # 是否仅在多周期共振时下单

TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "3.0"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "2.0"))

OHLCV_LIMIT = int(os.getenv("OHLCV_LIMIT", "200"))

# 报表时间（北京时间触发）
DAILY_REPORT_HOUR = int(os.getenv("DAILY_REPORT_HOUR", "8"))   # 每日 8:00
WEEKLY_REPORT_HOUR = int(os.getenv("WEEKLY_REPORT_HOUR", "20"))# 每周日 20:00
MONTHLY_REPORT_DAY = int(os.getenv("MONTHLY_REPORT_DAY", "1")) # 每月 1 号
MONTHLY_REPORT_HOUR = int(os.getenv("MONTHLY_REPORT_HOUR", "9"))

# ===========================
# 帮助函数
# ===========================
def now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def send_telegram(msg: str):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("⚠️ 未配置 Telegram 环境变量，跳过推送:", msg)
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": msg})
    except Exception as e:
        print("❌ Telegram 推送失败:", e)

# ===========================
# 初始化交易所（Binance Futures）
# ===========================
exchange = ccxt.binance({
    "apiKey": os.getenv("API_KEY"),
    "secret": os.getenv("API_SECRET"),
    "enableRateLimit": True,
    "options": {"defaultType": "future"}
})

def setup_account(symbol):
    """尝试设置杠杆/保证金模式，容错处理"""
    try:
        m = exchange.market(symbol)
        ex_symbol = m["id"]
        # 下面两个接口可能在 ccxt/bindings 中支持，也可能抛错，全部包在 try/except 中
        try:
            exchange.fapiPrivate_post_leverage({"symbol": ex_symbol, "leverage": LEVERAGE})
            exchange.fapiPrivate_post_margintype({"symbol": ex_symbol, "marginType": "ISOLATED"})
            print(f"✅ 已设置 {symbol} 杠杆与保证金模式")
        except Exception as e:
            print("⚠️ 设置杠杆/保证金失败（可忽略）:", e)
    except Exception as e:
        print("⚠️ setup_account 失败:", e)

# ===========================
# OHLCV 与指标
# ===========================
def fetch_ohlcv_df(symbol, timeframe="1h", limit=OHLCV_LIMIT):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    # 确保数值类型
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema20"] = ta.trend.EMAIndicator(df["close"], window=20, fillna=True).ema_indicator()
    df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50, fillna=True).ema_indicator()
    macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9, fillna=True)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14, fillna=True).rsi()
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14, fillna=True).average_true_range()
    df["vol_ma20"] = df["volume"].rolling(20, min_periods=1).mean()
    # Stochastic for KDJ might be added if needed
    return df

def signal_from_indicators(df: pd.DataFrame):
    """返回 (signal, score, reasons, last_row)
       signal: '买入' / '卖出' / None
    """
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last
    score = 0
    reasons = []

    # EMA
    if last["ema20"] > last["ema50"]:
        score += 2; reasons.append("EMA 多头")
    else:
        score -= 2; reasons.append("EMA 空头")

    # MACD
    if last["macd"] > last["macd_signal"]:
        score += 1; reasons.append("MACD 多头")
    else:
        score -= 1; reasons.append("MACD 空头")

    # RSI
    if last["rsi"] > 60:
        score += 1; reasons.append(f"RSI 偏强 {last['rsi']:.1f}")
    elif last["rsi"] < 40:
        score -= 1; reasons.append(f"RSI 偏弱 {last['rsi']:.1f}")

    # Volume spike
    if "vol_ma20" in df.columns and last["volume"] > last["vol_ma20"] * 1.5:
        score += 1; reasons.append("成交量放大")

    if score >= 3:
        return "买入", score, reasons, last
    elif score <= -3:
        return "卖出", score, reasons, last
    else:
        return None, score, reasons, last

# ===========================
# 仓位、平仓相关（兼容多种 ccxt 返回结构）
# ===========================
def fetch_all_positions():
    """返回 exchange.fetch_positions() 的结果（若不可用返回空）"""
    try:
        pos = exchange.fetch_positions()
        return pos if isinstance(pos, list) else []
    except Exception as e:
        # 有些 ccxt 版本/交易所不支持 fetch_positions
        print("⚠️ fetch_positions 不可用:", e)
        return []

def parse_position_entry(pos):
    """从 pos dict 解析 symbol, contracts, side, entryPrice
       返回 (symbol, contracts, side, entryPrice) 或 None
    """
    try:
        # 不同实现差异：尝试多种字段
        symbol = pos.get("symbol") or (pos.get("info") or {}).get("symbol")
        # 合约数量：contracts, positionAmt, amount
        contracts = None
        if "contracts" in pos:
            contracts = float(pos["contracts"])
        elif "positionAmt" in pos:
            contracts = float(pos["positionAmt"])
        elif "amount" in pos:
            contracts = float(pos["amount"])
        else:
            # try info
            info = pos.get("info", {})
            if "positionAmt" in info:
                contracts = float(info["positionAmt"])
        if contracts is None or contracts == 0:
            return None
        # side：
        side = None
        if "side" in pos and pos["side"]:
            side = pos["side"]  # 'long' / 'short'
        else:
            # positionAmt 正负判断
            if "positionAmt" in pos:
                amt = float(pos["positionAmt"])
                side = "long" if amt > 0 else "short"
            elif contracts > 0:
                # fallback: check pos.info.side or check entryPrice
                side = pos.get("side") or (pos.get("info") or {}).get("positionSide") or "long"
        entry = pos.get("entryPrice") or (pos.get("info") or {}).get("entryPrice") or None
        return (symbol, abs(contracts), side, float(entry) if entry else None)
    except Exception as e:
        print("⚠️ parse_position_entry 失败:", e)
        return None

def get_position(symbol):
    """尝试返回指定 symbol 的当前仓位 dict (解析后): 
       {'symbol':..., 'qty':..., 'side': 'long'/'short', 'entry':...}
       若无仓位返回 None
    """
    positions = fetch_all_positions()
    for p in positions:
        parsed = parse_position_entry(p)
        if not parsed:
            continue
        sym, qty, side, entry = parsed
        # match symbol names (e.g. 'BTC/USDT' vs 'BTCUSDT')
        if not sym:
            continue
        # normalize
        if sym.replace("/", "") == symbol.replace("/", "") or sym == symbol:
            return {"symbol": symbol, "qty": qty, "side": side, "entry": entry, "raw": p}
    return None

def close_position(symbol, position):
    """市价平掉给定的 position（会尽量解析数量），并发送通知"""
    try:
        qty = position.get("qty")
        if qty is None or qty == 0:
            send_telegram(f"❌ 平仓失败 {symbol}：无法解析仓位数量")
            return False
        # side: 若现有仓位为 long => 平仓需 sell；若 short => 需 buy
        pos_side = position.get("side", "")
        if pos_side and pos_side.lower().startswith("short"):
            side = "buy"
        else:
            side = "sell"
        # 下市价平仓
        if LIVE_TRADE:
            # 尝试精度化
            try:
                qty_precise = float(exchange.amount_to_precision(symbol, qty))
            except Exception:
                qty_precise = round(qty, 6)
            exchange.create_market_order(symbol, side, qty_precise)
            send_telegram(f"✅ 已市价平仓 {symbol} {pos_side} 数量={qty_precise}")
        else:
            send_telegram(f"📌 模拟平仓 {symbol} {pos_side} 数量={qty}")
        return True
    except Exception as e:
        send_telegram(f"❌ 平仓失败 {symbol}，原因: {e}")
        return False

# ===========================
# 下单（带止盈止损，中文推送）
# ===========================
def place_order(symbol, side_text, price, atr):
    """side_text: '买入' 或 '卖出'"""
    side = "buy" if side_text == "买入" else "sell"
    try:
        qty = BASE_USDT * LEVERAGE / price
        try:
            qty = float(exchange.amount_to_precision(symbol, qty))
        except Exception:
            qty = round(qty, 6)
    except Exception as e:
        send_telegram(f"❌ 计算下单数量失败 {symbol}：{e}")
        return

    if not LIVE_TRADE:
        send_telegram(f"📌 模拟下单 {symbol} {side_text} 数量={qty} @ {price:.2f}")
        return

    try:
        # 开仓市价单
        exchange.create_market_order(symbol, side, qty)
        # 计算止损、止盈
        if atr is None or np.isnan(atr):
            atr = price * 0.005
        if side == "buy":
            stop_loss = price - SL_ATR_MULT * atr
            take_profit = price + TP_ATR_MULT * atr
            close_side = "sell"
        else:
            stop_loss = price + SL_ATR_MULT * atr
            take_profit = price - TP_ATR_MULT * atr
            close_side = "buy"
        # 尝试挂单（交易所可能需要额外参数）
        try:
            exchange.create_order(symbol, "STOP_MARKET", close_side, qty, None, {"stopPrice": stop_loss})
            exchange.create_order(symbol, "TAKE_PROFIT_MARKET", close_side, qty, None, {"stopPrice": take_profit})
            send_telegram(
                f"✅ 已下单 {symbol} {side_text} 数量={qty} @ {price:.2f}\n🎯 止盈: {take_profit:.2f}\n🛡 止损: {stop_loss:.2f}"
            )
        except Exception as e:
            send_telegram(f"✅ 已下单 {symbol} {side_text} 数量={qty} @ {price:.2f}\n⚠️ 挂止盈/止损失败: {e}")
    except Exception as e:
        send_telegram(f"❌ 下单失败 {symbol}，原因: {e}")

# ===========================
# 趋势检测 & 启动汇报（含共振）
# ===========================
def check_trend_once(symbol):
    """检查 4h/1d EMA 金叉/死叉 以及返回当前方向"""
    result = {"alerts": [], "status": {}}
    for tf in ["4h", "1d"]:
        try:
            df = fetch_ohlcv_df(symbol, timeframe=tf, limit=100)
            df = compute_indicators(df)
            last = df.iloc[-1]; prev = df.iloc[-2]
            # status
            status = "多头" if last["ema20"] > last["ema50"] else "空头"
            result["status"][tf] = status
            # cross detection
            if last["ema20"] > last["ema50"] and prev["ema20"] <= prev["ema50"]:
                result["alerts"].append(f"⚡ 趋势提醒: {symbol} {tf} 出现金叉 → 趋势看多")
            elif last["ema20"] < last["ema50"] and prev["ema20"] >= prev["ema50"]:
                result["alerts"].append(f"⚡ 趋势提醒: {symbol} {tf} 出现死叉 → 趋势转空")
        except Exception as e:
            result["alerts"].append(f"❌ 趋势检测失败 {symbol} {tf}: {e}")
    # resonance
    if result["status"].get("4h") and result["status"].get("1d") and result["status"]["4h"] == result["status"]["1d"]:
        result["alerts"].append(f"🔥 趋势共振: {symbol} ({result['status']['4h']})")
    return result

def startup_trend_report():
    report = ["📌 启动时趋势检测:"]
    for symbol in SYMBOLS:
        r = check_trend_once(symbol)
        st4 = r["status"].get("4h", "未知"); st1 = r["status"].get("1d", "未知")
        report.append(f"{symbol} 4h:{st4} | 1d:{st1}")
        for a in r["alerts"]:
            report.append(a)
    send_telegram("\n".join(report))

# ===========================
# 报表：日/周/月
# ===========================
def daily_summary():
    try:
        balance = exchange.fetch_balance(params={"type": "future"})
        usdt = balance.get("total", {}).get("USDT") or balance.get("USDT") or 0
        # trades since yesterday
        since = int((datetime.now(timezone.utc) - timedelta(days=1)).timestamp() * 1000)
        total_pnl = 0
        trades_list = []
        for sym in SYMBOLS:
            try:
                t = exchange.fetch_my_trades(sym, since=since)
                for x in t:
                    pnl = float(x.get("realizedPnl", 0) or 0)
                    total_pnl += pnl
                    trades_list.append(f"{sym} {x.get('side')} {x.get('amount')} @ {x.get('price')} PnL={pnl:.2f}")
            except Exception:
                pass
        positions = fetch_all_positions()
        active = []
        for p in positions:
            parsed = parse_position_entry(p)
            if parsed:
                active.append(f"{parsed[0]} {parsed[2]} {parsed[1]} 张 @ {parsed[3]}")
        report = [
            "📊 每日总结",
            f"账户余额(USDT): {usdt:.2f}",
            f"昨日盈亏: {total_pnl:.2f} USDT",
            "",
            "昨日成交（最多10条）:",
            "\n".join(trades_list[-10:]) if trades_list else "无",
            "",
            "当前持仓:",
            "\n".join(active) if active else "无"
        ]
        send_telegram("\n".join(report))
    except Exception as e:
        send_telegram(f"❌ 每日总结生成失败: {e}")

def weekly_summary():
    try:
        balance = exchange.fetch_balance(params={"type": "future"})
        usdt = balance.get("total", {}).get("USDT") or balance.get("USDT") or 0
        since = int((datetime.now(timezone.utc) - timedelta(days=7)).timestamp() * 1000)
        total_pnl = 0
        pnl_per_symbol = {}
        trades_count = 0
        for sym in SYMBOLS:
            try:
                t = exchange.fetch_my_trades(sym, since=since)
                trades_count += len(t)
                for x in t:
                    pnl = float(x.get("realizedPnl", 0) or 0)
                    total_pnl += pnl
                    pnl_per_symbol[sym] = pnl_per_symbol.get(sym, 0) + pnl
            except Exception:
                pass
        best = max(pnl_per_symbol.items(), key=lambda x: x[1]) if pnl_per_symbol else ("无", 0)
        worst = min(pnl_per_symbol.items(), key=lambda x: x[1]) if pnl_per_symbol else ("无", 0)
        positions = fetch_all_positions()
        active = []
        for p in positions:
            parsed = parse_position_entry(p)
            if parsed:
                active.append(f"{parsed[0]} {parsed[2]} {parsed[1]} 张 @ {parsed[3]}")
        report = [
            "📅 周总结",
            f"账户余额(USDT): {usdt:.2f}",
            f"本周盈亏: {total_pnl:.2f} USDT",
            f"交易次数: {trades_count}",
            f"盈利最多: {best[0]} ({best[1]:.2f})",
            f"亏损最多: {worst[0]} ({worst[1]:.2f})",
            "",
            "当前持仓:",
            "\n".join(active) if active else "无"
        ]
        send_telegram("\n".join(report))
    except Exception as e:
        send_telegram(f"❌ 周总结生成失败: {e}")

def monthly_summary():
    try:
        balance = exchange.fetch_balance(params={"type": "future"})
        usdt = balance.get("total", {}).get("USDT") or balance.get("USDT") or 0
        now_dt = datetime.now(timezone.utc) + timedelta(hours=8)  # 北京时间
        # 上个月第一天
        first_day_this = datetime(now_dt.year, now_dt.month, 1, tzinfo=timezone.utc)
        last_month_end = first_day_this - timedelta(seconds=1)
        first_day_last = datetime(last_month_end.year, last_month_end.month, 1, tzinfo=timezone.utc)
        since = int(first_day_last.timestamp() * 1000)
        until = int(last_month_end.timestamp() * 1000)
        total_pnl = 0
        pnl_per_symbol = {}
        trades_count = 0
        for sym in SYMBOLS:
            try:
                t = exchange.fetch_my_trades(sym, since=since)
                # filter until
                t = [x for x in t if x["timestamp"] <= until]
                trades_count += len(t)
                for x in t:
                    pnl = float(x.get("realizedPnl", 0) or 0)
                    total_pnl += pnl
                    pnl_per_symbol[sym] = pnl_per_symbol.get(sym, 0) + pnl
            except Exception:
                pass
        best = max(pnl_per_symbol.items(), key=lambda x: x[1]) if pnl_per_symbol else ("无", 0)
        worst = min(pnl_per_symbol.items(), key=lambda x: x[1]) if pnl_per_symbol else ("无", 0)
        positions = fetch_all_positions()
        active = []
        for p in positions:
            parsed = parse_position_entry(p)
            if parsed:
                active.append(f"{parsed[0]} {parsed[2]} {parsed[1]} 张 @ {parsed[3]}")
        report = [
            "📆 月总结",
            f"账户余额(USDT): {usdt:.2f}",
            f"上月盈亏: {total_pnl:.2f} USDT",
            f"交易次数: {trades_count}",
            f"盈利最多: {best[0]} ({best[1]:.2f})",
            f"亏损最多: {worst[0]} ({worst[1]:.2f})",
            "",
            "当前持仓:",
            "\n".join(active) if active else "无"
        ]
        send_telegram("\n".join(report))
    except Exception as e:
        send_telegram(f"❌ 月总结生成失败: {e}")

# ===========================
# 主循环
# ===========================
def main():
    # 启动时报告与账号设置
    send_telegram("🚀 自动交易机器人启动")
    for s in SYMBOLS:
        try:
            setup_account(s)
        except Exception:
            pass
    startup_trend_report()

    last_summary_time = 0
    last_daily = None
    last_week = None
    last_month = None

    while True:
        summary_lines = []
        for symbol in SYMBOLS:
            try:
                # multi-timeframe signals
                df1h = compute_indicators(fetch_ohlcv_df(symbol, timeframe="1h"))
                sig1h, sc1h, reasons1h, last1h = signal_from_indicators(df1h)

                df4h = compute_indicators(fetch_ohlcv_df(symbol, timeframe="4h"))
                sig4h, sc4h, reasons4h, last4h = signal_from_indicators(df4h)

                df1d = compute_indicators(fetch_ohlcv_df(symbol, timeframe="1d"))
                sig1d, sc1d, reasons1d, last1d = signal_from_indicators(df1d)

                price = float(last1h["close"])
                atr = float(last1h["atr"]) if not pd.isna(last1h["atr"]) else None

                summary_lines.append(f"{symbol} {price:.2f} 分数={sc1h} 信号={sig1h}")

                # 趋势检测（4h/1d 金叉死叉 + 共振）
                trend_res = check_trend_once(symbol)
                for a in trend_res["alerts"]:
                    send_telegram(a)

                # 决定是否下单
                consensus = False
                if sig1h:
                    if ONLY_RESONANCE:
                        # 需要与 4h 或 1d 同向
                        if (sig1h == sig4h) or (sig1h == sig1d):
                            consensus = True
                    else:
                        # 允许 1h 单独下单
                        consensus = True

                position = get_position(symbol)

                if position:
                    # 已有仓位：判断是否需要反向平仓并换仓
                    pos_side = "买入" if (position.get("side","").lower().startswith("long")) else "卖出"
                    if sig1h and consensus:
                        # 当前持仓与信号方向不一致 -> 平仓 + 开新仓
                        if pos_side != sig1h:
                            send_telegram(f"🔄 信号反转：{symbol} 当前持仓 {pos_side} → 新信号 {sig1h}，准备平仓并开新仓")
                            closed = close_position(symbol, position)
                            if closed:
                                place_order(symbol, sig1h, price, atr)
                        else:
                            # 一致 -> 不重复开仓
                            print(f"{symbol} 已有持仓且方向一致，跳过开新仓")
                    else:
                        # 无强信号或不满足共振 -> 不做任何操作（保守）
                        pass
                else:
                    # 无仓位：若有共识则开仓
                    if sig1h and consensus:
                        send_telegram(f"⚡ 新信号触发：{symbol} {sig1h}，准备下单（若为 LIVE_TRADE=1 将真实下单）")
                        place_order(symbol, sig1h, price, atr)

            except Exception as e:
                print(f"❌ {symbo
