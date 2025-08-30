# trading_bot.py
"""
多周期共振策略 - 回测 + 实盘
"""

import os
import time
import ccxt
import pandas as pd
import ta
from datetime import datetime

# ================== 配置 ==================
MODE = os.getenv("MODE", "backtest")  # "backtest" / "live"
HEDGE_MODE = os.getenv("HEDGE_MODE", "false").lower() == "true"  # 是否开启双向持仓

# 从 ENV 读取交易对，默认用 BTC/USDT
SYMBOLS = os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT").split(",")
SYMBOLS = [s.strip() for s in SYMBOLS if s.strip()]
SYMBOL = SYMBOLS[0]  # 回测只跑第一个

TIMEFRAME = "1h"
HIGHER_TIMEFRAME = "4h"
LEVERAGE = 10
RISK_RATIO = 0.15
TP_ATR_MULT = 3.0
SL_ATR_MULT = 2.0
INITIAL_BALANCE = 1000  # 回测资金
BASE_USDT = 120  # 实盘资金
SLEEP_INTERVAL = 60  # 实盘循环等待时间

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

# ================== 初始化交易所 ==================
exchange = None
if MODE == "live":
    exchange = ccxt.binance({
        "apiKey": BINANCE_API_KEY,
        "secret": BINANCE_API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "future"}
    })
    exchange.load_markets()  # 避免 markets not loaded 错误

    # 设置双向/单向持仓模式
    try:
        if HEDGE_MODE:
            exchange.fapiPrivate_post_positionside_dual({"dualSidePosition": "true"})
            print("✅ 已切换为双向持仓模式 (HEDGE_MODE)")
        else:
            exchange.fapiPrivate_post_positionside_dual({"dualSidePosition": "false"})
            print("ℹ️ 使用单向持仓模式")
    except Exception as e:
        print(f"⚠️ 持仓模式设置失败: {e}")

# ================== 技术指标 ==================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=12).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=26).ema_indicator()
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["atr"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14
    ).average_true_range()
    df["vol_ma"] = df["volume"].rolling(window=20).mean()
    return df.dropna()

def signal_from_indicators(df_1h: pd.DataFrame, df_4h: pd.DataFrame):
    """1h 给信号，4h 做确认"""
    latest_1h = df_1h.iloc[-1]
    latest_4h = df_4h.iloc[-1]

    if latest_1h["volume"] < latest_1h["vol_ma"]:
        return "hold"

    if (
        latest_1h["macd"] > latest_1h["macd_signal"]
        and latest_1h["ema_fast"] > latest_1h["ema_slow"]
        and latest_1h["rsi"] > 50
    ):
        signal_1h = "buy"
    elif (
        latest_1h["macd"] < latest_1h["macd_signal"]
        and latest_1h["ema_fast"] < latest_1h["ema_slow"]
        and latest_1h["rsi"] < 50
    ):
        signal_1h = "sell"
    else:
        signal_1h = "hold"

    trend_4h = "buy" if latest_4h["ema_fast"] > latest_4h["ema_slow"] else "sell"

    if signal_1h == "buy" and trend_4h == "buy":
        return "buy"
    elif signal_1h == "sell" and trend_4h == "sell":
        return "sell"
    else:
        return "hold"

# ================== 回测账户类 ==================
class BacktestAccount:
    def __init__(self, initial_balance):
        self.balance = float(initial_balance)
        self.position = None
        self.trade_history = []

    def place_order(self, side, qty, price, atr, timestamp):
        if self.position:
            return
        cost = (qty * price) / LEVERAGE
        self.balance -= cost
        tp_price = price + TP_ATR_MULT * atr if side == "buy" else price - TP_ATR_MULT * atr
        sl_price = price - SL_ATR_MULT * atr if side == "buy" else price + SL_ATR_MULT * atr
        self.position = {"side": "long" if side == "buy" else "short",
                         "qty": qty, "entry": price, "tp": tp_price, "sl": sl_price}
        self.trade_history.append({"time": timestamp, "type": "Open", "side": side, "qty": qty, "price": price})

    def close_position(self, price, timestamp, reason="Signal"):
        if not self.position:
            return
        pos = self.position
        pnl = (price - pos["entry"]) * pos["qty"]
        if pos["side"] == "short":
            pnl *= -1
        self.balance += (pos["qty"] * pos["entry"] / LEVERAGE) + pnl
        self.trade_history.append({"time": timestamp, "type": "Close", "reason": reason,
                                   "side": pos["side"], "price": price, "pnl": pnl})
        self.position = None

    def check_tp_sl(self, high, low):
        if not self.position:
            return None, None
        pos = self.position
        if pos["side"] == "long":
            if high >= pos["tp"]: return pos["tp"], "TP"
            if low <= pos["sl"]: return pos["sl"], "SL"
        else:
            if low <= pos["tp"]: return pos["tp"], "TP"
            if high >= pos["sl"]: return pos["sl"], "SL"
        return None, None

# ================== 工具函数 ==================
def get_historical_data(symbol, timeframe="1h", limit=1000):
    ex = exchange if exchange else ccxt.binance()
    ex.load_markets()
    ohlcvs = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcvs, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df.set_index("time", inplace=True)
    return df

def calculate_position_size(balance, price):
    return (balance * RISK_RATIO * LEVERAGE) / price

def live_place_order(symbol, side, qty, price, atr):
    try:
        order = exchange.create_order(symbol, "MARKET", side.upper(), qty)
        if side == "buy":
            tp_price = price + TP_ATR_MULT * atr
            sl_price = price - SL_ATR_MULT * atr
            pos_side = "LONG"
        else:
            tp_price = price - TP_ATR_MULT * atr
            sl_price = price + SL_ATR_MULT * atr
            pos_side = "SHORT"

        exchange.create_order(symbol, "TAKE_PROFIT_MARKET",
                              "SELL" if side == "buy" else "BUY", qty,
                              params={"stopPrice": tp_price, "reduceOnly": True, "positionSide": pos_side})
        exchange.create_order(symbol, "STOP_MARKET",
                              "SELL" if side == "buy" else "BUY", qty,
                              params={"stopPrice": sl_price, "reduceOnly": True, "positionSide": pos_side})
        print(f"✅ 实盘下单 {side.upper()} {symbol} qty={qty} @ {price:.2f}")
    except Exception as e:
        print(f"❌ 下单失败 {symbol}: {e}")

# ================== 实盘 ==================
def run_live():
    print("🚀 启动实盘交易...")
    while True:
        for symbol in SYMBOLS:
            try:
                df_1h = compute_indicators(get_historical_data(symbol, TIMEFRAME, limit=200))
                df_4h = compute_indicators(get_historical_data(symbol, HIGHER_TIMEFRAME, limit=200))
                if df_1h.empty or df_4h.empty:
                    print(f"⚠️ 无法获取 {symbol} 的足够历史数据，跳过。")
                    continue

                signal = signal_from_indicators(df_1h, df_4h)
                price = df_1h["close"].iloc[-1]
                market = exchange.market(symbol)
                min_amount = market['limits']['amount']['min']

                qty = (BASE_USDT / len(SYMBOLS)) * RISK_RATIO * LEVERAGE / price
                if qty < min_amount:
                    print(f"❌ 下单失败 {symbol}: 数量 {qty} < 最小交易量 {min_amount}")
                    continue

                qty = float(exchange.amount_to_precision(symbol, qty))
                atr = df_1h["atr"].iloc[-1]

                if signal in ["buy", "sell"] and qty > 0:
                    live_place_order(symbol, signal, qty, price, atr)

            except Exception as e:
                print(f"主循环异常 {symbol}: {e}")

        time.sleep(SLEEP_INTERVAL)

# ================== 回测 ==================
def run_backtest():
    print("🤖 启动多周期回测...")
    df_1h = compute_indicators(get_historical_data(SYMBOL, TIMEFRAME, limit=1000))
    df_4h = compute_indicators(get_historical_data(SYMBOL, HIGHER_TIMEFRAME, limit=1000))
    account = BacktestAccount(INITIAL_BALANCE)

    for i in range(len(df_1h)):
        current_df_1h = df_1h.iloc[: i + 1]
        if len(current_df_1h) < 50:
            continue
        price = current_df_1h["close"].iloc[-1]
        atr = current_df_1h["atr"].iloc[-1]
        timestamp = current_df_1h.index[-1]
        current_df_4h = df_4h[df_4h.index <= timestamp]
        if current_df_4h.empty:
            continue

        if account.position:
            tp_sl_price, reason = account.check_tp_sl(
                current_df_1h["high"].iloc[-1], current_df_1h["low"].iloc[-1]
            )
            if tp_sl_price:
                account.close_position(tp_sl_price, timestamp, reason)
                continue

        signal = signal_from_indicators(current_df_1h, current_df_4h)
        if signal == "buy":
            if not account.position or account.position["side"] == "short":
                if account.position:
                    account.close_position(price, timestamp, "Reverse")
                qty = calculate_position_size(account.balance, price)
                account.place_order("buy", qty, price, atr, timestamp)
        elif signal == "sell":
            if not account.position or account.position["side"] == "long":
                if account.position:
                    account.close_position(price, timestamp, "Reverse")
                qty = calculate_position_size(account.balance, price)
                account.place_order("sell", qty, price, atr, timestamp)

    if account.position:
        last_price = df_1h["close"].iloc[-1]
        last_time = df_1h.index[-1]
        account.close_position(last_price, last_time, "Final")

    trade_df = pd.DataFrame(account.trade_history)
    print("\n--- 回测结果 ---")
    print(f"初始资金: {INITIAL_BALANCE:.2f}")
    print(f"最终资金: {account.balance:.2f}")
    closes = trade_df[trade_df["type"] == "Close"]
    if not closes.empty:
        total_trades = len(closes)
        win_trades = (closes["pnl"] > 0).sum()
        win_rate = win_trades / total_trades * 100
        print(f"交易次数: {total_trades}, 胜率: {win_rate:.2f}%")
        print(f"总盈亏: {closes['pnl'].sum():.2f}")

# ================== 主入口 ==================
if __name__ == "__main__":
    if MODE == "backtest":
        run_backtest()
    else:
        run_live()
