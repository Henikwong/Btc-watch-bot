# backtester.py
"""
回测脚本 - 多周期共振 + 成交量确认
"""

import ccxt
import pandas as pd
import ta
from datetime import datetime

# ================== 配置 ==================
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
HIGHER_TIMEFRAME = "4h"  # 高级别周期
LEVERAGE = 10
RISK_RATIO = 0.15
TP_ATR_MULT = 3.0
SL_ATR_MULT = 2.0
INITIAL_BALANCE = 1000

# ================== 技术指标 ==================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=12).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=26).ema_indicator()
    df["macd"] = ta.trend.MACD(df["close"]).macd()
    df["macd_signal"] = ta.trend.MACD(df["close"]).macd_signal()
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

    # 成交量过滤：当前成交量必须大于过去20根均量
    if latest_1h["volume"] < latest_1h["vol_ma"]:
        return "hold"

    # 1h 信号
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

    # 4h 趋势过滤
    if latest_4h["ema_fast"] > latest_4h["ema_slow"]:
        trend_4h = "buy"
    else:
        trend_4h = "sell"

    # 共振确认
    if signal_1h == "buy" and trend_4h == "buy":
        return "buy"
    elif signal_1h == "sell" and trend_4h == "sell":
        return "sell"
    else:
        return "hold"

# ================== 账户模拟 ==================
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
        self.position = {
            "side": "long" if side == "buy" else "short",
            "qty": qty,
            "entry": price,
            "tp": tp_price,
            "sl": sl_price,
        }
        self.trade_history.append(
            {"time": timestamp, "type": "Open", "side": self.position["side"], "qty": qty, "entry_price": price}
        )

    def close_position(self, price, timestamp, reason="Signal"):
        if not self.position:
            return
        pos = self.position
        pnl = (price - pos["entry"]) * pos["qty"]
        if pos["side"] == "short":
            pnl *= -1
        self.balance += (pos["qty"] * pos["entry"] / LEVERAGE) + pnl
        self.trade_history.append(
            {"time": timestamp, "type": "Close", "reason": reason, "side": pos["side"], "qty": pos["qty"], "close_price": price, "pnl": pnl}
        )
        self.position = None

    def check_tp_sl(self, high, low):
        if not self.position:
            return None, None
        pos = self.position
        if pos["side"] == "long":
            if high >= pos["tp"]:
                return pos["tp"], "TP"
            if low <= pos["sl"]:
                return pos["sl"], "SL"
        else:
            if low <= pos["tp"]:
                return pos["tp"], "TP"
            if high >= pos["sl"]:
                return pos["sl"], "SL"
        return None, None

# ================== 工具函数 ==================
def get_historical_data(symbol, timeframe="1h", limit=1000):
    exchange = ccxt.binance()
    ohlcvs = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcvs, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df.set_index("time", inplace=True)
    return df

def calculate_position_size(balance, current_price):
    return (balance * RISK_RATIO * LEVERAGE) / current_price

# ================== 回测主函数 ==================
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
        # 对齐 4h 数据（用当前 1h 对应的 4h bar）
        current_df_4h = df_4h[df_4h.index <= timestamp]
        if current_df_4h.empty:
            continue
        # TP/SL 检查
        if account.position:
            tp_sl_price, reason = account.check_tp_sl(
                current_df_1h["high"].iloc[-1], current_df_1h["low"].iloc[-1]
            )
            if tp_sl_price:
                account.close_position(tp_sl_price, timestamp, reason)
                continue
        # 信号
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

    # 输出结果
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

if __name__ == "__main__":
    run_backtest()
