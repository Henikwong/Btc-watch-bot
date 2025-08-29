# backtester.py
"""
自动交易机器人回测脚本

该脚本用于在历史数据上测试交易策略的有效性。
它模拟了账户的开仓、平仓、止损止盈逻辑，并最终计算策略的盈亏、胜率等指标。
"""
import ccxt
import pandas as pd
import numpy as np
import ta
from datetime import datetime

# ================== 配置 ==================
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
LEVERAGE = 10
RISK_RATIO = 0.15  # 动态仓位风险比例
TP_ATR_MULT = 3.0
SL_ATR_MULT = 2.0
INITIAL_BALANCE = 1000

# ================== 技术指标计算 ==================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # EMA
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=12).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=26).ema_indicator()

    # MACD
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    # ATR
    df["atr"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14
    ).average_true_range()

    return df.dropna()

def signal_from_indicators(df: pd.DataFrame):
    """根据指标生成交易信号"""
    latest = df.iloc[-1]

    # MACD金叉买入，死叉卖出
    if latest["macd"] > latest["macd_signal"] and latest["ema_fast"] > latest["ema_slow"] and latest["rsi"] > 50:
        return "buy", latest["macd"], latest["rsi"]
    elif latest["macd"] < latest["macd_signal"] and latest["ema_fast"] < latest["ema_slow"] and latest["rsi"] < 50:
        return "sell", latest["macd"], latest["rsi"]
    else:
        return "hold", latest["macd"], latest["rsi"]

# ================== 模拟账户 ==================
class BacktestAccount:
    def __init__(self, initial_balance):
        self.balance = float(initial_balance)
        self.position = None
        self.trade_history = []

    def place_order(self, symbol, side, qty, price, atr, timestamp):
        if self.position:
            return

        cost = (qty * price) / LEVERAGE
        self.balance -= cost

        if side == "buy":
            tp_price = price + TP_ATR_MULT * atr
            sl_price = price - SL_ATR_MULT * atr
        else:
            tp_price = price - TP_ATR_MULT * atr
            sl_price = price + SL_ATR_MULT * atr

        self.position = {
            "symbol": symbol,
            "side": "long" if side == "buy" else "short",
            "qty": qty,
            "entry": price,
            "tp": tp_price,
            "sl": sl_price,
        }

        self.trade_history.append(
            {
                "time": timestamp,
                "type": "Open",
                "side": self.position["side"],
                "qty": qty,
                "entry_price": price,
            }
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
            {
                "time": timestamp,
                "type": "Close",
                "reason": reason,
                "side": pos["side"],
                "qty": pos["qty"],
                "close_price": price,
                "pnl": pnl,
            }
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
        elif pos["side"] == "short":
            if low <= pos["tp"]:
                return pos["tp"], "TP"
            if high >= pos["sl"]:
                return pos["sl"], "SL"
        return None, None

def get_historical_data(symbol, timeframe="1h", limit=1000):
    exchange = ccxt.binance()
    ohlcvs = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(
        ohlcvs, columns=["time", "open", "high", "low", "close", "volume"]
    )
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df.set_index("time", inplace=True)
    return df

def calculate_position_size(balance, current_price):
    return (balance * RISK_RATIO * LEVERAGE) / current_price

# ================== 回测主函数 ==================
def run_backtest():
    print("🤖 启动回测...")
    df_raw = get_historical_data(SYMBOL, TIMEFRAME, limit=1000)
    if df_raw.empty:
        print("❌ 无法获取历史数据")
        return

    df = compute_indicators(df_raw)
    account = BacktestAccount(INITIAL_BALANCE)

    for i in range(len(df)):
        current_df = df.iloc[: i + 1]
        if len(current_df) < 50:
            continue

        price = current_df["close"].iloc[-1]
        atr = current_df["atr"].iloc[-1]
        timestamp = current_df.index[-1]

        # 检查TP/SL
        if account.position:
            tp_sl_price, reason = account.check_tp_sl(
                current_df["high"].iloc[-1], current_df["low"].iloc[-1]
            )
            if tp_sl_price:
                account.close_position(tp_sl_price, timestamp, reason)
                continue

        signal, _, _ = signal_from_indicators(current_df)

        if signal == "buy":
            if not account.position or account.position["side"] == "short":
                if account.position:
                    account.close_position(price, timestamp, "Reverse")
                qty = calculate_position_size(account.balance, price)
                account.place_order(SYMBOL, "buy", qty, price, atr, timestamp)

        elif signal == "sell":
            if not account.position or account.position["side"] == "long":
                if account.position:
                    account.close_position(price, timestamp, "Reverse")
                qty = calculate_position_size(account.balance, price)
                account.place_order(SYMBOL, "sell", qty, price, atr, timestamp)

    # 最终平仓
    if account.position:
        last_price = df["close"].iloc[-1]
        last_time = df.index[-1]
        account.close_position(last_price, last_time, "Final")

    # 结果统计
    trade_df = pd.DataFrame(account.trade_history)
    print("\n--- 回测结果 ---")
    print(f"初始资金: {INITIAL_BALANCE:.2f}")
    print(f"最终资金: {account.balance:.2f}")
    print(f"总盈亏: {account.balance - INITIAL_BALANCE:.2f}")

    closes = trade_df[trade_df["type"] == "Close"]
    if not closes.empty:
        total_trades = len(closes)
        win_trades = (closes["pnl"] > 0).sum()
        win_rate = win_trades / total_trades * 100
        print(f"交易次数: {total_trades}, 胜率: {win_rate:.2f}%")
        print(f"总交易盈亏: {closes['pnl'].sum():.2f}")

if __name__ == "__main__":
    run_backtest()
