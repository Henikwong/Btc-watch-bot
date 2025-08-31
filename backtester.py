# multi_backtest_bot_full.py
"""
多币种多周期共振策略回测（显示TP/SL百分比，保存CSV）
"""

import ccxt
import pandas as pd
import ta
import matplotlib.pyplot as plt

# ================== 配置 ==================
SYMBOLS = ["ETH/USDT", "LTC/USDT", "BNB/USDT", "DOGE/USDT",
           "XRP/USDT", "SOL/USDT", "TRX/USDT", "ADA/USDT", "LINK/USDT"]
TIMEFRAME = "1h"
HIGHER_TIMEFRAME = "4h"
LEVERAGE = 10
RISK_RATIO = 0.15
TP_ATR_MULT = 3.0
SL_ATR_MULT = 2.0
INITIAL_BALANCE = 300  # 总资金
FEE_RATE = 0.0004  # 0.04% 手续费

# ================== 数据 ==================
exchange = ccxt.binance()
def get_historical_data(symbol, timeframe="1h", limit=1000):
    ohlcvs = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcvs, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df.set_index("time", inplace=True)
    return df

# ================== 技术指标 ==================
def compute_indicators(df):
    df = df.copy()
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=12).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=26).ema_indicator()
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df["vol_ma"] = df["volume"].rolling(window=20).mean()
    df.dropna(inplace=True)
    return df

def signal_from_indicators(df_1h, df_4h):
    latest_1h = df_1h.iloc[-1]
    latest_1h_prev = df_1h.iloc[-2]
    latest_4h = df_4h.iloc[-1]

    if latest_1h["volume"] < latest_1h["vol_ma"]:
        return "hold"

    if latest_1h_prev["ema_fast"] < latest_1h_prev["ema_slow"] and latest_1h["ema_fast"] > latest_1h["ema_slow"] and latest_1h["rsi"] > 50:
        signal_1h = "buy"
    elif latest_1h_prev["ema_fast"] > latest_1h_prev["ema_slow"] and latest_1h["ema_fast"] < latest_1h["ema_slow"] and latest_1h["rsi"] < 50:
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

# ================== 回测账户 ==================
class BacktestAccount:
    def __init__(self, initial_balance):
        self.balance = float(initial_balance)
        self.positions = {}  # 多币种持仓
        self.trade_history = []
        self.balance_curve = []

    def place_order(self, symbol, side, qty, price, atr, timestamp):
        if symbol in self.positions and self.positions[symbol]:
            return
        cost = (qty * price) / LEVERAGE
        self.balance -= cost + cost * FEE_RATE
        tp_price = price + TP_ATR_MULT * atr if side == "buy" else price - TP_ATR_MULT * atr
        sl_price = price - SL_ATR_MULT * atr if side == "buy" else price + SL_ATR_MULT * atr
        tp_pct = (tp_price - price) / price * 100 if side == "buy" else (price - tp_price) / price * 100
        sl_pct = (price - sl_price) / price * 100 if side == "buy" else (sl_price - price) / price * 100

        self.positions[symbol] = {"side": "long" if side == "buy" else "short",
                                  "qty": qty, "entry": price, "tp": tp_price, "sl": sl_price,
                                  "tp_pct": tp_pct, "sl_pct": sl_pct}
        self.trade_history.append({"time": timestamp, "symbol": symbol, "type": "Open", "side": side,
                                   "qty": qty, "price": price, "tp_price": tp_price, "tp_pct": tp_pct,
                                   "sl_price": sl_price, "sl_pct": sl_pct})
        self.balance_curve.append(self.balance)
        print(f"🟢 开仓 {symbol} {side.upper()} qty={qty:.4f} @ {price:.2f} | TP: {tp_price:.2f} ({tp_pct:.2f}%), SL: {sl_price:.2f} ({sl_pct:.2f}%)")

    def close_position(self, symbol, price, timestamp, reason="Signal"):
        if symbol not in self.positions or not self.positions[symbol]:
            return
        pos = self.positions[symbol]
        pnl = (price - pos["entry"]) * pos["qty"]
        if pos["side"] == "short":
            pnl *= -1
        pnl_pct = pnl / (pos["entry"] * pos["qty"] / LEVERAGE) * 100  # 盈亏百分比
        self.balance += (pos["qty"] * pos["entry"] / LEVERAGE) + pnl - (price * pos["qty"] / LEVERAGE * FEE_RATE)
        self.trade_history.append({"time": timestamp, "symbol": symbol, "type": "Close", "reason": reason,
                                   "side": pos["side"], "price": price, "pnl": pnl, "pnl_pct": pnl_pct})
        self.positions[symbol] = None
        self.balance_curve.append(self.balance)
        print(f"🔴 平仓 {symbol} {pos['side'].upper()} @ {price:.2f} | PnL: {pnl:.2f} ({pnl_pct:.2f}%) | 原因: {reason}")

    def check_tp_sl(self, symbol, high, low):
        if symbol not in self.positions or not self.positions[symbol]:
            return None, None
        pos = self.positions[symbol]
        if pos["side"] == "long":
            if high >= pos["tp"]: return pos["tp"], "TP"
            if low <= pos["sl"]: return pos["sl"], "SL"
        else:
            if low <= pos["tp"]: return pos["tp"], "TP"
            if high >= pos["sl"]: return pos["sl"], "SL"
        return None, None

def calculate_position_size(balance, price, num_symbols):
    return (balance * RISK_RATIO * LEVERAGE / num_symbols) / price

# ================== 批量回测 ==================
def run_multi_backtest():
    print("🤖 启动多币种回测...")
    account = BacktestAccount(INITIAL_BALANCE)
    dfs_1h = {}
    dfs_4h = {}

    # 获取数据
    for symbol in SYMBOLS:
        dfs_1h[symbol] = compute_indicators(get_historical_data(symbol, TIMEFRAME, limit=1000))
        dfs_4h[symbol] = compute_indicators(get_historical_data(symbol, HIGHER_TIMEFRAME, limit=1000))

    max_len = max([len(dfs_1h[symbol]) for symbol in SYMBOLS])

    for i in range(1, max_len):
        for symbol in SYMBOLS:
            df_1h = dfs_1h[symbol].iloc[:i + 1] if i < len(dfs_1h[symbol]) else dfs_1h[symbol]
            df_4h = dfs_4h[symbol][dfs_4h[symbol].index <= df_1h.index[-1]]
            if len(df_1h) < 50 or df_4h.empty:
                continue

            price = df_1h["close"].iloc[-1]
            atr = df_1h["atr"].iloc[-1]
            timestamp = df_1h.index[-1]

            # TP/SL 检查
            tp_sl_price, reason = account.check_tp_sl(symbol, df_1h["high"].iloc[-1], df_1h["low"].iloc[-1])
            if tp_sl_price:
                account.close_position(symbol, tp_sl_price, timestamp, reason)
                continue

            # 信号
            signal = signal_from_indicators(df_1h, df_4h)
            if signal == "buy":
                if not account.positions.get(symbol) or account.positions[symbol]["side"] == "short":
                    if account.positions.get(symbol):
                        account.close_position(symbol, price, timestamp, "Reverse")
                    qty = calculate_position_size(account.balance, price, len(SYMBOLS
