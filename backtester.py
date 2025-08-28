# backtester.py
"""
自动交易机器人回测脚本

该脚本用于在历史数据上测试交易策略的有效性。
它模拟了账户的开仓、平仓、止损止盈逻辑，并最终计算策略的盈亏、胜率等指标。
"""
import os
import time
import math
import ccxt
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timezone, timedelta

# 从 autotrader.py 中导入核心逻辑
from autotrader import compute_indicators, signal_from_indicators

# ================== 配置 ==================
# 这里的参数可以根据你的需要进行调整和优化
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
LEVERAGE = 10
RISK_RATIO = 0.15  # 动态仓位风险比例
TP_ATR_MULT = 3.0
SL_ATR_MULT = 2.0
PARTIAL_TP_RATIO = 0.0 # 回测暂时不启用分批止盈，方便计算

# 初始资金
INITIAL_BALANCE = 1000

# ================== 模拟账户 ==================
class BacktestAccount:
    def __init__(self, initial_balance):
        self.balance = float(initial_balance)  # 可用资金
        self.equity = float(initial_balance)   # 总资产 (资金 + 仓位盈亏)
        self.position = None                  # 当前仓位 {'symbol', 'side', 'qty', 'entry', 'tp', 'sl'}
        self.trade_history = []               # 交易记录
        self.last_update_prices = {}          # 缓存最新价格

    def update_equity(self, symbol, current_price):
        if not self.position:
            self.equity = self.balance
            return

        pos = self.position
        if pos['symbol'] == symbol:
            pnl = (current_price - pos['entry']) * pos['qty']
            if pos['side'] == 'short':
                pnl *= -1
            self.equity = self.balance + pnl

    def place_order(self, symbol, side, qty, price, current_ohlcv):
        if not self.position:
            # 计算开仓成本，这里简化为不考虑手续费
            cost = (qty * price) / LEVERAGE
            self.balance -= cost
            
            # 计算 TP/SL
            atr = current_ohlcv['atr'].iloc[-1]
            if side == 'buy':
                tp_price = price + TP_ATR_MULT * atr
                sl_price = price - SL_ATR_MULT * atr
            else: # side == 'sell'
                tp_price = price - TP_ATR_MULT * atr
                sl_price = price + SL_ATR_MULT * atr
            
            self.position = {
                'symbol': symbol,
                'side': 'long' if side == 'buy' else 'short',
                'qty': qty,
                'entry': price,
                'tp': tp_price,
                'sl': sl_price
            }
            
            self.trade_history.append({
                'time': current_ohlcv.index[-1],
                'type': 'Open',
                'symbol': symbol,
                'side': self.position['side'],
                'qty': qty,
                'entry_price': price
            })
            print(f"✅ 开仓: {self.position['side']} {symbol} qty={qty:.4f} @{price:.2f} | TP:{tp_price:.2f} SL:{sl_price:.2f}")

    def close_position(self, current_price, current_ohlcv, reason="Signal"):
        if not self.position:
            return
        
        pos = self.position
        pnl = (current_price - pos['entry']) * pos['qty']
        if pos['side'] == 'short':
            pnl *= -1
        
        # 将盈亏加回账户余额
        self.balance += (pos['qty'] * pos['entry'] / LEVERAGE) + pnl
        
        self.trade_history.append({
            'time': current_ohlcv.index[-1],
            'type': 'Close',
            'reason': reason,
            'symbol': pos['symbol'],
            'side': pos['side'],
            'qty': pos['qty'],
            'close_price': current_price,
            'pnl': pnl,
            'pnl_percent': (pnl / (pos['qty'] * pos['entry'] / LEVERAGE))
        })
        print(f"❌ 平仓: {pos['side']} {pos['symbol']} @{current_price:.2f} | 盈亏: {pnl:.2f} ({pnl*100/(pos['qty']*pos['entry']/LEVERAGE):.2f}%) | 理由: {reason}")
        
        self.position = None

    def check_tp_sl(self, high, low):
        if not self.position:
            return None
        
        pos = self.position
        
        if pos['side'] == 'long':
            if high >= pos['tp']:
                return pos['tp'], "TP"
            if low <= pos['sl']:
                return pos['sl'], "SL"
        elif pos['side'] == 'short':
            if low <= pos['tp']:
                return pos['tp'], "TP"
            if high >= pos['sl']:
                return pos['sl'], "SL"
        
        return None, None

def get_historical_data(symbol, timeframe="1h", limit=1000):
    """
    通过 ccxt 获取历史数据并缓存到本地。
    """
    try:
        exchange = ccxt.binance()
        ohlcvs = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcvs, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df.set_index('time', inplace=True)
        return df
    except Exception as e:
        print(f"❌ 获取历史数据失败: {e}")
        return pd.DataFrame()

def calculate_position_size(balance, current_price):
    """根据风险比例计算仓位大小"""
    return (balance * RISK_RATIO * LEVERAGE) / current_price

# ================== 回测主函数 ==================
def run_backtest():
    print("🤖 正在启动回测...")
    df_raw = get_historical_data(SYMBOL, TIMEFRAME, limit=1000)
    if df_raw.empty:
        print("❌ 无法获取历史数据，回测结束。")
        return

    # 预先计算所有指标，减少循环内计算开销
    df = compute_indicators(df_raw)
    
    account = BacktestAccount(INITIAL_BALANCE)
    
    for i in range(len(df)):
        current_ohlcv = df.iloc[0:i+1]
        
        # 确保数据帧有足够的数据量来计算指标
        if len(current_ohlcv) < 50: # MACD 和 EMA 需要一定数据量
            continue

        current_price = current_ohlcv['close'].iloc[-1]
        
        # 检查是否触及 TP/SL
        if account.position:
            tp_sl_price, reason = account.check_tp_sl(current_ohlcv['high'].iloc[-1], current_ohlcv['low'].iloc[-1])
            if tp_sl_price:
                account.close_position(tp_sl_price, current_ohlcv, reason=reason)
                # 如果是 TP/SL 平仓，不需要再检查信号
                continue

        # 检查信号
        signal, _, _ = signal_from_indicators(current_ohlcv)

        # 交易逻辑
        if signal == "buy":
            if not account.position or account.position['side'] == 'short':
                if account.position:
                    account.close_position(current_price, current_ohlcv, reason="Reverse Signal")
                
                qty = calculate_position_size(account.balance, current_price)
                account.place_order(SYMBOL, "buy", qty, current_price, current_ohlcv)
        
        elif signal == "sell":
            if not account.position or account.position['side'] == 'long':
                if account.position:
                    account.close_position(current_price, current_ohlcv, reason="Reverse Signal")
                
                qty = calculate_position_size(account.balance, current_price)
                account.place_order(SYMBOL, "sell", qty, current_price, current_ohlcv)

    # 回测结果分析
    print("\n--- 回测结束 ---")
    
    # 最终平仓（如果有）
    last_price = df['close'].iloc[-1]
    if account.position:
        account.close_position(last_price, df.iloc[-1:], reason="Final Close")

    final_equity = account.balance
    
    trade_df = pd.DataFrame(account.trade_history)
    
    print(f"初始资金: ${INITIAL_BALANCE:.2f}")
    print(f"最终资金: ${final_equity:.2f}")
    print(f"总盈亏: ${(final_equity - INITIAL_BALANCE):.2f}")
    
    if not trade_df.empty:
        total_trades = len(trade_df[trade_df['type'] == 'Close'])
        winning_trades = len(trade_df[(trade_df['type'] == 'Close') & (trade_df['pnl'] > 0)])
        
        print(f"交易总数: {total_trades}")
        if total_trades > 0:
            win_rate = (winning_trades / total_trades) * 100
            print(f"胜率: {win_rate:.2f}%")
        
        total_pnl = trade_df[trade_df['type'] == 'Close']['pnl'].sum()
        print(f"总盈亏 (交易): ${total_pnl:.2f}")

        # 计算最大回撤
        equity_curve = [INITIAL_BALANCE]
        for _, trade in trade_df.iterrows():
            if trade['type'] == 'Close':
                equity_curve.append(equity_curve[-1] + trade['pnl'])
        
        equity_series = pd.Series(equity_curve)
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min() * -100
        
        print(f"最大回撤: {max_drawdown:.2f}%")

if __name__ == "__main__":
    run_backtest()
