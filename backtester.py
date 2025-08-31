# ultimate_backtest_engine.py
"""
终极版多币种回测引擎 - 解决所有潜在问题
支持本地数据缓存、精确止盈止损、高级分析
"""

import ccxt
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pickle
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ================== 配置 ==================
class BacktestConfig:
    # 交易对配置
    SYMBOLS = ["ETH/USDT", "LTC/USDT", "BNB/USDT", "DOGE/USDT",
               "XRP/USDT", "SOL/USDT", "TRX/USDT", "ADA/USDT", "LINK/USDT"]
    
    # 时间框架
    TIMEFRAME = "1h"
    HIGHER_TIMEFRAME = "4h"
    
    # 风险参数
    LEVERAGE = 10
    RISK_RATIO = 0.15
    TP_ATR_MULT = 3.0
    SL_ATR_MULT = 2.0
    INITIAL_BALANCE = 10000
    FEE_RATE = 0.0004  # 0.04% 手续费
    
    # 回测参数
    DATA_LIMIT = 1000
    MIN_DATA_POINTS = 50
    CACHE_DIR = "data_cache"
    RESULTS_DIR = "backtest_results"

# ================== 数据管理 ==================
class DataManager:
    """高级数据管理器，支持本地缓存"""
    
    def __init__(self):
        self.exchange = ccxt.binance()
        self.config = BacktestConfig
        self._ensure_dirs()
    
    def _ensure_dirs(self):
        """确保目录存在"""
        Path(self.config.CACHE_DIR).mkdir(exist_ok=True)
        Path(self.config.RESULTS_DIR).mkdir(exist_ok=True)
    
    def _get_cache_path(self, symbol: str, timeframe: str) -> str:
        """获取缓存文件路径"""
        symbol_clean = symbol.replace('/', '_')
        return f"{self.config.CACHE_DIR}/{symbol_clean}_{timeframe}.pkl"
    
    def fetch_data_with_cache(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """带缓存的数据获取"""
        cache_path = self._get_cache_path(symbol, timeframe)
        
        # 检查缓存
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    if len(cached_data) >= limit * 0.8:  # 缓存数据足够
                        print(f"📦 使用缓存数据: {symbol} {timeframe}")
                        return cached_data.tail(limit)
            except:
                pass
        
        # 从交易所获取数据
        print(f"🌐 下载数据: {symbol} {timeframe}")
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["datetime"] = pd.to_datetime(df["timestamp"], unit='ms')
            df.set_index("datetime", inplace=True)
            
            # 保存到缓存
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)
            
            return df
        except Exception as e:
            print(f"❌ 获取数据失败 {symbol}: {e}")
            return pd.DataFrame()
    
    def get_aligned_data(self, symbols: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """获取对齐的时间序列数据"""
        all_data = {}
        
        for symbol in symbols:
            df_1h = self.fetch_data_with_cache(symbol, self.config.TIMEFRAME, self.config.DATA_LIMIT)
            df_4h = self.fetch_data_with_cache(symbol, self.config.HIGHER_TIMEFRAME, self.config.DATA_LIMIT)
            
            if not df_1h.empty and not df_4h.empty:
                # 对齐时间戳
                common_index = df_1h.index.intersection(df_4h.index)
                if len(common_index) > self.config.MIN_DATA_POINTS:
                    all_data[symbol] = {
                        '1h': df_1h.loc[common_index],
                        '4h': df_4h.loc[common_index]
                    }
        
        return all_data

# ================== 指标系统 ==================
class AdvancedIndicatorSystem:
    """高级指标计算系统"""
    
    @staticmethod
    def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标"""
        df = df.copy()
        
        # 趋势指标
        df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        
        # 动量指标
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        # 超买超卖
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # 波动率
        df['atr'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], window=14
        ).average_true_range()
        
        # 成交量
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        
        # 价格变化
        df['returns'] = df['close'].pct_change()
        
        return df.dropna()
    
    @staticmethod
    def generate_signal(df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> str:
        """生成交易信号"""
        if df_1h.empty or df_4h.empty:
            return "hold"
        
        latest_1h = df_1h.iloc[-1]
        latest_4h = df_4h.iloc[-1]
        
        # 成交量过滤
        if latest_1h['volume_ratio'] < 0.8:
            return "hold"
        
        # 多条件共振
        bullish_conditions = [
            latest_1h['macd'] > latest_1h['macd_signal'],
            latest_1h['ema_12'] > latest_1h['ema_26'],
            40 < latest_1h['rsi'] < 70,
            latest_4h['ema_12'] > latest_4h['ema_26'],
            latest_1h['close'] > latest_1h['ema_50']  # 趋势过滤
        ]
        
        bearish_conditions = [
            latest_1h['macd'] < latest_1h['macd_signal'],
            latest_1h['ema_12'] < latest_1h['ema_26'],
            30 < latest_1h['rsi'] < 60,
            latest_4h['ema_12'] < latest_4h['ema_26'],
            latest_1h['close'] < latest_1h['ema_50']  # 趋势过滤
        ]
        
        if all(bullish_conditions):
            return "buy"
        elif all(bearish_conditions):
            return "sell"
        
        return "hold"

# ================== 高级回测账户 ==================
class AdvancedBacktestAccount:
    """高级回测账户管理系统"""
    
    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.positions: Dict[str, Optional[Dict]] = {}
        self.trade_history = []
        self.equity_curve = []
        self.drawdown_curve = []
        self.timestamps = []
        
        for symbol in BacktestConfig.SYMBOLS:
            self.positions[symbol] = None
    
    def calculate_position_size(self, price: float, atr: float) -> float:
        """基于风险的仓位计算"""
        risk_amount = self.equity * BacktestConfig.RISK_RATIO
        risk_per_share = atr * BacktestConfig.SL_ATR_MULT
        position_size = risk_amount / risk_per_share
        max_position = (self.equity * BacktestConfig.LEVERAGE) / price
        return min(position_size, max_position)
    
    def open_position(self, symbol: str, side: str, price: float, atr: float, timestamp: datetime) -> bool:
        """开仓"""
        if self.positions[symbol] is not None:
            return False
        
        quantity = self.calculate_position_size(price, atr)
        if quantity <= 0:
            return False
        
        # 计算保证金和手续费
        margin_required = (quantity * price) / BacktestConfig.LEVERAGE
        open_fee = margin_required * BacktestConfig.FEE_RATE
        
        if margin_required + open_fee > self.balance:
            return False
        
        # 计算止盈止损
        if side == "buy":
            tp_price = price + atr * BacktestConfig.TP_ATR_MULT
            sl_price = price - atr * BacktestConfig.SL_ATR_MULT
        else:
            tp_price = price - atr * BacktestConfig.TP_ATR_MULT
            sl_price = price + atr * BacktestConfig.SL_ATR_MULT
        
        # 更新账户
        self.balance -= open_fee
        self.positions[symbol] = {
            'side': side,
            'quantity': quantity,
            'entry_price': price,
            'entry_time': timestamp,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'margin': margin_required,
            'open_fee': open_fee
        }
        
        # 记录交易
        trade_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': 'OPEN',
            'side': side,
            'quantity': quantity,
            'price': price,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'fee': open_fee,
            'balance': self.balance,
            'equity': self.equity
        }
        
        self.trade_history.append(trade_record)
        return True
    
    def close_position(self, symbol: str, price: float, timestamp: datetime, reason: str):
        """平仓"""
        if self.positions[symbol] is None:
            return
        
        position = self.positions[symbol]
        quantity = position['quantity']
        entry_price = position['entry_price']
        
        # 计算盈亏
        if position['side'] == 'buy':
            pnl = (price - entry_price) * quantity
        else:
            pnl = (entry_price - price) * quantity
        
        # 计算手续费
        close_fee = (price * quantity / BacktestConfig.LEVERAGE) * BacktestConfig.FEE_RATE
        total_fee = position['open_fee'] + close_fee
        
        # 更新账户
        self.balance += pnl - close_fee
        self.equity = self.balance
        self.positions[symbol] = None
        
        # 记录交易
        trade_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': 'CLOSE',
            'side': position['side'],
            'quantity': quantity,
            'price': price,
            'pnl': pnl,
            'fee': close_fee,
            'total_fee': total_fee,
            'balance': self.balance,
            'equity': self.equity,
            'reason': reason
        }
        
        self.trade_history.append(trade_record)
        self._update_equity_curve(timestamp)
    
    def _update_equity_curve(self, timestamp: datetime):
        """更新权益曲线"""
        # 计算当前总权益（余额 + 持仓市值）
        total_equity = self.balance
        for symbol, position in self.positions.items():
            if position is not None:
                # 这里需要当前价格来计算持仓市值，简化处理用入场价
                total_equity += position['margin'] * BacktestConfig.LEVERAGE
        
        self.equity = total_equity
        self.equity_curve.append(self.equity)
        self.timestamps.append(timestamp)
        
        # 计算回撤
        if self.equity_curve:
            peak = max(self.equity_curve)
            drawdown = (peak - self.equity) / peak * 100
            self.drawdown_curve.append(drawdown)
    
    def check_tp_sl(self, symbol: str, high: float, low: float, current_price: float) -> Tuple[Optional[float], Optional[str]]:
        """检查止盈止损（包含K线内模拟）"""
        if self.positions[symbol] is None:
            return None, None
        
        position = self.positions[symbol]
        
        if position['side'] == 'buy':
            # 检查是否触发止盈止损
            if high >= position['tp_price']:
                return position['tp_price'], 'TP'
            if low <= position['sl_price']:
                return position['sl_price'], 'SL'
            # K线内模拟：如果开盘就触发
            if current_price >= position['tp_price']:
                return position['tp_price'], 'TP'
            if current_price <= position['sl_price']:
                return position['sl_price'], 'SL'
        else:
            if low <= position['tp_price']:
                return position['tp_price'], 'TP'
            if high >= position['sl_price']:
                return position['sl_price'], 'SL'
            if current_price <= position['tp_price']:
                return position['tp_price'], 'TP'
            if current_price >= position['sl_price']:
                return position['sl_price'], 'SL'
        
        return None, None

# ================== 回测引擎 ==================
class UltimateBacktestEngine:
    """终极回测引擎"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.indicator_system = AdvancedIndicatorSystem()
        self.config = BacktestConfig
    
    def run_backtest(self):
        """运行回测"""
        print("🚀 启动终极回测引擎...")
        
        # 获取数据
        print("📊 获取并预处理数据...")
        aligned_data = self.data_manager.get_aligned_data(self.config.SYMBOLS)
        
        if not aligned_data:
            print("❌ 没有有效数据")
            return None
        
        # 计算指标
        print("📈 计算技术指标...")
        indicator_data = {}
        for symbol, data in aligned_data.items():
            indicator_data[symbol] = {
                '1h': self.indicator_system.compute_indicators(data['1h']),
                '4h': self.indicator_system.compute_indicators(data['4h'])
            }
        
        # 初始化账户
        account = AdvancedBacktestAccount(self.config.INITIAL_BALANCE)
        
        # 获取共同时间索引
        common_timestamps = self._get_common_timestamps(indicator_data)
        
        print(f"⏰ 回测时间范围: {common_timestamps[0]} 到 {common_timestamps[-1]}")
        print(f"📅 总K线数量: {len(common_timestamps)}")
        
        # 主回测循环
        print("🔁 开始回测...")
        for i, timestamp in enumerate(common_timestamps):
            if i % 100 == 0:
                print(f"📊 处理进度: {i}/{len(common_timestamps)}")
            
            for symbol in self.config.SYMBOLS:
                if symbol not in indicator_data:
                    continue
                
                data_1h = indicator_data[symbol]['1h']
                data_4h = indicator_data[symbol]['4h']
                
                # 获取当前数据
                current_data_1h = data_1h[data_1h.index <= timestamp]
                current_data_4h = data_4h[data_4h.index <= timestamp]
                
                if len(current_data_1h) < self.config.MIN_DATA_POINTS or len(current_data_4h) < 10:
                    continue
                
                current_price = current_data_1h['close'].iloc[-1]
                current_high = current_data_1h['high'].iloc[-1]
                current_low = current_data_1h['low'].iloc[-1]
                current_atr = current_data_1h['atr'].iloc[-1]
                
                # 检查止盈止损
                exit_price, exit_reason = account.check_tp_sl(
                    symbol, current_high, current_low, current_price
                )
                if exit_price is not None:
                    account.close_position(symbol, exit_price, timestamp, exit_reason)
                    continue
                
                # 生成信号
                signal = self.indicator_system.generate_signal(current_data_1h, current_data_4h)
                
                if signal == "buy":
                    self._handle_buy_signal(account, symbol, current_price, current_atr, timestamp)
                elif signal == "sell":
                    self._handle_sell_signal(account, symbol, current_price, current_atr, timestamp)
        
        # 清理剩余仓位
        self._close_all_positions(account, indicator_data)
        
        # 分析结果
        results = self.analyze_results(account)
        
        return results
    
    def _handle_buy_signal(self, account, symbol, price, atr, timestamp):
        """处理买入信号"""
        if account.positions[symbol] is None:
            account.open_position(symbol, "buy", price, atr, timestamp)
        elif account.positions[symbol]['side'] == 'sell':
            account.close_position(symbol, price, timestamp, "Reverse")
            account.open_position(symbol, "buy", price, atr, timestamp)
    
    def _handle_sell_signal(self, account, symbol, price, atr, timestamp):
        """处理卖出信号"""
        if account.positions[symbol] is None:
            account.open_position(symbol, "sell", price, atr, timestamp)
        elif account.positions[symbol]['side'] == 'buy':
            account.close_position(symbol, price, timestamp, "Reverse")
            account.open_position(symbol, "sell", price, atr, timestamp)
    
    def _get_common_timestamps(self, indicator_data: Dict) -> pd.DatetimeIndex:
        """获取共同时间索引"""
        common_index = None
        for symbol, data in indicator_data.items():
            symbol_index = data['1h'].index.intersection(data['4h'].index)
            if common_index is None:
                common_index = symbol_index
            else:
                common_index = common_index.intersection(symbol_index)
        return common_index
    
    def _close_all_positions(self, account: AdvancedBacktestAccount, indicator_data: Dict):
        """平掉所有仓位"""
        for symbol in self.config.SYMBOLS:
            if account.positions[symbol] is not None and symbol in indicator_data:
                last_price = indicator_data[symbol]['1h']['close'].iloc[-1]
                last_time = indicator_data[symbol]['1h'].index[-1]
                account.close_position(symbol, last_price, last_time, "End of Backtest")
    
    def analyze_results(self, account: AdvancedBacktestAccount) -> Dict:
        """分析回测结果"""
        trade_df = pd.DataFrame(account.trade_history)
        
        if trade_df.empty:
            print("❌ 没有交易记录")
            return {}
        
        # 保存详细结果
        results_path = f"{self.config.RESULTS_DIR}/backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trade_df.to_csv(results_path, index=False)
        print(f"💾 交易记录已保存到: {results_path}")
        
        # 计算性能指标
        close_trades = trade_df[trade_df['action'] == 'CLOSE']
        
        results = {
            'initial_balance': account.initial_balance,
            'final_equity': account.equity,
            'total_trades': len(close_trades),
            'winning_trades': len(close_trades[close_trades['pnl'] > 0]),
            'losing_trades': len(close_trades[close_trades['pnl'] <= 0]),
            'total_pnl': close_trades['pnl'].sum() if not close_trades.empty else 0,
            'total_fees': trade_df['fee'].sum(),
            'equity_curve': account.equity_curve,
            'drawdown_curve': account.drawdown_curve,
            'timestamps': account.timestamps
        }
        
        results['profit'] = results['final_equity'] - results['initial_balance']
        results['return_pct'] = (results['profit'] / results['initial_balance']) * 100
        
        if results['total_trades'] > 0:
            results['win_rate'] = (results['winning_trades'] / results['total_trades']) * 100
            results['avg_win'] = close_trades[close_trades['pnl'] > 0]['pnl'].mean() if results['winning_trades'] > 0 else 0
            results['avg_loss'] = close_trades[close_trades['pnl'] <= 0]['pnl'].mean() if results['losing_trades'] > 0 else 0
            results['profit_factor'] = abs(results['avg_win'] * results['winning_trades'] / 
                                         (results['avg_loss'] * results['losing_trades'])) if results['losing_trades'] > 0 else float('inf')
        else:
            results['win_rate'] = 0
            results['avg_win'] = 0
            results['avg_loss'] = 0
            results['profit_factor'] = 0
        
        results['max_drawdown'] = max(account.drawdown_curve) if account.drawdown_curve else 0
        
        # 打印结果
        self._print_results(results)
        self._plot_results(account, results)
        
        return results
    
    def _print_results(self, results: Dict):
        """打印回测结果"""
        print("\n" + "="*60)
        print("📊 终极回测结果摘要")
        print("="*60)
        print(f"初始资金: ${results['initial_balance']:,.2f}")
        print(f"最终权益: ${results['final_equity']:,.2f}")
        print(f"净利润: ${results['profit']:,.2f} ({results['return_pct']:.2f}%)")
        print(f"总交易次数: {results['total_trades']}")
        print(f"盈利交易: {results['winning_trades']}")
        print(f"亏损交易: {results['losing_trades']}")
        print(f
