# ================== 优化后的参数优化回测系统 ==================
import numpy as np
import pandas as pd
import ta
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any, Union
import itertools
import os
from datetime import datetime
import warnings
import json
import pickle
warnings.filterwarnings('ignore')

# 导入Config配置（与主脚本保持一致）
try:
    from autotrader_enhanced import Config
except ImportError:
    # 如果无法导入，定义相同的配置类
    class Config:
        # 技术指标参数
        EMA_SHORT = 12
        EMA_LONG = 26
        MACD_FAST = 12
        MACD_SLOW = 26
        MACD_SIGNAL = 9
        RSI_WINDOW = 14
        ATR_WINDOW = 14
        
        # 风险管理和止损参数
        SL_ATR_MULT = 2.0
        TP_ATR_MULT = 3.0
        RISK_RATIO = 0.05
        LEVERAGE = 5
        
        # 交易参数
        INITIAL_BALANCE = 100

# 设置日志
def setup_logger(name='backtest_optimizer'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger

logger = setup_logger()

# 独立技术指标计算函数
def calculate_technical_indicators_func(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """独立的技术指标计算函数"""
    df = df.copy()
    
    # EMA - 使用Config默认值或参数值
    ema_short = params.get('ema_short', Config.EMA_SHORT)
    ema_long = params.get('ema_long', Config.EMA_LONG)
    df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=ema_short).ema_indicator()
    df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=ema_long).ema_indicator()
    
    # MACD - 使用Config默认值或参数值
    macd_fast = params.get('macd_fast', Config.MACD_FAST)
    macd_slow = params.get('macd_slow', Config.MACD_SLOW)
    macd_signal = params.get('macd_signal', Config.MACD_SIGNAL)
    macd = ta.trend.MACD(df['close'], window_slow=macd_slow, 
                        window_fast=macd_fast, window_sign=macd_signal)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    
    # RSI - 使用Config默认值或参数值
    rsi_window = params.get('rsi_window', Config.RSI_WINDOW)
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=rsi_window).rsi()
    
    # ATR - 使用Config默认值或参数值
    atr_window = params.get('atr_window', Config.ATR_WINDOW)
    df['atr'] = ta.volatility.AverageTrueRange(
        df['high'], df['low'], df['close'], window=atr_window
    ).average_true_range()
    
    return df.dropna()

# 将回测函数提取为独立函数，提高多进程兼容性
def run_backtest_optimized_func(df: pd.DataFrame, params: Dict, 
                               precomputed_indicators: Optional[pd.DataFrame] = None,
                               logger: Optional[logging.Logger] = None,
                               initial_balance: Optional[float] = None) -> Dict:
    """优化版回测函数，包含止损止盈逻辑（独立函数版本）"""
    try:
        # 使用传入的初始资金或Config默认值
        initial_balance = initial_balance if initial_balance is not None else Config.INITIAL_BALANCE
        
        if precomputed_indicators is not None:
            df_indicators = precomputed_indicators
        else:
            df_indicators = calculate_technical_indicators_func(df, params)
        
        if df_indicators.empty:
            return {'net_profit': -9999, 'sharpe_ratio': -9999, 'max_drawdown': 9999, 'trades': 0}
        
        # 模拟交易逻辑
        balance = initial_balance
        position = 0  # 0: 无仓位, >0: 多头仓位, <0: 空头仓位
        entry_price = 0
        trades = []
        equity_curve = [initial_balance]
        
        # 使用Config默认值或参数值
        sl_mult = params.get('sl_mult', Config.SL_ATR_MULT)
        tp_mult = params.get('tp_mult', Config.TP_ATR_MULT)
        risk_ratio = params.get('risk_ratio', Config.RISK_RATIO)
        leverage = params.get('leverage', Config.LEVERAGE)
        
        # 止损止盈价格
        stop_loss_price = 0
        take_profit_price = 0
        
        for i in range(len(df_indicators)):
            current = df_indicators.iloc[i]
            price = current['close']
            high = current['high']
            low = current['low']
            atr = current['atr']
            
            # 检查止损止盈条件
            if position > 0:  # 多头仓位
                # 检查止损
                if low <= stop_loss_price:
                    pnl = (stop_loss_price - entry_price) * position
                    balance += pnl
                    trades.append(('STOP_LOSS_LONG', stop_loss_price, current.name, pnl))
                    position = 0
                    stop_loss_price = 0
                    take_profit_price = 0
                # 检查止盈
                elif high >= take_profit_price:
                    pnl = (take_profit_price - entry_price) * position
                    balance += pnl
                    trades.append(('TAKE_PROFIT_LONG', take_profit_price, current.name, pnl))
                    position = 0
                    stop_loss_price = 0
                    take_profit_price = 0
            
            elif position < 0:  # 空头仓位
                # 检查止损
                if high >= stop_loss_price:
                    pnl = (entry_price - stop_loss_price) * abs(position)
                    balance += pnl
                    trades.append(('STOP_LOSS_SHORT', stop_loss_price, current.name, pnl))
                    position = 0
                    stop_loss_price = 0
                    take_profit_price = 0
                # 检查止盈
                elif low <= take_profit_price:
                    pnl = (entry_price - take_profit_price) * abs(position)
                    balance += pnl
                    trades.append(('TAKE_PROFIT_SHORT', take_profit_price, current.name, pnl))
                    position = 0
                    stop_loss_price = 0
                    take_profit_price = 0
            
            # 计算当前权益
            if position != 0:
                if position > 0:  # 多头
                    pnl = (price - entry_price) * position
                else:  # 空头
                    pnl = (entry_price - price) * abs(position)
                current_balance = balance + pnl
            else:
                current_balance = balance
            
            equity_curve.append(current_balance)
            
            # 如果没有持仓，检查交易信号
            if position == 0:
                # 交易信号
                bullish = all([
                    current['macd'] > current['macd_signal'],
                    current['ema_12'] > current['ema_26'],
                    40 < current['rsi'] < 70
                ])
                
                bearish = all([
                    current['macd'] < current['macd_signal'],
                    current['ema_12'] < current['ema_26'],
                    30 < current['rsi'] < 60
                ])
                
                # 执行交易
                if bullish:
                    # 开多仓
                    risk_amount = balance * risk_ratio
                    risk_per_share = atr * sl_mult
                    if risk_per_share <= 0:
                        continue
                        
                    position_size = risk_amount / risk_per_share
                    # 考虑杠杆
                    max_position = (balance * leverage) / price
                    position_size = min(position_size, max_position)
                    
                    if position_size > 0:
                        position = position_size
                        entry_price = price
                        # 设置止损止盈价格
                        stop_loss_price = entry_price - atr * sl_mult
                        take_profit_price = entry_price + atr * tp_mult
                        trades.append(('OPEN_LONG', price, current.name, 0))
                
                elif bearish:
                    # 开空仓
                    risk_amount = balance * risk_ratio
                    risk_per_share = atr * sl_mult
                    if risk_per_share <= 0:
                        continue
                        
                    position_size = risk_amount / risk_per_share
                    # 考虑杠杆
                    max_position = (balance * leverage) / price
                    position_size = min(position_size, max_position)
                    
                    if position_size > 0:
                        position = -position_size  # 负数表示空头
                        entry_price = price
                        # 设置止损止盈价格
                        stop_loss_price = entry_price + atr * sl_mult
                        take_profit_price = entry_price - atr * tp_mult
                        trades.append(('OPEN_SHORT', price, current.name, 0))
        
        # 最后一天平仓
        if position != 0:
            if position > 0:
                pnl = (price - entry_price) * position
            else:
                pnl = (entry_price - price) * abs(position)
            balance += pnl
            equity_curve[-1] = balance
        
        # 计算性能指标
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()
        
        if len(returns) == 0:
            return {'net_profit': -9999, 'sharpe_ratio': -9999, 'max_drawdown': 9999, 'trades': 0}
        
        net_profit = equity_series.iloc[-1] - initial_balance
        # 小时数据年化因子为√(24*365)
        annual_factor = np.sqrt(24 * 365)
        sharpe_ratio = returns.mean() / returns.std() * annual_factor if returns.std() > 0 else 0
        
        # 计算最大回撤
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # 计算胜率
        winning_trades = [t for t in trades if t[3] > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        # 计算盈亏比
        profit_trades = [t[3] for t in trades if t[3] > 0]
        loss_trades = [t[3] for t in trades if t[3] < 0]
        profit_factor = abs(sum(profit_trades) / sum(loss_trades)) if loss_trades and profit_trades else 0
        
        return {
            'net_profit': net_profit,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'params': params,
            'equity_curve': equity_series
        }
        
    except Exception as e:
        if logger:
            logger.error(f"回测失败: {e}")
        import traceback
        traceback.print_exc()
        return {'net_profit': -9999, 'sharpe_ratio': -9999, 'max_drawdown': 9999, 'trades': 0}

class ParameterOptimizer:
    """并行参数优化系统"""
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logger()
        self.results = []
        self.best_params = {}
        self.optimization_history = []
    
    def calculate_technical_indicators(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """计算技术指标（使用独立函数）"""
        return calculate_technical_indicators_func(df, params)
    
    def run_backtest_optimized(self, df: pd.DataFrame, params: Dict, 
                              precomputed_indicators: Optional[pd.DataFrame] = None,
                              initial_balance: Optional[float] = None) -> Dict:
        """优化版回测函数，包含止损止盈逻辑"""
        return run_backtest_optimized_func(
            df, params, precomputed_indicators, self.logger, initial_balance
        )
    
    def run_single_parameter_combination(self, df: pd.DataFrame, params: Dict, 
                                       precomputed_indicators: Optional[pd.DataFrame] = None,
                                       initial_balance: Optional[float] = None) -> Dict:
        """运行单个参数组合的回测"""
        return self.run_backtest_optimized(df, params, precomputed_indicators, initial_balance)
    
    def optimize_parameters(self, data_dict: Dict[str, pd.DataFrame], param_grid: Dict, 
                          objective: str = 'composite_score',
                          max_workers: int = 4, 
                          max_combinations: int = 1000,
                          initial_balance: Optional[float] = None) -> Dict:
        """并行参数优化，支持多币种"""
        symbols = list(data_dict.keys())
        self.logger.info(f"开始参数优化，币种: {symbols}，目标函数: {objective}")
        
        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_param_combinations = list(itertools.product(*param_values))
        
        # 限制参数组合数量，避免内存问题
        if len(all_param_combinations) > max_combinations:
            self.logger.warning(f"参数组合数量({len(all_param_combinations)})超过限制({max_combinations})，进行随机抽样")
            np.random.seed(42)  # 固定随机种子以便复现
            indices = np.random.choice(len(all_param_combinations), max_combinations, replace=False)
            all_param_combinations = [all_param_combinations[i] for i in indices]
        
        param_dicts = []
        for combination in all_param_combinations:
            param_dict = dict(zip(param_names, combination))
            param_dicts.append(param_dict)
        
        # 并行执行回测 - 修复日志问题，不传递logger到子进程
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            for params in param_dicts:
                # 为每个币种运行回测
                for symbol in symbols:
                    df = data_dict[symbol]
                    future = executor.submit(
                        run_backtest_optimized_func, 
                        df, params, None, None, initial_balance  # 不传递logger到子进程
                    )
                    futures[future] = (params, symbol)
            
            # 收集结果
            for i, future in enumerate(as_completed(futures)):
                params, symbol = futures[future]
                try:
                    result = future.result()
                    result['symbol'] = symbol
                    result['params'] = params
                    results.append(result)
                    
                    # 每完成10%输出一次进度
                    if (i + 1) % max(1, len(futures) // 10) == 0:
                        self.logger.info(f"进度: {i+1}/{len(futures)} ({((i+1)/len(futures)*100):.1f}%)")
                        
                except Exception as e:
                    self.logger.error(f"参数组合 {params} 回测失败: {e}")
        
        # 按币种和参数组合聚合结果
        aggregated_results = self._aggregate_results(results, symbols)
        
        # 计算综合评分
        for result in aggregated_results:
            result['composite_score'] = self._calculate_composite_score(result, aggregated_results)
        
        # 记录优化历史
        self.optimization_history.append({
            'timestamp': datetime.now(),
            'symbols': symbols,
            'param_grid': param_grid,
            'objective': objective,
            'results': aggregated_results
        })
        
        # 选择最佳参数
        if objective == 'sharpe_ratio':
            best_result = max(aggregated_results, key=lambda x: x['avg_sharpe_ratio'])
        elif objective == 'net_profit':
            best_result = max(aggregated_results, key=lambda x: x['avg_net_profit'])
        elif objective == 'calmar_ratio':
            best_result = max(aggregated_results, key=lambda x: x['avg_calmar_ratio'])
        elif objective == 'win_rate':
            best_result = max(aggregated_results, key=lambda x: x['avg_win_rate'])
        elif objective == 'composite_score':
            best_result = max(aggregated_results, key=lambda x: x['composite_score'])
        else:
            best_result = min(aggregated_results, key=lambda x: x['avg_max_drawdown'])
        
        self.best_params = best_result['params']
        self.results = aggregated_results
        
        # 保存结果
        self.save_optimization_results()
        
        self.logger.info(f"优化完成，最佳参数: {self.best_params}")
        return self.best_params
    
    def _calculate_composite_score(self, result: Dict, all_results: List[Dict]) -> float:
        """计算综合评分，结合多个指标（使用传入的结果列表）"""
        # 权重可以调整
        weights = {
            'sharpe_ratio': 0.4,
            'calmar_ratio': 0.3,
            'win_rate': 0.2,
            'profit_factor': 0.1
        }
        
        # 归一化指标
        max_sharpe = max(r['avg_sharpe_ratio'] for r in all_results) if all_results else 1
        max_calmar = max(r['avg_calmar_ratio'] for r in all_results) if all_results else 1
        max_win_rate = max(r['avg_win_rate'] for r in all_results) if all_results else 1
        max_profit_factor = max(r['avg_profit_factor'] for r in all_results) if all_results else 1
        
        # 避免除以零
        max_sharpe = max(max_sharpe, 0.001)
        max_calmar = max(max_calmar, 0.001)
        max_win_rate = max(max_win_rate, 0.001)
        max_profit_factor = max(max_profit_factor, 0.001)
        
        # 计算评分
        score = (
            weights['sharpe_ratio'] * (result['avg_sharpe_ratio'] / max_sharpe) +
            weights['calmar_ratio'] * (result['avg_calmar_ratio'] / max_calmar) +
            weights['win_rate'] * (result['avg_win_rate'] / max_win_rate) +
            weights['profit_factor'] * (result['avg_profit_factor'] / max_profit_factor)
        )
        
        return score
    
    def _aggregate_results(self, results: List[Dict], symbols: List[str]) -> List[Dict]:
        """聚合多个币种的结果"""
        aggregated = {}
        
        for result in results:
            params_key = str(result['params'])
            
            if params_key not in aggregated:
                aggregated[params_key] = {
                    'params': result['params'],
                    'sharpe_ratios': [],
                    'net_profits': [],
                    'max_drawdowns': [],
                    'trade_counts': [],
                    'win_rates': [],
                    'profit_factors': []
                }
            
            aggregated[params_key]['sharpe_ratios'].append(result['sharpe_ratio'])
            aggregated[params_key]['net_profits'].append(result['net_profit'])
            aggregated[params_key]['max_drawdowns'].append(result['max_drawdown'])
            aggregated[params_key]['trade_counts'].append(result['trades'])
            aggregated[params_key]['win_rates'].append(result.get('win_rate', 0))
            aggregated[params_key]['profit_factors'].append(result.get('profit_factor', 0))
        
        # 计算平均指标
        final_results = []
        for params_key, data in aggregated.items():
            avg_sharpe = np.mean(data['sharpe_ratios'])
            avg_profit = np.mean(data['net_profits'])
            avg_drawdown = np.mean(data['max_drawdowns'])
            avg_trades = np.mean(data['trade_counts'])
            avg_win_rate = np.mean(data['win_rates'])
            avg_profit_factor = np.mean(data['profit_factors'])
            
            # 计算Calmar比率
            calmar_ratio = avg_profit / abs(avg_drawdown) if avg_drawdown != 0 else 0
            
            final_results.append({
                'params': data['params'],
                'avg_sharpe_ratio': avg_sharpe,
                'avg_net_profit': avg_profit,
                'avg_max_drawdown': avg_drawdown,
                'avg_calmar_ratio': calmar_ratio,
                'avg_trades': avg_trades,
                'avg_win_rate': avg_win_rate,
                'avg_profit_factor': avg_profit_factor,
                'symbol_count': len(symbols)
            })
        
        return final_results
    
    def save_optimization_results(self, filename: Optional[str] = None):
        """保存优化结果到文件"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_results_{timestamp}.pkl"
        
        # 确保目录存在
        os.makedirs("results", exist_ok=True)
        filepath = os.path.join("results", filename)
        
        # 保存结果
        with open(filepath, 'wb') as f:
            pickle.dump({
                'best_params': self.best_params,
                'results': self.results,
                'optimization_history': self.optimization_history
            }, f)
        
        self.logger.info(f"优化结果已保存到: {filepath}")
    
    def load_optimization_results(self, filename: str):
        """从文件加载优化结果"""
        filepath = os.path.join("results", filename)
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.best_params = data['best_params']
            self.results = data['results']
            self.optimization_history = data.get('optimization_history', [])
        
        self.logger.info(f"优化结果已从 {filepath} 加载")
    
    def plot_parameter_sensitivity(self, save_path: Optional[str] = None):
        """绘制参数敏感性分析"""
        if not self.results:
            self.logger.warning("没有优化结果可可视化")
            return
        
        if not save_path:
            os.makedirs("plots", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"plots/parameter_sensitivity_{timestamp}.png"
        
        # 创建热力图数据
        param_names = list(self.results[0]['params'].keys())
        n_params = len(param_names)
        
        # 创建2x2的子图布局
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Parameter Sensitivity Analysis', fontsize=16)
        
        # 热力图
        heatmap_data = []
        for result in self.results:
            row = [result['params'][param] for param in param_names]
            row.append(result['avg_sharpe_ratio'])
            heatmap_data.append(row)
        
        heatmap_df = pd.DataFrame(heatmap_data, columns=param_names + ['sharpe_ratio'])
        
        if n_params >= 2:
            try:
                pivot_table = heatmap_df.pivot_table(
                    values='sharpe_ratio', 
                    index=param_names[0], 
                    columns=param_names[1],
                    aggfunc='mean'
                )
                
                sns.heatmap(pivot_table, annot=True, fmt='.3f', ax=axes[0, 0], cmap='viridis')
                axes[0, 0].set_title('Parameter Sensitivity Heatmap')
            except Exception as e:
                self.logger.warning(f"无法创建热力图: {e}")
                axes[0, 0].text(0.5, 0.5, "无法创建热力图", ha='center', va='center')
                axes[0, 0].set_title('Parameter Sensitivity Heatmap')
        else:
            axes[0, 0].set_visible(False)
        
        # 参数敏感性曲线（前3个参数）
        for i, param in enumerate(param_names[:3]):
            try:
                x_values = [result['params'][param] for result in self.results]
                y_values = [result['avg_sharpe_ratio'] for result in self.results]
                
                row = (i + 1) // 2
                col = (i + 1) % 2
                axes[row, col].plot(x_values, y_values, 'o-', label=f'{param}')
                axes[row, col].set_xlabel(param)
                axes[row, col].set_ylabel('Sharpe Ratio')
                axes[row, col].set_title(f'{param} vs Sharpe Ratio')
                axes[row, col].legend()
                axes[row, col].grid(True, linestyle='--', alpha=0.7)
            except Exception as e:
                self.logger.warning(f"绘制参数敏感性曲线失败: {e}")
                axes[row, col].text(0.5, 0.5, f"无法绘制{param}敏感性曲线", ha='center', va='center')
        
        # 隐藏多余的子图
        for i in range(n_params, 4):
            row = i // 2
            col = i % 2
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"参数敏感性图已保存到: {save_path}")
    
    def plot_single_equity_curve(self, result: Dict, save_path: Optional[str] = None):
        """绘制单个回测结果的净值曲线"""
        if 'equity_curve' not in result:
            self.logger.warning("没有净值曲线数据")
            return
        
        if not save_path:
            os.makedirs("plots", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"plots/equity_curve_{timestamp}.png"

        plt.figure(figsize=(12, 6))
        plt.plot(result['equity_curve'].values, label="Equity Curve", linewidth=2)
        plt.title("Equity Curve")
        plt.xlabel("Time")
        plt.ylabel("Balance")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加性能指标
        metrics_text = f"Net Profit: {result['net_profit']:.2f}, Sharpe: {result['sharpe_ratio']:.2f}, Max DD: {result['max_drawdown']:.2f}"
        plt.figtext(0.5, 0.01, metrics_text, ha='center', fontsize=10, 
                   bbox={'facecolor': 'lightgray', 'alpha': 0.7, 'pad': 5})
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"净值曲线已保存到: {save_path}")
    
    def plot_equity_curve(self, data_dict: Dict[str, pd.DataFrame], 
                         save_path: Optional[str] = None,
                         initial_balance: Optional[float] = None):
        """绘制最佳参数下的净值曲线"""
        if not self.best_params:
            self.logger.warning("没有最佳参数可绘制净值曲线")
            return
        
        if not save_path:
            os.makedirs("plots", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"plots/equity_curve_{timestamp}.png"
        
        # 运行回测获取净值曲线
        equity_curves = {}
        for symbol, df in data_dict.items():
            result = self.run_backtest_optimized(df, self.best_params, initial_balance=initial_balance)
            equity_curves[symbol] = result.get('equity_curve', pd.Series([initial_balance or Config.INITIAL_BALANCE]))
        
        # 绘制净值曲线
        plt.figure(figsize=(12, 8))
        
        for symbol, equity in equity_curves.items():
            # 归一化到初始资金
            normalized_equity = equity / equity.iloc[0]
            plt.plot(normalized_equity, label=symbol, linewidth=2)
        
        plt.title('Normalized Equity Curve with Optimized Parameters')
        plt.xlabel('Time')
        plt.ylabel('Normalized Equity')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加性能指标
        metrics_text = f"Best Parameters: {self.best_params}"
        plt.figtext(0.5, 0.01, metrics_text, ha='center', fontsize=10, 
                   bbox={'facecolor': 'lightgray', 'alpha': 0.7, 'pad': 5})
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"净值曲线图已保存到: {save_path}")
    
    def plot_best_parameters_comparison(self, data_dict: Dict[str, pd.DataFrame], 
                                      save_path: Optional[str] = None,
                                      initial_balance: Optional[float] = None):
        """绘制最佳参数对比"""
        if not self.best_params:
            self.logger.warning("没有最佳参数可对比")
            return
        
        if not save_path:
            os.makedirs("plots", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"plots/parameter_comparison_{timestamp}.png"
        
        # 运行基准参数回测（使用Config默认值）
        baseline_params = {
            'ema_short': Config.EMA_SHORT,
            'ema_long': Config.EMA_LONG,
            'macd_fast': Config.MACD_FAST,
            'macd_slow': Config.MACD_SLOW,
            'macd_signal': Config.MACD_SIGNAL,
            'rsi_window': Config.RSI_WINDOW,
            'atr_window': Config.ATR_WINDOW,
            'sl_mult': Config.SL_ATR_MULT,
            'tp_mult': Config.TP_ATR_MULT,
            'risk_ratio': Config.RISK_RATIO,  # 修复了这里的拼写错误
            'leverage': Config.LEVERAGE
        }
        
        baseline_results = {}
        optimized_results = {}
        
        for symbol, df in data_dict.items():
            baseline_results[symbol] = self.run_backtest_optimized(df, baseline_params, initial_balance=initial_balance)
            optimized_results[symbol] = self.run_backtest_optimized(df, self.best_params, initial_balance=initial_balance)
        
        # 绘制对比图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Parameter Optimization Comparison', fontsize=16)
        
        # 性能指标对比
        metrics = ['Sharpe Ratio', 'Net Profit', 'Max Drawdown', 'Win Rate']
        baseline_values = []
        optimized_values = []
        
        for symbol in data_dict.keys():
            baseline_values.append([
                baseline_results[symbol]['sharpe_ratio'],
                baseline_results[symbol]['net_profit'],
                baseline_results[symbol]['max_drawdown'],
                baseline_results[symbol].get('win_rate', 0)
            ])
            optimized_values.append([
                optimized_results[symbol]['sharpe_ratio'],
                optimized_results[symbol]['net_profit'],
                optimized_results[symbol]['max_drawdown'],
                optimized_results[symbol].get('win_rate', 0)
            ])
        
        # 计算平均指标
        baseline_avg = np.mean(baseline_values, axis=0)
        optimized_avg = np.mean(optimized_values, axis=0)
        
        x_pos = np.arange(len(metrics))
        axes[0, 0].bar(x_pos - 0.2, baseline_avg, 0.4, label='Baseline', alpha=0.8)
        axes[0, 0].bar(x_pos + 0.2, optimized_avg, 0.4, label='Optimized', alpha=0.8)
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(metrics)
        axes[0, 0].set_title('Performance Metrics Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # 参数对比
        param_names = list(self.best_params.keys())
        baseline_params_values = [baseline_params.get(name, 0) for name in param_names]
        optimized_params_values = [self.best_params.get(name, 0) for name in param_names]
        
        x_pos = np.arange(len(param_names))
        axes[0, 1].bar(x_pos - 0.2, baseline_params_values, 0.4, label='Baseline', alpha=0.8)
        axes[0, 1].bar(x_pos + 0.2, optimized_params_values, 0.4, label='Optimized', alpha=0.8)
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(param_names, rotation=45, ha='right')
        axes[0, 1].set_title('Parameters Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # 各币种性能提升对比
        symbols = list(data_dict.keys())
        sharpe_improvements = [
            optimized_results[symbol]['sharpe_ratio'] - baseline_results[symbol]['sharpe_ratio'] 
            for symbol in symbols
        ]
        
        axes[1, 0].bar(symbols, sharpe_improvements, alpha=0.8)
        axes[1, 0].set_title('Sharpe Ratio Improvement by Symbol')
        axes[1, 0].set_ylabel('Improvement')
        axes[1, 0].grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # 添加数值标签
        for i, v in enumerate(sharpe_improvements):
            axes[1, 0].text(i, v + (0.01 if v >= 0 else -0.03), f'{v:.2f}', 
                           ha='center', va='bottom' if v >= 0 else 'top')
        
        # 权益曲线对比（使用第一个币种）
        first_symbol = symbols[0]
        if 'equity_curve' in baseline_results[first_symbol] and 'equity_curve' in optimized_results[first_symbol]:
            baseline_equity = baseline_results[first_symbol]['equity_curve'].values
            optimized_equity = optimized_results[first_symbol]['equity_curve'].values
            
            # 确保长度一致
            min_len = min(len(baseline_equity), len(optimized_equity))
            baseline_equity = baseline_equity[:min_len]
            optimized_equity = optimized_equity[:min_len]
            
            # 归一化到初始资金
            baseline_equity = baseline_equity / baseline_equity[0]
            optimized_equity = optimized_equity / optimized_equity[0]
            
            axes[1, 1].plot(baseline_equity, label='Baseline', linewidth=2)
            axes[1, 1].plot(optimized_equity, label='Optimized', linewidth=2)
            axes[1, 1].set_title(f'Normalized Equity Curve ({first_symbol})')
            axes[1, 1].legend()
            axes[1, 1].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"参数对比图已保存到: {save_path}")

# ================== 使用示例 ==================
def generate_sample_data(symbols: List[str], periods: int = 1000) -> Dict[str, pd.DataFrame]:
    """生成示例数据"""
    data_dict = {}
    
    for symbol in symbols:
        np.random.seed(hash(symbol) % 1000)  # 不同币种使用不同随机种子
        
        dates = pd.date_range('2023-01-01', periods=periods, freq='1h')
        
        # 生成更真实的价格序列（带趋势和波动）
        base_price = 100 if symbol == 'BTCUSDT' else 50
        trend = np.linspace(0, 0.5, periods)  # 向上趋势
        noise = np.random.normal(0, 0.01, periods)
        
        close_prices = base_price * (1 + trend + noise).cumprod()
        
        # 生成高、低、开盘价格
        volatility = 0.02  # 2% 波动率
        returns = np.random.normal(0, volatility, periods)
        close_prices = base_price * np.exp(np.cumsum(returns))
        
        # 添加一些趋势
        trend = np.linspace(0, 0.2, periods)
        close_prices = close_prices * (1 + trend)
        
        # 生成OHLC数据
        open_prices = close_prices * (1 + np.random.normal(0, 0.005, periods))
        high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, periods)))
        low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, periods)))
        
        # 确保价格合理
        high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
        low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
        
        df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': np.random.lognormal(5, 1, periods)
        }, index=dates)
        
        data_dict[symbol] = df
    
    return data_dict

if __name__ == "__main__":
    # 示例使用
    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
    
    # 生成示例数据
    data_dict = generate_sample_data(symbols, periods=2000)
    
    # 参数网格
    param_grid = {
        'ema_short': [8, 12, 16],
        'ema_long': [20, 26, 32],
        'macd_fast': [8, 12, 16],
        'macd_slow': [20, 26, 32],
        'macd_signal': [7, 9, 11],
        'rsi_window': [10, 14, 18],
        'atr_window': [10, 14, 18],
        'sl_mult': [1.5, 2.0, 2.5],
        'tp_mult': [2.0, 3.0, 4.0],
        'risk_ratio': [0.03, 0.05, 0.07],
        'leverage': [3, 5, 7]
    }
    
    # 初始化优化器
    optimizer = ParameterOptimizer()
    
    # 运行优化
    best_params = optimizer.optimize_parameters(
        data_dict=data_dict,
        param_grid=param_grid,
        objective='composite_score',
        max_workers=4,
        max_combinations=500,  # 限制组合数量
        initial_balance=1000
    )
    
    # 绘制结果
    optimizer.plot_parameter_sensitivity()
    optimizer.plot_equity_curve(data_dict)
    optimizer.plot_best_parameters_comparison(data_dict)
    
    print(f"最佳参数: {best_params}")
    print("优化完成!")
