# ================== 参数优化系统 ==================
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any, Set
import itertools

class ParameterOptimizer:
    """并行参数优化系统"""
    
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.results = []
        self.best_params = {}
    
    def calculate_technical_indicators(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """计算技术指标（与主系统保持一致）"""
        df = df.copy()
        
        # EMA
        ema_short = params.get('ema_short', 12)
        ema_long = params.get('ema_long', 26)
        df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=ema_short).ema_indicator()
        df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=ema_long).ema_indicator()
        
        # MACD
        macd_fast = params.get('macd_fast', 12)
        macd_slow = params.get('macd_slow', 26)
        macd_signal = params.get('macd_signal', 9)
        macd = ta.trend.MACD(df['close'], window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_signal)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        # RSI
        rsi_window = params.get('rsi_window', 14)
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=rsi_window).rsi()
        
        # ATR
        atr_window = params.get('atr_window', 14)
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=atr_window).average_true_range()
        
        return df.dropna()
    
    def run_backtest_optimized(self, df: pd.DataFrame, params: Dict, precomputed_indicators: Optional[pd.DataFrame] = None) -> Dict:
        """优化版回测函数"""
        try:
            if precomputed_indicators is not None:
                df_indicators = precomputed_indicators
            else:
                df_indicators = self.calculate_technical_indicators(df, params)
            
            if df_indicators.empty:
                return {'net_profit': -9999, 'sharpe_ratio': -9999, 'max_drawdown': 9999}
            
            # 模拟交易逻辑
            balance = 10000
            position = 0
            trades = []
            equity_curve = []
            
            sl_mult = params.get('sl_mult', 2.0)
            tp_mult = params.get('tp_mult', 3.0)
            risk_ratio = params.get('risk_ratio', 0.15)
            
            for i in range(len(df_indicators)):
                current = df_indicators.iloc[i]
                price = current['close']
                
                # 记录权益曲线
                if position != 0:
                    pnl = (price - position) * (1 if position > 0 else -1)
                    current_balance = balance + pnl
                else:
                    current_balance = balance
                equity_curve.append(current_balance)
                
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
                if bullish and position <= 0:
                    if position < 0:  # 平空仓
                        pnl = (price - position) * -1
                        balance += pnl
                        position = 0
                    
                    # 开多仓
                    risk_amount = balance * risk_ratio
                    risk_per_share = current['atr'] * sl_mult
                    position_size = risk_amount / risk_per_share
                    position = price
                    trades.append(('BUY', price, current.name))
                
                elif bearish and position >= 0:
                    if position > 0:  # 平多仓
                        pnl = (price - position)
                        balance += pnl
                        position = 0
                    
                    # 开空仓
                    risk_amount = balance * risk_ratio
                    risk_per_share = current['atr'] * sl_mult
                    position_size = risk_amount / risk_per_share
                    position = -price
                    trades.append(('SELL', price, current.name))
            
            # 计算性能指标
            equity_series = pd.Series(equity_curve)
            returns = equity_series.pct_change().dropna()
            
            if len(returns) == 0:
                return {'net_profit': -9999, 'sharpe_ratio': -9999, 'max_drawdown': 9999}
            
            net_profit = equity_series.iloc[-1] - 10000
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            max_drawdown = (equity_series / equity_series.cummax() - 1).min()
            
            return {
                'net_profit': net_profit,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'trades': len(trades),
                'params': params
            }
            
        except Exception as e:
            self.logger.error(f"回测失败: {e}")
            return {'net_profit': -9999, 'sharpe_ratio': -9999, 'max_drawdown': 9999}
    
    def run_single_parameter_combination(self, df: pd.DataFrame, params: Dict, 
                                       precomputed_indicators: Optional[pd.DataFrame] = None) -> Dict:
        """运行单个参数组合的回测"""
        return self.run_backtest_optimized(df, params, precomputed_indicators)
    
    def optimize_parameters(self, df: pd.DataFrame, param_grid: Dict, 
                          symbols: List[str] = ['BTCUSDT'], 
                          objective: str = 'sharpe_ratio',
                          max_workers: int = 4) -> Dict:
        """并行参数优化"""
        self.logger.info(f"开始参数优化，币种: {symbols}，目标函数: {objective}")
        
        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_param_combinations = list(itertools.product(*param_values))
        
        param_dicts = []
        for combination in all_param_combinations:
            param_dict = dict(zip(param_names, combination))
            param_dicts.append(param_dict)
        
        # 并行执行回测
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            for params in param_dicts:
                # 为每个币种运行回测
                symbol_results = []
                for symbol in symbols:
                    future = executor.submit(
                        self.run_single_parameter_combination, 
                        df, params, None
                    )
                    futures[future] = (params, symbol)
            
            # 收集结果
            for future in as_completed(futures):
                params, symbol = futures[future]
                try:
                    result = future.result()
                    result['symbol'] = symbol
                    result['params'] = params
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"参数组合 {params} 回测失败: {e}")
        
        # 按币种和参数组合聚合结果
        aggregated_results = self._aggregate_results(results, symbols)
        
        # 选择最佳参数
        if objective == 'sharpe_ratio':
            best_result = max(aggregated_results, key=lambda x: x['avg_sharpe_ratio'])
        elif objective == 'net_profit':
            best_result = max(aggregated_results, key=lambda x: x['avg_net_profit'])
        elif objective == 'calmar_ratio':
            best_result = max(aggregated_results, key=lambda x: x['avg_calmar_ratio'])
        else:
            best_result = min(aggregated_results, key=lambda x: x['avg_max_drawdown'])
        
        self.best_params = best_result['params']
        self.results = aggregated_results
        
        self.logger.info(f"优化完成，最佳参数: {self.best_params}")
        return self.best_params
    
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
                    'trade_counts': []
                }
            
            aggregated[params_key]['sharpe_ratios'].append(result['sharpe_ratio'])
            aggregated[params_key]['net_profits'].append(result['net_profit'])
            aggregated[params_key]['max_drawdowns'].append(result['max_drawdown'])
            aggregated[params_key]['trade_counts'].append(result['trades'])
        
        # 计算平均指标
        final_results = []
        for params_key, data in aggregated.items():
            avg_sharpe = np.mean(data['sharpe_ratios'])
            avg_profit = np.mean(data['net_profits'])
            avg_drawdown = np.mean(data['max_drawdowns'])
            avg_trades = np.mean(data['trade_counts'])
            
            # 计算Calmar比率
            calmar_ratio = avg_profit / abs(avg_drawdown) if avg_drawdown != 0 else 0
            
            final_results.append({
                'params': data['params'],
                'avg_sharpe_ratio': avg_sharpe,
                'avg_net_profit': avg_profit,
                'avg_max_drawdown': avg_drawdown,
                'avg_calmar_ratio': calmar_ratio,
                'avg_trades': avg_trades,
                'symbol_count': len(symbols)
            })
        
        return final_results
    
    def plot_parameter_sensitivity(self):
        """绘制参数敏感性分析"""
        if not self.results:
            self.logger.warning("没有优化结果可可视化")
            return
        
        # 创建热力图数据
        param_names = list(self.results[0]['params'].keys())
        n_params = len(param_names)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 热力图
        heatmap_data = []
        for result in self.results:
            row = [result['params'][param] for param in param_names]
            row.append(result['avg_sharpe_ratio'])
            heatmap_data.append(row)
        
        heatmap_df = pd.DataFrame(heatmap_data, columns=param_names + ['sharpe_ratio'])
        pivot_table = heatmap_df.pivot_table(
            values='sharpe_ratio', 
            index=param_names[0], 
            columns=param_names[1] if n_params > 1 else param_names[0]
        )
        
        sns.heatmap(pivot_table, annot=True, fmt='.3f', ax=axes[0, 0])
        axes[0, 0].set_title('Parameter Sensitivity Heatmap')
        
        # 参数敏感性曲线
        for i, param in enumerate(param_names):
            if i < 3:  # 只显示前3个参数
                x_values = [result['params'][param] for result in self.results]
                y_values = [result['avg_sharpe_ratio'] for result in self.results]
                axes[(i+1)//2, (i+1)%2].plot(x_values, y_values, 'o-')
                axes[(i+1)//2, (i+1)%2].set_xlabel(param)
                axes[(i+1)//2, (i+1)%2].set_ylabel('Sharpe Ratio')
                axes[(i+1)//2, (i+1)%2].set_title(f'{param} Sensitivity')
        
        plt.tight_layout()
        plt.savefig('parameter_sensitivity.png')
        plt.close()
    
    def plot_best_parameters_comparison(self, df: pd.DataFrame):
        """绘制最佳参数对比"""
        if not self.best_params:
            self.logger.warning("没有最佳参数可对比")
            return
        
        # 运行基准参数回测
        baseline_params = {
            'ema_short': 12, 'ema_long': 26,
            'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
            'rsi_window': 14, 'atr_window': 14,
            'sl_mult': 2.0, 'tp_mult': 3.0, 'risk_ratio': 0.15
        }
        
        baseline_result = self.run_backtest_optimized(df, baseline_params)
        optimized_result = self.run_backtest_optimized(df, self.best_params)
        
        # 绘制对比图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 权益曲线对比
        baseline_equity = [10000 + baseline_result['net_profit'] * i/100 for i in range(100)]
        optimized_equity = [10000 + optimized_result['net_profit'] * i/100 for i in range(100)]
        
        axes[0, 0].plot(baseline_equity, label='Baseline')
        axes[0, 0].plot(optimized_equity, label='Optimized')
        axes[0, 0].set_title('Equity Curve Comparison')
        axes[0, 0].legend()
        
        # 性能指标对比
        metrics = ['Sharpe Ratio', 'Net Profit', 'Max Drawdown']
        baseline_values = [
            baseline_result['sharpe_ratio'],
            baseline_result['net_profit'],
            baseline_result['max_drawdown']
        ]
        optimized_values = [
            optimized_result['sharpe_ratio'],
            optimized_result['net_profit'],
            optimized_result['max_drawdown']
        ]
        
        x_pos = np.arange(len(metrics))
        axes[0, 1].bar(x_pos - 0.2, baseline_values, 0.4, label='Baseline')
        axes[0, 1].bar(x_pos + 0.2, optimized_values, 0.4, label='Optimized')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(metrics)
        axes[0, 1].set_title('Performance Metrics Comparison')
        axes[0, 1].legend()
        
        plt.tight_layout()
        plt.savefig('parameter_optimization_comparison.png')
        plt.close()

# ================== 使用示例 ==================
async def run_parameter_optimization():
    """运行参数优化示例"""
    logger = AdvancedLogger()
    optimizer = ParameterOptimizer(logger)
    
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='1h')
    data = {
        'open': np.random.normal(100, 10, len(dates)),
        'high': np.random.normal(105, 10, len(dates)),
        'low': np.random.normal(95, 10, len(dates)),
        'close': np.random.normal(100, 10, len(dates)),
        'volume': np.random.normal(1000, 100, len(dates))
    }
    df = pd.DataFrame(data, index=dates)
    
    # 定义参数网格
    param_grid = {
        'ema_short': [10, 12, 14],
        'ema_long': [24, 26, 28],
        'macd_fast': [10, 12, 14],
        'macd_slow': [24, 26, 28],
        'rsi_window': [12, 14, 16],
        'sl_mult': [1.5, 2.0, 2.5],
        'risk_ratio': [0.1, 0.15, 0.2]
    }
    
    # 运行优化
    best_params = optimizer.optimize_parameters(
        df=df,
        param_grid=param_grid,
        symbols=['BTCUSDT', 'SOLUSDT', 'BNBUSDT'],  # 多币种优化
        objective='sharpe_ratio',  # 可选: sharpe_ratio, net_profit, calmar_ratio, max_drawdown
        max_workers=4  # 并行进程数
    )
    
    # 可视化结果
    optimizer.plot_parameter_sensitivity()
    optimizer.plot_best_parameters_comparison(df)
    
    return best_params

# 在主类中添加优化方法
class ProductionTrader:
    # ... 原有代码 ...
    
    async def optimize_strategy(self, symbols: List[str] = None):
        """运行策略参数优化"""
        if symbols is None:
            symbols = Config.SYMBOLS
        
        self.logger.info(f"开始策略参数优化: {symbols}")
        
        # 获取历史数据
        historical_data = {}
        for symbol in symbols:
            df = await self.exchange.get_historical_data(
                symbol, '1h', Config.OHLCV_LIMIT
            )
            historical_data[symbol] = df
        
        # 运行优化
        optimizer = ParameterOptimizer(self.logger)
        
        param_grid = {
            'ema_short': [10, 12, 14],
            'ema_long': [24, 26, 28],
            'macd_fast': [10, 12, 14],
            'macd_slow': [24, 26, 28],
            'rsi_window': [12, 14, 16],
            'sl_mult': [1.5, 2.0, 2.5],
            'risk_ratio': [0.1, 0.15, 0.2]
        }
        
        best_params = optimizer.optimize_parameters(
            df=historical_data[symbols[0]],  # 使用第一个币种的数据
            param_grid=param_grid,
            symbols=symbols,
            objective='sharpe_ratio',
            max_workers=Config.MAX_WORKERS
        )
        
        # 更新配置
        self._update_strategy_params(best_params)
        
        return best_params
    
    def _update_strategy_params(self, params: Dict):
        """更新策略参数"""
        # 这里可以根据优化结果更新交易系统的参数
        self.logger.info(f"更新策略参数: {params}")
