import pandas as pd
import numpy as np
import json
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from typing import List, Dict, Optional
from datetime import datetime

# 添加模式开关
MODE = "research"  # 可选 "research" 或 "bot"

# 模拟 Backtester 和 Optimizer 类
# 实际项目中，这些类会包含更复杂的交易逻辑和优化算法
class Config:
    INITIAL_BALANCE = 1000
    ema_short = 12
    ema_long = 26
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    rsi_window = 14
    atr_window = 14
    sl_mult = 1.5
    tp_mult = 3.0
    risk_ratio = 0.01
    leverage = 1

class Logger:
    def info(self, message):
        print(f"[INFO] {message}")
    
    def warning(self, message):
        print(f"[WARNING] {message}")
        
    def error(self, message):
        print(f"[ERROR] {message}")

class OptimizerStub:
    """一个用于演示的存根（Stub）类，模拟真实的优化器"""
    def __init__(self):
        self.logger = Logger()
        self.best_params = None
        self.results = []
        self.optimization_history = []

    def optimize_parameters(self, data_dict: Dict[str, pd.DataFrame], param_grid: Dict, objective: str, max_workers: int, max_combinations: int, initial_balance: float) -> Dict:
        """模拟参数优化过程，生成结果"""
        self.logger.info("模拟运行参数优化...")
        
        # 模拟生成结果
        all_combinations = []
        param_names = list(param_grid.keys())
        
        from itertools import product
        params_list = list(product(*param_grid.values()))
        
        for params_tuple in params_list[:max_combinations]:
            params = dict(zip(param_names, params_tuple))
            
            # 模拟回测结果
            results_for_symbols = []
            symbols = list(data_dict.keys())
            for symbol in symbols:
                backtest_result = self.run_backtest_optimized(data_dict[symbol], params, initial_balance)
                results_for_symbols.append(backtest_result)
            
            # 模拟聚合结果
            aggregated = self._aggregate_results(results_for_symbols, symbols)
            self.results.extend(aggregated)
        
        if self.results:
            self.results.sort(key=lambda x: x['avg_sharpe_ratio'], reverse=True)
            self.best_params = self.results[0]['params']

        # 记录优化历史
        self.optimization_history.append({
            'timestamp': datetime.now(),
            'objective': objective,
            'symbols': list(data_dict.keys()),
            'best_params': self.best_params,
            'best_score': self.results[0]['avg_sharpe_ratio'] if self.results else None
        })
        
        # 如果是 bot 模式，保存最佳参数到 JSON
        if MODE == "bot":
            self.save_best_params()
        
        return self.best_params

    def run_backtest_optimized(self, df: pd.DataFrame, params: Dict, initial_balance: float) -> Dict:
        """模拟单次回测，生成回测结果"""
        
        # 确保 df 不为空
        if df.empty:
            return {'params': params, 'sharpe_ratio': 0, 'net_profit': 0, 'max_drawdown': 0, 'calmar_ratio': 0, 
                    'sortino_ratio': 0, 'mar_ratio': 0, 'trades': 0, 'win_rate': 0, 
                    'profit_factor': 0, 'avg_holding_period': 0, 'trade_frequency': 0,
                    'equity_curve': pd.Series([initial_balance])}

        # 模拟计算净值曲线
        length = len(df)
        equity = pd.Series(
            np.cumprod(1 + np.random.normal(0.0001, 0.005, length)),
            index=df.index
        ) * initial_balance
        
        # 模拟计算性能指标
        returns = equity.pct_change().dropna()
        daily_returns = returns.resample('D').sum()
        
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365) if np.std(daily_returns) != 0 else 0
        net_profit = equity.iloc[-1] - equity.iloc[0]
        max_drawdown = (equity.div(equity.cummax()) - 1).min()
        
        return {
            'params': params,
            'sharpe_ratio': sharpe_ratio,
            'net_profit': net_profit,
            'max_drawdown': max_drawdown,
            'calmar_ratio': abs(net_profit / max_drawdown) if max_drawdown != 0 else 0,
            'sortino_ratio': abs(np.mean(returns) / np.std(returns[returns < 0])) * np.sqrt(252) if np.std(returns[returns < 0]) != 0 else 0,
            'mar_ratio': abs(net_profit / max_drawdown) if max_drawdown != 0 else 0,
            'trades': np.random.randint(50, 200),
            'win_rate': np.random.uniform(0.4, 0.6),
            'profit_factor': np.random.uniform(1.2, 2.0),
            'avg_holding_period': np.random.randint(3600, 10800),
            'trade_frequency': np.random.uniform(100, 300),
            'equity_curve': equity
        }
    
    def _aggregate_results(self, results: List[Dict], symbols: List[str]) -> List[Dict]:
        """聚合多个币种的结果"""
        aggregated = {}
        
        for result in results:
            params_key = json.dumps(result.get('params', {}), sort_keys=True)
            
            if params_key not in aggregated:
                aggregated[params_key] = {
                    'params': result.get('params', {}),
                    'sharpe_ratios': [], 'net_profits': [], 'max_drawdowns': [],
                    'calmar_ratios': [], 'sortino_ratios': [], 'mar_ratios': [],
                    'trade_counts': [], 'win_rates': [], 'profit_factors': [],
                    'avg_holding_periods': [], 'trade_frequencies': []
                }
            
            aggregated[params_key]['sharpe_ratios'].append(result.get('sharpe_ratio', 0))
            aggregated[params_key]['net_profits'].append(result.get('net_profit', 0))
            aggregated[params_key]['max_drawdowns'].append(result.get('max_drawdown', 0))
            aggregated[params_key]['calmar_ratios'].append(result.get('calmar_ratio', 0))
            aggregated[params_key]['sortino_ratios'].append(result.get('sortino_ratio', 0))
            aggregated[params_key]['mar_ratios'].append(result.get('mar_ratio', 0))
            aggregated[params_key]['trade_counts'].append(result.get('trades', 0))
            aggregated[params_key]['win_rates'].append(result.get('win_rate', 0))
            aggregated[params_key]['profit_factors'].append(result.get('profit_factor', 0))
            aggregated[params_key]['avg_holding_periods'].append(result.get('avg_holding_period', 0))
            aggregated[params_key]['trade_frequencies'].append(result.get('trade_frequency', 0))

        final_results = []
        for params_key, data in aggregated.items():
            final_results.append({
                'params': data['params'],
                'avg_sharpe_ratio': np.mean(data['sharpe_ratios']),
                'avg_sortino_ratio': np.mean(data['sortino_ratios']),
                'avg_net_profit': np.mean(data['net_profits']),
                'avg_max_drawdown': np.mean(data['max_drawdowns']),
                'avg_calmar_ratio': np.mean(data['calmar_ratios']),
                'avg_mar_ratio': np.mean(data['mar_ratios']),
                'avg_trades': np.mean(data['trade_counts']),
                'avg_win_rate': np.mean(data['win_rates']),
                'avg_profit_factor': np.mean(data['profit_factors']),
                'avg_holding_period': np.mean(data['avg_holding_periods']),
                'avg_trade_frequency': np.mean(data['trade_frequencies']),
                'symbol_count': len(symbols)
            })
        
        return final_results
    
    def save_best_params(self, filename: Optional[str] = None):
        """保存最佳参数到JSON文件"""
        if not self.best_params:
            self.logger.warning("没有最佳参数可保存")
            return
            
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"best_params_{timestamp}.json"
        
        os.makedirs("results", exist_ok=True)
        filepath = os.path.join("results", filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.best_params, f, indent=4)
        
        self.logger.info(f"最佳参数已保存到: {filepath}")
    
class ReportGenerator:
    def __init__(self, optimizer: OptimizerStub):
        self.optimizer = optimizer
        self.logger = Logger()
        
    def save_optimization_results(self, filename: Optional[str] = None):
        """保存优化结果到文件"""
        if MODE != "research":
            return
            
        if not self.optimizer.results:
            self.logger.warning("没有优化结果可保存")
            return
            
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_results_{timestamp}.pkl"
        
        os.makedirs("results", exist_ok=True)
        filepath = os.path.join("results", filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'best_params': self.optimizer.best_params,
                'results': self.optimizer.results,
                'optimization_history': self.optimizer.optimization_history
            }, f)
        
        self.logger.info(f"优化结果已保存到: {filepath}")
    
    def plot_parameter_sensitivity(self, save_path: Optional[str] = None):
        """绘制参数敏感性分析"""
        if MODE != "research":
            return ""
            
        if not self.optimizer.results:
            self.logger.warning("没有优化结果可可视化")
            return ""
        
        if not save_path:
            os.makedirs("plots", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"plots/parameter_sensitivity_{timestamp}.png"
        
        param_names = list(self.optimizer.results[0]['params'].keys())
        
        if len(param_names) >= 2:
            fig, axes = plt.subplots(1, 1, figsize=(10, 8))
            fig.suptitle('Parameter Sensitivity Analysis: Heatmap', fontsize=16)
            try:
                heatmap_df = pd.DataFrame([r['params'] for r in self.optimizer.results])
                heatmap_df['avg_sharpe_ratio'] = [r['avg_sharpe_ratio'] for r in self.optimizer.results]
                
                pivot_table = heatmap_df.pivot_table(
                    index=param_names[0],
                    columns=param_names[1],
                    values='avg_sharpe_ratio',
                    aggfunc='mean'
                )
                sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="viridis", ax=axes)
                axes.set_title(f"Sharpe Heatmap: {param_names[0]} vs {param_names[1]}")
            except Exception as e:
                self.logger.error(f"绘制热力图失败: {e}")
                axes.text(0.5, 0.5, "无法创建热力图", ha='center', va='center', transform=axes.transAxes)
                axes.set_title('Parameter Sensitivity Heatmap')
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            fig.suptitle(f"Parameter Sensitivity Analysis: {param_names[0]}", fontsize=16)
            
            values = [r['params'][param_names[0]] for r in self.optimizer.results]
            sharpe_vals = [r['avg_sharpe_ratio'] for r in self.optimizer.results]
            
            ax.scatter(values, sharpe_vals, alpha=0.6)
            ax.set_title(f"{param_names[0]} vs Avg Sharpe Ratio")
            ax.set_xlabel(param_names[0])
            ax.set_ylabel("Avg Sharpe Ratio")
            ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"参数敏感性分析已保存到: {save_path}")
        plt.close()
        return save_path

    def plot_equity_curve(self, data_dict: Dict[str, pd.DataFrame], 
                         save_path: Optional[str] = None,
                         initial_balance: Optional[float] = None) -> str:
        """绘制最佳参数下的净值曲线，并与 Buy & Hold 对比"""
        if MODE != "research":
            return ""
            
        if not self.optimizer.best_params:
            self.logger.warning("没有最佳参数可绘制净值曲线")
            return ""
        
        if not save_path:
            os.makedirs("plots", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"plots/equity_curve_{timestamp}.png"
        
        equity_curves = {}
        buy_hold_curves = {}
        initial_balance = initial_balance or Config.INITIAL_BALANCE
        
        for symbol, df in data_dict.items():
            result = self.optimizer.run_backtest_optimized(df, self.optimizer.best_params, initial_balance=initial_balance)
            equity_curves[symbol] = result.get('equity_curve', pd.Series([initial_balance]))
            
            # Buy & Hold 逻辑
            buy_hold_df = df.iloc[len(df)-len(result.get('equity_curve', [])):].copy()
            if not buy_hold_df.empty:
                buy_hold_return = (buy_hold_df['close'] / buy_hold_df['close'].iloc[0]).values
                buy_hold_curves[symbol] = pd.Series(buy_hold_return * initial_balance, index=buy_hold_df.index)
            else:
                buy_hold_curves[symbol] = pd.Series([initial_balance])

        plt.figure(figsize=(12, 8))
        
        for symbol, equity in equity_curves.items():
            normalized_equity = equity / equity.iloc[0]
            plt.plot(normalized_equity, label=f'{symbol} Strategy', linewidth=2)
        
        for symbol, buy_hold in buy_hold_curves.items():
            # 确保 buy_hold 不为空
            if not buy_hold.empty and buy_hold.iloc[0] != 0:
                normalized_buy_hold = buy_hold / buy_hold.iloc[0]
                plt.plot(normalized_buy_hold, label=f'{symbol} Buy & Hold', linestyle='--', linewidth=1)
        
        plt.title('Normalized Equity Curve: Strategy vs Buy & Hold')
        plt.xlabel('Time')
        plt.ylabel('Normalized Equity')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        metrics_text = f"Best Parameters: {self.optimizer.best_params}"
        plt.figtext(0.5, 0.01, metrics_text, ha='center', fontsize=10, 
                   bbox={'facecolor': 'lightgray', 'alpha': 0.7, 'pad': 5})
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"净值曲线图已保存到: {save_path}")
        return save_path

    def plot_best_parameters_comparison(self, data_dict: Dict[str, pd.DataFrame], 
                                      save_path: Optional[str] = None) -> str:
        """绘制最佳参数对比"""
        if MODE != "research":
            return ""
            
        if not self.optimizer.best_params:
            self.logger.warning("没有最佳参数可对比")
            return ""
        
        if not save_path:
            os.makedirs("plots", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"plots/parameter_comparison_{timestamp}.png"
        
        # 确保参数名一致（使用小写）
        baseline_params = {
            'ema_short': Config.ema_short, 'ema_long': Config.ema_long, 'macd_fast': Config.macd_fast,
            'macd_slow': Config.macd_slow, 'macd_signal': Config.macd_signal, 'rsi_window': Config.rsi_window,
            'atr_window': Config.atr_window, 'sl_mult': Config.sl_mult, 'tp_mult': Config.tp_mult,
            'risk_ratio': Config.risk_ratio, 'leverage': Config.leverage
        }
        
        baseline_results = {}
        optimized_results = {}
        initial_balance = Config.INITIAL_BALANCE

        for symbol, df in data_dict.items():
            baseline_results[symbol] = self.optimizer.run_backtest_optimized(df, baseline_params, initial_balance=initial_balance)
            optimized_results[symbol] = self.optimizer.run_backtest_optimized(df, self.optimizer.best_params, initial_balance=initial_balance)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Parameter Optimization Comparison', fontsize=16)
        
        # 避免索引错误
        if not baseline_results or not optimized_results:
            self.logger.warning("没有回测结果可用于对比")
            plt.close(fig)
            return ""

        # 使用一个通用列表来确保指标顺序一致
        metrics_to_plot = ['sharpe_ratio', 'net_profit', 'max_drawdown', 'win_rate']
        metrics_display_names = ['Sharpe Ratio', 'Net Profit', 'Max Drawdown', 'Win Rate']

        baseline_values = [np.mean([baseline_results[s].get(m, 0) for s in data_dict]) for m in metrics_to_plot]
        optimized_values = [np.mean([optimized_results[s].get(m, 0) for s in data_dict]) for m in metrics_to_plot]
        
        x_pos = np.arange(len(metrics_display_names))
        axes[0, 0].bar(x_pos - 0.2, baseline_values, 0.4, label='Baseline', alpha=0.8)
        axes[0, 0].bar(x_pos + 0.2, optimized_values, 0.4, label='Optimized', alpha=0.8)
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(metrics_display_names)
        axes[0, 0].set_title('Performance Metrics Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, axis='y', linestyle='--', alpha=0.7)
        
        param_names = list(self.optimizer.best_params.keys())
        baseline_params_values = [baseline_params.get(name, 0) for name in param_names]
        optimized_params_values = [self.optimizer.best_params.get(name, 0) for name in param_names]
        
        x_pos = np.arange(len(param_names))
        axes[0, 1].bar(x_pos - 0.2, baseline_params_values, 0.4, label='Baseline', alpha=0.8)
        axes[0, 1].bar(x_pos + 0.2, optimized_params_values, 0.4, label='Optimized', alpha=0.8)
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(param_names, rotation=45, ha='right')
        axes[0, 1].set_title('Parameters Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True, axis='y', linestyle='--', alpha=0.7)
        
        symbols = list(data_dict.keys())
        sharpe_improvements = [
            optimized_results[symbol].get('sharpe_ratio', 0) - baseline_results[symbol].get('sharpe_ratio', 0)
            for symbol in symbols
        ]
        
        axes[1, 0].bar(symbols, sharpe_improvements, alpha=0.8)
        axes[1, 0].set_title('Sharpe Ratio Improvement by Symbol')
        axes[1, 0].set_ylabel('Improvement')
        axes[1, 0].grid(True, axis='y', linestyle='--', alpha=0.7)
        for i, v in enumerate(sharpe_improvements):
            axes[1, 0].text(i, v + (0.01 if v >= 0 else -0.03), f'{v:.2f}', ha='center', va='bottom' if v >= 0 else 'top')
        
        first_symbol = symbols[0]
        if 'equity_curve' in baseline_results[first_symbol] and 'equity_curve' in optimized_results[first_symbol]:
            baseline_equity = baseline_results[first_symbol]['equity_curve'].values
            optimized_equity = optimized_results[first_symbol]['equity_curve'].values
            min_len = min(len(baseline_equity), len(optimized_equity))
            if min_len > 0:
                baseline_equity = baseline_equity[:min_len] / baseline_equity[0]
                optimized_equity = optimized_equity[:min_len] / optimized_equity[0]
                axes[1, 1].plot(baseline_equity, label='Baseline', linewidth=2)
                axes[1, 1].plot(optimized_equity, label='Optimized', linewidth=2)
                axes[1, 1].set_title(f'Normalized Equity Curve ({first_symbol})')
                axes[1, 1].legend()
                axes[1, 1].grid(True, linestyle='--', alpha=0.7)
            else:
                axes[1, 1].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"参数对比图已保存到: {save_path}")
        return save_path

    def generate_backtest_report(self, data_dict: Dict[str, pd.DataFrame], initial_balance: Optional[float] = None):
        """生成并导出 HTML 格式的完整回测报告"""
        if MODE != "research":
            return
            
        # 检查是否有优化结果
        if not self.optimizer.best_params or not self.optimizer.results:
            self.logger.warning("没有可用的优化结果，无法生成报告。")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join("results", f"backtest_report_{timestamp}.html")
        os.makedirs("results", exist_ok=True)
        os.makedirs("plots", exist_ok=True)

        # 生成图片并获取路径
        sens_plot_path = self.plot_parameter_sensitivity(save_path=os.path.join("plots", f"sensitivity_{timestamp}.png"))
        equity_plot_path = self.plot_equity_curve(data_dict, save_path=os.path.join("plots", f"equity_curve_{timestamp}.png"), initial_balance=initial_balance)
        comp_plot_path = self.plot_best_parameters_comparison(data_dict, save_path=os.path.join("plots", f"comparison_{timestamp}.png"))
        
        # 将图片转换为 base64 编码以嵌入 HTML
        def img_to_base64(path):
            if not path or not os.path.exists(path): return ""
            with open(path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        
        sens_img_base64 = img_to_base64(sens_plot_path)
        equity_img_base64 = img_to_base64(equity_plot_path)
        comp_img_base64 = img_to_base64(comp_plot_path)

        # 准备性能指标表格
        best_result_aggregated = next((r for r in self.optimizer.results if r['params'] == self.optimizer.best_params), None)
        metrics_table_html = ""
        if best_result_aggregated:
            metrics_table_html = """
            <table>
                <thead>
                    <tr><th>Metric</th><th>Value</th></tr>
                </thead>
                <tbody>
            """
            metrics = {
                'Avg Net Profit': f"{best_result_aggregated['avg_net_profit']:.2f}",
                'Avg Sharpe Ratio': f"{best_result_aggregated['avg_sharpe_ratio']:.2f}",
                'Avg Sortino Ratio': f"{best_result_aggregated['avg_sortino_ratio']:.2f}",
                'Avg Calmar Ratio': f"{best_result_aggregated['avg_calmar_ratio']:.2f}",
                'Avg MAR Ratio': f"{best_result_aggregated['avg_mar_ratio']:.2f}",
                'Avg Max Drawdown': f"{best_result_aggregated['avg_max_drawdown']:.2f}",
                'Avg Win Rate': f"{best_result_aggregated['avg_win_rate']:.2%}",
                'Avg Profit Factor': f"{best_result_aggregated['avg_profit_factor']:.2f}",
                'Avg Total Trades': f"{best_result_aggregated['avg_trades']:.0f}",
                'Avg Holding Period': f"{best_result_aggregated['avg_holding_period']/3600:.2f} hours",
                'Avg Trade Frequency (annual)': f"{best_result_aggregated['avg_trade_frequency']:.2f}",
            }
            for metric, value in metrics.items():
                metrics_table_html += f"<tr><td>{metric}</td><td>{value}</td></tr>"
            metrics_table_html += "</tbody></table>"
        else:
            metrics_table_html = "<p>无法找到最佳参数的详细指标。</p>"

        # 准备参数表格
        params_table_html = ""
        if self.optimizer.best_params:
            params_table_html = """
                <table>
                    <thead>
                        <tr><th>Parameter</th><th>Value</th></tr>
                    </thead>
                    <tbody>
            """
            for param, value in self.optimizer.best_params.items():
                params_table_html += f"<tr><td>{param}</td><td>{value}</td></tr>"
            params_table_html += "</tbody></table>"
        else:
            params_table_html = "<p>无最佳参数。</p>"
        
        # HTML 模板 - 完整版
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <title>回测优化报告 - {timestamp}</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 2rem; background-color: #f4f7f9; }}
                .container {{ max-width: 1200px; margin: auto; background: #fff; padding: 2rem; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                h1, h2 {{ color: #004085; border-bottom: 2px solid #e9ecef; padding-bottom: 0.5rem; margin-bottom: 1.5rem; }}
                h1 {{ text-align: center; font-size: 2.5rem; }}
                table {{ width: 100%; border-collapse: collapse; margin-bottom: 2rem; background-color: #fff; }}
                th, td {{ border: 1px solid #dee2e6; padding: 0.75rem; text-align: left; vertical-align: top; }}
                th {{ background-color: #e9ecef; font-weight: bold; }}
                .image-container {{ text-align: center; margin-bottom: 2rem; }}
                img {{ max-width: 100%; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
                .highlight-box {{ background-color: #e2f0fb; border: 1px solid #b3d1f3; padding: 1rem; border-radius: 6px; margin-bottom: 1.5rem; }}
                .flex-container {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
                .flex-item {{ flex: 1; min-width: 45%; margin: 1rem; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>回测优化报告</h1>
                <p><strong>生成时间:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p><strong>优化目标:</strong> {self.optimizer.optimization_history[-1]['objective'] if self.optimizer.optimization_history else 'N/A'}</p>
                <p><strong>交易品种:</strong> {', '.join(self.optimizer.optimization_history[-1]['symbols']) if self.optimizer.optimization_history and 'symbols' in self.optimizer.optimization_history[-1] else 'N/A'}</p>

                <h2>最佳参数</h2>
                <div class="highlight-box">
                    {params_table_html}
                </div>

                <h2>核心绩效指标</h2>
                <div class="highlight-box">
                    {metrics_table_html}
                </div>

                <h2>可视化分析</h2>
                <div class="image-container">
                    <h3>策略净值曲线 (与 Buy & Hold 对比)</h3>
                    <img src="data:image/png;base64,{equity_img_base64}" alt="Equity Curve">
                </div>

                <div class="image-container">
                    <h3>参数敏感性分析</h3>
                    <img src="data:image/png;base64,{sens_img_base64}" alt="Parameter Sensitivity">
                </div>

                <div class="image-container">
                    <h3>优化前后性能对比</h3>
                    <img src="data:image/png;base64,{comp_img_base64}" alt="Comparison">
                </div>

                <h2>优化历史</h2>
                <div class="highlight-box">
                    <pre>{json.dumps(self.optimizer.optimization_history, indent=2, default=str)}</pre>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        self.logger.info(f"回测报告已成功导出至：{report_path}")

# ================== 示例数据生成和使用 ==================
def generate_sample_data(symbols: List[str], periods: int = 1000) -> Dict[str, pd.DataFrame]:
    """生成示例数据"""
    np.random.seed(42) # 增加随机种子以确保可重复性
    data_dict = {}
    
    for symbol in symbols:
        np.random.seed(hash(symbol) % 1000)
        dates = pd.date_range('2023-01-01', periods=periods, freq='1h')
        
        base_price = 100 if symbol == 'BTCUSDT' else 50
        trend = np.linspace(0, 0.2, periods)
        returns = np.random.normal(0, 0.02, periods)
        close_prices = base_price * np.exp(np.cumsum(returns)) * (1 + trend)
        
        open_prices = close_prices * (1 + np.random.normal(0, 0.005, periods))
        high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, periods)))
        low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, periods)))
        
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
    data_dict = generate_sample_data(symbols, periods=2000)
    
    # 两种参数网格示例
    # 1. 单参数网格 (会生成散点图)
    param_grid_single = {
        'ema_short': [8, 12, 16, 20],
    }

    # 2. 多参数网格 (会生成热力图和散点图)
    param_grid_multi = {
        'ema_short': [8, 12, 16],
        'ema_long': [20, 26, 32],
    }
    
    # 实例化优化器并运行优化
    optimizer = OptimizerStub()
    best_params = optimizer.optimize_parameters(
        data_dict=data_dict,
        param_grid=param_grid_multi,
        objective='composite_score',
        max_workers=4,
        max_combinations=500,
        initial_balance=1000
    )
    
    # 实例化报告生成器，并使用优化器实例
    report_generator = ReportGenerator(optimizer)
    
    # 保存优化结果和生成报告
    report_generator.save_optimization_results()
    
    # 只在研究模式下生成报告
    if MODE == "research":
        report_generator.generate_backtest_report(data_dict)
    
    print(f"优化完成，最佳参数: {best_params}")
    
    if MODE == "research":
        print("已生成回测报告，请查看 results 文件夹。")
    else:
        print("已保存最佳参数到JSON文件。")
