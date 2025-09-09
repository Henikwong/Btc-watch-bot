import os
import sys
import time
import ccxt
import pandas as pd
import numpy as np
import ta
import logging
import asyncio
import signal
import json
import math
import requests
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ================== 配置参数 ==================
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
SYMBOLS_CONFIG = [s.strip() for s in os.getenv("SYMBOLS", "LTC/USDT,DOGE/USDT,XRP/USDT,ADA/USDT,LINK/USDT").split(",") if s.strip()]
TIMEFRAME = os.getenv("MACD_FILTER_TIMEFRAME", "4h")
LEVERAGE = int(os.getenv("LEVERAGE", "15"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))
TELEGRAM_SUMMARY_INTERVAL = int(os.getenv("TELEGRAM_SUMMARY_INTERVAL", "3600"))  # 默认每小时发送一次摘要

# 风险管理配置
MAX_ACCOUNT_RISK = float(os.getenv("MAX_ACCOUNT_RISK", "0.3"))  # 最大账户风险30%
MAX_SYMBOL_RISK = float(os.getenv("MAX_SYMBOL_RISK", "0.1"))  # 单币种最大风险10%
DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", "0.05"))  # 单日最大亏损5%
POSITION_SIZING_MODE = os.getenv("POSITION_SIZING_MODE", "fixed")  # fixed或percentage

# 止损配置
STOP_LOSS_PER_SYMBOL = -1000  # 单币种亏损1000USDT时止损

# Telegram 配置
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# 重试参数
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.0"))

# 币安最小名义价值要求（USDT）
MIN_NOTIONAL = {
    "LTC/USDT": 20,
    "XRP/USDT": 5,
    "ADA/USDT": 5,
    "DOGE/USDT": 20,
    "LINK/USDT": 20,
    "BTC/USDT": 10,
    "ETH/USDT": 10,
    "BNB/USDT": 10,
    "SOL/USDT": 10,
    "DOT/USDT": 10,
    "AVAX/USDT": 10,
    "MATIC/USDT": 10,
    "UNI/USDT": 10,
    "SUI/USDT": 10,
}

# ================== 日志设置 ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('cointech2u_bot.log')]
)
logger = logging.getLogger("CoinTech2uBot")

# ================== 明确的层级配置 ==================
class LayerConfiguration:
    def __init__(self):
        # 明确的层级配置：从第1层到第10层
        self.layers = {
            1: {
                'trigger_percentage': 0.02678,  # 2.678%
                'base_size': 6.0,               # 基础交易大小
                'trend_catch_size': 5.0,        # 趋势捕捉加仓大小
                'description': '第一层基础仓位'
            },
            2: {
                'trigger_percentage': 0.05,     # 5%
                'base_size': 12.0,              # 2倍基础
                'trend_catch_size': 7.0,        # 趋势捕捉加仓大小
                'description': '第二层加仓'
            },
            3: {
                'trigger_percentage': 0.06,     # 6%
                'base_size': 24.0,              # 4倍基础
                'trend_catch_size': 10.0,       # 趋势捕捉加仓大小
                'description': '第三层加仓'
            },
            4: {
                'trigger_percentage': 0.07,     # 7%
                'base_size': 48.0,              # 8倍基础
                'trend_catch_size': 15.0,       # 趋势捕捉加仓大小
                'description': '第四层加仓'
            },
            5: {
                'trigger_percentage': 0.08,     # 8%
                'base_size': 96.0,              # 16倍基础
                'trend_catch_size': 20.0,       # 趋势捕捉加仓大小
                'description': '第五层加仓'
            },
            6: {
                'trigger_percentage': 0.09,     # 9%
                'base_size': 192.0,             # 32倍基础
                'trend_catch_size': 25.0,       # 趋势捕捉加仓大小
                'description': '第六层加仓'
            },
            7: {
                'trigger_percentage': 0.10,     # 10%
                'base_size': 384.0,             # 64倍基础
                'trend_catch_size': 30.0,       # 趋势捕捉加仓大小
                'description': '第七层加仓'
            },
            8: {
                'trigger_percentage': 0.13,     # 13%
                'base_size': 768.0,             # 128倍基础
                'trend_catch_size': 35.0,       # 趋势捕捉加仓大小
                'description': '第八层加仓'
            },
            9: {
                'trigger_percentage': 0.14,     # 14%
                'base_size': 1536.0,            # 256倍基础
                'trend_catch_size': 40.0,       # 趋势捕捉加仓大小
                'description': '第九层加仓'
            },
            10: {
                'trigger_percentage': 0.15,     # 15%
                'base_size': 3072.0,            # 512倍基础
                'trend_catch_size': 45.0,       # 趋势捕捉加仓大小
                'description': '第十层加仓'
            }
        }
        
        # 最大层数
        self.max_layers = len(self.layers)
        
    def get_trigger_percentage(self, layer: int) -> float:
        """获取指定层级的触发百分比"""
        if layer > self.max_layers:
            return self.layers[self.max_layers]['trigger_percentage']
        return self.layers[layer]['trigger_percentage']
    
    def get_base_size(self, layer: int) -> float:
        """获取指定层级的基础交易大小"""
        if layer > self.max_layers:
            return self.layers[self.max_layers]['base_size']
        return self.layers[layer]['base_size']
    
    def get_trend_catch_size(self, layer: int) -> float:
        """获取指定层级的趋势捕捉加仓大小"""
        if layer > self.max_layers:
            return self.layers[self.max_layers]['trend_catch_size']
        return self.layers[layer]['trend_catch_size']
    
    def get_description(self, layer: int) -> str:
        """获取指定层级的描述"""
        if layer > self.max_layers:
            return self.layers[self.max_layers]['description']
        return self.layers[layer]['description']
    
    def print_configuration(self):
        """打印层级配置"""
        logger.info("📋 层级配置:")
        for layer in range(1, self.max_layers + 1):
            config = self.layers[layer]
            logger.info(f"  第{layer}层: {config['description']}, "
                       f"触发阈值={config['trigger_percentage']*100}%, "
                       f"基础大小=${config['base_size']}, "
                       f"趋势捕捉大小=${config['trend_catch_size']}")

# ================== Telegram 通知类 ==================
class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
    def send_message(self, message: str) -> bool:
        """发送消息到Telegram"""
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram 配置未设置，跳过发送消息")
            return False
            
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info("Telegram 消息发送成功")
                return True
            else:
                logger.error(f"Telegram 消息发送失败: {response.text}")
                return False
        except Exception as e:
            logger.error(f"发送 Telegram 消息时出错: {e}")
            return False

# ================== 技术分析函数 ==================
def analyze_trend(df: pd.DataFrame) -> Tuple[float, str]:
    """分析趋势方向和强度，使用多时间框架确认
    
    Returns:
        Tuple[float, str]: (趋势强度, 趋势方向) 方向为 'long', 'short' 或 'neutral'
    """
    try:
        # 计算多时间框架EMA指标
        ema_fast = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        ema_slow = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        
        # 计算RSI指标
        rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # 计算MACD
        macd = ta.trend.MACD(df['close'])
        macd_line = macd.macd()
        macd_signal = macd.macd_signal()
        
        # 计算ADX趋势强度
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
        
        # 计算成交量指标
        volume = df['volume']
        volume_ma = volume.rolling(window=20).mean()
        
        # 获取最新值
        latest_ema_fast = ema_fast.iloc[-1]
        latest_ema_slow = ema_slow.iloc[-1]
        latest_rsi = rsi.iloc[-1]
        latest_macd_line = macd_line.iloc[-1]
        latest_macd_signal = macd_signal.iloc[-1]
        latest_adx = adx.iloc[-1]
        latest_volume = volume.iloc[-1]
        latest_volume_ma = volume_ma.iloc[-1]
        
        # 判断趋势方向
        trend_direction = "neutral"
        if latest_ema_fast > latest_ema_slow and latest_macd_line > latest_macd_signal:
            trend_direction = "long"
        elif latest_ema_fast < latest_ema_slow and latest_macd_line < latest_macd_signal:
            trend_direction = "short"
            
        # 计算综合趋势强度 (0-1之间)
        trend_strength = min(latest_adx / 100, 1.0)  # ADX归一化
        trend_strength = max(trend_strength, 0)
        
        # 考虑RSI极端值
        if (trend_direction == "long" and latest_rsi > 70) or (trend_direction == "short" and latest_rsi < 30):
            trend_strength *= 0.7  # 在超买超卖区域减弱信号强度
            
        # 成交量确认：如果成交量没有放大，减弱信号强度
        if latest_volume < latest_volume_ma * 1.2:
            trend_strength *= 0.8
            
        return trend_strength, trend_direction
        
    except Exception as e:
        logger.error(f"趋势分析错误: {e}")
        return 0, "neutral"

# ================== 工具函数 ==================
def quantize_amount(amount: float, market) -> float:
    """量化交易量到交易所允许的精度"""
    try:
        # 尝试从filters获取stepSize
        step = None
        for f in market['info'].get('filters', []):
            if f.get('filterType') == 'LOT_SIZE':
                step = float(f.get('stepSize'))
                break
        
        if step is None:
            # 回退到精度
            prec = market.get('precision', {}).get('amount')
            if isinstance(prec, int):
                return round(amount, prec)
            # 默认精度
            return float(Decimal(amount).quantize(Decimal('0.000001'), rounding=ROUND_DOWN))
        
        # 使用Decimal进行精确计算
        step_dec = Decimal(str(step))
        amount_dec = Decimal(str(amount))
        # 向下取整到step的倍数
        quantized = (amount_dec // step_dec) * step_dec
        return float(quantized)
    except Exception as e:
        logger.error(f"量化数量失败: {e}")
        # 回退到简单舍入
        return round(amount, 6)

# ================== 交易所接口 ==================
class BinanceFutureAPI:
    def __init__(self, api_key: str, api_secret: str, symbols: List[str]):
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbols = symbols
        self.exchange = None
        self.symbol_info = {}  # 缓存交易对信息

    def initialize(self) -> bool:
        """同步初始化交易所连接"""
        try:
            self.exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'options': {'defaultType': 'future'},
                'enableRateLimit': True
            })
            
            # 加载所有交易对信息
            markets = self.exchange.load_markets()
            valid_symbols = []
            
            for symbol in self.symbols:
                if symbol in markets:
                    self.symbol_info[symbol] = markets[symbol]
                    try:
                        self.exchange.set_leverage(LEVERAGE, symbol)
                        logger.info(f"设置杠杆 {symbol} {LEVERAGE}x")
                        valid_symbols.append(symbol)
                    except Exception as e:
                        logger.warning(f"设置杠杆失败 {symbol}: {e}")
                else:
                    logger.warning(f"交易对 {symbol} 不存在，跳过")
            
            # 更新有效的交易对列表
            self.symbols = valid_symbols
            
            logger.info("交易所初始化成功")
            return True
        except Exception as e:
            logger.error(f"交易所初始化失败: {e}")
            return False

    def get_balance(self) -> float:
        """获取账户余额"""
        try:
            balance = self.exchange.fetch_balance()
            return float(balance['USDT']['free'])
        except Exception as e:
            logger.error(f"获取余额失败: {e}")
            return 0.0

    def get_ohlcv_data(self, symbol: str, timeframe: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """获取K线数据"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"K线获取失败 {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker['last'])
        except Exception as e:
            logger.error(f"获取价格失败 {symbol}: {e}")
            return None

    def get_positions(self, symbol: str) -> Dict[str, dict]:
        """获取当前持仓信息"""
        try:
            positions = self.exchange.fetch_positions([symbol])
            result = {}
            for pos in positions:
                if float(pos['contracts']) > 0:
                    # 使用side作为键，而不是positionSide
                    side = pos['side'].lower()
                    result[side] = {
                        'size': float(pos['contracts']),
                        'entry_price': float(pos['entryPrice']),
                        'side': pos['side'],
                    }
            return result
        except Exception as e:
            logger.error(f"获取持仓失败 {symbol}: {e}")
            return {}

    def create_order_with_fallback(self, symbol: str, side: str, contract_size: float, position_side: str):
        """创建订单，如果失败则尝试回退到单向模式"""
        for attempt in range(MAX_RETRIES):
            try:
                # 尝试带positionSide下单
                params = {"positionSide": position_side}
                order = self.exchange.create_order(
                    symbol,
                    'market',
                    side.lower(),
                    contract_size,
                    None,
                    params
                )
                return order
            except Exception as e:
                err_msg = str(e)
                # 如果是position side不匹配的错误，尝试不带positionSide下单
                if "-4061" in err_msg or "position side does not match" in err_msg.lower():
                    logger.warning(f"positionSide与账户设置不符，尝试不带positionSide重试")
                    try:
                        order = self.exchange.create_order(
                            symbol,
                            'market',
                            side.lower(),
                            contract_size
                        )
                        return order
                    except Exception as e2:
                        logger.error(f"重试不带positionSide失败: {e2}")
                        if attempt == MAX_RETRIES - 1:
                            return None
                else:
                    logger.error(f"下单失败: {e}")
                    if attempt == MAX_RETRIES - 1:
                        return None
            
            # 等待一段时间后重试
            time.sleep(RETRY_DELAY * (2 ** attempt))
        
        return None

    def execute_market_order(self, symbol: str, side: str, amount: float, position_side: str) -> bool:
        """执行市价订单"""
        try:
            # 获取交易对信息
            market = self.symbol_info.get(symbol)
            if not market:
                logger.error(f"找不到交易对信息: {symbol}")
                return False
                
            # 获取当前价格
            current_price = self.get_current_price(symbol)
            if current_price is None:
                logger.error(f"无法获取 {symbol} 的价格")
                return False
                
            # 计算合约数量
            contract_size = amount / current_price
            
            # 量化到交易所精度
            contract_size = quantize_amount(contract_size, market)
            
            # 确保不低于最小交易量
            min_amount = market['limits']['amount']['min']
            if contract_size < min_amount:
                contract_size = min_amount
                logger.warning(f"交易量低于最小值，使用最小值: {min_amount}")

            # 检查最小名义价值
            min_notional = MIN_NOTIONAL.get(symbol, 10)  # 默认10 USDT
            notional_value = contract_size * current_price
            
            # 如果名义价值不足，调整合约数量
            if notional_value < min_notional:
                # 计算需要的最小合约数量
                min_contract_size = min_notional / current_price
                contract_size = max(contract_size, min_contract_size)
                
                # 重新量化到交易所精度
                contract_size = quantize_amount(contract_size, market)
                
                # 重新计算名义价值
                notional_value = contract_size * current_price
                
                # 如果仍然不足，继续增加直到满足要求
                step = 0.001  # 默认步长
                for f in market['info'].get('filters', []):
                    if f.get('filterType') == 'LOT_SIZE':
                        step = float(f.get('stepSize'))
                        break
                
                while notional_value < min_notional:
                    contract_size += step
                    contract_size = quantize_amount(contract_size, market)
                    notional_value = contract_size * current_price
                    
                    # 安全保护，避免无限循环
                    if contract_size > min_contract_size * 10:
                        logger.error(f"无法满足最小名义价值要求: {notional_value:.2f} < {min_notional}")
                        return False
                
                logger.warning(f"调整合约数量以满足最小名义价值: {contract_size:.6f}, 名义价值: {notional_value:.2f} USDT")
            
            # 创建订单
            order = self.create_order_with_fallback(symbol, side, contract_size, position_side)
            if order:
                logger.info(f"订单成功 {symbol} {side} {contract_size:.6f} ({position_side}) - 订单ID: {order['id']}")
                return True
            else:
                logger.error(f"下单失败 {symbol} {side}: 所有重试均失败")
                return False
                
        except Exception as e:
            logger.error(f"下单失败 {symbol} {side}: {e}")
            return False

# ================== 风险管理类 ==================
class RiskManager:
    def __init__(self, api: BinanceFutureAPI):
        self.api = api
        self.initial_balance = 0.0
        self.daily_balance = 0.0
        self.daily_loss_limit = 0.0
        self.max_symbol_risk = 0.0
        self.max_account_risk = 0.0
        self.today = datetime.now().date()
        
    def initialize(self):
        """初始化风险管理"""
        balance = self.api.get_balance()
        self.initial_balance = balance
        self.daily_balance = balance
        self.daily_loss_limit = balance * DAILY_LOSS_LIMIT
        self.max_symbol_risk = balance * MAX_SYMBOL_RISK
        self.max_account_risk = balance * MAX_ACCOUNT_RISK
        
        logger.info(f"💰 风险管理初始化: 初始余额=${balance:.2f}, 单日最大亏损=${self.daily_loss_limit:.2f}")
        logger.info(f"📊 单币种最大风险=${self.max_symbol_risk:.2f}, 账户最大风险=${self.max_account_risk:.2f}")
    
    def check_daily_loss(self):
        """检查每日亏损限制"""
        current_balance = self.api.get_balance()
        daily_pnl = self.daily_balance - current_balance
        
        if daily_pnl >= self.daily_loss_limit:
            logger.warning(f"🚨 达到每日亏损限制: ${daily_pnl:.2f} >= ${self.daily_loss_limit:.2f}")
            return False
        
        return True
    
    def check_symbol_risk(self, symbol: str, position_side: str, current_price: float, positions: List[dict]) -> bool:
        """检查单币种风险"""
        if not positions:
            return True
            
        total_size = sum(p['size'] for p in positions)
        total_value = sum(p['size'] * p['entry_price'] for p in positions)
        avg_price = total_value / total_size
        
        # 计算当前浮动盈亏
        if position_side == 'long':
            unrealized_pnl = total_size * (current_price - avg_price)
        else:  # short
            unrealized_pnl = total_size * (avg_price - current_price)
        
        # 如果浮动亏损超过单币种风险限制，停止加仓
        if unrealized_pnl <= -self.max_symbol_risk:
            logger.warning(f"🚨 {symbol} {position_side.upper()} 达到单币种风险限制: ${unrealized_pnl:.2f} <= -${self.max_symbol_risk:.2f}")
            return False
            
        return True
    
    def check_account_risk(self) -> bool:
        """检查账户整体风险"""
        current_balance = self.api.get_balance()
        total_loss = self.initial_balance - current_balance
        
        if total_loss >= self.max_account_risk:
            logger.warning(f"🚨 达到账户最大风险限制: ${total_loss:.2f} >= ${self.max_account_risk:.2f}")
            return False
            
        return True
    
    def calculate_position_size(self, symbol: str, current_price: float, base_size: float) -> float:
        """根据风险管理计算仓位大小"""
        if POSITION_SIZING_MODE == "percentage":
            # 基于账户余额的百分比计算仓位大小
            balance = self.api.get_balance()
            risk_per_trade = balance * 0.01  # 每笔交易风险1%
            position_size_usdt = risk_per_trade * 2  # 因为是双仓，所以乘以2
        else:
            # 固定仓位大小
            position_size_usdt = base_size
        
        # 确保不超过单币种风险限制
        position_size = position_size_usdt / current_price
        
        logger.info(f"📏 {symbol} 风险调整后仓位: USDT价值={position_size_usdt:.3f}, 数量={position_size:.6f}")
        return position_size
    
    def should_trade(self, symbol: str) -> bool:
        """检查是否允许交易"""
        # 检查每日亏损限制
        if not self.check_daily_loss():
            return False
            
        # 检查账户整体风险
        if not self.check_account_risk():
            return False
            
        return True
    
    def reset_daily_balance(self):
        """重置每日余额（在每天开始时调用）"""
        today = datetime.now().date()
        if today != self.today:
            self.daily_balance = self.api.get_balance()
            self.today = today
            logger.info(f"📅 新的一天开始，重置每日余额: ${self.daily_balance:.2f}")

# ================== 精准加仓监控系统 ==================
class PrecisionLayerMonitor:
    def __init__(self, martingale_manager, api, telegram_notifier=None):
        self.martingale = martingale_manager
        self.api = api
        self.telegram = telegram_notifier
        # 记录每个仓位的最后检查时间和价格
        self.last_check = {}
        # 设置更频繁的检查间隔（秒）
        self.check_interval = 30
        
    def initialize_symbol(self, symbol: str):
        """初始化交易对监控"""
        if symbol not in self.last_check:
            self.last_check[symbol] = {
                'long': {'last_check_time': 0, 'last_price': 0},
                'short': {'last_check_time': 0, 'last_price': 0}
            }
    
    def monitor_all_symbols(self):
        """监控所有交易对的加仓条件"""
        current_time = time.time()
        
        for symbol in self.martingale.symbols:
            self.initialize_symbol(symbol)
            
            # 获取当前价格
            current_price = self.api.get_current_price(symbol)
            if current_price is None:
                continue
                
            # 监控多仓
            self.monitor_position(symbol, 'long', current_price, current_time)
            
            # 监控空仓
            self.monitor_position(symbol, 'short', current_price, current_time)
    
    def monitor_position(self, symbol: str, position_side: str, current_price: float, current_time: float):
        """监控特定方向的仓位"""
        # 检查是否有该方向的仓位
        if not self.martingale.positions[symbol][position_side]:
            return
            
        # 检查是否达到检查间隔
        last_check = self.last_check[symbol][position_side]['last_check_time']
        if current_time - last_check < self.check_interval:
            return
            
        # 更新最后检查时间
        self.last_check[symbol][position_side]['last_check_time'] = current_time
        self.last_check[symbol][position_side]['last_price'] = current_price
        
        # 计算当前盈亏百分比
        positions = self.martingale.positions[symbol][position_side]
        total_size = sum(p['size'] for p in positions)
        total_value = sum(p['size'] * p['entry_price'] for p in positions)
        avg_price = total_value / total_size
        
        if position_side == 'long':
            pnl_pct = (current_price - avg_price) / avg_price
        else:  # short
            pnl_pct = (avg_price - current_price) / avg_price
            
        current_layers = len(positions)
        
        # 记录详细监控信息
        logger.info(f"🔍 {symbol} {position_side.upper()} 第{current_layers}层监控: "
                   f"均价={avg_price:.6f}, 现价={current_price:.6f}, 盈亏={pnl_pct*100:.4f}%")
        
        # 检查是否达到加仓阈值
        threshold = self.martingale.layer_config.get_trigger_percentage(current_layers + 1)
            
        # 只有当亏损达到触发阈值时才加仓
        if pnl_pct <= -threshold:
            logger.info(f"🎯 {symbol} {position_side.upper()} 第{current_layers}层达到加仓阈值: "
                       f"亏损{pnl_pct*100:.4f}% >= 阈值{threshold*100:.4f}%")
            
            # 检查是否已达到最大层数
            if current_layers >= self.martingale.layer_config.max_layers:
                logger.info(f"⛔ {symbol} {position_side.upper()} 已达到最大层数 {self.martingale.layer_config.max_layers}")
                return
                
            # 执行加仓
            self.execute_add_layer(symbol, position_side, current_price)
    
    def execute_add_layer(self, symbol: str, position_side: str, current_price: float):
        """执行加仓操作"""
        side = "buy" if position_side == "long" else "sell"
        position_side_param = "LONG" if position_side == "long" else "SHORT"
        
        # 计算加仓大小
        layer_size = self.martingale.calculate_layer_size(symbol, position_side, current_price, False)
        
        current_layers = len(self.martingale.positions[symbol][position_side])
        logger.info(f"📈 {symbol} {position_side.upper()} 第{current_layers}层准备加仓第{current_layers+1}层，"
                   f"方向: {side}, 大小: {layer_size:.8f}")
        
        # 执行市价订单
        success = self.api.execute_market_order(symbol, side, layer_size, position_side_param)
        if success:
            self.martingale.add_position(symbol, side, layer_size, current_price, False)
            logger.info(f"✅ {symbol} {position_side.upper()} 第{current_layers+1}层加仓成功")
            
            # 发送Telegram通知
            if self.telegram:
                telegram_msg = (f"<b>📈 加仓成功</b>\n"
                               f"{symbol} {position_side.upper()} 第{current_layers+1}层\n"
                               f"操作: {side.upper()}\n"
                               f"数量: {layer_size:.8f}\n"
                               f"价格: ${current_price:.6f}")
                self.telegram.send_message(telegram_msg)
        else:
            logger.error(f"❌ {symbol} {position_side.upper()} 加仓失败")
            
            # 发送Telegram通知
            if self.telegram:
                telegram_msg = (f"<b>❌ 加仓失败</b>\n"
                               f"{symbol} {position_side.upper()} 第{current_layers+1}层\n"
                               f"操作: {side.upper()}\n"
                               f"数量: {layer_size:.8f}\n"
                               f"价格: ${current_price:.6f}")
                self.telegram.send_message(telegram_msg)

# ================== 双仓马丁策略管理 ==================
class DualMartingaleManager:
    def __init__(self, telegram_notifier: TelegramNotifier = None, symbols: List[str] = None, risk_manager: RiskManager = None):
        # 仓位结构: {symbol: {'long': [], 'short': []}}
        self.positions: Dict[str, Dict[str, List[dict]]] = {}
        # 仓位状态文件
        self.positions_file = "positions.json"
        # Telegram 通知器
        self.telegram = telegram_notifier
        # 风险管理器
        self.risk_manager = risk_manager
        # 交易对列表
        self.symbols = symbols or []
        # 层级配置
        self.layer_config = LayerConfiguration()
        # 初始化所有交易对
        for symbol in self.symbols:
            self.initialize_symbol(symbol)
        # 加载保存的仓位
        self.load_positions()

    def initialize(self):
        """初始化时打印层级配置"""
        self.layer_config.print_configuration()
        
    def initialize_symbol(self, symbol: str):
        """初始化交易对仓位结构"""
        if symbol not in self.positions:
            self.positions[symbol] = {'long': [], 'short': []}

    def add_position(self, symbol: str, side: str, size: float, price: float, is_trend_catch: bool = False):
        """添加仓位到对应方向"""
        self.initialize_symbol(symbol)
        position_side = 'long' if side.lower() == 'buy' else 'short'
        layer = len(self.positions[symbol][position_side]) + 1
        
        self.positions[symbol][position_side].append({
            'side': side,
            'size': size,
            'entry_price': price,
            'timestamp': datetime.now(),
            'layer': layer,
            'is_trend_catch': is_trend_catch
        })
        
        # 记录日志
        log_msg = f"📊 {symbol} {position_side.upper()} 第{layer}层仓位: {side} {size:.6f} @ {price:.2f}"
        if is_trend_catch:
            log_msg += " (趋势捕捉)"
        logger.info(log_msg)
        
        # 发送 Telegram 通知
        if self.telegram:
            if is_trend_catch:
                telegram_msg = f"<b>🎯 趋势捕捉加仓</b>\n{symbol} {position_side.upper()} 第{layer}层\n操作: {side.upper()}\n数量: {size:.6f}\n价格: ${price:.2f}"
            else:
                telegram_msg = f"<b>🔄 常规加仓</b>\n{symbol} {position_side.upper()} 第{layer}层\n操作: {side.upper()}\n数量: {size:.6f}\n价格: ${price:.2f}"
            self.telegram.send_message(telegram_msg)
        
        # 保存仓位状态
        self.save_positions()

    def should_add_trend_catch_layer(self, symbol: str, position_side: str, trend_strength: float) -> Tuple[bool, int]:
        """检查是否应该进行趋势捕捉加仓"""
        self.initialize_symbol(symbol)
        
        # 检查是否有持仓
        if not self.positions[symbol][position_side]:
            return False, 0
            
        # 检查趋势强度
        if trend_strength < 0.7:  # 趋势信号强度阈值
            return False, 0
            
        # 获取当前仓位层数
        current_layers = len(self.positions[symbol][position_side])
        next_layer = current_layers + 1
        
        # 检查是否已达到最大层数
        if next_layer > self.layer_config.max_layers:
            logger.info(f"⚠️ {symbol} {position_side.upper()} 已达到最大层数 {self.layer_config.max_layers}")
            return False, 0
            
        return True, next_layer

    def should_add_layer(self, symbol: str, position_side: str, current_price: float) -> bool:
        """检查是否应该加仓（使用明确的层级配置）"""
        self.initialize_symbol(symbol)
        
        # 检查是否已达到最大层数
        current_layers = len(self.positions[symbol][position_side])
        if current_layers >= self.layer_config.max_layers:
            logger.info(f"⚠️ {symbol} {position_side.upper()} 已达到最大层数 {self.layer_config.max_layers}")
            return False
            
        positions = self.positions[symbol][position_side]
        if not positions:
            return False
            
        # 检查风险管理
        if self.risk_manager and not self.risk_manager.check_symbol_risk(symbol, position_side, current_price, positions):
            return False
            
        total_size = sum(p['size'] for p in positions)
        total_value = sum(p['size'] * p['entry_price'] for p in positions)
        avg_price = total_value / total_size
        
        # 计算当前盈亏百分比
        if position_side == 'long':
            pnl_pct = (current_price - avg_price) / avg_price
        else:  # short
            pnl_pct = (avg_price - current_price) / avg_price
            
        # 获取当前层对应的触发阈值
        threshold = self.layer_config.get_trigger_percentage(current_layers + 1)
        
        logger.info(f"📈 {symbol} {position_side.upper()} 第{current_layers}层仓位监控: "
                   f"均价={avg_price:.6f}, 现价={current_price:.6f}, "
                   f"盈亏={pnl_pct*100:.2f}%, 第{current_layers+1}层触发阈值={threshold*100:.2f}%")
        
        # 检查止损条件
        unrealized_pnl = total_size * (current_price - avg_price) if position_side == 'long' else total_size * (avg_price - current_price)
        if unrealized_pnl <= STOP_LOSS_PER_SYMBOL:
            logger.warning(f"🚨 {symbol} {position_side.upper()} 达到止损条件: {unrealized_pnl:.2f} USDT <= {STOP_LOSS_PER_SYMBOL} USDT")
            return False
        
        # 只有当亏损达到触发阈值时才加仓
        return pnl_pct <= -threshold

    def calculate_layer_size(self, symbol: str, position_side: str, current_price: float, is_trend_catch: bool = False) -> float:
        """计算加仓大小（使用明确的层级配置）"""
        self.initialize_symbol(symbol)
        current_layers = len(self.positions[symbol][position_side])
        next_layer = current_layers + 1
        
        if next_layer > self.layer_config.max_layers:
            logger.error(f"层数 {next_layer} 超过最大层数 {self.layer_config.max_layers}")
            return 0
        
        if is_trend_catch:
            # 使用趋势捕捉加仓配置
            size_in_usdt = self.layer_config.get_trend_catch_size(next_layer)
        else:
            # 使用基础加仓配置
            size_in_usdt = self.layer_config.get_base_size(next_layer)
        
        # 如果有风险管理器，使用风险管理计算仓位大小
        if self.risk_manager:
            size = self.risk_manager.calculate_position_size(symbol, current_price, size_in_usdt)
        else:
            size = size_in_usdt / current_price
        
        layer_desc = self.layer_config.get_description(next_layer)
        logger.info(f"📏 {symbol} {position_side.upper()} {layer_desc}: "
                   f"USDT价值={size_in_usdt:.3f}, 数量={size:.6f}")
        return size

    def calculate_initial_size(self, current_price: float, symbol: str = "") -> float:
        """计算初始仓位大小（使用明确的层级配置）"""
        # 使用第一层的基础交易大小
        size_in_usdt = self.layer_config.get_base_size(1)
        
        # 如果有风险管理器，使用风险管理计算仓位大小
        if self.risk_manager:
            size = self.risk_manager.calculate_position_size(symbol, current_price, size_in_usdt)
        else:
            size = size_in_usdt / current_price
        
        logger.info(f"📏 {symbol} 初始仓位计算: USDT价值={size_in_usdt:.3f}, 数量={size:.6f}")
        return size
        
    def should_close_position(self, symbol: str, position_side: str, current_price: float) -> bool:
        """检查是否应该平仓（止盈）"""
        self.initialize_symbol(symbol)
        if not self.positions[symbol][position_side]:
            return False
            
        positions = self.positions[symbol][position_side]
        total_size = sum(p['size'] for p in positions)
        total_value = sum(p['size'] * p['entry_price'] for p in positions)
        avg_price = total_value / total_size
        
        # 计算当前盈亏百分比
        if position_side == 'long':
            pnl_pct = (current_price - avg_price) / avg_price
        else:  # short
            pnl_pct = (avg_price - current_price) / avg_price
            
        # 如果盈利超过止盈点，止盈平仓
        if pnl_pct >= 0.015:  # 1.5%止盈
            current_layers = len(positions)
            logger.info(f"🎯 {symbol} {position_side.upper()} 第{current_layers}层仓位 盈利超过1.5%，止盈平仓")
            
            # 发送 Telegram 通知
            if self.telegram:
                profit_usdt = total_size * (current_price - avg_price) if position_side == 'long' else total_size * (avg_price - current_price)
                telegram_msg = f"<b>🎯 止盈触发</b>\n{symbol} {position_side.upper()} 第{current_layers}层\n盈利: {pnl_pct*100:.2f}%\n收益: ${profit_usdt:.2f}\n平均成本: ${avg_price:.2f}\n当前价格: ${current_price:.2f}"
                self.telegram.send_message(telegram_msg)
                
            return True
            
        return False

    def get_position_size(self, symbol: str, position_side: str) -> float:
        """获取某个方向的仓位总大小"""
        self.initialize_symbol(symbol)
        return sum(p['size'] for p in self.positions[symbol][position_side])
    
    def get_position_layers(self, symbol: str, position_side: str) -> int:
        """获取某个方向的仓位层数"""
        self.initialize_symbol(symbol)
        return len(self.positions[symbol][position_side])
    
    def clear_positions(self, symbol: str, position_side: str):
        """清空某个方向的仓位记录"""
        self.initialize_symbol(symbol)
        self.positions[symbol][position_side] = []
        logger.info(f"🔄 {symbol} {position_side.upper()} 仓位记录已清空")
        # 保存仓位状态
        self.save_positions()
        
    def has_open_positions(self, symbol: str) -> bool:
        """检查是否有任何方向的仓位"""
        self.initialize_symbol(symbol)
        return len(self.positions[symbol]['long']) > 0 or len(self.positions[symbol]['short']) > 0
    
    def save_positions(self):
        """保存仓位状态到文件"""
        try:
            # 转换为可序列化的格式
            serializable_positions = {}
            for symbol, sides in self.positions.items():
                serializable_positions[symbol] = {}
                for side, positions in sides.items():
                    serializable_positions[symbol][side] = []
                    for pos in positions:
                        serializable_positions[symbol][side].append({
                            'side': pos['side'],
                            'size': pos['size'],
                            'entry_price': pos['entry_price'],
                            'timestamp': pos['timestamp'].isoformat(),
                            'layer': pos['layer'],
                            'is_trend_catch': pos.get('is_trend_catch', False)
                        })
            
            serializable_data = {
                'positions': serializable_positions,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(self.positions_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
        except Exception as e:
            logger.error(f"保存仓位状态失败: {e}")
    
    def load_positions(self):
        """从文件加载仓位状态"""
        try:
            if os.path.exists(self.positions_file):
                with open(self.positions_file, 'r') as f:
                    data = json.load(f)
                
                # 加载仓位数据
                serializable_positions = data.get('positions', {})
                for symbol, sides in serializable_positions.items():
                    self.positions[symbol] = {}
                    for side, positions in sides.items():
                        self.positions[symbol][side] = []
                        for pos in positions:
                            self.positions[symbol][side].append({
                                'side': pos['side'],
                                'size': pos['size'],
                                'entry_price': pos['entry_price'],
                                'timestamp': datetime.fromisoformat(pos['timestamp']),
                                'layer': pos['layer'],
                                'is_trend_catch': pos.get('is_trend_catch', False)
                            })
                
                logger.info("仓位状态已从文件加载")
        except Exception as e:
            logger.error(f"加载仓位状态失败: {e}")
            
    def check_and_fill_base_position(self, api: BinanceFutureAPI, symbol: str):
        """检查并填充基础仓位 - 核心功能：一测试到没有仓位就补上"""
        try:
            # 获取交易所当前仓位
            exchange_positions = api.get_positions(symbol)
            has_long = exchange_positions.get('long') and exchange_positions['long']['size'] > 0
            has_short = exchange_positions.get('short') and exchange_positions['short']['size'] > 0
            
            # 检查本地记录
            self.initialize_symbol(symbol)
            local_has_long = len(self.positions[symbol]['long']) > 0
            local_has_short = len(self.positions[symbol]['short']) > 0
            
            # 如果交易所和本地记录不一致，以交易所为准
            if has_long != local_has_long or has_short != local_has_short:
                logger.warning(f"⚠️ {symbol} 本地与交易所仓位记录不一致，同步中...")
                # 清空本地记录
                self.positions[symbol]['long'] = []
                self.positions[symbol]['short'] = []
                
                # 重新记录仓位
                if has_long:
                    self.add_position(symbol, "buy", exchange_positions['long']['size'], exchange_positions['long']['entry_price'])
                if has_short:
                    self.add_position(symbol, "sell", exchange_positions['short']['size'], exchange_positions['short']['entry_price'])
            
            # 检查是否需要补仓
            if not has_long or not has_short:
                logger.info(f"🔄 {symbol} 检测到仓位不完整，准备补仓")
                
                # 获取当前价格
                current_price = api.get_current_price(symbol)
                if current_price is None:
                    logger.error(f"无法获取 {symbol} 的价格，跳过补仓")
                    return
                
                # 计算初始仓位大小
                position_size = self.calculate_initial_size(current_price, symbol)
                if position_size <= 0:
                    logger.error(f"{symbol} 仓位大小计算错误，跳过补仓")
                    return
                
                # 补多仓
                if not has_long:
                    logger.info(f"📈 {symbol} 补多仓，大小: {position_size:.6f}")
                    success = api.execute_market_order(symbol, "buy", position_size, "LONG")
                    if success:
                        self.add_position(symbol, "buy", position_size, current_price)
                        logger.info(f"✅ {symbol} 多仓补充成功")
                    else:
                        logger.error(f"❌ {symbol} 多仓补充失败")
                
                # 补空仓
                if not has_short:
                    logger.info(f"📉 {symbol} 补空仓，大小: {position_size:.6f}")
                    success = api.execute_market_order(symbol, "sell", position_size, "SHORT")
                    if success:
                        self.add_position(symbol, "sell", position_size, current_price)
                        logger.info(f"✅ {symbol} 空仓补充成功")
                    else:
                        logger.error(f"❌ {symbol} 空仓补充失败")
        except Exception as e:
            logger.error(f"检查并填充基础仓位错误 {symbol}: {e}")

    def get_position_summary(self, symbol: str) -> str:
        """获取仓位摘要信息"""
        self.initialize_symbol(symbol)
        long_layers = len(self.positions[symbol]['long'])
        short_layers = len(self.positions[symbol]['short'])
        
        if long_layers == 0 and short_layers == 0:
            return f"{symbol}: 无仓位"
        
        long_size = sum(p['size'] for p in self.positions[symbol]['long'])
        short_size = sum(p['size'] for p in self.positions[symbol]['short'])
        
        return f"{symbol}: 多仓{long_layers}层({long_size:.6f}) | 空仓{short_layers}层({short_size:.6f})"
    
    def get_all_positions_summary(self) -> str:
        """获取所有交易对的仓位摘要"""
        summary = "📊 <b>仓位摘要</b>\n\n"
        for symbol in self.symbols:
            self.initialize_symbol(symbol)
            long_layers = len(self.positions[symbol]['long'])
            short_layers = len(self.positions[symbol]['short'])
            
            if long_layers > 0 or short_layers > 0:
                long_size = sum(p['size'] for p in self.positions[symbol]['long'])
                short_size = sum(p['size'] for p in self.positions[symbol]['short'])
                
                # 计算平均入场价格
                long_avg_price = 0
                if long_layers > 0:
                    long_total_value = sum(p['size'] * p['entry_price'] for p in self.positions[symbol]['long'])
                    long_avg_price = long_total_value / long_size
                
                short_avg_price = 0
                if short_layers > 0:
                    short_total_value = sum(p['size'] * p['entry_price'] for p in self.positions[symbol]['short'])
                    short_avg_price = short_total_value / short_size
                
                summary += f"<b>{symbol}</b>\n"
                summary += f"  多仓: {long_layers}层, 数量: {long_size:.6f}, 均价: ${long_avg_price:.4f}\n"
                summary += f"  空仓: {short_layers}层, 数量: {short_size:.6f}, 均价: ${short_avg_price:.4f}\n\n"
        
        if summary == "📊 <b>仓位摘要</b>\n\n":
            summary += "暂无持仓"
            
        return summary

# ================== 主交易机器人 ==================
class CoinTech2uBot:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.api = BinanceFutureAPI(BINANCE_API_KEY, BINANCE_API_SECRET, symbols)
        
        # 初始化风险管理器
        self.risk_manager = RiskManager(self.api)
        
        # 初始化 Telegram 通知器
        self.telegram = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        
        # 初始化策略管理器（传入风险管理器）
        self.martingale = DualMartingaleManager(self.telegram, symbols, self.risk_manager)
        
        # 初始化精准加仓监控系统
        self.layer_monitor = PrecisionLayerMonitor(self.martingale, self.api, self.telegram)
        
        # 上次发送摘要的时间
        self.last_summary_time = 0
        
        self.running = True
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        
        # 初始化层级配置
        self.martingale.initialize()

    def shutdown(self, signum, frame):
        logger.info("收到关闭信号，停止交易...")
        self.running = False
        self.martingale.save_positions()
        
        # 发送关闭通知
        if self.telegram:
            self.telegram.send_message("<b>🛑 交易机器人已停止</b>")

    def run(self):
        if not self.api.initialize():
            logger.error("交易所初始化失败，程序退出")
            # 发送错误通知
            if self.telegram:
                self.telegram.send_message("<b>❌ 交易所初始化失败，程序退出</b>")
            return
        
        # 初始化风险管理
        self.risk_manager.initialize()
            
        logger.info("🚀 开始CoinTech2u策略交易...")
        
        # 发送启动通知
        if self.telegram:
            # 获取第一层配置
            first_layer_config = self.martingale.layer_config.layers[1]
            telegram_msg = (f"<b>🚀 CoinTech2u交易机器人已启动</b>\n"
                           f"交易对: {', '.join(self.symbols)}\n"
                           f"杠杆: {LEVERAGE}x\n"
                           f"基础仓位: ${first_layer_config['base_size']}\n"
                           f"最大层数: {self.martingale.layer_config.max_layers}\n"
                           f"风险管理: 单日最大亏损{DAILY_LOSS_LIMIT*100}%, 单币种最大风险{MAX_SYMBOL_RISK*100}%")
            self.telegram.send_message(telegram_msg)
        
        # 程序启动时立即对所有币对开双仓
        logger.info("🔄 程序启动时对所有币对开双仓")
        for symbol in self.symbols:
            self.open_immediate_hedge(symbol)
        
        # 记录启动时间
        start_time = time.time()
        self.last_summary_time = start_time
        
        while self.running:
            try:
                # 重置每日余额（如果需要）
                self.risk_manager.reset_daily_balance()
                
                # 检查是否允许交易
                if not self.risk_manager.should_trade(""):
                    logger.warning("⚠️ 交易被风险管理阻止")
                    # 发送警告通知
                    if self.telegram:
                        self.telegram.send_message("<b>⚠️ 交易被风险管理阻止</b>\n已达到风险限制，暂停交易")
                    time.sleep(POLL_INTERVAL)
                    continue
                
                balance = self.api.get_balance()
                logger.info(f"当前余额: {balance:.2f} USDT")
                
                # 监控所有仓位状态
                self.layer_monitor.monitor_all_symbols()
                
                # 打印所有币种的仓位摘要
                self.print_position_summary()
                
                # 检查是否需要发送Telegram摘要
                current_time = time.time()
                if current_time - self.last_summary_time >= TELEGRAM_SUMMARY_INTERVAL:
                    self.send_telegram_summary(balance)
                    self.last_summary_time = current_time
                
                for symbol in self.symbols:
                    # 检查并填充基础仓位 - 核心功能：一测试到没有仓位就补上
                    self.martingale.check_and_fill_base_position(self.api, symbol)
                    # 处理交易逻辑
                    self.process_symbol(symbol)
                    
                time.sleep(POLL_INTERVAL)
            except Exception as e:
                logger.error(f"交易循环错误: {e}")
                # 发送错误通知
                if self.telegram:
                    self.telegram.send_message(f"<b>❌ 交易循环错误</b>\n{str(e)}")
                time.sleep(10)

    def send_telegram_summary(self, balance: float):
        """发送仓位摘要到Telegram"""
        if not self.telegram:
            return
            
        summary = self.martingale.get_all_positions_summary()
        summary += f"\n💰 <b>账户余额</b>: ${balance:.2f} USDT"
        summary += f"\n⏰ <b>更新时间</b>: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        self.telegram.send_message(summary)

    def print_position_summary(self):
        """打印所有币种的仓位摘要"""
        logger.info("📋 仓位摘要:")
        for symbol in self.symbols:
            summary = self.martingale.get_position_summary(symbol)
            logger.info(f"   {summary}")

    def open_immediate_hedge(self, symbol: str):
        """程序启动时立即开双仓（增加风险管理）"""
        # 检查是否允许交易
        if not self.risk_manager.should_trade(symbol):
            logger.warning(f"⚠️ {symbol} 开仓被风险管理阻止")
            return
            
        # 检查交易所是否已有仓位
        exchange_positions = self.api.get_positions(symbol)
        has_long = exchange_positions.get('long') and exchange_positions['long']['size'] > 0
        has_short = exchange_positions.get('short') and exchange_positions['short']['size'] > 0
        
        if has_long or has_short:
            logger.info(f"⏩ {symbol} 交易所已有仓位，跳过开仓")
            # 同步本地记录
            if has_long:
                self.martingale.add_position(symbol, "buy", exchange_positions['long']['size'], exchange_positions['long']['entry_price'])
            if has_short:
                self.martingale.add_position(symbol, "sell", exchange_positions['short']['size'], exchange_positions['short']['entry_price'])
            return
        
        # 获取当前价格
        current_price = self.api.get_current_price(symbol)
        if current_price is None:
            logger.error(f"无法获取 {symbol} 的价格，跳过")
            return
        
        # 计算初始仓位大小（使用风险管理）
        position_size = self.martingale.calculate_initial_size(current_price, symbol)
        if position_size <= 0:
            logger.error(f"{symbol} 仓位大小计算错误，跳过")
            return
        
        logger.info(f"📊 {symbol} 准备开双仓，价格: {current_price:.2f}, 大小: {position_size:.6f}")
        
        # 同时开多仓和空仓
        long_success = self.api.execute_market_order(symbol, "buy", position_size, "LONG")
        short_success = self.api.execute_market_order(symbol, "sell", position_size, "SHORT")
        
        if long_success and short_success:
            logger.info(f"✅ {symbol} 已同时开多空仓位: 多单 {position_size:.6f} | 空单 {position_size:.6f}")
            # 记录仓位
            self.martingale.add_position(symbol, "buy", position_size, current_price)
            self.martingale.add_position(symbol, "sell", position_size, current
