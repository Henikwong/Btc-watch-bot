import os
import time
import ccxt
import pandas as pd
import numpy as np
import ta
import logging
import json
from datetime import datetime, timedelta
import signal
import sys
import asyncio
import aiohttp
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Any, Set
import requests
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from queue import Queue
import cachetools
from abc import ABC, abstractmethod
import uuid
import hashlib
import sqlite3
from contextlib import contextmanager
import math

# 修复WebSocket导入问题
try:
    from websockets import connect
    from websockets import exceptions as ws_exceptions
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("警告: websockets 库未安装，WebSocket功能将不可用")

# ================== 环境检测 ==================
IS_RAILWAY = os.environ.get('RAILWAY_ENVIRONMENT') is not None

# ================== Railway优化的日志配置 ==================
# 清除任何现有的日志处理器
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Railway特定的日志格式化器
class RailwayLogFormatter(logging.Formatter):
    LEVEL_MAP = {
        logging.DEBUG: 'DEBUG',
        logging.INFO: 'INFO',
        logging.WARNING: 'WARNING',
        logging.ERROR: 'ERROR',
        logging.CRITICAL: 'CRITICAL'
    }
    
    def format(self, record):
        # 在Railway环境中，使用更简洁的日志格式
        if IS_RAILWAY:
            record.levelname = self.LEVEL_MAP.get(record.levelno, record.levelname)
            return super().format(record)
        return super().format(record)

# 配置根日志记录器
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_level = logging.INFO

# 创建处理器
handler = logging.StreamHandler(sys.stdout)
formatter = RailwayLogFormatter(log_format)
handler.setFormatter(formatter)

# 配置根日志记录器
logging.basicConfig(
    level=log_level,
    handlers=[handler]
)

# 禁用过于详细的库日志
logging.getLogger("ccxt").setLevel(logging.INFO)
logging.getLogger("websockets").setLevel(logging.INFO)
logging.getLogger("asyncio").setLevel(logging.INFO)

# ================== 数据类型定义 ==================
class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

class Mode(Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"

@dataclass
class TradeSignal:
    symbol: str
    side: OrderSide
    price: float
    atr: float
    quantity: float
    timestamp: datetime
    confidence: float = 1.0
    timeframe: str = "1h"

@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str] = None
    error: Optional[str] = None
    symbol: Optional[str] = None
    side: Optional[OrderSide] = None
    amount_usdt: Optional[float] = None

@dataclass
class BalanceInfo:
    total: float
    free: float
    used: float

@dataclass
class HealthStatus:
    total_symbols: int
    connected_symbols: int
    disconnected_symbols: int
    last_check: datetime
    error_count: int

# ================== 配置管理 ==================
class Config:
    """完整的配置管理"""
    def __init__(self):
        self.mode = Mode.LIVE
        self.hedge_mode = True
        self.leverage = 15
        self.max_position_size_percent = 5.0  # 单笔最大仓位百分比
        self.max_portfolio_risk_percent = 20.0  # 最大组合风险百分比
        self.symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "XRP/USDT", 
                       "SOL/USDT", "DOT/USDT", "DOGE/USDT", "AVAX/USDT", "MATIC/USDT", "LTC/USDT"]
        self.timeframes = ["1h", "4h"]
        self.atr_period = 14
        self.atr_multiplier = 1.5
        self.risk_per_trade = 2.0  # 每笔交易风险百分比
        self.min_order_value = 10.0  # 最小订单价值(USDT)
        self.db_path = "trading_bot.db"
        self.telegram_bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        self.max_retries = 3
        self.retry_delay = 2
        self.health_check_interval = 1800  # 健康检查间隔(秒)

# ================== 数据库管理 ==================
class DatabaseManager:
    """数据库管理器"""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        with self._get_connection() as conn:
            # 创建状态表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS bot_state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')
            
            # 创建交易记录表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    symbol TEXT,
                    side TEXT,
                    price REAL,
                    quantity REAL,
                    timestamp DATETIME,
                    order_id TEXT,
                    amount_usdt REAL
                )
            ''')
            
            # 创建信号记录表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id TEXT PRIMARY KEY,
                    symbol TEXT,
                    side TEXT,
                    price REAL,
                    atr REAL,
                    quantity REAL,
                    timestamp DATETIME,
                    confidence REAL,
                    timeframe TEXT
                )
            ''')
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def save_state(self, state: Dict):
        """保存状态到数据库"""
        with self._get_connection() as conn:
            for key, value in state.items():
                conn.execute(
                    "INSERT OR REPLACE INTO bot_state (key, value) VALUES (?, ?)",
                    (key, json.dumps(value))
                )
            conn.commit()
    
    def load_state(self) -> Dict:
        """从数据库加载状态"""
        state = {}
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT key, value FROM bot_state")
            for row in cursor:
                try:
                    state[row[0]] = json.loads(row[1])
                except:
                    state[row[0]] = row[1]
        return state
    
    def save_trade(self, trade: OrderResult, signal: TradeSignal):
        """保存交易记录"""
        with self._get_connection() as conn:
            trade_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO trades (id, symbol, side, price, quantity, timestamp, order_id, amount_usdt) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trade_id,
                    signal.symbol,
                    signal.side.value,
                    signal.price,
                    signal.quantity,
                    signal.timestamp.isoformat(),
                    trade.order_id,
                    trade.amount_usdt
                )
            )
            conn.commit()
    
    def save_signal(self, signal: TradeSignal):
        """保存信号记录"""
        with self._get_connection() as conn:
            signal_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO signals (id, symbol, side, price, atr, quantity, timestamp, confidence, timeframe) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    signal_id,
                    signal.symbol,
                    signal.side.value,
                    signal.price,
                    signal.atr,
                    signal.quantity,
                    signal.timestamp.isoformat(),
                    signal.confidence,
                    signal.timeframe
                )
            )
            conn.commit()

# ================== 日志系统 ==================
class AdvancedLogger:
    """高级日志系统"""
    def __init__(self, name: str, db_manager: DatabaseManager):
        self.logger = logging.getLogger(name)
        self.db_manager = db_manager
    
    def debug(self, msg: str):
        self.logger.debug(msg)
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
    
    def critical(self, msg: str):
        self.logger.critical(msg)

# ================== 缓存系统 ==================
class TimedCache:
    """带时间戳的缓存系统"""
    def __init__(self, ttl: int = 300, maxsize: int = 1000):
        self.cache = cachetools.TTLCache(maxsize=maxsize, ttl=ttl)
    
    def get(self, key: str) -> Any:
        return self.cache.get(key)
    
    def set(self, key: str, value: Any):
        self.cache[key] = value

# ================== 交易所接口 ==================
# 装饰器，用于封装重试逻辑
def retry_with_exponential_backoff(retries=3, delay=2, backoff=2):
    """指数退避重试装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == retries - 1:
                        raise e
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            return None
        return wrapper
    return decorator

class ExchangeInterface(ABC):
    """交易所接口抽象类"""
    @abstractmethod
    async def initialize(self):
        pass
    
    @abstractmethod
    async def get_balance(self) -> BalanceInfo:
        pass
    
    @abstractmethod
    async def get_positions(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: Optional[float] = None) -> OrderResult:
        pass
    
    @abstractmethod
    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> List[List]:
        pass

class BinanceExchange(ExchangeInterface):
    """币安交易所实现"""
    def __init__(self, config: Config, mode: Mode = Mode.LIVE):
        self.config = config
        self.mode = mode
        self.exchange = None
        self.initialized = False
        self.db_manager = DatabaseManager(config.db_path)
        self.logger = AdvancedLogger("BinanceExchange", self.db_manager)
    
    async def initialize(self):
        """初始化交易所连接"""
        try:
            if self.mode == Mode.LIVE:
                api_key = os.environ.get('BINANCE_API_KEY')
                api_secret = os.environ.get('BINANCE_API_SECRET')
                
                if not api_key or not api_secret:
                    raise ValueError("币安API密钥未设置")
                
                self.exchange = ccxt.binance({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'future',
                        'adjustForTimeDifference': True,
                    }
                })
                
                # 设置杠杆和模式
                for symbol in self.config.symbols:
                    market = self.exchange.market(symbol)
                    if market['future']:
                        await asyncio.get_event_loop().run_in_executor(
                            None, 
                            lambda: self.exchange.set_leverage(self.config.leverage, symbol)
                        )
                        
                        if self.config.hedge_mode:
                            await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: self.exchange.set_position_mode(True, symbol)
                            )
            else:
                # 模拟/回测模式
                self.exchange = ccxt.binance({
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'future',
                        'adjustForTimeDifference': True,
                    }
                })
            
            self.initialized = True
            self.logger.info("交易所初始化成功")
            
        except Exception as e:
            self.logger.error(f"交易所初始化失败: {e}")
            raise
    
    async def get_balance(self) -> BalanceInfo:
        """获取余额信息"""
        if not self.initialized:
            raise RuntimeError("交易所未初始化")
        
        try:
            balance = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.exchange.fetch_balance()
            )
            
            total = float(balance['total']['USDT'])
            free = float(balance['free']['USDT'])
            used = float(balance['used']['USDT'])
            
            return BalanceInfo(total=total, free=free, used=used)
        
        except Exception as e:
            self.logger.error(f"获取余额失败: {e}")
            raise
    
    async def get_positions(self) -> Dict[str, Any]:
        """获取所有持仓"""
        if not self.initialized:
            raise RuntimeError("交易所未初始化")
        
        try:
            positions = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.exchange.fetch_positions()
            )
            
            result = {}
            for pos in positions:
                if float(pos['contracts']) > 0:
                    result[pos['symbol']] = {
                        'side': PositionSide.LONG if pos['side'] == 'long' else PositionSide.SHORT,
                        'contracts': float(pos['contracts']),
                        'entryPrice': float(pos['entryPrice']),
                        'notional': float(pos['notional']),
                        'unrealizedPnl': float(pos['unrealizedPnl'])
                    }
            
            return result
        
        except Exception as e:
            self.logger.error(f"获取持仓失败: {e}")
            raise
    
    @retry_with_exponential_backoff()
    async def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: Optional[float] = None) -> OrderResult:
        """创建订单"""
        if not self.initialized:
            return OrderResult(success=False, error="交易所未初始化")
        
        try:
            order = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.exchange.create_order(symbol, order_type, side, amount, price) if price else 
                       self.exchange.create_order(symbol, order_type, side, amount)
            )
            
            return OrderResult(
                success=True,
                order_id=order['id'],
                symbol=symbol,
                side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
                amount_usdt=amount * (price if price else await self.get_current_price(symbol))
            )
        
        except Exception as e:
            self.logger.error(f"创建订单失败: {e}")
            return OrderResult(success=False, error=str(e))
    
    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> List[List]:
        """获取K线数据"""
        if not self.initialized:
            raise RuntimeError("交易所未初始化")
        
        try:
            ohlcv = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            )
            
            return ohlcv
        
        except Exception as e:
            self.logger.error(f"获取K线数据失败: {e}")
            raise
    
    async def get_current_price(self, symbol: str) -> float:
        """获取当前价格"""
        if not self.initialized:
            raise RuntimeError("交易所未初始化")
        
        try:
            ticker = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.exchange.fetch_ticker(symbol)
            )
            
            return float(ticker['last'])
        
        except Exception as e:
            self.logger.error(f"获取当前价格失败: {e}")
            raise

# ================== WebSocket数据处理器 ==================
class WebSocketDataHandler:
    """增强的WebSocket实时数据处理器"""
    def __init__(self, config: Config, exchange: ExchangeInterface):
        self.config = config
        self.exchange = exchange
        self.websockets = {}
        self.last_prices = {}
        self.connected = False
        self.db_manager = DatabaseManager(config.db_path)
        self.logger = AdvancedLogger("WebSocketDataHandler", self.db_manager)
    
    async def initialize(self):
        """初始化WebSocket连接"""
        if not WEBSOCKETS_AVAILABLE:
            self.logger.warning("WebSocket库不可用，使用REST API轮询")
            return
        
        try:
            # 为每个交易对创建WebSocket连接
            for symbol in self.config.symbols:
                symbol_normalized = symbol.replace('/', '').lower()
                ws_url = f"wss://fstream.binance.com/ws/{symbol_normalized}@ticker"
                
                try:
                    self.websockets[symbol] = await connect(ws_url)
                    self.last_prices[symbol] = 0.0
                    self.logger.info(f"WebSocket连接已建立: {symbol}")
                except Exception as e:
                    self.logger.error(f"WebSocket连接失败 {symbol}: {e}")
            
            self.connected = True
            self.logger.info("WebSocket处理器初始化完成")
            
            # 启动数据接收任务
            asyncio.create_task(self._receive_data())
        
        except Exception as e:
            self.logger.error(f"WebSocket初始化失败: {e}")
    
    async def _receive_data(self):
        """接收WebSocket数据"""
        while self.connected:
            for symbol, ws in self.websockets.items():
                try:
                    data = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    data = json.loads(data)
                    
                    if 'c' in data:  # 最新价格
                        self.last_prices[symbol] = float(data['c'])
                
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self.logger.error(f"接收WebSocket数据失败 {symbol}: {e}")
            
            await asyncio.sleep(0.1)
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格（优先使用WebSocket）"""
        if self.connected and symbol in self.last_prices:
            return self.last_prices[symbol]
        
        # 回退到REST API
        try:
            return await self.exchange.get_current_price(symbol)
        except:
            return None
    
    async def close(self):
        """关闭所有WebSocket连接"""
        self.connected = False
        for ws in self.websockets.values():
            await ws.close()
        self.websockets.clear()
        self.logger.info("所有WebSocket连接已关闭")

# ================== 动态ATR计算器 ==================
class DynamicATRCalculator:
    """动态ATR计算器"""
    def __init__(self, exchange: ExchangeInterface, config: Config):
        self.exchange = exchange
        self.config = config
        self.atr_cache = TimedCache(ttl=300)  # 5分钟缓存
        self.db_manager = DatabaseManager(config.db_path)
        self.logger = AdvancedLogger("ATRCalculator", self.db_manager)
    
    async def calculate_atr(self, symbol: str, timeframe: str, period: int = None) -> float:
        """计算ATR指标"""
        if period is None:
            period = self.config.atr_period
        
        cache_key = f"{symbol}_{timeframe}_{period}"
        cached_atr = self.atr_cache.get(cache_key)
        
        if cached_atr is not None:
            return cached_atr
        
        try:
            # 获取K线数据
            ohlcv = await self.exchange.get_ohlcv(symbol, timeframe, limit=period + 20)
            
            if len(ohlcv) < period + 1:
                self.logger.warning(f"数据不足，无法计算ATR: {symbol}")
                return 0.0
            
            # 转换为DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 计算ATR
            atr_indicator = ta.volatility.AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=period
            )
            
            atr = atr_indicator.average_true_range().iloc[-1]
            
            # 缓存结果
            self.atr_cache.set(cache_key, atr)
            
            return atr
        
        except Exception as e:
            self.logger.error(f"计算ATR失败 {symbol}: {e}")
            return 0.0

# ================== 多周期信号生成器 ==================
class MultiTimeframeSignalGenerator:
    """多周期信号生成器"""
    def __init__(self, exchange: ExchangeInterface, atr_calculator: DynamicATRCalculator, config: Config):
        self.exchange = exchange
        self.atr_calculator = atr_calculator
        self.config = config
        self.signal_cache = TimedCache(ttl=60)  # 1分钟缓存
        self.db_manager = DatabaseManager(config.db_path)
        self.logger = AdvancedLogger("SignalGenerator", self.db_manager)
    
    async def generate_signals(self, symbol: str) -> List[TradeSignal]:
        """为指定交易对生成交易信号"""
        signals = []
        
        for timeframe in self.config.timeframes:
            try:
                signal = await self._generate_signal_for_timeframe(symbol, timeframe)
                if signal:
                    signals.append(signal)
            except Exception as e:
                self.logger.error(f"生成信号失败 {symbol} {timeframe}: {e}")
        
        return signals
    
    async def _generate_signal_for_timeframe(self, symbol: str, timeframe: str) -> Optional[TradeSignal]:
        """为指定交易对和时间框架生成交易信号"""
        cache_key = f"{symbol}_{timeframe}"
        cached_signal = self.signal_cache.get(cache_key)
        
        if cached_signal is not None:
            return cached_signal
        
        try:
            # 获取K线数据
            ohlcv = await self.exchange.get_ohlcv(symbol, timeframe, limit=100)
            
            if len(ohlcv) < 50:  # 确保有足够的数据
                return None
            
            # 转换为DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 计算指标
            df['sma20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
            df['sma50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            
            # 获取当前价格
            current_price = await self.exchange.get_current_price(symbol)
            
            # 计算ATR
            atr = await self.atr_calculator.calculate_atr(symbol, timeframe)
            
            # 生成信号
            last_row = df.iloc[-1]
            prev_row = df.iloc[-2]
            
            # 简单的双均线策略
            if last_row['sma20'] > last_row['sma50'] and prev_row['sma20'] <= prev_row['sma50']:
                # 金叉 - 买入信号
                signal = TradeSignal(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    price=current_price,
                    atr=atr,
                    quantity=0,  # 将在执行时计算
                    timestamp=datetime.now(),
                    timeframe=timeframe
                )
                self.signal_cache.set(cache_key, signal)
                return signal
            
            elif last_row['sma20'] < last_row['sma50'] and prev_row['sma20'] >= prev_row['sma50']:
                # 死叉 - 卖出信号
                signal = TradeSignal(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    price=current_price,
                    atr=atr,
                    quantity=0,  # 将在执行时计算
                    timestamp=datetime.now(),
                    timeframe=timeframe
                )
                self.signal_cache.set(cache_key, signal)
                return signal
        
        except Exception as e:
            self.logger.error(f"生成时间框架信号失败 {symbol} {timeframe}: {e}")
        
        return None

# ================== 指标系统 ==================
class IndicatorSystem:
    """完整的指标计算系统"""
    def __init__(self, config: Config):
        self.config = config
        self.db_manager = DatabaseManager(config.db_path)
        self.logger = AdvancedLogger("IndicatorSystem", self.db_manager)
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标"""
        try:
            # 趋势指标
            df['sma20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
            df['sma50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
            df['ema20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
            
            # 动量指标
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            df['macd'] = ta.trend.MACD(df['close']).macd()
            df['macd_signal'] = ta.trend.MACD(df['close']).macd_signal()
            df['macd_diff'] = ta.trend.MACD(df['close']).macd_diff()
            
            # 波动率指标
            df['bb_upper'] = ta.volatility.BollingerBands(df['close']).bollinger_hband()
            df['bb_lower'] = ta.volatility.BollingerBands(df['close']).bollinger_lband()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_upper']
            
            # 成交量指标
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            
            return df
        
        except Exception as e:
            self.logger.error(f"计算指标失败: {e}")
            return df

# ================== 交易执行器 ==================
class TradeExecutor:
    """优化的交易执行器"""
    def __init__(self, exchange: BinanceExchange, config: Config, db_manager: DatabaseManager):
        self.exchange = exchange
        self.config = config
        self.db_manager = db_manager
        self.open_orders = {}
        self.logger = AdvancedLogger("TradeExecutor", db_manager)
    
    async def execute_signal(self, signal: TradeSignal) -> OrderResult:
        """执行交易信号"""
        try:
            # 计算仓位大小
            position_size = await self.calculate_position_size(signal.symbol, signal.atr)
            
            if position_size < self.config.min_order_value:
                self.logger.info(f"订单价值低于最小值: {position_size} < {self.config.min_order_value}")
                return OrderResult(success=False, error="订单价值过低")
            
            # 计算数量
            quantity = position_size / signal.price
            
            # 更新信号中的数量
            signal.quantity = quantity
            
            # 创建订单
            order_type = "market"  # 使用市价单
            side = signal.side.value
            
            result = await self.exchange.create_order(
                signal.symbol, order_type, side, quantity
            )
            
            # 保存交易记录
            if result.success:
                self.db_manager.save_trade(result, signal)
                self.logger.info(f"订单执行成功: {signal.symbol} {side} {quantity}")
            else:
                self.logger.error(f"订单执行失败: {result.error}")
            
            return result
        
        except Exception as e:
            self.logger.error(f"执行信号失败: {e}")
            return OrderResult(success=False, error=str(e))
    
    async def calculate_position_size(self, symbol: str, atr: float) -> float:
        """根据风险计算仓位大小"""
        try:
            # 获取账户余额
            balance = await self.exchange.get_balance()
            account_balance = balance.total
            
            # 计算风险资金
            risk_amount = account_balance * (self.config.risk_per_trade / 100)
            
            # 计算基于ATR的仓位大小
            current_price = await self.exchange.get_current_price(symbol)
            atr_stop_loss = current_price * (atr * self.config.atr_multiplier) / current_price
            
            position_size = risk_amount / atr_stop_loss
            
            # 应用最大仓位限制
            max_position_size = account_balance * (self.config.max_position_size_percent / 100)
            position_size = min(position_size, max_position_size)
            
            return position_size
        
        except Exception as e:
            self.logger.error(f"计算仓位大小失败: {e}")
            return 0.0

# ================== 增强的风险管理系统 ==================
class EnhancedRiskManager:
    """增强的风险管理系统 - 修复了只计算亏损的问题"""
    def __init__(self, exchange: ExchangeInterface, config: Config, db_manager: DatabaseManager):
        self.exchange = exchange
        self.config = config
        self.db_manager = db_manager
        self.logger = AdvancedLogger("RiskManager", db_manager)
        self.max_drawdown = config.max_portfolio_risk_percent / 100.0
    
    async def check_portfolio_risk(self) -> bool:
        """检查投资组合风险 - 修复版本，考虑盈亏总额"""
        try:
            # 获取当前持仓
            positions = await self.exchange.get_positions()
            
            # 获取当前余额
            balance = await self.exchange.get_balance()
            total_balance = balance.total
            
            # 计算总盈亏（包括浮动盈亏）
            total_pnl = 0.0
            for symbol, position in positions.items():
                total_pnl += position['unrealizedPnl']
            
            # 计算当前权益（余额 + 浮动盈亏）
            equity = total_balance + total_pnl
            
            # 计算回撤（相对于最高权益）
            state = self.db_manager.load_state()
            peak_equity = state.get('peak_equity', total_balance)
            
            # 更新最高权益
            if equity > peak_equity:
                state['peak_equity'] = equity
                self.db_manager.save_state(state)
            
            # 计算回撤百分比
            drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
            
            self.logger.info(f"权益: {equity:.2f}, 最高权益: {peak_equity:.2f}, 回撤: {drawdown*100:.2f}%")
            
            # 检查是否超过最大回撤
            if drawdown > self.max_drawdown:
                self.logger.warning(f"投资组合回撤超过限制: {drawdown*100:.2f}% > {self.max_drawdown*100:.2f}%")
                return False
            
            return True
        
        except Exception as e:
            self.logger.error(f"检查投资组合风险失败: {e}")
            return True  # 出错时允许继续交易
    
    async def check_symbol_risk(self, symbol: str, signal: TradeSignal) -> bool:
        """检查单个交易对风险"""
        try:
            # 获取当前持仓
            positions = await self.exchange.get_positions()
            
            if symbol not in positions:
                return True  # 没有持仓，允许交易
            
            position = positions[symbol]
            
            # 检查是否已经有同方向持仓
            if (signal.side == OrderSide.BUY and position['side'] == PositionSide.LONG) or \
               (signal.side == OrderSide.SELL and position['side'] == PositionSide.SHORT):
                self.logger.info(f"已有同方向持仓: {symbol} {signal.side}")
                return False
            
            return True
        
        except Exception as e:
            self.logger.error(f"检查交易对风险失败 {symbol}: {e}")
            return True
    
    async def emergency_stop(self):
        """紧急停止 - 平掉所有持仓"""
        self.logger.critical("执行紧急停止!")
        
        try:
            positions = await self.exchange.get_positions()
            
            # 使用线程池并行平仓
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for symbol, position in positions.items():
                    if position['contracts'] > 0:
                        side = 'sell' if position['side'] == PositionSide.LONG else 'buy'
                        futures.append(
                            executor.submit(
                                self._close_position,
                                symbol,
                                side,
                                position['contracts']
                            )
                        )
                
                # 等待所有平仓完成
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if result.success:
                            self.logger.info(f"平仓成功: {result.symbol}")
                        else:
                            self.logger.error(f"平仓失败: {result.error}")
                    except Exception as e:
                        self.logger.error(f"平仓异常: {e}")
        
        except Exception as e:
            self.logger.error(f"紧急停止失败: {e}")
    
    def _close_position(self, symbol: str, side: str, amount: float) -> OrderResult:
        """同步方法平仓"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(
                self.exchange.create_order(symbol, 'market', side, amount)
            )
            
            loop.close()
            return result
        except Exception as e:
            return OrderResult(success=False, error=str(e))

# ================== 警报系统 ==================
class AlertSystem:
    """警报系统"""
    def __init__(self, config: Config, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager
        self.logger = AdvancedLogger("AlertSystem", db_manager)
    
    async def send_telegram_alert(self, message: str):
        """发送Telegram警报"""
        if not self.config.telegram_bot_token or not self.config.telegram_chat_id:
            return
        
        try:
            url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
            payload = {
                'chat_id': self.config.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        self.logger.error(f"Telegram警报发送失败: {await response.text()}")
        
        except Exception as e:
            self.logger.error(f"发送Telegram警报失败: {e}")
    
    async def send_trade_alert(self, signal: TradeSignal, executed: bool, amount_usdt: float = 0):
        """发送交易警报"""
        side_emoji = "🟢" if signal.side == OrderSide.BUY else "🔴"
        status = "执行成功" if executed else "执行失败"
        
        message = f"""
        {side_emoji} <b>交易信号</b> {side_emoji}
        
        🪙 交易对: {signal.symbol}
        📈 方向: {signal.side.value.upper()}
        💰 价格: ${signal.price:.2f}
        📊 数量: {signal.quantity:.4f}
        💵 价值: ${amount_usdt:.2f}
        ⏰ 时间: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
        📶 时间框架: {signal.timeframe}
        
        🚦 状态: {status}
        """
        
        await self.send_telegram_alert(message)
    
    async def send_error_alert(self, error_msg: str):
        """发送错误警报"""
        message = f"""
        🚨 <b>错误警报</b> 🚨
        
        ❌ 错误信息: {error_msg}
        ⏰ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        await self.send_telegram_alert(message)
    
    async def send_health_alert(self, health_status: HealthStatus):
        """发送健康状态警报"""
        message = f"""
        🏥 <b>系统健康状态</b> 🏥
        
        📊 总交易对: {health_status.total_symbols}
        ✅ 已连接: {health_status.connected_symbols}
        ❌ 断开连接: {health_status.disconnected_symbols}
        🐛 错误计数: {health_status.error_count}
        ⏰ 最后检查: {health_status.last_check.strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        await self.send_telegram_alert(message)

# ================== 状态管理器 ==================
class StateManager:
    """增强的状态管理器"""
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.state = self.db_manager.load_state()
        self.logger = AdvancedLogger("StateManager", db_manager)
    
    def save_state(self):
        """保存状态到数据库"""
        self.db_manager.save_state(self.state)
    
    def update_state(self, key: str, value: Any):
        """更新状态值"""
        self.state[key] = value
        self.save_state()
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """获取状态值"""
        return self.state.get(key, default)

# ================== 增强的错误处理 ==================
class EnhancedErrorHandler:
    """增强的错误处理"""
    def __init__(self, alert_system: AlertSystem, db_manager: DatabaseManager):
        self.alert_system = alert_system
        self.db_manager = db_manager
        self.error_count = 0
        self.last_error_time = None
        self.logger = AdvancedLogger("ErrorHandler", db_manager)
    
    async def handle_error(self, error: Exception, context: str = ""):
        """处理错误"""
        self.error_count += 1
        self.last_error_time = datetime.now()
        
        error_msg = f"{context}: {str(error)}" if context else str(error)
        self.logger.error(error_msg)
        
        # 每5个错误发送一次警报
        if self.error_count % 5 == 0:
            await self.alert_system.send_error_alert(f"错误计数: {self.error_count}, 最后错误: {error_msg}")
    
    def reset_error_count(self):
        """重置错误计数"""
        self.error_count = 0
        self.last_error_time = None

# ================== 主交易机器人 ==================
class EnhancedProductionTrader:
    """增强的生产环境交易机器人"""
    def __init__(self):
        self.config = Config()
        self.db_manager = DatabaseManager(self.config.db_path)
        self.logger = AdvancedLogger("Main", self.db_manager)
        self.exchange = BinanceExchange(self.config, self.config.mode)
        self.ws_handler = WebSocketDataHandler(self.config, self.exchange)
        self.atr_calculator = DynamicATRCalculator(self.exchange, self.config)
        self.signal_generator = MultiTimeframeSignalGenerator(self.exchange, self.atr_calculator, self.config)
        self.indicator_system = IndicatorSystem(self.config)
        self.trade_executor = TradeExecutor(self.exchange, self.config, self.db_manager)
        self.risk_manager = EnhancedRiskManager(self.exchange, self.config, self.db_manager)
        self.alert_system = AlertSystem(self.config, self.db_manager)
        self.state_manager = StateManager(self.db_manager)
        self.error_handler = EnhancedErrorHandler(self.alert_system, self.db_manager)
        
        self.running = False
        self.health_check_task = None
    
    async def run(self):
        """运行交易机器人"""
        self.running = True
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        
        try:
            # 初始化交易所
            await self.exchange.initialize()
            
            # 初始化WebSocket连接
            await self.ws_handler.initialize()
            
            # 加载状态
            self.state_manager.state = self.db_manager.load_state()
            
            self.logger.info("交易机器人启动成功")
            await self.alert_system.send_telegram_alert("🚀 交易机器人启动成功")
            
            # 启动健康检查任务
            self.health_check_task = asyncio.create_task(self.health_check_loop())
            
            # 主交易循环
            while self.running:
                try:
                    await self.trading_loop()
                    await asyncio.sleep(60)  # 每分钟运行一次
                except Exception as e:
                    await self.error_handler.handle_error(e, "交易循环")
                    await asyncio.sleep(30)  # 出错后等待30秒
        
        except Exception as e:
            self.logger.critical(f"机器人启动失败: {e}")
            await self.alert_system.send_error_alert(f"机器人启动失败: {e}")
            self.stop()
    
    async def trading_loop(self):
        """交易循环"""
        # 检查投资组合风险
        risk_ok = await self.risk_manager.check_portfolio_risk()
        if not risk_ok:
            self.logger.warning("投资组合风险超过限制，暂停交易")
            await self.alert_system.send_error_alert("投资组合风险超过限制，暂停交易")
            return
        
        # 为每个交易对生成信号
        for symbol in self.config.symbols:
            try:
                signals = await self.signal_generator.generate_signals(symbol)
                
                for signal in signals:
                    # 检查交易对风险
                    symbol_risk_ok = await self.risk_manager.check_symbol_risk(symbol, signal)
                    if not symbol_risk_ok:
                        continue
                    
                    # 执行信号
                    result = await self.trade_executor.execute_signal(signal)
                    
                    # 发送警报
                    await self.alert_system.send_trade_alert(
                        signal, result.success, result.amount_usdt or 0
                    )
                    
                    # 保存信号
                    self.db_manager.save_signal(signal)
                    
                    # 短暂延迟，避免速率限制
                    await asyncio.sleep(1)
            
            except Exception as e:
                await self.error_handler.handle_error(e, f"处理交易对 {symbol}")
    
    async def health_check_loop(self):
        """健康检查循环"""
        while self.running:
            try:
                health_status = await self.check_health()
                
                # 每6次检查发送一次健康报告（3小时一次）
                if self.error_handler.error_count % 6 == 0:
                    await self.alert_system.send_health_alert(health_status)
                
                # 如果断开连接数量超过一半，尝试重新连接
                if health_status.disconnected_symbols > health_status.total_symbols // 2:
                    self.logger.warning("过多WebSocket断开连接，尝试重新连接")
                    await self.ws_handler.close()
                    await self.ws_handler.initialize()
                
                await asyncio.sleep(self.config.health_check_interval)
            
            except Exception as e:
                await self.error_handler.handle_error(e, "健康检查")
                await asyncio.sleep(300)  # 出错后等待5分钟
    
    async def check_health(self) -> HealthStatus:
        """检查系统健康状态"""
        total_symbols = len(self.config.symbols)
        connected_symbols = len(self.ws_handler.last_prices) if self.ws_handler.connected else 0
        disconnected_symbols = total_symbols - connected_symbols
        
        return HealthStatus(
            total_symbols=total_symbols,
            connected_symbols=connected_symbols,
            disconnected_symbols=disconnected_symbols,
            last_check=datetime.now(),
            error_count=self.error_handler.error_count
        )
    
    def stop(self, signum=None, frame=None):
        """停止交易机器人"""
        self.logger.info("正在停止交易机器人...")
        self.running = False
        
        # 取消健康检查任务
        if self.health_check_task:
            self.health_check_task.cancel()
        
        # 保存状态
        self.state_manager.save_state()
        
        # 关闭WebSocket连接
        asyncio.create_task(self.ws_handler.close())
        
        self.logger.info("交易机器人已停止")
        
        # 发送停止通知
        asyncio.create_task(
            self.alert_system.send_telegram_alert("🛑 交易机器人已停止")
        )

# ================== 启动入口 ==================
if __name__ == "__main__":
    trader = EnhancedProductionTrader()
    try:
        asyncio.run(trader.run())
    except KeyboardInterrupt:
        trader.stop()
    except Exception as e:
        logging.critical(f"未处理的异常: {e}")
        sys.exit(1)
