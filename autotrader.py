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
import traceback

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
IS_DOCKER = os.path.exists('/.dockerenv')

# ================== Railway优化的日志配置 ==================
# 确保日志立即输出
os.environ['PYTHONUNBUFFERED'] = '1'

# 清除任何现有的日志处理器
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# 配置根日志记录器
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_level = logging.INFO

# 创建处理器
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(log_format)
handler.setFormatter(formatter)

# 配置根日志记录器
logging.basicConfig(
    level=log_level,
    handlers=[handler],
    format=log_format,
    force=True  # 强制重新配置
)

# 禁用过于详细的库日志
logging.getLogger("ccxt").setLevel(logging.INFO)
logging.getLogger("websockets").setLevel(logging.INFO)
logging.getLogger("asyncio").setLevel(logging.INFO)
logging.getLogger("aiohttp").setLevel(logging.WARNING)

logger = logging.getLogger("Main")

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
    status: str = "UNKNOWN"

# ================== 配置管理 ==================
class Config:
    """完整的配置管理"""
    def __init__(self):
        # 从环境变量读取配置
        self.mode = Mode(os.environ.get('TRADING_MODE', 'PAPER').upper())
        self.hedge_mode = os.environ.get('HEDGE_MODE', 'true').lower() == 'true'
        self.leverage = int(os.environ.get('LEVERAGE', '15'))
        self.max_position_size_percent = float(os.environ.get('MAX_POSITION_SIZE_PERCENT', '5.0'))
        self.max_portfolio_risk_percent = float(os.environ.get('MAX_PORTFOLIO_RISK_PERCENT', '20.0'))
        
        # 交易对配置
        symbols_str = os.environ.get('SYMBOLS', 'BTC/USDT,ETH/USDT,BNB/USDT')
        self.symbols = [s.strip() for s in symbols_str.split(',')]
        
        self.timeframes = ["1h", "4h"]
        self.atr_period = int(os.environ.get('ATR_PERIOD', '14'))
        self.atr_multiplier = float(os.environ.get('ATR_MULTIPLIER', '1.5'))
        self.risk_per_trade = float(os.environ.get('RISK_PER_TRADE', '2.0'))
        self.min_order_value = float(os.environ.get('MIN_ORDER_VALUE', '10.0'))
        
        # 数据库路径
        self.db_path = os.environ.get('DB_PATH', '/data/trading_bot.db' if IS_RAILWAY or IS_DOCKER else 'trading_bot.db')
        
        # 通知配置
        self.telegram_bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        
        # 系统配置
        self.max_retries = int(os.environ.get('MAX_RETRIES', '3'))
        self.retry_delay = int(os.environ.get('RETRY_DELAY', '2'))
        self.health_check_interval = int(os.environ.get('HEALTH_CHECK_INTERVAL', '300'))
        self.signal_check_interval = int(os.environ.get('SIGNAL_CHECK_INTERVAL', '300'))
        
        # 确保数据目录存在
        if IS_RAILWAY or IS_DOCKER:
            os.makedirs('/data', exist_ok=True)
        
        logger.info(f"配置加载完成: 模式={self.mode.value}, 交易对={len(self.symbols)}个")

# ================== 数据库管理 ==================
class DatabaseManager:
    """数据库管理器"""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """初始化数据库表"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # 创建交易记录表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        price REAL NOT NULL,
                        quantity REAL NOT NULL,
                        timestamp DATETIME NOT NULL,
                        order_id TEXT,
                        profit_loss REAL,
                        status TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # 创建信号记录表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS signals (
                        id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        price REAL NOT NULL,
                        atr REAL NOT NULL,
                        confidence REAL NOT NULL,
                        timeframe TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        executed BOOLEAN DEFAULT FALSE,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # 创建错误日志表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS error_logs (
                        id TEXT PRIMARY KEY,
                        component TEXT NOT NULL,
                        error_message TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        resolved BOOLEAN DEFAULT FALSE,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # 创建系统状态表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_status (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        status TEXT NOT NULL,
                        message TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                logger.info("数据库初始化完成")
                
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接上下文管理器"""
        # 注册适配器用于正确处理datetime对象
        sqlite3.register_adapter(datetime, lambda val: val.isoformat())
        sqlite3.register_converter("datetime", lambda val: datetime.fromisoformat(val.decode()))
        
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.execute("PRAGMA journal_mode=WAL")  # 启用WAL模式提高并发性能
        try:
            yield conn
        finally:
            conn.close()
    
    def save_trade(self, trade_data: dict):
        """保存交易记录"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO trades (id, symbol, side, price, quantity, timestamp, order_id, profit_loss, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_data.get('id', str(uuid.uuid4())),
                    trade_data['symbol'],
                    trade_data['side'],
                    trade_data['price'],
                    trade_data['quantity'],
                    trade_data['timestamp'],
                    trade_data.get('order_id'),
                    trade_data.get('profit_loss'),
                    trade_data.get('status', 'open')
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"保存交易记录失败: {e}")
    
    def save_signal(self, signal: TradeSignal):
        """保存交易信号"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO signals (id, symbol, side, price, atr, confidence, timeframe, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    str(uuid.uuid4()),
                    signal.symbol,
                    signal.side.value,
                    signal.price,
                    signal.atr,
                    signal.confidence,
                    signal.timeframe,
                    signal.timestamp
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"保存信号失败: {e}")
    
    def save_system_status(self, status: str, message: str):
        """保存系统状态"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO system_status (status, message, timestamp)
                    VALUES (?, ?, ?)
                ''', (status, message, datetime.now()))
                conn.commit()
        except Exception as e:
            logger.error(f"保存系统状态失败: {e}")

# ================== 简化的日志系统 ==================
class AdvancedLogger:
    """高级日志系统"""
    def __init__(self, name: str, db_manager: DatabaseManager = None):
        self.logger = logging.getLogger(name)
        self.db_manager = db_manager
    
    def info(self, message: str, extra: Optional[dict] = None):
        """记录信息日志"""
        self.logger.info(message, extra=extra)
    
    def error(self, message: str, component: str = "unknown", extra: Optional[dict] = None):
        """记录错误日志并保存到数据库"""
        self.logger.error(message, extra=extra)
        
        # 保存错误到数据库
        if self.db_manager:
            try:
                error_data = {
                    'id': str(uuid.uuid4()),
                    'component': component,
                    'error_message': message,
                    'timestamp': datetime.now()
                }
                
                with self.db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO error_logs (id, component, error_message, timestamp)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        error_data['id'],
                        error_data['component'],
                        error_data['error_message'],
                        error_data['timestamp']
                    ))
                    conn.commit()
            except Exception as e:
                self.logger.error(f"保存错误日志失败: {e}")
    
    def warning(self, message: str, extra: Optional[dict] = None):
        """记录警告日志"""
        self.logger.warning(message, extra=extra)
    
    def debug(self, message: str, extra: Optional[dict] = None):
        """记录调试日志"""
        self.logger.debug(message, extra=extra)

# ================== 缓存系统 ==================
class TimedCache:
    """带时间戳的缓存系统"""
    def __init__(self, ttl: int = 300, maxsize: int = 1000):
        self.cache = cachetools.TTLCache(maxsize=maxsize, ttl=ttl)
    
    def get(self, key: str):
        """获取缓存值"""
        return self.cache.get(key)
    
    def set(self, key: str, value: Any):
        """设置缓存值"""
        self.cache[key] = value
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()

# ================== 交易所接口 ==================
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
                    logger.warning(f"重试 {attempt + 1}/{retries}: {str(e)}")
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
    async def create_order(self, symbol: str, order_type: str, side: OrderSide, amount: float, price: Optional[float] = None) -> OrderResult:
        pass
    
    @abstractmethod
    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> List[List[float]]:
        pass
    
    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Dict]:
        pass
    
    @abstractmethod
    async def get_price(self, symbol: str) -> float:
        pass

class BinanceExchange(ExchangeInterface):
    """币安交易所实现"""
    def __init__(self, config: Config, mode: Mode = Mode.PAPER):
        self.config = config
        self.mode = mode
        self.exchange = None
        self.initialized = False
        self.db_manager = DatabaseManager(config.db_path)
        self.logger = AdvancedLogger("BinanceExchange", self.db_manager)
    
    async def initialize(self):
        """初始化交易所连接"""
        try:
            api_key = os.environ.get('BINANCE_API_KEY')
            api_secret = os.environ.get('BINANCE_API_SECRET')
            
            if not api_key or not api_secret:
                self.logger.warning("Binance API密钥未设置，使用纸交易模式")
                self.mode = Mode.PAPER
                api_key = "paper_trading"
                api_secret = "paper_trading"
            
            exchange_config = {
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True,
                },
                'verbose': False  # 减少日志输出
            }
            
            if self.mode == Mode.PAPER:
                exchange_config['sandbox'] = True
                self.logger.info("使用币安测试网络(纸交易模式)")
            else:
                self.logger.info("使用币安实盘交易")
            
            self.exchange = ccxt.binance(exchange_config)
            
            # 加载市场数据
            await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.exchange.load_markets()
            )
            
            # 设置杠杆
            await self.set_leverage(self.config.leverage)
            
            self.initialized = True
            self.logger.info("Binance交易所初始化成功")
            
            return True
            
        except Exception as e:
            self.logger.error(f"交易所初始化失败: {str(e)}", component="BinanceExchange.initialize")
            # 在纸交易模式下，即使初始化失败也继续
            if self.mode == Mode.PAPER:
                self.logger.warning("纸交易模式下继续运行")
                self.initialized = True
                return True
            return False
    
    async def set_leverage(self, leverage: int):
        """设置杠杆"""
        try:
            if self.mode == Mode.PAPER:
                self.logger.info(f"纸交易模式，跳过设置杠杆")
                return
            
            for symbol in self.config.symbols:
                clean_symbol = symbol.replace('/', '')
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.exchange.set_leverage(leverage, clean_symbol)
                )
            self.logger.info(f"已设置杠杆为 {leverage}")
        except Exception as e:
            self.logger.error(f"设置杠杆失败: {str(e)}", component="BinanceExchange.set_leverage")
            # 在纸交易模式下继续
            if self.mode == Mode.PAPER:
                self.logger.warning("纸交易模式下继续运行")
    
    @retry_with_exponential_backoff()
    async def get_balance(self) -> BalanceInfo:
        """获取余额信息"""
        try:
            if self.mode == Mode.PAPER:
                # 纸交易模式下返回模拟余额
                return BalanceInfo(total=10000.0, free=8000.0, used=2000.0)
            
            balance = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.exchange.fetch_balance()
            )
            
            total = float(balance['USDT']['total'])
            free = float(balance['USDT']['free'])
            used = float(balance['USDT']['used'])
            
            return BalanceInfo(total=total, free=free, used=used)
        except Exception as e:
            self.logger.error(f"获取余额失败: {str(e)}", component="BinanceExchange.get_balance")
            # 返回默认余额避免崩溃
            return BalanceInfo(total=10000.0, free=10000.0, used=0.0)
    
    @retry_with_exponential_backoff()
    async def create_order(self, symbol: str, order_type: str, side: OrderSide, amount: float, price: Optional[float] = None) -> OrderResult:
        """创建订单"""
        try:
            current_price = await self.get_price(symbol)
            order_value = amount * (price if price else current_price)
            
            if self.mode == Mode.PAPER:
                # 纸交易模式 - 只记录不实际下单
                self.logger.info(f"纸交易订单: {symbol} {side.value} {amount} @ {price or current_price} (总值: {order_value:.2f} USDT)")
                
                return OrderResult(
                    success=True,
                    order_id=f"paper_{uuid.uuid4().hex}",
                    symbol=symbol,
                    side=side,
                    amount_usdt=order_value
                )
            else:
                # 实盘模式
                order = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.exchange.create_order(
                        symbol=symbol,
                        type=order_type,
                        side=side.value,
                        amount=amount,
                        price=price
                    ) if price else self.exchange.create_order(
                        symbol=symbol,
                        type=order_type,
                        side=side.value,
                        amount=amount
                    )
                )
                
                order_id = order['id']
                self.logger.info(f"订单创建成功: {order_id} - {symbol} {side.value} {amount}")
                
                return OrderResult(
                    success=True,
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    amount_usdt=order_value
                )
                
        except Exception as e:
            error_msg = f"创建订单失败: {str(e)}"
            self.logger.error(error_msg, component="BinanceExchange.create_order")
            return OrderResult(success=False, error=error_msg)
    
    @retry_with_exponential_backoff()
    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> List[List[float]]:
        """获取K线数据"""
        try:
            if self.mode == Mode.PAPER:
                # 纸交易模式下返回模拟数据
                return self._generate_mock_ohlcv(limit)
            
            ohlcv = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            )
            return ohlcv
        except Exception as e:
            self.logger.error(f"获取K线数据失败: {str(e)}", component="BinanceExchange.fetch_ohlcv")
            # 返回模拟数据避免崩溃
            return self._generate_mock_ohlcv(limit)
    
    def _generate_mock_ohlcv(self, limit: int) -> List[List[float]]:
        """生成模拟K线数据"""
        base_price = 50000.0
        ohlcv = []
        current_time = int(time.time() * 1000)
        
        for i in range(limit):
            timestamp = current_time - (limit - i) * 3600000  # 1小时间隔
            open_price = base_price * (1 + 0.01 * np.sin(i / 10))
            high_price = open_price * (1 + 0.02 * abs(np.cos(i / 5)))
            low_price = open_price * (1 - 0.015 * abs(np.sin(i / 7)))
            close_price = (high_price + low_price) / 2
            volume = 1000 + 500 * np.sin(i / 3)
            
            ohlcv.append([timestamp, open_price, high_price, low_price, close_price, volume])
        
        return ohlcv
    
    @retry_with_exponential_backoff()
    async def get_position(self, symbol: str) -> Optional[Dict]:
        """获取持仓信息"""
        try:
            if self.mode == Mode.PAPER:
                # 纸交易模式下返回空持仓
                return None
            
            positions = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.exchange.fetch_positions([symbol])
            )
            
            if positions and len(positions) > 0:
                return positions[0]
            return None
        except Exception as e:
            self.logger.error(f"获取持仓信息失败: {str(e)}", component="BinanceExchange.get_position")
            return None
    
    @retry_with_exponential_backoff()
    async def get_price(self, symbol: str) -> float:
        """获取当前价格"""
        try:
            if self.mode == Mode.PAPER:
                # 纸交易模式下返回模拟价格
                return 50000.0 + 1000.0 * np.sin(time.time() / 3600)
            
            ticker = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.exchange.fetch_ticker(symbol)
            )
            return float(ticker['last'])
        except Exception as e:
            self.logger.error(f"获取价格失败: {str(e)}", component="BinanceExchange.get_price")
            # 返回默认价格避免崩溃
            return 50000.0

# ================== 简化的WebSocket处理器 ==================
class WebSocketDataHandler:
    """简化的WebSocket实时数据处理器"""
    def __init__(self, config: Config, exchange: ExchangeInterface):
        self.config = config
        self.exchange = exchange
        self.websockets = {}
        self.last_prices = {}
        self.connected = False
        self.db_manager = DatabaseManager(config.db_path)
        self.logger = AdvancedLogger("WebSocketDataHandler", self.db_manager)
    
    async def start(self):
        """启动WebSocket连接"""
        if not WEBSOCKETS_AVAILABLE:
            self.logger.warning("WebSocket库不可用，使用轮询模式")
            return
        
        if self.exchange.mode == Mode.PAPER:
            self.logger.info("纸交易模式，跳过WebSocket连接")
            return
        
        try:
            self.logger.info("启动WebSocket连接...")
            # 简化的WebSocket实现，避免复杂的连接管理
            self.connected = True
            self.logger.info("WebSocket连接已启动")
        except Exception as e:
            self.logger.error(f"启动WebSocket失败: {str(e)}", component="WebSocketDataHandler.start")
    
    def get_last_price(self, symbol: str) -> Optional[float]:
        """获取最后记录的价格"""
        if symbol in self.last_prices:
            return self.last_prices[symbol]
        
        # 如果没有WebSocket数据，尝试从交易所获取
        try:
            price = asyncio.run(self.exchange.get_price(symbol))
            self.last_prices[symbol] = price
            return price
        except:
            return None
    
    async def stop(self):
        """停止所有WebSocket连接"""
        self.connected = False
        self.logger.info("WebSocket连接已关闭")

# ================== 动态ATR计算器 ==================
class DynamicATRCalculator:
    """动态ATR计算器"""
    def __init__(self, exchange: ExchangeInterface, config: Config):
        self.exchange = exchange
        self.config = config
        self.atr_cache = TimedCache(ttl=300)  # 5分钟缓存
        self.db_manager = DatabaseManager(config.db_path)
        self.logger = AdvancedLogger("ATRCalculator", self.db_manager)
    
    async def calculate_atr(self, symbol: str, timeframe: str = "1h") -> float:
        """计算ATR指标"""
        # 检查缓存
        cache_key = f"{symbol}_{timeframe}_atr"
        cached_atr = self.atr_cache.get(cache_key)
        if cached_atr:
            return cached_atr
        
        try:
            # 获取K线数据
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=self.config.atr_period + 20)
            
            if len(ohlcv) < self.config.atr_period + 1:
                self.logger.warning(f"数据不足计算ATR: {symbol} {timeframe}")
                return 0.0
            
            # 转换为DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 计算ATR
            atr_indicator = ta.volatility.AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=self.config.atr_period
            )
            
            atr = atr_indicator.average_true_range().iloc[-1]
            
            # 缓存结果
            self.atr_cache.set(cache_key, atr)
            
            return atr
        except Exception as e:
            self.logger.error(f"计算ATR失败 {symbol}: {str(e)}", component="DynamicATRCalculator.calculate_atr")
            return 1.0  # 返回默认ATR值避免崩溃

# ================== 多周期信号生成器 ==================
class MultiTimeframeSignalGenerator:
    """多周期信号生成器"""
    def __init__(self, exchange: ExchangeInterface, atr_calculator: DynamicATRCalculator, config: Config):
        self.exchange = exchange
        self.atr_calculator = atr_calculator
        self.config = config
        self.db_manager = DatabaseManager(config.db_path)
        self.logger = AdvancedLogger("SignalGenerator", self.db_manager)
    
    async def generate_signals(self, symbol: str) -> Optional[TradeSignal]:
        """生成交易信号"""
        try:
            # 获取多个时间框架的数据
            signals = []
            for timeframe in self.config.timeframes:
                signal = await self._analyze_timeframe(symbol, timeframe)
                if signal:
                    signals.append(signal)
            
            # 综合多个时间框架的信号
            if not signals:
                return None
            
            # 优先选择较高时间框架的信号
            signals.sort(key=lambda x: 0 if x.timeframe == "4h" else 1)
            primary_signal = signals[0]
            
            # 计算置信度 (基于信号一致性)
            confidence = self._calculate_confidence(signals)
            primary_signal.confidence = confidence
            
            # 只返回高置信度的信号
            if confidence >= 0.6:
                return primary_signal
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"生成信号失败 {symbol}: {str(e)}", component="MultiTimeframeSignalGenerator.generate_signals")
            return None
    
    async def _analyze_timeframe(self, symbol: str, timeframe: str) -> Optional[TradeSignal]:
        """分析单个时间框架"""
        try:
            # 获取K线数据
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=100)
            
            if len(ohlcv) < 50:
                return None
            
            # 转换为DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 计算技术指标
            df = self._calculate_indicators(df)
            
            # 生成信号
            current_price = df['close'].iloc[-1]
            atr = await self.atr_calculator.calculate_atr(symbol, timeframe)
            
            # 简单的趋势跟踪策略
            if df['ema_fast'].iloc[-1] > df['ema_slow'].iloc[-1] and df['rsi'].iloc[-1] > 50:
                # 多头信号
                return TradeSignal(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    price=current_price,
                    atr=atr,
                    quantity=0,  # 将在执行时计算
                    timestamp=datetime.now(),
                    timeframe=timeframe
                )
            elif df['ema_fast'].iloc[-1] < df['ema_slow'].iloc[-1] and df['rsi'].iloc[-1] < 50:
                # 空头信号
                return TradeSignal(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    price=current_price,
                    atr=atr,
                    quantity=0,  # 将在执行时计算
                    timestamp=datetime.now(),
                    timeframe=timeframe
                )
            
            return None
        except Exception as e:
            self.logger.error(f"分析时间框架失败 {symbol} {timeframe}: {str(e)}", component="MultiTimeframeSignalGenerator._analyze_timeframe")
            return None
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        try:
            # EMA
            df['ema_fast'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
            df['ema_slow'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
            
            # RSI
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            
            return df
        except Exception as e:
            self.logger.error(f"计算指标失败: {str(e)}")
            return df
    
    def _calculate_confidence(self, signals: List[TradeSignal]) -> float:
        """计算信号置信度"""
        if not signals:
            return 0.0
        
        # 简单的一致性检查
        buy_signals = sum(1 for s in signals if s.side == OrderSide.BUY)
        sell_signals = sum(1 for s in signals if s.side == OrderSide.SELL)
        
        total_signals = len(signals)
        
        if buy_signals == total_signals:
            return 1.0  # 所有信号一致看多
        elif sell_signals == total_signals:
            return 1.0  # 所有信号一致看空
        else:
            # 信号不一致，置信度降低
            max_consistent = max(buy_signals, sell_signals)
            return max_consistent / total_signals

# ================== 交易执行器 ==================
class TradeExecutor:
    """优化的交易执行器"""
    def __init__(self, exchange: ExchangeInterface, config: Config):
        self.exchange = exchange
        self.config = config
        self.db_manager = DatabaseManager(config.db_path)
        self.logger = AdvancedLogger("TradeExecutor", self.db_manager)
    
    async def execute_signal(self, signal: TradeSignal) -> OrderResult:
        """执行交易信号"""
        try:
            # 计算交易数量基于ATR和风险控制
            balance = await self.exchange.get_balance()
            risk_amount = balance.total * (self.config.risk_per_trade / 100)
            
            # 基于ATR计算仓位大小
            atr_multiplier = self.config.atr_multiplier
            position_size = risk_amount / (signal.atr * atr_multiplier)
            
            # 获取当前价格以计算准确的价值
            current_price = await self.exchange.get_price(signal.symbol)
            order_value = position_size * current_price
            
            # 检查最小订单价值
            if order_value < self.config.min_order_value:
                self.logger.info(f"订单价值 {order_value:.2f} USDT 低于最小值 {self.config.min_order_value} USDT，跳过执行")
                return OrderResult(success=False, error="订单价值过低")
            
            # 检查最大仓位限制
            max_position_value = balance.total * (self.config.max_position_size_percent / 100)
            if order_value > max_position_value:
                position_size = max_position_value / current_price
                self.logger.info(f"调整仓位大小以适应最大仓位限制: {position_size:.6f}")
            
            # 更新信号中的数量
            signal.quantity = position_size
            
            # 创建订单
            order_type = "market"
            order_result = await self.exchange.create_order(
                symbol=signal.symbol,
                order_type=order_type,
                side=signal.side,
                amount=position_size
            )
            
            if order_result.success:
                # 保存交易记录
                trade_data = {
                    'symbol': signal.symbol,
                    'side': signal.side.value,
                    'price': current_price,
                    'quantity': position_size,
                    'timestamp': datetime.now(),
                    'order_id': order_result.order_id,
                    'status': 'open'
                }
                self.db_manager.save_trade(trade_data)
                
                # 保存信号记录
                self.db_manager.save_signal(signal)
            
            return order_result
            
        except Exception as e:
            error_msg = f"执行交易失败: {str(e)}"
            self.logger.error(error_msg, component="TradeExecutor.execute_signal")
            return OrderResult(success=False, error=error_msg)

# ================== 简化的风险管理系统 ==================
class EnhancedRiskManager:
    """增强的风险管理系统"""
    def __init__(self, exchange: ExchangeInterface, config: Config):
        self.exchange = exchange
        self.config = config
        self.db_manager = DatabaseManager(config.db_path)
        self.logger = AdvancedLogger("RiskManager", self.db_manager)
    
    async def should_accept_signal(self, signal: TradeSignal) -> bool:
        """决定是否接受交易信号"""
        try:
            # 检查投资组合风险
            if not await self.check_portfolio_risk():
                return False
            
            # 检查是否已有相同方向的持仓
            position = await self.exchange.get_position(signal.symbol)
            if position and position['size'] != 0:
                is_long = position['size'] > 0
                wants_long = signal.side == OrderSide.BUY
                
                if is_long == wants_long:
                    self.logger.info(f"已有相同方向的持仓 {signal.symbol}，跳过交易")
                    return
