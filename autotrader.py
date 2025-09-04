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
        record.levelname = self.LEVEL_MAP.get(record.levelno, record.levelname)
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
        """初始化数据库表"""
        with sqlite3.connect(self.db_path) as conn:
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
                    status TEXT
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
                    executed BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # 创建错误日志表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS error_logs (
                    id TEXT PRIMARY KEY,
                    component TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接上下文管理器"""
        # 注册适配器用于正确处理datetime对象
        sqlite3.register_adapter(datetime, lambda val: val.isoformat())
        sqlite3.register_converter("datetime", lambda val: datetime.fromisoformat(val.decode()))
        
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        try:
            yield conn
        finally:
            conn.close()
    
    def save_trade(self, trade_data: dict):
        """保存交易记录"""
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
    
    def save_signal(self, signal: TradeSignal):
        """保存交易信号"""
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

# ================== 日志系统 ==================
class AdvancedLogger:
    """高级日志系统"""
    def __init__(self, name: str, db_manager: DatabaseManager):
        self.logger = logging.getLogger(name)
        self.db_manager = db_manager
    
    def info(self, message: str, extra: Optional[dict] = None):
        """记录信息日志"""
        self.logger.info(message, extra=extra)
    
    def error(self, message: str, component: str = "unknown", extra: Optional[dict] = None):
        """记录错误日志并保存到数据库"""
        self.logger.error(message, extra=extra)
        
        # 保存错误到数据库
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
            api_key = os.environ.get('BINANCE_API_KEY')
            api_secret = os.environ.get('BINANCE_API_SECRET')
            
            if not api_key or not api_secret:
                raise ValueError("Binance API密钥未设置")
            
            # 根据模式选择不同的交易所配置
            if self.mode == Mode.LIVE:
                self.exchange = ccxt.binance({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'future',
                        'adjustForTimeDifference': True,
                    }
                })
            else:  # 纸交易模式
                self.exchange = ccxt.binance({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'future',
                        'adjustForTimeDifference': True,
                    },
                    'sandbox': True  # 启用测试网络
                })
            
            # 设置杠杆和保证金模式
            if self.config.hedge_mode:
                await self.set_hedge_mode()
            
            await self.set_leverage(self.config.leverage)
            
            self.initialized = True
            self.logger.info("Binance交易所初始化成功")
            
        except Exception as e:
            self.logger.error(f"交易所初始化失败: {str(e)}", component="BinanceExchange.initialize")
            raise
    
    async def set_hedge_mode(self):
        """设置对冲模式"""
        try:
            # 使用CCXT的标准方法设置持仓模式
            # 币安期货API需要特定的参数来设置对冲模式
            params = {'dualSidePosition': 'true'}
            
            # 使用CCXT的统一方法
            await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.exchange.set_position_mode(True, params=params)
            )
            self.logger.info("已设置对冲模式")
        except Exception as e:
            # 如果标准方法失败，尝试使用替代方法
            try:
                # 使用私密API调用的替代方式
                await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self.exchange.private_post_position_side_dual({
                        'dualSidePosition': 'true'
                    })
                )
                self.logger.info("已设置对冲模式（使用替代方法）")
            except Exception as e2:
                self.logger.error(f"设置对冲模式失败: {str(e2)}", component="BinanceExchange.set_hedge_mode")
                # 在某些情况下，如果账户已经是对冲模式，我们可以继续
                self.logger.warning("继续运行，假设账户已处于对冲模式")
    
    async def set_leverage(self, leverage: int):
        """设置杠杆"""
        try:
            for symbol in self.config.symbols:
                # 为每个交易对设置杠杆
                clean_symbol = symbol.replace('/', '')
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.exchange.set_leverage(leverage, clean_symbol)
                )
            self.logger.info(f"已设置杠杆为 {leverage}")
        except Exception as e:
            self.logger.error(f"设置杠杆失败: {str(e)}", component="BinanceExchange.set_leverage")
            raise
    
    @retry_with_exponential_backoff()
    async def get_balance(self) -> BalanceInfo:
        """获取余额信息"""
        try:
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
            raise
    
    @retry_with_exponential_backoff()
    async def create_order(self, symbol: str, order_type: str, side: OrderSide, amount: float, price: Optional[float] = None) -> OrderResult:
        """创建订单"""
        try:
            # 在实盘模式下实际下单，在纸交易模式下只记录日志
            if self.mode == Mode.LIVE:
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
                    amount_usdt=amount * (price if price else await self.get_price(symbol))
                )
            else:
                # 纸交易模式 - 只记录不实际下单
                current_price = await self.get_price(symbol)
                order_value = amount * (price if price else current_price)
                
                self.logger.info(f"纸交易订单: {symbol} {side.value} {amount} @ {price or current_price} (总值: {order_value:.2f} USDT)")
                
                return OrderResult(
                    success=True,
                    order_id=f"paper_{uuid.uuid4().hex}",
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
            ohlcv = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            )
            return ohlcv
        except Exception as e:
            self.logger.error(f"获取K线数据失败: {str(e)}", component="BinanceExchange.fetch_ohlcv")
            raise
    
    @retry_with_exponential_backoff()
    async def get_position(self, symbol: str) -> Optional[Dict]:
        """获取持仓信息"""
        try:
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
            ticker = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.exchange.fetch_ticker(symbol)
            )
            return float(ticker['last'])
        except Exception as e:
            self.logger.error(f"获取价格失败: {str(e)}", component="BinanceExchange.get_price")
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
    
    async def start(self):
        """启动WebSocket连接"""
        if not WEBSOCKETS_AVAILABLE:
            self.logger.warning("WebSocket库不可用，无法启动实时数据流")
            return
        
        try:
            # 为每个交易对创建WebSocket连接
            for symbol in self.config.symbols:
                await self._create_websocket(symbol)
            
            self.connected = True
            self.logger.info("WebSocket连接已启动")
        except Exception as e:
            self.logger.error(f"启动WebSocket失败: {str(e)}", component="WebSocketDataHandler.start")
    
    async def _create_websocket(self, symbol: str):
        """创建单个交易对的WebSocket连接"""
        try:
            # 币安WebSocket端点
            stream_name = f"{symbol.lower().replace('/', '')}@ticker"
            ws_url = f"wss://fstream.binance.com/ws/{stream_name}"
            
            # 创建WebSocket连接
            self.websockets[symbol] = await connect(ws_url)
            
            # 启动消息处理循环
            asyncio.create_task(self._handle_messages(symbol))
            
            self.logger.info(f"已为 {symbol} 创建WebSocket连接")
        except Exception as e:
            self.logger.error(f"创建WebSocket连接失败 {symbol}: {str(e)}", component="WebSocketDataHandler._create_websocket")
    
    async def _handle_messages(self, symbol: str):
        """处理WebSocket消息"""
        ws = self.websockets[symbol]
        try:
            async for message in ws:
                data = json.loads(message)
                if 'c' in data:  # 最新价格字段
                    price = float(data['c'])
                    self.last_prices[symbol] = price
        except Exception as e:
            self.logger.error(f"处理WebSocket消息失败 {symbol}: {str(e)}", component="WebSocketDataHandler._handle_messages")
    
    def get_last_price(self, symbol: str) -> Optional[float]:
        """获取最后记录的价格"""
        return self.last_prices.get(symbol)
    
    async def stop(self):
        """停止所有WebSocket连接"""
        for symbol, ws in self.websockets.items():
            try:
                await ws.close()
            except Exception as e:
                self.logger.error(f"关闭WebSocket连接失败 {symbol}: {str(e)}", component="WebSocketDataHandler.stop")
        
        self.connected = False
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
            return 0.0

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
        # EMA
        df['ema_fast'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        df['ema_slow'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
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

# ================== 指标系统 ==================
class IndicatorSystem:
    """完整的指标计算系统"""
    def __init__(self):
        self.indicators = {}
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有指标"""
        # 这里可以实现更复杂的指标计算逻辑
        return df

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
            order_type = "market"  # 使用市价单
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

# ================== 增强的风险管理系统 ==================
class EnhancedRiskManager:
    """增强的风险管理系统"""
    def __init__(self, exchange: ExchangeInterface, config: Config):
        self.exchange = exchange
        self.config = config
        self.db_manager = DatabaseManager(config.db_path)
        self.logger = AdvancedLogger("RiskManager", self.db_manager)
    
    async def check_portfolio_risk(self) -> bool:
        """检查整体投资组合风险"""
        try:
            balance = await self.exchange.get_balance()
            positions = await self.get_all_positions()
            
            total_risk = 0.0
            for symbol, position in positions.items():
                if position['size'] != 0:
                    current_price = await self.exchange.get_price(symbol)
                    position_value = abs(position['size']) * current_price
                    risk_ratio = position_value / balance.total * 100
                    total_risk += risk_ratio
            
            if total_risk > self.config.max_portfolio_risk_percent:
                self.logger.warning(f"投资组合风险 {total_risk:.2f}% 超过限制 {self.config.max_portfolio_risk_percent}%")
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"检查投资组合风险失败: {str(e)}", component="RiskManager.check_portfolio_risk")
            return False
    
    async def get_all_positions(self) -> Dict[str, Dict]:
        """获取所有持仓"""
        positions = {}
        for symbol in self.config.symbols:
            position = await self.exchange.get_position(symbol)
            if position:
                positions[symbol] = position
        return positions
    
    async def should_accept_signal(self, signal: TradeSignal) -> bool:
        """决定是否接受交易信号"""
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
                return False
        
        return True

# ================== 警报系统 ==================
class AlertSystem:
    """警报系统"""
    def __init__(self, config: Config):
        self.config = config
        self.db_manager = DatabaseManager(config.db_path)
        self.logger = AdvancedLogger("AlertSystem", self.db_manager)
    
    async def send_telegram_alert(self, message: str):
        """发送Telegram警报"""
        if not self.config.telegram_bot_token or not self.config.telegram_chat_id:
            self.logger.warning("Telegram配置缺失，无法发送警报")
            return
        
        try:
            url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
            payload = {
                'chat_id': self.config.telegram_chat_id,
                'text': message
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        self.logger.error(f"发送Telegram警报失败: {response.status} - {response_text}")
        except Exception as e:
            self.logger.error(f"发送Telegram警报异常: {str(e)}", component="AlertSystem.send_telegram_alert")
    
    async def send_trade_alert(self, signal: TradeSignal, order_result: OrderResult):
        """发送交易警报"""
        message = f"""🚀 交易执行 {'成功' if order_result.success else '失败'}
交易对: {signal.symbol}
方向: {signal.side.value}
价格: {signal.price:.4f}
数量: {signal.quantity:.6f}
时间: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
订单ID: {order_result.order_id or 'N/A'}"""
        
        if not order_result.success:
            message += f"\n错误: {order_result.error}"
        
        await self.send_telegram_alert(message)
    
    async def send_error_alert(self, component: str, error: str):
        """发送错误警报"""
        message = f"""⚠️ 系统错误
组件: {component}
错误: {error}
时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        
        await self.send_telegram_alert(message)

# ================== 状态管理器 ==================
class StateManager:
    """增强的状态管理器"""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.state = {}
        self._load_state()
    
    def _load_state(self):
        """从数据库加载状态"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("CREATE TABLE IF NOT EXISTS bot_state (key TEXT PRIMARY KEY, value TEXT)")
                
                cursor.execute("SELECT key, value FROM bot_state")
                rows = cursor.fetchall()
                
                for key, value in rows:
                    self.state[key] = json.loads(value)
        except Exception as e:
            logging.error(f"加载状态失败: {str(e)}")
    
    def save_state(self):
        """保存状态到数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for key, value in self.state.items():
                    cursor.execute(
                        "INSERT OR REPLACE INTO bot_state (key, value) VALUES (?, ?)",
                        (key, json.dumps(value))
                    )
                
                conn.commit()
        except Exception as e:
            logging.error(f"保存状态失败: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取状态值"""
        return self.state.get(key, default)
    
    def set(self, key: str, value: Any):
        """设置状态值"""
        self.state[key] = value
        self.save_state()

# ================== 增强的错误处理 ==================
class EnhancedErrorHandler:
    """增强的错误处理"""
    def __init__(self, alert_system: AlertSystem, db_manager: DatabaseManager):
        self.alert_system = alert_system
        self.db_manager = db_manager
        self.logger = AdvancedLogger("ErrorHandler", db_manager)
    
    async def handle_error(self, component: str, error: str, is_critical: bool = False):
        """处理错误"""
        self.logger.error(error, component=component)
        
        # 发送错误警报
        if is_critical:
            await self.alert_system.send_error_alert(component, error)
        
        # 保存错误到数据库
        error_data = {
            'id': str(uuid.uuid4()),
            'component': component,
            'error_message': error,
            'timestamp': datetime.now(),
            'resolved': False
        }
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO error_logs (id, component, error_message, timestamp, resolved)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                error_data['id'],
                error_data['component'],
                error_data['error_message'],
                error_data['timestamp'],
                error_data['resolved']
            ))
            conn.commit()

# ================== 主交易机器人 ==================
class EnhancedProductionTrader:
    """增强的生产环境交易机器人"""
    def __init__(self):
        self.config = Config()
        self.db_manager = DatabaseManager(self.config.db_path)
        self.logger = AdvancedLogger("ProductionTrader", self.db_manager)
        
        # 初始化组件
        self.exchange = BinanceExchange(self.config, self.config.mode)
        self.ws_handler = WebSocketDataHandler(self.config, self.exchange)
        self.atr_calculator = DynamicATRCalculator(self.exchange, self.config)
        self.signal_generator = MultiTimeframeSignalGenerator(self.exchange, self.atr_calculator, self.config)
        self.trade_executor = TradeExecutor(self.exchange, self.config)
        self.risk_manager = EnhancedRiskManager(self.exchange, self.config)
        self.alert_system = AlertSystem(self.config)
        self.error_handler = EnhancedErrorHandler(self.alert_system, self.db_manager)
        self.state_manager = StateManager(self.config.db_path)
        
        self.running = False
        self.health_status = HealthStatus(
            total_symbols=len(self.config.symbols),
            connected_symbols=0,
            disconnected_symbols=len(self.config.symbols),
            last_check=datetime.now(),
            error_count=0
        )
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        """处理终止信号"""
        self.logger.info(f"接收到信号 {signum}，正在停止...")
        self.stop()
    
    async def initialize(self):
        """初始化交易机器人"""
        try:
            self.logger.info("正在初始化交易机器人...")
            
            # 初始化交易所连接
            await self.exchange.initialize()
            
            # 启动WebSocket连接
            if WEBSOCKETS_AVAILABLE:
                await self.ws_handler.start()
            
            self.logger.info("交易机器人初始化完成")
            return True
        except Exception as e:
            await self.error_handler.handle_error("ProductionTrader.initialize", f"初始化失败: {str(e)}", True)
            return False
    
    async def run(self):
        """运行交易机器人的主循环"""
        if not await self.initialize():
            self.logger.error("初始化失败，无法启动交易机器人")
            return
        
        self.running = True
        self.logger.info("交易机器人开始运行")
        
        # 发送启动通知
        await self.alert_system.send_telegram_alert("✅ 交易机器人已启动")
        
        # 主循环
        last_health_check = datetime.now()
        last_signal_check = datetime.now()
        
        while self.running:
            try:
                current_time = datetime.now()
                
                # 定期健康检查
                if (current_time - last_health_check).total_seconds() >= self.config.health_check_interval:
                    await self._perform_health_check()
                    last_health_check = current_time
                
                # 定期检查信号
                if (current_time - last_signal_check).total_seconds() >= 300:  # 每5分钟检查一次信号
                    await self._check_signals()
                    last_signal_check = current_time
                
                # 其他定期任务可以在这里添加
                
                await asyncio.sleep(10)  # 短暂休眠以减少CPU使用
                
            except Exception as e:
                await self.error_handler.handle_error("ProductionTrader.run", f"主循环异常: {str(e)}")
                await asyncio.sleep(30)  # 发生错误后等待更长时间
    
    async def _perform_health_check(self):
        """执行健康检查"""
        try:
            # 检查交易所连接
            try:
                balance = await self.exchange.get_balance()
                exchange_connected = True
            except Exception:
                exchange_connected = False
            
            # 检查WebSocket连接
            ws_connected = self.ws_handler.connected if WEBSOCKETS_AVAILABLE else False
            
            # 更新健康状态
            self.health_status.connected_symbols = len(self.ws_handler.websockets) if ws_connected else 0
            self.health_status.disconnected_symbols = self.health_status.total_symbols - self.health_status.connected_symbols
            self.health_status.last_check = datetime.now()
            
            # 记录健康状态
            self.logger.info(
                f"健康检查: 交易所连接={exchange_connected}, "
                f"WebSocket连接={ws_connected}, "
                f"连接交易对={self.health_status.connected_symbols}/{self.health_status.total_symbols}"
            )
            
            # 如果有问题，发送警报
            if not exchange_connected or not ws_connected:
                await self.alert_system.send_telegram_alert(
                    f"⚠️ 健康检查警报: 交易所连接={exchange_connected}, WebSocket连接={ws_connected}"
                )
                
        except Exception as e:
            await self.error_handler.handle_error("ProductionTrader._perform_health_check", f"健康检查失败: {str(e)}")
    
    async def _check_signals(self):
        """检查并处理交易信号"""
        try:
            self.logger.info("开始检查交易信号...")
            
            for symbol in self.config.symbols:
                if not self.running:
                    break
                
                try:
                    # 生成信号
                    signal = await self.signal_generator.generate_signals(symbol)
                    if not signal:
                        continue
                    
                    self.logger.info(f"发现交易信号: {signal.symbol} {signal.side.value}")
                    
                    # 风险检查
                    if not await self.risk_manager.should_accept_signal(signal):
                        self.logger.info(f"风险检查未通过: {signal.symbol}")
                        continue
                    
                    # 执行交易
                    order_result = await self.trade_executor.execute_signal(signal)
                    
                    # 发送警报
                    await self.alert_system.send_trade_alert(signal, order_result)
                    
                    # 短暂休眠以避免速率限制
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    await self.error_handler.handle_error("ProductionTrader._check_signals", f"处理信号失败 {symbol}: {str(e)}")
            
            self.logger.info("交易信号检查完成")
            
        except Exception as e:
            await self.error_handler.handle_error("ProductionTrader._check_signals", f"检查信号失败: {str(e)}")
    
    def stop(self):
        """停止交易机器人"""
        self.logger.info("正在停止交易机器人...")
        self.running = False
        
        # 关闭WebSocket连接
        if WEBSOCKETS_AVAILABLE:
            asyncio.create_task(self.ws_handler.stop())
        
        # 发送停止通知
        asyncio.create_task(self.alert_system.send_telegram_alert("🛑 交易机器人已停止"))

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
