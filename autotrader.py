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
        
        # 从环境变量加载API密钥
        self.api_key = os.environ.get('BINANCE_API_KEY')
        self.api_secret = os.environ.get('BINANCE_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            logging.warning("Binance API密钥未设置，将使用纸交易模式")
            self.mode = Mode.PAPER

# ================== 数据库管理 ==================
class DatabaseManager:
    """数据库管理器"""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
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
                    status TEXT NOT NULL,
                    profit_loss REAL DEFAULT 0,
                    close_price REAL,
                    close_time DATETIME
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
                    quantity REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    confidence REAL NOT NULL,
                    timeframe TEXT NOT NULL,
                    executed BOOLEAN DEFAULT FALSE
                )
            ''')
            # 创建状态记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS bot_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            ''')
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def save_trade(self, trade_data: Dict):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO trades (id, symbol, side, price, quantity, timestamp, order_id, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data['id'],
                trade_data['symbol'],
                trade_data['side'],
                trade_data['price'],
                trade_data['quantity'],
                trade_data['timestamp'],
                trade_data.get('order_id'),
                trade_data['status']
            ))
            conn.commit()
    
    def update_trade(self, trade_id: str, updates: Dict):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
            values = list(updates.values())
            values.append(trade_id)
            cursor.execute(f'UPDATE trades SET {set_clause} WHERE id = ?', values)
            conn.commit()
    
    def save_signal(self, signal_data: Dict):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO signals (id, symbol, side, price, atr, quantity, timestamp, confidence, timeframe)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_data['id'],
                signal_data['symbol'],
                signal_data['side'],
                signal_data['price'],
                signal_data['atr'],
                signal_data['quantity'],
                signal_data['timestamp'],
                signal_data['confidence'],
                signal_data['timeframe']
            ))
            conn.commit()
    
    def mark_signal_executed(self, signal_id: str):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE signals SET executed = TRUE WHERE id = ?', (signal_id,))
            conn.commit()
    
    def save_state(self, key: str, value: str):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO bot_state (key, value)
                VALUES (?, ?)
            ''', (key, value))
            conn.commit()
    
    def load_state(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT value FROM bot_state WHERE key = ?', (key,))
            result = cursor.fetchone()
            return result[0] if result else default

# ================== 日志系统 ==================
class AdvancedLogger:
    """高级日志系统"""
    def __init__(self, name: str, db_manager: DatabaseManager):
        self.logger = logging.getLogger(name)
        self.db_manager = db_manager
    
    def info(self, msg: str, extra: Optional[Dict] = None):
        self.logger.info(msg, extra=extra)
    
    def warning(self, msg: str, extra: Optional[Dict] = None):
        self.logger.warning(msg, extra=extra)
    
    def error(self, msg: str, extra: Optional[Dict] = None):
        self.logger.error(msg, extra=extra)
    
    def critical(self, msg: str, extra: Optional[Dict] = None):
        self.logger.critical(msg, extra=extra)
    
    def debug(self, msg: str, extra: Optional[Dict] = None):
        self.logger.debug(msg, extra=extra)

# ================== 缓存系统 ==================
class TimedCache:
    """带时间戳的缓存系统"""
    def __init__(self, ttl: int = 300, maxsize: int = 1000):
        self.cache = cachetools.TTLCache(maxsize=maxsize, ttl=ttl)
    
    def get(self, key: str) -> Optional[Any]:
        return self.cache.get(key)
    
    def set(self, key: str, value: Any):
        self.cache[key] = value
    
    def clear(self):
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
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100):
        pass
    
    @abstractmethod
    async def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: Optional[float] = None, params: Optional[Dict] = None):
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str):
        pass
    
    @abstractmethod
    async def fetch_balance(self):
        pass
    
    @abstractmethod
    async def fetch_positions(self, symbols: Optional[List[str]] = None):
        pass
    
    @abstractmethod
    async def fetch_ticker(self, symbol: str):
        pass
    
    @abstractmethod
    async def set_leverage(self, leverage: int, symbol: Optional[str] = None):
        pass
    
    @abstractmethod
    async def set_hedge_mode(self, enabled: bool):
        pass

class BinanceExchange(ExchangeInterface):
    """币安交易所实现"""
    def __init__(self, config: Config, mode: Mode = Mode.LIVE):
        self.config = config
        self.mode = mode
        self.exchange = None
        self.initialized = False
        self.logger = AdvancedLogger("BinanceExchange", DatabaseManager(config.db_path))
    
    async def initialize(self):
        try:
            if self.mode == Mode.LIVE:
                self.exchange = ccxt.binance({
                    'apiKey': self.config.api_key,
                    'secret': self.config.api_secret,
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'future',
                        'adjustForTimeDifference': True,
                    }
                })
                # 设置对冲模式
                try:
                    await self.set_hedge_mode(self.config.hedge_mode)
                except Exception as e:
                    if "No need to change position side" not in str(e):
                        self.logger.warning(f"设置对冲模式失败: {e}")
                
                # 设置杠杆
                try:
                    await self.set_leverage(self.config.leverage)
                except Exception as e:
                    self.logger.error(f"设置杠杆失败: {e}")
            else:
                # 纸交易模式
                self.exchange = ccxt.binance({
                    'apiKey': '',
                    'secret': '',
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'future',
                        'adjustForTimeDifference': True,
                    }
                })
                self.exchange.set_sandbox_mode(True)
            
            self.initialized = True
            self.logger.info("交易所接口初始化成功")
        except Exception as e:
            self.logger.error(f"交易所接口初始化失败: {e}")
            raise
    
    async def set_hedge_mode(self, enabled: bool):
        if self.mode != Mode.LIVE:
            return
        
        try:
            params = {'dualSidePosition': 'true' if enabled else 'false'}
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.exchange.fapiPrivate_post_positionside_dual(params)
            )
            return response
        except Exception as e:
            self.logger.warning(f"设置对冲模式失败: {e}")
            raise
    
    async def set_leverage(self, leverage: int, symbol: Optional[str] = None):
        if self.mode != Mode.LIVE:
            return
        
        try:
            symbols = [symbol] if symbol else self.config.symbols
            for sym in symbols:
                params = {'symbol': sym.replace('/', ''), 'leverage': leverage}
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.exchange.fapiPrivate_post_leverage(params)
                )
                self.logger.info(f"设置{sym}杠杆为{leverage}")
        except Exception as e:
            self.logger.error(f"设置杠杆失败: {e}")
            raise
    
    @retry_with_exponential_backoff()
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100):
        try:
            ohlcv = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            )
            return ohlcv
        except Exception as e:
            self.logger.error(f"获取{symbol}K线数据失败: {e}")
            raise
    
    @retry_with_exponential_backoff()
    async def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: Optional[float] = None, params: Optional[Dict] = None):
        if self.mode == Mode.PAPER:
            order_id = f"paper_{int(time.time() * 1000)}"
            self.logger.info(f"纸交易订单: {symbol} {side} {amount} @ {price}")
            return {'id': order_id, 'info': {'orderId': order_id}}
        
        try:
            order_params = params or {}
            order = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.exchange.create_order(symbol, order_type, side, amount, price, order_params)
            )
            return order
        except Exception as e:
            self.logger.error(f"创建订单失败: {e}")
            raise
    
    @retry_with_exponential_backoff()
    async def cancel_order(self, order_id: str, symbol: str):
        if self.mode == Mode.PAPER:
            self.logger.info(f"纸交易取消订单: {order_id}")
            return True
        
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.exchange.cancel_order(order_id, symbol)
            )
            return result
        except Exception as e:
            self.logger.error(f"取消订单失败: {e}")
            raise
    
    @retry_with_exponential_backoff()
    async def fetch_balance(self):
        try:
            balance = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.exchange.fetch_balance()
            )
            return BalanceInfo(
                total=balance['total']['USDT'],
                free=balance['free']['USDT'],
                used=balance['used']['USDT']
            )
        except Exception as e:
            self.logger.error(f"获取余额失败: {e}")
            raise
    
    @retry_with_exponential_backoff()
    async def fetch_positions(self, symbols: Optional[List[str]] = None):
        try:
            positions = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.exchange.fetch_positions(symbols)
            )
            return positions
        except Exception as e:
            self.logger.error(f"获取持仓失败: {e}")
            raise
    
    @retry_with_exponential_backoff()
    async def fetch_ticker(self, symbol: str):
        try:
            ticker = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.exchange.fetch_ticker(symbol)
            )
            return ticker
        except Exception as e:
            self.logger.error(f"获取行情失败: {e}")
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
        self.logger = AdvancedLogger("WebSocketDataHandler", DatabaseManager(config.db_path))
    
    async def connect(self):
        if not WEBSOCKETS_AVAILABLE:
            self.logger.warning("WebSocket库不可用，无法建立实时连接")
            return
        
        try:
            for symbol in self.config.symbols:
                await self._connect_symbol(symbol)
            
            self.connected = True
            self.logger.info("WebSocket连接已建立")
        except Exception as e:
            self.logger.error(f"WebSocket连接失败: {e}")
    
    async def _connect_symbol(self, symbol: str):
        if not WEBSOCKETS_AVAILABLE:
            return
        
        try:
            # 币安WebSocket连接
            stream_name = f"{symbol.lower().replace('/', '')}@ticker"
            ws_url = f"wss://fstream.binance.com/ws/{stream_name}"
            
            self.websockets[symbol] = await connect(ws_url)
            self.logger.info(f"WebSocket连接已建立: {symbol}")
            
            # 启动消息处理循环
            asyncio.create_task(self._message_loop(symbol))
        except Exception as e:
            self.logger.error(f"建立{symbol}的WebSocket连接失败: {e}")
    
    async def _message_loop(self, symbol: str):
        if symbol not in self.websockets:
            return
        
        ws = self.websockets[symbol]
        try:
            async for message in ws:
                data = json.loads(message)
                if 'c' in data:  # 最新价格
                    self.last_prices[symbol] = float(data['c'])
        except Exception as e:
            self.logger.error(f"WebSocket消息处理错误({symbol}): {e}")
            # 尝试重新连接
            await asyncio.sleep(5)
            await self._connect_symbol(symbol)
    
    def get_last_price(self, symbol: str) -> Optional[float]:
        return self.last_prices.get(symbol)
    
    async def close(self):
        for ws in self.websockets.values():
            await ws.close()
        self.websockets = {}
        self.connected = False
        self.logger.info("WebSocket连接已关闭")

# ================== 动态ATR计算器 ==================
class DynamicATRCalculator:
    """动态ATR计算器"""
    def __init__(self, exchange: ExchangeInterface, config: Config):
        self.exchange = exchange
        self.config = config
        self.atr_cache = TimedCache(ttl=300)  # 5分钟缓存
        self.logger = AdvancedLogger("ATRCalculator", DatabaseManager(config.db_path))
    
    async def calculate_atr(self, symbol: str, timeframe: str = '1h') -> Optional[float]:
        cache_key = f"{symbol}_{timeframe}_atr"
        cached_atr = self.atr_cache.get(cache_key)
        if cached_atr is not None:
            return cached_atr
        
        try:
            # 获取K线数据
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=100)
            if len(ohlcv) < self.config.atr_period + 1:
                self.logger.warning(f"数据不足，无法计算{symbol}的ATR")
                return None
            
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
            self.logger.error(f"计算{symbol}的ATR失败: {e}")
            return None

# ================== 多周期信号生成器 ==================
class MultiTimeframeSignalGenerator:
    """多周期信号生成器"""
    def __init__(self, exchange: ExchangeInterface, atr_calculator: DynamicATRCalculator, config: Config):
        self.exchange = exchange
        self.atr_calculator = atr_calculator
        self.config = config
        self.signal_cache = TimedCache(ttl=60)  # 1分钟缓存
        self.logger = AdvancedLogger("SignalGenerator", DatabaseManager(config.db_path))
    
    async def generate_signal(self, symbol: str) -> Optional[TradeSignal]:
        cache_key = f"{symbol}_signal"
        cached_signal = self.signal_cache.get(cache_key)
        if cached_signal is not None:
            return cached_signal
        
        try:
            # 获取当前价格
            ticker = await self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # 计算ATR
            atr = await self.atr_calculator.calculate_atr(symbol, '1h')
            if atr is None or atr == 0:
                self.logger.warning(f"无法计算{symbol}的ATR，跳过信号生成")
                return None
            
            # 获取余额以计算仓位大小
            balance = await self.exchange.fetch_balance()
            if balance is None or balance.total <= 0:
                self.logger.error("无法获取余额，跳过信号生成")
                return None
            
            # 计算仓位大小 - 使用严格的仓位控制
            risk_amount = balance.total * (self.config.risk_per_trade / 100.0)
            position_size = risk_amount / (atr * self.config.atr_multiplier)
            
            # 确保最小订单价值
            order_value = position_size * current_price
            if order_value < self.config.min_order_value:
                self.logger.info(f"订单价值{order_value:.2f}小于最小限制{self.config.min_order_value}，跳过{symbol}")
                return None
            
            # 确保不超过最大仓位百分比
            max_position_value = balance.total * (self.config.max_position_size_percent / 100.0)
            if order_value > max_position_value:
                position_size = max_position_value / current_price
                self.logger.info(f"调整{symbol}的仓位大小以符合最大仓位限制")
            
            # 简化信号生成逻辑 - 实际应根据策略生成
            # 这里使用随机信号作为示例
            import random
            side = OrderSide.BUY if random.random() > 0.5 else OrderSide.SELL
            
            signal = TradeSignal(
                symbol=symbol,
                side=side,
                price=current_price,
                atr=atr,
                quantity=position_size,
                timestamp=datetime.now(),
                confidence=0.7,  # 置信度
                timeframe="1h"
            )
            
            # 缓存信号
            self.signal_cache.set(cache_key, signal)
            
            return signal
        except Exception as e:
            self.logger.error(f"生成{symbol}的交易信号失败: {e}")
            return None

# ================== 指标系统 ==================
class IndicatorSystem:
    """完整的指标计算系统"""
    def __init__(self, config: Config):
        self.config = config
        self.logger = AdvancedLogger("IndicatorSystem", DatabaseManager(config.db_path))
    
    def calculate_rsi(self, closes: List[float], period: int = 14) -> Optional[float]:
        try:
            series = pd.Series(closes)
            rsi = ta.momentum.RSIIndicator(series, window=period).rsi()
            return rsi.iloc[-1] if not rsi.empty else None
        except Exception as e:
            self.logger.error(f"计算RSI失败: {e}")
            return None
    
    def calculate_macd(self, closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        try:
            series = pd.Series(closes)
            macd_indicator = ta.trend.MACD(series, window_fast=fast, window_slow=slow, window_sign=signal)
            macd = macd_indicator.macd().iloc[-1] if not macd_indicator.macd().empty else None
            signal_line = macd_indicator.macd_signal().iloc[-1] if not macd_indicator.macd_signal().empty else None
            histogram = macd_indicator.macd_diff().iloc[-1] if not macd_indicator.macd_diff().empty else None
            return macd, signal_line, histogram
        except Exception as e:
            self.logger.error(f"计算MACD失败: {e}")
            return None, None, None
    
    def calculate_bollinger_bands(self, closes: List[float], period: int = 20, std_dev: int = 2) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        try:
            series = pd.Series(closes)
            bb_indicator = ta.volatility.BollingerBands(series, window=period, window_dev=std_dev)
            bb_high = bb_indicator.bollinger_hband().iloc[-1] if not bb_indicator.bollinger_hband().empty else None
            bb_mid = bb_indicator.bollinger_mavg().iloc[-1] if not bb_indicator.bollinger_mavg().empty else None
            bb_low = bb_indicator.bollinger_lband().iloc[-1] if not bb_indicator.bollinger_lband().empty else None
            return bb_high, bb_mid, bb_low
        except Exception as e:
            self.logger.error(f"计算布林带失败: {e}")
            return None, None, None

# ================== 交易执行器 ==================
class TradeExecutor:
    """优化的交易执行器"""
    def __init__(self, exchange: ExchangeInterface, config: Config, db_manager: DatabaseManager):
        self.exchange = exchange
        self.config = config
        self.db_manager = db_manager
        self.open_orders = {}
        self.logger = AdvancedLogger("TradeExecutor", db_manager)
    
    async def execute_signal(self, signal: TradeSignal) -> OrderResult:
        # 检查信号有效性
        if signal.quantity <= 0 or signal.price <= 0:
            self.logger.error(f"无效的信号参数: 数量={signal.quantity}, 价格={signal.price}")
            return OrderResult(success=False, error="无效的信号参数")
        
        # 检查最小订单价值
        order_value = signal.quantity * signal.price
        if order_value < self.config.min_order_value:
            self.logger.info(f"订单价值{order_value:.2f}小于最小限制{self.config.min_order_value}，跳过执行")
            return OrderResult(success=False, error="订单价值太小")
        
        # 保存信号到数据库
        signal_id = str(uuid.uuid4())
        signal_data = {
            'id': signal_id,
            'symbol': signal.symbol,
            'side': signal.side.value,
            'price': signal.price,
            'atr': signal.atr,
            'quantity': signal.quantity,
            'timestamp': signal.timestamp.isoformat(),
            'confidence': signal.confidence,
            'timeframe': signal.timeframe
        }
        self.db_manager.save_signal(signal_data)
        
        # 在执行前检查当前持仓
        try:
            positions = await self.exchange.fetch_positions([signal.symbol])
            current_position = next((p for p in positions if p['symbol'] == signal.symbol), None)
            
            if current_position and abs(current_position['contracts']) > 0:
                self.logger.info(f"{signal.symbol}已有持仓，数量: {current_position['contracts']}")
                # 根据策略决定是否平仓或对冲
        except Exception as e:
            self.logger.error(f"获取{signal.symbol}持仓失败: {e}")
        
        # 执行订单
        try:
            order_type = "market"  # 使用市价单
            order = await self.exchange.create_order(
                symbol=signal.symbol,
                order_type=order_type,
                side=signal.side.value,
                amount=signal.quantity,
                price=None  # 市价单不需要价格
            )
            
            # 保存交易记录
            trade_id = str(uuid.uuid4())
            trade_data = {
                'id': trade_id,
                'symbol': signal.symbol,
                'side': signal.side.value,
                'price': signal.price,
                'quantity': signal.quantity,
                'timestamp': datetime.now().isoformat(),
                'order_id': order['id'],
                'status': 'open'
            }
            self.db_manager.save_trade(trade_data)
            
            # 标记信号已执行
            self.db_manager.mark_signal_executed(signal_id)
            
            self.logger.info(f"已执行{signal.symbol} {signal.side.value}订单，数量: {signal.quantity:.6f}")
            return OrderResult(success=True, order_id=order['id'], symbol=signal.symbol, side=signal.side)
        except Exception as e:
            self.logger.error(f"执行{signal.symbol}订单失败: {e}")
            return OrderResult(success=False, error=str(e), symbol=signal.symbol, side=signal.side)
    
    async def close_position(self, symbol: str, side: OrderSide, quantity: float) -> OrderResult:
        try:
            close_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
            order = await self.exchange.create_order(
                symbol=symbol,
                order_type="market",
                side=close_side.value,
                amount=quantity
            )
            
            self.logger.info(f"已平仓{symbol}，数量: {quantity:.6f}")
            return OrderResult(success=True, order_id=order['id'], symbol=symbol, side=close_side)
        except Exception as e:
            self.logger.error(f"平仓{symbol}失败: {e}")
            return OrderResult(success=False, error=str(e), symbol=symbol)

# ================== 增强的风险管理系统 ==================
class EnhancedRiskManager:
    """增强的风险管理系统"""
    def __init__(self, exchange: ExchangeInterface, config: Config, db_manager: DatabaseManager):
        self.exchange = exchange
        self.config = config
        self.db_manager = db_manager
        self.logger = AdvancedLogger("RiskManager", db_manager)
        self.max_drawdown = config.max_portfolio_risk_percent / 100.0
    
    async def check_portfolio_risk(self) -> bool:
        """检查投资组合风险，返回True如果风险在可接受范围内"""
        try:
            # 获取当前余额
            balance = await self.exchange.fetch_balance()
            if balance is None:
                return False
            
            # 获取所有持仓
            positions = await self.exchange.fetch_positions()
            if positions is None:
                return True  # 没有持仓，风险为0
            
            # 计算总风险和权益
            total_equity = balance.total
            total_risk = 0
            
            for position in positions:
                if abs(position['contracts']) > 0:
                    # 简化计算：使用初始风险(ATR * 乘数)
                    # 实际应根据当前价格和入场价格计算
                    symbol = position['symbol']
                    entry_price = position['entryPrice']
                    contracts = position['contracts']
                    
                    # 获取当前价格
                    ticker = await self.exchange.fetch_ticker(symbol)
                    current_price = ticker['last']
                    
                    # 计算当前盈亏
                    if position['side'] == 'long':
                        pnl = (current_price - entry_price) * contracts
                    else:
                        pnl = (entry_price - current_price) * contracts
                    
                    total_risk += max(0, -pnl)  # 只计算亏损部分
            
            # 计算回撤百分比
            drawdown_pct = total_risk / total_equity if total_equity > 0 else 0
            
            self.logger.info(f"投资组合风险检查: 回撤={drawdown_pct*100:.2f}%, 最大允许={self.max_drawdown*100}%")
            
            if drawdown_pct > self.max_drawdown:
                self.logger.warning(f"投资组合回撤超过最大限制: {drawdown_pct*100:.2f}% > {self.max_drawdown*100}%")
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"投资组合风险检查失败: {e}")
            return False
    
    async def check_symbol_risk(self, symbol: str, signal: TradeSignal) -> bool:
        """检查单个交易对的风险"""
        try:
            # 获取当前持仓
            positions = await self.exchange.fetch_positions([symbol])
            position = next((p for p in positions if p['symbol'] == symbol), None)
            
            if position and abs(position['contracts']) > 0:
                # 已有持仓，检查是否超过最大仓位限制
                current_value = abs(position['contracts']) * position['entryPrice']
                balance = await self.exchange.fetch_balance()
                max_position_value = balance.total * (self.config.max_position_size_percent / 100.0)
                
                if current_value >= max_position_value:
                    self.logger.warning(f"{symbol}已超过最大仓位限制: {current_value:.2f} >= {max_position_value:.2f}")
                    return False
            
            return True
        except Exception as e:
            self.logger.error(f"检查{symbol}风险失败: {e}")
            return False
    
    async def emergency_stop(self):
        """紧急停止所有交易"""
        self.logger.critical("执行紧急停止程序")
        
        try:
            # 获取所有持仓
            positions = await self.exchange.fetch_positions()
            for position in positions:
                if abs(position['contracts']) > 0:
                    symbol = position['symbol']
                    side = OrderSide.SELL if position['side'] == 'long' else OrderSide.BUY
                    quantity = abs(position['contracts'])
                    
                    self.logger.warning(f"紧急平仓: {symbol} {side.value} {quantity}")
                    await self.exchange.create_order(
                        symbol=symbol,
                        order_type="market",
                        side=side.value,
                        amount=quantity
                    )
        except Exception as e:
            self.logger.error(f"紧急停止失败: {e}")

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
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
            payload = {
                'chat_id': self.config.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        return True
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Telegram发送失败: {error_text}")
                        return False
        except Exception as e:
            self.logger.error(f"发送Telegram警报失败: {e}")
            return False
    
    async def send_trade_alert(self, signal: TradeSignal, executed: bool = False):
        """发送交易警报"""
        status = "已执行" if executed else "生成"
        message = f"<b>交易信号{status}</b>\n" \
                 f"品种: {signal.symbol}\n" \
                 f"方向: {signal.side.value}\n" \
                 f"价格: ${signal.price:.4f}\n" \
                 f"数量: {signal.quantity:.6f}\n" \
                 f"时间: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        
        await self.send_telegram_alert(message)
    
    async def send_error_alert(self, error_msg: str):
        """发送错误警报"""
        message = f"<b>❌ 交易错误</b>\n{error_msg}"
        await self.send_telegram_alert(message)
    
    async def send_health_alert(self, health_status: HealthStatus):
        """发送健康状态警报"""
        message = f"<b>🤖 交易机器人健康状态报告</b>\n" \
                 f"• 总交易对: {health_status.total_symbols}\n" \
                 f"• 已连接: {health_status.connected_symbols}\n" \
                 f"• 已断开: {health_status.disconnected_symbols}\n" \
                 f"• 错误计数: {health_status.error_count}\n" \
                 f"• 最后检查: {health_status.last_check.strftime('%Y-%m-%d %H:%M:%S')}"
        
        await self.send_telegram_alert(message)

# ================== 状态管理器 ==================
class StateManager:
    """增强的状态管理器"""
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.state = {}
        self.logger = AdvancedLogger("StateManager", db_manager)
    
    def save_state(self):
        """保存状态到数据库"""
        try:
            for key, value in self.state.items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                self.db_manager.save_state(key, str(value))
        except Exception as e:
            self.logger.error(f"保存状态失败: {e}")
    
    def load_state(self):
        """从数据库加载状态"""
        try:
            # 加载常用状态键
            state_keys = ['last_signal_time', 'open_positions', 'last_health_check', 'error_count']
            for key in state_keys:
                value = self.db_manager.load_state(key)
                if value:
                    # 尝试解析JSON
                    try:
                        self.state[key] = json.loads(value)
                    except:
                        self.state[key] = value
        except Exception as e:
            self.logger.error(f"加载状态失败: {e}")
    
    def update_state(self, key: str, value: Any):
        """更新状态"""
        self.state[key] = value
        self.save_state()
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """获取状态"""
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
        
        # 如果错误严重，发送警报
        if self.error_count % 5 == 0:  # 每5个错误发送一次警报
            await self.alert_system.send_error_alert(f"错误计数: {self.error_count}\n最近错误: {error_msg}")
    
    def reset_error_count(self):
        """重置错误计数"""
        self.error_count = 0
        self.logger.info("错误计数已重置")

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
        self.health_check_interval = self.config.health_check_interval
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """处理终止信号"""
        self.logger.info(f"收到信号 {signum}，正在停止...")
        self.stop()
    
    async def initialize(self):
        """初始化交易机器人"""
        try:
            self.logger.info("🚀 启动增强版交易机器人")
            
            # 初始化交易所
            await self.exchange.initialize()
            
            # 加载状态
            self.state_manager.load_state()
            self.logger.info("状态已加载")
            
            # 连接WebSocket
            await self.ws_handler.connect()
            
            self.running = True
            self.logger.info(f"交易机器人初始化完成，模式: {self.config.mode}, 对冲: {self.config.hedge_mode}, 杠杆: {self.config.leverage}")
            
        except Exception as e:
            self.logger.critical(f"初始化失败: {e}")
            await self.error_handler.handle_error(e, "初始化")
            raise
    
    async def run(self):
        """运行交易机器人的主循环"""
        try:
            await self.initialize()
            
            # 启动健康检查任务
            health_task = asyncio.create_task(self.health_check_loop())
            
            # 主交易循环
            while self.running:
                try:
                    # 生成并执行交易信号
                    await self.trading_cycle()
                    
                    # 等待一段时间后再进行下一轮
                    await asyncio.sleep(60)  # 每分钟检查一次
                    
                except Exception as e:
                    await self.error_handler.handle_error(e, "主循环")
                    await asyncio.sleep(30)  # 出错后等待30秒再继续
            
            # 等待健康检查任务结束
            health_task.cancel()
            try:
                await health_task
            except asyncio.CancelledError:
                pass
                
        except Exception as e:
            self.logger.critical(f"主循环失败: {e}")
            await self.error_handler.handle_error(e, "主循环")
        finally:
            await self.shutdown()
    
    async def trading_cycle(self):
        """执行交易周期"""
        # 检查投资组合风险
        risk_ok = await self.risk_manager.check_portfolio_risk()
        if not risk_ok:
            self.logger.warning("投资组合风险超过限制，跳过交易周期")
            # 可选：执行紧急停止
            # await self.risk_manager.emergency_stop()
            return
        
        # 为每个交易对生成信号
        for symbol in self.config.symbols:
            try:
                signal = await self.signal_generator.generate_signal(symbol)
                if signal is None:
                    continue
                
                # 检查交易对风险
                symbol_risk_ok = await self.risk_manager.check_symbol_risk(symbol, signal)
                if not symbol_risk_ok:
                    self.logger.info(f"{symbol}风险检查未通过，跳过执行")
                    continue
                
                # 发送信号警报
                await self.alert_system.send_trade_alert(signal, executed=False)
                
                # 执行信号
                result = await self.trade_executor.execute_signal(signal)
                
                if result.success:
                    self.logger.info(f"成功执行{signal.symbol} {signal.side.value}订单")
                    # 发送执行警报
                    await self.alert_system.send_trade_alert(signal, executed=True)
                else:
                    self.logger.error(f"执行{signal.symbol}订单失败: {result.error}")
                    
            except Exception as e:
                await self.error_handler.handle_error(e, f"处理{symbol}交易")
    
    async def health_check_loop(self):
        """健康检查循环"""
        while self.running:
            try:
                await self.perform_health_check()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self.error_handler.handle_error(e, "健康检查")
                await asyncio.sleep(300)  # 出错后等待5分钟再继续
    
    async def perform_health_check(self):
        """执行健康检查"""
        try:
            # 检查WebSocket连接
            connected_symbols = len(self.ws_handler.websockets)
            disconnected_symbols = len(self.config.symbols) - connected_symbols
            
            # 检查交易所连接
            balance = await self.exchange.fetch_balance()
            exchange_connected = balance is not None
            
            health_status = HealthStatus(
                total_symbols=len(self.config.symbols),
                connected_symbols=connected_symbols,
                disconnected_symbols=disconnected_symbols,
                last_check=datetime.now(),
                error_count=self.error_handler.error_count
            )
            
            self.logger.info(f"健康检查: {connected_symbols}/{len(self.config.symbols)} 连接正常, 错误计数: {self.error_handler.error_count}")
            
            # 发送健康状态警报
            await self.alert_system.send_health_alert(health_status)
            
            # 如果连接数不足，尝试重新连接
            if disconnected_symbols > len(self.config.symbols) / 2:
                self.logger.warning("超过一半的交易对连接断开，尝试重新连接")
                await self.ws_handler.close()
                await self.ws_handler.connect()
            
        except Exception as e:
            await self.error_handler.handle_error(e, "健康检查")
    
    async def shutdown(self):
        """关闭交易机器人"""
        self.logger.info("正在关闭交易机器人...")
        self.running = False
        
        # 关闭WebSocket连接
        await self.ws_handler.close()
        
        # 保存状态
        self.state_manager.save_state()
        
        self.logger.info("交易机器人已关闭")
    
    def stop(self):
        """停止交易机器人"""
        self.running = False

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
