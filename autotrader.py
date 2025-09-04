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
        # 确保levelname被正确映射
        if record.levelno in self.LEVEL_MAP:
            record.levelname = self.LEVEL_MAP[record.levelno]
        
        # 在Railway环境中使用简化格式
        if IS_RAILWAY:
            return f"{record.levelname} - {record.name} - {record.getMessage()}"
        
        # 本地开发使用详细格式
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
    
    def to_dict(self):
        """转换为字典以便JSON序列化"""
        data = asdict(self)
        data['side'] = self.side.value
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data):
        """从字典还原TradeSignal"""
        data = data.copy()
        data['side'] = OrderSide(data['side'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

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
    
    # 基础配置
    EXCHANGE = os.getenv("EXCHANGE", "binance")
    MARKET_TYPE = os.getenv("MARKET_TYPE", "future")
    
    # 修改这里：同时支持 LIVE_TRADE 和 MODE 变量
    mode_str = os.getenv("MODE") or ("live" if os.getenv("LIVE_TRADE", "").lower() == "true" else "paper")
    MODE = Mode(mode_str)
    
    HEDGE_MODE = os.getenv("HEDGE_MODE", "false").lower() == "true"
    
    # 交易对
    SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT").split(",") if s.strip()]
    
    # 风险参数
    RISK_RATIO = float(os.getenv("RISK_RATIO", "0.05"))
    LEVERAGE = int(os.getenv("LEVERAGE", "5"))
    SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "2.0"))
    TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "3.0"))
    RISK_ATR_MULT = float(os.getenv("RISK_ATR_MULT", "1.5"))
    PARTIAL_TP_RATIO = float(os.getenv("PARTIAL_TP_RATIO", "0.3"))
    MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "5"))
    MAX_RISK_RATIO = float(os.getenv("MAX_RISK_RATIO", "0.02"))  # 最大风险比
    
    # 时间参数
    POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))
    MARGIN_COOLDOWN = int(os.getenv("MARGIN_COOLDOWN", "3600"))
    SUMMARY_INTERVAL = int(os.getenv("SUMMARY_INTERVAL", "3600"))
    OHLCV_LIMIT = int(os.getenv("OHLCV_LIMIT", "200"))
    MACD_FILTER_TIMEFRAME = os.getenv("MACD_FILTER_TIMEFRAME", "4h")
    ENTRY_TIMEFRAME = os.getenv("ENTRY_TIMEFRAME", "15m")  # 入场时间框架
    
    # API配置
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
    
    # 性能配置
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY = float(os.getenv("RETRY_DELAY", "2.0"))
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "5"))
    CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))
    VOLUME_FILTER_MULTIPLIER = float(os.getenv("VOLUME_FILTER_MULTIPLIER", "0.8"))
    MAX_CONCURRENT_CONNECTIONS = int(os.getenv("MAX_CONCURRENT_CONNECTIONS", "3"))  # 最大并发连接数
    
    # 风控参数
    MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", "0.2"))
    DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", "0.1"))
    ORDER_TIMEOUT = int(os.getenv("ORDER_TIMEOUT", "30"))
    
    # 新增配置
    USE_WEBSOCKET = os.getenv("USE_WEBSOCKET", "true").lower() == "true" and WEBSOCKETS_AVAILABLE
    BAYESIAN_OPTIMIZATION = os.getenv("BAYESIAN_OPTIMIZATION", "false").lower() == "true"
    CROSS_VALIDATION_FOLDS = int(os.getenv("CROSS_VALIDATION_FOLDS", "3"))
    SLIPPAGE_RATIO = float(os.getenv("SLIPPAGE_RATIO", "0.0005"))
    COMMISSION_RATE = float(os.getenv("COMMISSION_RATE", "0.001"))
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
    OPTUNA_STORAGE = os.getenv("OPTUNA_STORAGE", "sqlite:///optuna.db")
    OPTUNA_N_TRIALS = int(os.getenv("OPTUNA_N_TRIALS", "100"))
    
    # 新增: 状态保存间隔(秒)
    STATE_SAVE_INTERVAL = int(os.getenv("STATE_SAVE_INTERVAL", "300"))
    
    # WebSocket配置
    WEBSOCKET_RECONNECT_DELAY = int(os.getenv("WEBSOCKET_RECONNECT_DELAY", "5"))
    WEBSOCKET_TIMEOUT = int(os.getenv("WEBSOCKET_TIMEOUT", "30"))
    WEBSOCKET_MAX_RETRIES = int(os.getenv("WEBSOCKET_MAX_RETRIES", "5"))  # 最大重试次数
    WEBSOCKET_RETRY_WINDOW = int(os.getenv("WEBSOCKET_RETRY_WINDOW", "300"))  # 重试窗口(秒)
    
    # 健康检查配置
    HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "1800"))  # 30分钟

# ================== 数据库管理 ==================
class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, db_path="trading_bot.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        with self.get_connection() as conn:
            # 创建仓位表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建订单表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建状态表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS app_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def save_position(self, position_data):
        """保存仓位信息"""
        with self.get_connection() as conn:
            conn.execute('''
                INSERT INTO positions (symbol, side, entry_price, quantity, stop_loss, take_profit)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                position_data['symbol'],
                position_data['side'],
                position_data['entry_price'],
                position_data['quantity'],
                position_data.get('stop_loss'),
                position_data.get('take_profit')
            ))
            conn.commit()
    
    def get_active_positions(self):
        """获取活跃仓位"""
        with self.get_connection() as conn:
            cursor = conn.execute('SELECT * FROM positions')
            return [dict(row) for row in cursor.fetchall()]
    
    def save_state(self, key, value):
        """保存应用状态"""
        with self.get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO app_state (key, value)
                VALUES (?, ?)
            ''', (key, json.dumps(value)))
            conn.commit()
    
    def load_state(self, key, default=None):
        """加载应用状态"""
        with self.get_connection() as conn:
            cursor = conn.execute('SELECT value FROM app_state WHERE key = ?', (key,))
            row = cursor.fetchone()
            return json.loads(row['value']) if row else default

# ================== 日志系统 ==================
class AdvancedLogger:
    """高级日志系统"""
    
    def __init__(self):
        self.setup_logging()
    
    def setup_logging(self):
        """配置日志"""
        # 使用全局配置的日志系统，不再重复配置
        self.logger = logging.getLogger(__name__)
        
        # 根据模式设置日志级别
        log_level = logging.DEBUG if Config.MODE == Mode.BACKTEST else logging.INFO
        self.logger.setLevel(log_level)
        
        # 在生产环境中添加文件处理器
        if Config.MODE != Mode.BACKTEST and not IS_RAILWAY:
            log_file = f'trading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            file_handler.setLevel(log_level)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def critical(self, message: str):
        self.logger.critical(message)
    
    def structured_log(self, event_type: str, data: Dict):
        """结构化日志记录"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            **data
        }
        self.info(f"STRUCTURED_LOG: {json.dumps(log_data)}")

# ================== 缓存系统 ==================
class TimedCache:
    """带时间戳的缓存系统"""
    
    def __init__(self, maxsize=100, ttl=300):
        self.cache = cachetools.TTLCache(maxsize=maxsize, ttl=ttl)
        self.timestamps = {}
        self.lock = threading.RLock()
    
    def get(self, key):
        with self.lock:
            data = self.cache.get(key)
            timestamp = self.timestamps.get(key)
            return data, timestamp
    
    def set(self, key, value):
        with self.lock:
            self.cache[key] = value
            self.timestamps[key] = datetime.now()
    
    def is_fresh(self, key, max_age_seconds=60):
        with self.lock:
            timestamp = self.timestamps.get(key)
            if not timestamp:
                return False
            return (datetime.now() - timestamp).total_seconds() < max_age_seconds

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
    async def get_historical_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        pass
    
    @abstractmethod
    async def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: Optional[float] = None, params: Optional[Dict] = None) -> OrderResult:
        pass
    
    @abstractmethod
    async def fetch_positions(self) -> List[Dict]:
        pass
    
    @abstractmethod
    async def fetch_balance(self) -> BalanceInfo:
        pass

class BinanceExchange(ExchangeInterface):
    """币安交易所实现"""
    
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.exchange = ccxt.binance({
            "apiKey": Config.BINANCE_API_KEY,
            "secret": Config.BINANCE_API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": Config.MARKET_TYPE}
        })
        self.exchange.load_markets()

        # 设置对冲模式
        if Config.HEDGE_MODE and hasattr(self.exchange, 'set_position_mode'):
            try:
                self.exchange.set_position_mode(True)
                self.logger.info("已设置对冲模式")
            except Exception as e:
                self.logger.warning(f"设置对冲模式失败: {e}")

        # 为每个交易对设置杠杆与保证金模式
        mode = 'CROSS'  # 或 'ISOLATED'
        for sym in Config.SYMBOLS:
            try:
                # ccxt 统一方法
                if hasattr(self.exchange, 'set_leverage'):
                    self.exchange.set_leverage(Config.LEVERAGE, sym)
                # 保证金模式
                if hasattr(self.exchange, 'set_margin_mode'):
                    self.exchange.set_margin_mode(mode, sym, params={})
            except Exception as e:
                self.logger.warning(f"设置杠杆/保证金模式失败 {sym}: {e}")

    @retry_with_exponential_backoff(retries=Config.MAX_RETRIES, delay=Config.RETRY_DELAY)
    async def get_historical_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """异步获取历史数据"""
        try:
            # 使用线程池执行同步IO操作
            ohlcv = await asyncio.to_thread(
                self.exchange.fetch_ohlcv, symbol, timeframe, None, limit
            )
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            return df
            
        except Exception as e:
            self.logger.error(f"获取历史数据失败 {symbol}: {e}")
            raise
    
    @retry_with_exponential_backoff(retries=Config.MAX_RETRIES, delay=Config.RETRY_DELAY)
    async def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: Optional[float] = None, params: Optional[Dict] = None) -> OrderResult:
        """异步创建订单"""
        try:
            # 生成唯一的客户端订单ID
            order_params = params.copy() if params else {}
            order_params['newClientOrderId'] = f"bot-{uuid.uuid4().hex[:16]}"
            
            # 使用线程池执行同步IO操作
            order = await asyncio.to_thread(
                self.exchange.create_order, symbol, order_type, side, amount, price, order_params
            )
            return OrderResult(success=True, order_id=order['id'], symbol=symbol, side=OrderSide(side))
            
        except Exception as e:
            error_msg = str(e)
            return OrderResult(success=False, error=error_msg, symbol=symbol, side=OrderSide(side))
    
    @retry_with_exponential_backoff(retries=Config.MAX_RETRIES, delay=Config.RETRY_DELAY)
    async def fetch_positions(self) -> List[Dict]:
        """异步获取持仓信息"""
        try:
            return await asyncio.to_thread(self.exchange.fetch_positions)
        except Exception as e:
            self.logger.error(f"获取持仓失败: {e}")
            return []
    
    @retry_with_exponential_backoff(retries=Config.MAX_RETRIES, delay=Config.RETRY_DELAY)
    async def fetch_balance(self) -> BalanceInfo:
        """异步获取余额信息"""
        try:
            balance_data = await asyncio.to_thread(self.exchange.fetch_balance)
            usdt_balance = balance_data.get('USDT', {})
            return BalanceInfo(
                total=float(usdt_balance.get('total', 0)),
                free=float(usdt_balance.get('free', 0)),
                used=float(usdt_balance.get('used', 0))
            )
        except Exception as e:
            self.logger.error(f"获取余额失败: {e}")
            return BalanceInfo(total=0, free=0, used=0)

# ================== WebSocket数据处理器 ==================
class WebSocketDataHandler:
    """增强的WebSocket实时数据处理器"""
    
    def __init__(self, exchange: ExchangeInterface, logger: AdvancedLogger, symbols: List[str]):
        self.exchange = exchange
        self.logger = logger
        self.symbols = symbols
        self.data_queue = asyncio.Queue()
        self.running = False
        self.ohlcv_data = {}
        self.ws_connections = {}
        self.retry_counts = {symbol: 0 for symbol in symbols}
        self.last_retry_time = {symbol: 0 for symbol in symbols}
        self.connection_semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_CONNECTIONS)
        
    async def start(self):
        """启动WebSocket连接"""
        self.running = True
        if Config.USE_WEBSOCKET and WEBSOCKETS_AVAILABLE:
            await self._start_websocket()
        else:
            if not WEBSOCKETS_AVAILABLE:
                self.logger.warning("websockets库未安装，使用REST API轮询模式")
            await self._start_polling()
    
    async def _start_websocket(self):
        """启动真正的WebSocket连接"""
        self.logger.info("启动WebSocket连接")
        
        # 为每个交易对创建WebSocket连接
        for symbol in self.symbols:
            asyncio.create_task(self._websocket_listener(symbol))
    
    async def _websocket_listener(self, symbol: str):
        """WebSocket监听器"""
        symbol_lower = symbol.replace('/', '').lower()
        ws_url = f"wss://fstream.binance.com/ws/{symbol_lower}@kline_1h"
        
        while self.running:
            # 检查重试次数
            current_time = time.time()
            if (self.retry_counts[symbol] >= Config.WEBSOCKET_MAX_RETRIES and 
                current_time - self.last_retry_time[symbol] < Config.WEBSOCKET_RETRY_WINDOW):
                self.logger.warning(f"{symbol} WebSocket连接重试次数过多，切换到REST模式")
                await self._start_polling_for_symbol(symbol)
                break
                
            async with self.connection_semaphore:
                try:
                    async with connect(ws_url, ping_interval=20, ping_timeout=10) as websocket:
                        self.logger.info(f"WebSocket连接已建立: {symbol}")
                        self.ws_connections[symbol] = websocket
                        self.retry_counts[symbol] = 0  # 重置重试计数
                        
                        while self.running:
                            try:
                                message = await asyncio.wait_for(websocket.recv(), timeout=Config.WEBSOCKET_TIMEOUT)
                                data = json.loads(message)
                                
                                if 'k' in data:
                                    kline = data['k']
                                    if kline['x']:  # 如果是收盘
                                        # 给REST端一点点同步时间
                                        await asyncio.sleep(1.0)
                                        
                                        ohlcv = {
                                            'timestamp': kline['t'],
                                            'open': float(kline['o']),
                                            'high': float(kline['h']),
                                            'low': float(kline['l']),
                                            'close': float(kline['c']),
                                            'volume': float(kline['v'])
                                        }
                                        
                                        # 创建DataFrame
                                        df = pd.DataFrame([ohlcv])
                                        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                                        df.set_index('datetime', inplace=True)
                                        
                                        await self.data_queue.put((symbol, df))
                            except asyncio.TimeoutError:
                                self.logger.debug(f"WebSocket接收超时: {symbol}")
                            except ws_exceptions.ConnectionClosed:
                                self.logger.warning(f"WebSocket连接已关闭: {symbol}")
                                break
                                
                except Exception as e:
                    self.logger.warning(f"WebSocket连接错误 {symbol}: {e}")
                    self.retry_counts[symbol] += 1
                    self.last_retry_time[symbol] = current_time
                    await asyncio.sleep(Config.WEBSOCKET_RECONNECT_DELAY)
    
    async def _start_polling_for_symbol(self, symbol: str):
        """为单个交易对启动轮询模式"""
        self.logger.info(f"为 {symbol} 使用REST API轮询模式")
        while self.running:
            try:
                ohlcv = await self.exchange.get_historical_data(symbol, "1h", 1)
                if not ohlcv.empty:
                    await self.data_queue.put((symbol, ohlcv.iloc[-1:]))
                await asyncio.sleep(Config.POLL_INTERVAL)
            except Exception as e:
                self.logger.error(f"轮询数据失败 {symbol}: {e}")
                await asyncio.sleep(5)
    
    async def _start_polling(self):
        """启动轮询模式"""
        self.logger.info("使用REST API轮询模式")
        while self.running:
            try:
                for symbol in Config.SYMBOLS:
                    ohlcv = await self.exchange.get_historical_data(symbol, "1h", 1)
                    if not ohlcv.empty:
                        await self.data_queue.put((symbol, ohlcv.iloc[-1:]))
                await asyncio.sleep(Config.POLL_INTERVAL)
            except Exception as e:
                self.logger.error(f"轮询数据失败: {e}")
                await asyncio.sleep(5)
    
    async def get_next_data(self):
        """获取下一个数据点"""
        return await self.data_queue.get()
    
    async def stop(self):
        """停止数据流"""
        self.running = False
        # 关闭所有WebSocket连接
        for symbol, ws in self.ws_connections.items():
            try:
                await ws.close()
            except:
                pass
    
    def get_health_status(self) -> HealthStatus:
        """获取健康状态"""
        connected = len(self.ws_connections)
        total = len(self.symbols)
        return HealthStatus(
            total_symbols=total,
            connected_symbols=connected,
            disconnected_symbols=total - connected,
            last_check=datetime.now(),
            error_count=sum(self.retry_counts.values())
        )

# ================== 动态ATR计算器 ==================
class DynamicATRCalculator:
    """动态ATR计算器"""
    
    def __init__(self):
        self.atr_history = {}
        self.volatility_threshold = 0.02  # 波动率阈值
    
    def calculate_dynamic_multipliers(self, symbol: str, current_atr: float, price: float) -> Tuple[float, float]:
        """计算动态的止损和止盈倍数"""
        # 初始化历史记录
        if symbol not in self.atr_history:
            self.atr_history[symbol] = []
        
        # 添加当前ATR到历史记录
        self.atr_history[symbol].append(current_atr)
        
        # 保持历史记录长度
        if len(self.atr_history[symbol]) > 20:
            self.atr_history[symbol].pop(0)
        
        # 计算ATR波动率
        if len(self.atr_history[symbol]) > 5:
            atr_std = np.std(self.atr_history[symbol])
            atr_mean = np.mean(self.atr_history[symbol])
            volatility_ratio = atr_std / atr_mean if atr_mean > 0 else 0
            
            # 根据波动率调整倍数
            if volatility_ratio > self.volatility_threshold:
                # 高波动率环境，使用较小的倍数以保护利润
                sl_mult = max(1.5, Config.SL_ATR_MULT * 0.8)
                tp_mult = max(2.0, Config.TP_ATR_MULT * 0.8)
            else:
                # 低波动率环境，使用较大的倍数以给价格更多空间
                sl_mult = Config.SL_ATR_MULT * 1.2
                tp_mult = Config.TP_ATR_MULT * 1.2
        else:
            # 历史数据不足，使用默认值
            sl_mult = Config.SL_ATR_MULT
            tp_mult = Config.TP_ATR_MULT
        
        return sl_mult, tp_mult

# ================== 多周期信号生成器 ==================
class MultiTimeframeSignalGenerator:
    """多周期信号生成器"""
    
    def __init__(self, indicator_system):
        self.indicator_system = indicator_system
    
    async def generate_signal(self, symbol: str, exchange: ExchangeInterface) -> Optional[TradeSignal]:
        """生成多周期信号"""
        try:
            # 获取多时间框架数据
            df_15m = await exchange.get_historical_data(symbol, Config.ENTRY_TIMEFRAME, Config.OHLCV_LIMIT)
            df_1h = await exchange.get_historical_data(symbol, "1h", Config.OHLCV_LIMIT)
            df_4h = await exchange.get_historical_data(symbol, Config.MACD_FILTER_TIMEFRAME, Config.OHLCV_LIMIT)
            
            if df_15m.empty or df_1h.empty or df_4h.empty:
                return None
            
            # 计算指标
            df_15m = self.indicator_system.compute_indicators(df_15m, symbol, Config.ENTRY_TIMEFRAME)
            df_1h = self.indicator_system.compute_indicators(df_1h, symbol, "1h")
            df_4h = self.indicator_system.compute_indicators(df_4h, symbol, Config.MACD_FILTER_TIMEFRAME)
            
            if df_15m.empty or df_1h.empty or df_4h.empty:
                return None
            
            # 获取最新数据
            current_15m = df_15m.iloc[-1]
            current_1h = df_1h.iloc[-1]
            current_4h = df_4h.iloc[-1]
            
            # 检查是否有NaN值
            if (pd.isna(current_15m.get('volume_ma', 0)) or 
                pd.isna(current_15m.get('volume', 0))):
                return None
            
            # 动态成交量过滤
            vol_threshold = current_15m.get('volume_ma', 0) * Config.VOLUME_FILTER_MULTIPLIER
            if current_15m['volume'] < vol_threshold:
                return None
            
            # 多周期信号逻辑
            price = current_15m['close']
            atr = current_15m['atr']
            
            # 1. 入场信号 (15分钟)
            entry_bullish = all([
                current_15m['macd'] > current_15m['macd_signal'],
                current_15m['ema_12'] > current_15m['ema_26'],
                40 < current_15m['rsi'] < 70
            ])
            
            entry_bearish = all([
                current_15m['macd'] < current_15m['macd_signal'],
                current_15m['ema_12'] < current_15m['ema_26'],
                30 < current_15m['rsi'] < 60
            ])
            
            # 2. 趋势方向 (1小时和4小时)
            trend_bullish = all([
                current_1h['ema_12'] > current_1h['ema_26'],
                current_4h['ema_12'] > current_4h['ema_26']
            ])
            
            trend_bearish = all([
                current_1h['ema_12'] < current_1h['ema_26'],
                current_4h['ema_12'] < current_4h['ema_26']
            ])
            
            # 3. 只有趋势和入场信号一致时才生成信号
            if entry_bullish and trend_bullish:
                return TradeSignal(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    price=price,
                    atr=atr,
                    quantity=0,
                    timestamp=datetime.now(),
                    timeframe=Config.ENTRY_TIMEFRAME
                )
            elif entry_bearish and trend_bearish:
                return TradeSignal(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    price=price,
                    atr=atr,
                    quantity=0,
                    timestamp=datetime.now(),
                    timeframe=Config.ENTRY_TIMEFRAME
                )
            
            return None
            
        except Exception as e:
            logging.error(f"多周期信号生成失败 {symbol}: {e}")
            return None

# ================== 指标系统 ==================
class IndicatorSystem:
    """完整的指标计算系统"""
    
    def __init__(self, cache: TimedCache):
        self.cache = cache
    
    def compute_indicators(self, df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
        if df.empty:
            return df
            
        # 使用最后时间戳作为缓存键的一部分
        last_ts = int(df.index[-1].timestamp())
        cache_key = f"{symbol}_{timeframe}_{last_ts}"
        
        # 检查缓存有效性
        cached_data, timestamp = self.cache.get(cache_key)
        if cached_data is not None and self.cache.is_fresh(cache_key, 60):
            return cached_data
        
        result = self._compute_indicators(df)
        self.cache.set(cache_key, result)
        return result
    
    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # EMA
        df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        
        # Volume
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_lower']
        
        return df.dropna()

# ================== 交易执行器 ==================
class TradeExecutor:
    """优化的交易执行器"""
    
    def __init__(self, exchange: ExchangeInterface, logger: AdvancedLogger):
        self.exchange = exchange
        self.logger = logger
        self.dynamic_atr = DynamicATRCalculator()
        self.consecutive_losses = 0  # 连续亏损次数
        # 定义最小交易量（根据币安期货规则）
        self.min_quantities = {
            'BTC/USDT': 0.001,
            'ETH/USDT': 0.01,
            'LTC/USDT': 0.1,
            'BNB/USDT': 0.1,
            'DOGE/USDT': 100,
            'XRP/USDT': 10,
            'SOL/USDT': 0.1,
            'TRX/USDT': 100,
            'ADA/USDT': 10,
            'LINK/USDT': 0.1,
        }

    def _apply_exchange_filters(self, symbol: str, qty: float, price: float) -> float:
        """应用交易所规则修正数量"""
        ex = self.exchange.exchange
        market = ex.market(symbol)
        contract_size = market.get('contractSize', 1) or 1
        notional = qty * price * contract_size

        # 交易所限制
        limits = market.get('limits', {})
        min_qty = (limits.get('amount', {}) or {}).get('min', None)
        min_cost = (limits.get('cost', {}) or {}).get('min', None)
        
        # 先按最小数量抬一档
        if min_qty and qty < min_qty:
            qty = min_qty

        # 再按名义价值抬到 min_notional
        if min_cost and notional < min_cost:
            target_qty = (min_cost / (price * contract_size)) * 1.02
            qty = max(qty, target_qty)

        # 按精度对齐
        qty = float(ex.amount_to_precision(symbol, qty))
        return max(0.0, qty)

    def _cap_by_available_margin(self, symbol: str, qty: float, price: float, free_usdt: float) -> float:
        """根据可用保证金限制数量"""
        leverage = Config.LEVERAGE
        ex = self.exchange.exchange
        market = ex.market(symbol)
        contract_size = market.get('contractSize', 1) or 1

        # 预估初始保证金
        notional = qty * price * contract_size
        init_margin = (notional / leverage) * 1.02
        if init_margin <= 0:
            return 0.0

        if init_margin <= free_usdt:
            return qty

        # 超出余额 -> 按比例缩小
        scale = max(0.0, (free_usdt / init_margin) * 0.98)
        capped = qty * scale
        capped = float(ex.amount_to_precision(symbol, capped))
        return max(0.0, capped)
    
    def calculate_position_size(self, balance: float, price: float, atr: float, symbol: str) -> float:
        """
        根据账户余额、价格和ATR计算仓位大小
        - balance: 可用余额（USDT）
        - price: 当前标的价格
        - atr: 平均真实波幅，用于估算风险
        """
        try:
            if atr <= 0 or price <= 0:
                return 0.0
            
            # 获取最小交易量
            min_qty = self.min_quantities.get(symbol, 0.001)
            
            # 账户风险资金（考虑连续亏损调整）
            risk_adjustment = max(0.5, 1.0 - (self.consecutive_losses * 0.1))  # 每连续亏损一次减少10%风险
            risk_amount = balance * Config.RISK_RATIO * risk_adjustment
            
            # 每份仓位的风险（假设止损距离 = ATR * SL倍数）
            risk_per_unit = atr * Config.SL_ATR_MULT
            if risk_per_unit <= 0:
                return 0.0
            
            # 理论仓位数量（币的数量）
            position_size = risk_amount / risk_per_unit
            
            # 确保不低于最小交易量
            position_size = max(position_size, min_qty)
            
            # 考虑杠杆的最大允许仓位（超出余额会被强制缩小）
            max_notional = balance * Config.LEVERAGE
            max_position = max_notional / price
            position_size = min(position_size, max_position)
            
            return max(0.0, position_size)
        
        except Exception:
            return 0.0
    
    async def execute_signal(self, signal: TradeSignal, free_balance: float) -> Tuple[bool, Optional[TradeSignal]]:
        try:
            # 重新获取最新余额，避免使用过时数据
            balance_info = await self.exchange.fetch_balance()
            free_usdt = balance_info.free
            
            # 计算理论仓位
            raw_qty = self.calculate_position_size(free_usdt, signal.price, signal.atr, signal.symbol)
            if raw_qty <= 0:
                self.logger.warning(f"仓位计算为0或负数: {signal.symbol}")
                return False, None

            # 先按交易规则/精度/最小名义价值修正
            qty_rules = self._apply_exchange_filters(signal.symbol, raw_qty, signal.price)
            if qty_rules <= 0:
                self.logger.warning(f"{signal.symbol} 数量在交易规则收敛后为0（可能余额过低或低于最小名义价值）")
                return False, None

            # 再按可用保证金收敛
            qty_cap = self._cap_by_available_margin(signal.symbol, qty_rules, signal.price, free_usdt)
            if qty_cap <= 0:
                self.logger.error(f"{signal.symbol} 可用保证金不足，放弃下单")
                return False, None

            # 计算动态ATR倍数
            sl_mult, tp_mult = self.dynamic_atr.calculate_dynamic_multipliers(
                signal.symbol, signal.atr, signal.price
            )
            
            # 计算止损和止盈价格
            sl_price = signal.price - signal.atr * sl_mult if signal.side == OrderSide.BUY else signal.price + signal.atr * sl_mult
            tp_price = signal.price + signal.atr * tp_mult if signal.side == OrderSide.BUY else signal.price - signal.atr * tp_mult
            
            # 记录详细的调试信息，包括ATR和止盈止损价格
            self.logger.info(
                f"[{signal.symbol}] 价格={signal.price:.2f}, ATR={signal.atr:.2f}, "
                f"SL={sl_price:.2f}({sl_mult:.1f}×ATR), TP={tp_price:.2f}({tp_mult:.1f}×ATR), "
                f"qty(raw→rules→cap)={raw_qty:.6f}→{qty_rules:.6f}→{qty_cap:.6f}, "
                f"freeUSDT={free_usdt:.2f}, leverage={Config.LEVERAGE}"
            )

            signal.quantity = qty_cap

            order_params = {}
            if Config.HEDGE_MODE:
                order_params['positionSide'] = 'LONG' if signal.side == OrderSide.BUY else 'SHORT'

            # 首次尝试下单
            result = await self.exchange.create_order(
                signal.symbol, 'market', signal.side.value, signal.quantity, None, order_params
            )

            # 如果保证金不足，自动降 30% 再试一次
            errmsg = (result.error or "").lower()
            if (not result.success) and any(k in errmsg for k in ['-2019', 'insufficient', 'margin']):
                self.logger.warning(f"{signal.symbol} 首次下单保证金不足，自动缩小 30% 再试")
                signal.quantity = float(self.exchange.exchange.amount_to_precision(signal.symbol, signal.quantity * 0.7))
                if signal.quantity <= 0:
                    self.update_consecutive_losses(True)
                    return False, None
                result = await self.exchange.create_order(
                    signal.symbol, 'market', signal.side.value, signal.quantity, None, order_params
                )

            if not result.success:
                self.logger.error(f"订单执行失败 {signal.symbol}: {result.error}")
                self.update_consecutive_losses(True)
                return False, None

            # 设置止盈止损（使用动态倍数）
            tp_success = await self.place_tp_order(signal, tp_price)
            sl_success = await self.place_sl_order(signal, sl_price)
            
            if tp_success and sl_success:
                self.logger.info(f"交易执行成功: {signal.symbol} {signal.side.value} 数量: {signal.quantity:.6f}")
                
                # 记录结构化日志
                self.logger.structured_log("order_executed", {
                    "symbol": signal.symbol,
                    "side": signal.side.value,
                    "quantity": signal.quantity,
                    "price": signal.price,
                    "atr": atr,
                    "sl_mult": sl_mult,
                    "tp_mult": tp_mult,
                    "order_id": result.order_id
                })
                
                self.update_consecutive_losses(False) # 交易成功，重置或减少连续亏损计数
                return True, signal
            else:
                self.logger.warning(f"止盈止损设置部分失败: {signal.symbol}")
                # 如果止盈止损设置失败，尝试撤销订单
                try:
                    await self.exchange.create_order(
                        signal.symbol, 'market', 
                        'sell' if signal.side == OrderSide.BUY else 'buy', 
                        signal.quantity, None, 
                        {'reduceOnly': True}
                    )
                    self.logger.info(f"已撤销订单: {signal.symbol}")
                except Exception as e:
                    self.logger.error(f"撤销订单失败: {e}")
                
                self.update_consecutive_losses(True) # 交易失败，增加连续亏损计数
                return False, None

        except Exception as e:
            self.logger.error(f"执行信号失败 {signal.symbol}: {e}")
            self.update_consecutive_losses(True) # 异常情况也视为亏损
            return False, None
    
    async def place_tp_order(self, signal: TradeSignal, tp_price: float) -> bool:
        """完整的止盈单设置"""
        # 精度处理
        tp_price = float(self.exchange.exchange.price_to_precision(signal.symbol, tp_price))
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                params = {
                    'stopPrice': tp_price,
                    'reduceOnly': True
                }
                if Config.HEDGE_MODE:
                    params['positionSide'] = 'LONG' if signal.side == OrderSide.BUY else 'SHORT'
                
                order_side = 'sell' if signal.side == OrderSide.BUY else 'buy'
                
                # 尝试使用take_profit_market订单类型
                try:
                    result = await self.exchange.create_order(
                        signal.symbol,
                        'take_profit_market',
                        order_side,
                        signal.quantity,
                        None,
                        params
                    )
                except Exception:
                    # 如果不支持take_profit_market，尝试使用限价单+条件
                    self.logger.warning(f"take_profit_market订单类型不支持，尝试替代方案")
                    params['type'] = 'LIMIT'
                    params['timeInForce'] = 'GTC'
                    result = await self.exchange.create_order(
                        signal.symbol,
                        'limit',
                        order_side,
                        signal.quantity,
                        tp_price,
                        params
                    )
                
                if result.success:
                    self.logger.info(f"止盈单设置成功: {signal.symbol} @ {tp_price:.2f}")
                    return True
                else:
                    raise Exception(result.error)
                    
            except Exception as e:
                self.logger.warning(f"止盈单设置失败(尝试{attempt+1}): {e}")
                if attempt == Config.MAX_RETRIES - 1:
                    self.logger.error(f"止盈单设置失败: {signal.symbol} - {e}")
                    return False
                await asyncio.sleep(Config.RETRY_DELAY)
        
        return False
    
    async def place_sl_order(self, signal: TradeSignal, sl_price: float) -> bool:
        """完整的止损单设置"""
        # 精度处理
        sl_price = float(self.exchange.exchange.price_to_precision(signal.symbol, sl_price))
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                params = {
                    'stopPrice': sl_price,
                    'reduceOnly': True
                }
                if Config.HEDGE_MODE:
                    params['positionSide'] = 'LONG' if signal.side == OrderSide.BUY else 'SHORT'
                
                order_side = 'sell' if signal.side == OrderSide.BUY else 'buy'
                
                result = await self.exchange.create_order(
                    signal.symbol,
                    'stop_market',
                    order_side,
                    signal.quantity,
                    None,
                    params
                )
                
                if result.success:
                    self.logger.info(f"止损单设置成功: {signal.symbol} @ {sl_price:.2f}")
                    return True
                else:
                    raise Exception(result.error)
                    
            except Exception as e:
                self.logger.warning(f"止损单设置失败(尝试{attempt+1}): {e}")
                if attempt == Config.MAX_RETRIES - 1:
                    self.logger.error(f"止损单设置失败: {signal.symbol} - {e}")
                    return False
                await asyncio.sleep(Config.RETRY_DELAY)
        
        return False
    
    def update_consecutive_losses(self, is_loss: bool):
        """更新连续亏损计数"""
        if is_loss:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = max(0, self.consecutive_losses - 1)

# ================== 增强的风险管理系统 ==================
class EnhancedRiskManager:
    """增强的风险管理系统"""
    
    def __init__(self, exchange: ExchangeInterface, logger: AdvancedLogger):
        self.exchange = exchange
        self.logger = logger
        self.alert_system = AlertSystem(logger)
        self.max_drawdown = 0
        self.equity_high = 0
        self.daily_start_equity = 0
        self.daily_start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.floating_pnl = 0  # 浮动盈亏
        
    async def check_risk_limits(self, balance: float, positions: List[Dict] = None) -> bool:
        """检查风险限制"""
        # 计算浮动盈亏
        if positions:
            self.floating_pnl = await self.calculate_floating_pnl(positions)
        
        # 检查最大回撤
        total_equity = balance + self.floating_pnl
        if total_equity > self.equity_high:
            self.equity_high = total_equity
        
        drawdown = (self.equity_high - total_equity) / self.equity_high if self.equity_high > 0 else 0
        
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        if drawdown > Config.MAX_DRAWDOWN:
            self.logger.critical(f"超过最大回撤限制: {drawdown:.2%} > {Config.MAX_DRAWDOWN:.2%}")
            self.alert_system.send_alert(f"超过最大回撤限制: {drawdown:.2%}")
            return False
        
        # 检查日亏损 - 使用每日起始权益作为基准
        daily_pnl = await self.calculate_daily_pnl(total_equity)
        if daily_pnl < -Config.DAILY_LOSS_LIMIT * self.daily_start_equity:
            self.logger.critical(f"超过日亏损限制: {daily_pnl:.2f}")
            self.alert_system.send_alert(f"超过日亏损限制: {daily_pnl:.2f}")
            return False
        
        # 检查最大风险比
        if self.floating_pnl < -Config.MAX_RISK_RATIO * total_equity:
            self.logger.critical(f"超过最大风险比限制: {self.floating_pnl:.2f}")
            self.alert_system.send_alert(f"超过最大风险比限制: {self.floating_pnl:.2f}")
            return False
        
        return True
    
    async def calculate_floating_pnl(self, positions: List[Dict]) -> float:
        """计算浮动盈亏"""
        floating_pnl = 0
        for pos in positions:
            if 'unrealizedPnl' in pos:
                floating_pnl += float(pos['unrealizedPnl'])
        return floating_pnl
    
    async def calculate_daily_pnl(self, current_equity: float) -> float:
        """计算当日盈亏"""
        # 如果是新的一天，重置起始权益
        now = datetime.now()
        if now.date() != self.daily_start_time.date():
            self.daily_start_equity = current_equity
            self.daily_start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # 计算当日盈亏
        return current_equity - self.daily_start_equity

# ================== 警报系统 ==================
class AlertSystem:
    """警报系统"""
    
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
    
    def send_alert(self, message: str):
        """发送警报"""
        self.logger.critical(f"警报: {message}")
        
        # 发送到Telegram
        if Config.TELEGRAM_BOT_TOKEN and Config.TELEGRAM_CHAT_ID:
            try:
                url = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendMessage"
                payload = {
                    "chat_id": Config.TELEGRAM_CHAT_ID,
                    "text": f"交易警报: {message}",
                    "parse_mode": "HTML"
                }
                response = requests.post(url, json=payload, timeout=10)
                if response.status_code != 200:
                    self.logger.error(f"Telegram消息发送失败: {response.text}")
            except Exception as e:
                self.logger.error(f"发送Telegram警报失败: {e}")
    
    def send_health_status(self, status: HealthStatus):
        """发送健康状态"""
        message = (
            f"🤖 交易机器人健康状态报告:\n"
            f"• 总交易对: {status.total_symbols}\n"
            f"• 已连接: {status.connected_symbols}\n"
            f"• 已断开: {status.disconnected_symbols}\n"
            f"• 错误计数: {status.error_count}\n"
            f"• 最后检查: {status.last_check.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        self.logger.info(message)
        
        # 发送到Telegram
        if Config.TELEGRAM_BOT_TOKEN and Config.TELEGRAM_CHAT_ID:
            try:
                url = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendMessage"
                payload = {
                    "chat_id": Config.TELEGRAM_CHAT_ID,
                    "text": message,
                    "parse_mode": "HTML"
                }
                response = requests.post(url, json=payload, timeout=10)
                if response.status_code != 200:
                    self.logger.error(f"Telegram健康状态发送失败: {response.text}")
            except Exception as e:
                self.logger.error(f"发送Telegram健康状态失败: {e}")

# ================== 状态管理器 ==================
class StateManager:
    """增强的状态管理器"""
    
    def __init__(self, logger: AdvancedLogger, db_manager: DatabaseManager):
        self.logger = logger
        self.db_manager = db_manager
        self.state_file = "trading_state.json"
        self.state = {}
        self.last_save_time = 0
        self.lock = threading.RLock()
        
    def load_state(self):
        """加载状态"""
        with self.lock:
            try:
                # 从数据库加载状态
                self.state = self.db_manager.load_state('app_state', {})
                
                # 恢复活跃持仓
                if 'active_positions' in self.state:
                    active_positions = {}
                    for symbol, pos_data in self.state['active_positions'].items():
                        try:
                            active_positions[symbol] = TradeSignal.from_dict(pos_data)
                        except Exception as e:
                            self.logger.error(f"恢复持仓状态失败 {symbol}: {e}")
                    self.state['active_positions'] = active_positions
                
                self.logger.info("状态已加载")
            except Exception as e:
                self.logger.error(f"加载状态失败: {e}")
                self.state = {}
    
    def save_state(self, force: bool = False):
        """保存状态"""
        with self.lock:
            current_time = time.time()
            if not force and current_time - self.last_save_time < Config.STATE_SAVE_INTERVAL:
                return
                
            try:
                # 创建要写入的副本，不修改原始状态
                payload = dict(self.state)
                if 'active_positions' in payload:
                    serializable_positions = {}
                    for k, v in payload['active_positions'].items():
                        serializable_positions[k] = v.to_dict() if isinstance(v, TradeSignal) else v
                    payload['active_positions'] = serializable_positions
                
                # 保存到数据库
                self.db_manager.save_state('app_state', payload)
                    
                self.last_save_time = current_time
                self.logger.debug("状态已保存")
            except Exception as e:
                self.logger.error(f"保存状态失败: {e}")
    
    def get_state(self, key, default=None):
        """获取状态值"""
        with self.lock:
            return self.state.get(key, default)
    
    def set_state(self, key, value):
        """设置状态值"""
        with self.lock:
            self.state[key] = value
            self.save_state()
    
    def update_state(self, key, updater):
        """原子更新状态值"""
        with self.lock:
            current = self.state.get(key, {})
            if callable(updater):
                new_value = updater(current)
            else:
                new_value = updater
            self.state[key] = new_value
            self.save_state()

# ================== 增强的错误处理 ==================
class EnhancedErrorHandler:
    """增强的错误处理"""
    
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.error_counts = {}
        self.last_alert_time = {}
        
    def handle_error(self, error: Exception, context: str = ""):
        """处理错误"""
        error_type = type(error).__name__
        error_key = f"{error_type}_{context}"
        
        # 计数错误
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # 分类处理错误
        if "Network" in error_type or "Connection" in error_type:
            self.handle_network_error(error, context)
        elif "Insufficient" in error_type or "Balance" in error_type:
            self.handle_balance_error(error, context)
        elif "RateLimit" in error_type:
            self.handle_rate_limit_error(error, context)
        else:
            self.handle_general_error(error, context)
        
        # 如果错误频繁发生，发送警报
        if self.error_counts[error_key] > 5:
            current_time = time.time()
            last_alert = self.last_alert_time.get(error_key, 0)
            
            if current_time - last_alert > 3600:  # 每小时最多报警一次
                self.logger.critical(f"频繁错误警报: {error_key} (count: {self.error_counts[error_key]})")
                self.last_alert_time[error_key] = current_time
    
    def handle_network_error(self, error: Exception, context: str):
        """处理网络错误"""
        self.logger.warning(f"网络错误 {context}: {error}")
        # 实现指数退避重试逻辑
    
    def handle_balance_error(self, error: Exception, context: str):
        """处理余额不足错误"""
        self.logger.error(f"余额不足 {context}: {error}")
        # 可能需要停止交易或调整仓位大小
    
    def handle_rate_limit_error(self, error: Exception, context: str):
        """处理速率限制错误"""
        self.logger.warning(f"速率限制 {context}: {error}")
        # 实现适当的等待和重试逻辑
    
    def handle_general_error(self, error: Exception, context: str):
        """处理一般错误"""
        self.logger.error(f"一般错误 {context}: {e}")

# ================== 主交易机器人 ==================
class EnhancedProductionTrader:
    """增强的生产环境交易机器人"""
    
    def __init__(self):
        self.logger = AdvancedLogger()
        self.cache = TimedCache()
        self.exchange = BinanceExchange(self.logger)
        self.indicators = IndicatorSystem(self.cache)
        self.executor = TradeExecutor(self.exchange, self.logger)
        self.websocket_handler = WebSocketDataHandler(self.exchange, self.logger, Config.SYMBOLS)
        self.risk_manager = EnhancedRiskManager(self.exchange, self.logger)
        self.error_handler = EnhancedErrorHandler(self.logger)
        
        # 数据库和状态管理
        self.db_manager = DatabaseManager()
        self.state_manager = StateManager(self.logger, self.db_manager)
        
        self.active_positions: Dict[str, TradeSignal] = {}
        self.last_state_save = 0
        self.position_check_interval = 300  # 每5分钟检查一次持仓状态
        self.last_position_check = 0
        self.last_health_check = 0
        self.signal_generator = MultiTimeframeSignalGenerator(self.indicators)

        # 加载保存的状态
        self.state_manager.load_state()
        self.active_positions = self.state_manager.get_state('active_positions', {})
        self.risk_manager.equity_high = self.state_manager.get_state('equity_high', 0)
        self.executor.consecutive_losses = self.state_manager.get_state('consecutive_losses', 0)
        
        daily_start_time_str = self.state_manager.get_state('daily_start_time')
        if daily_start_time_str:
            self.risk_manager.daily_start_time = datetime.fromisoformat(daily_start_time_str)

        # 注册优雅退出
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        self.running = True

    async def process_symbol(self, symbol: str):
        """处理单一交易对 - 使用多周期信号"""
        try:
            signal_data = await self.signal_generator.generate_signal(symbol, self.exchange)
            
            # 记录信号生成日志
            if signal_data:
                self.logger.info(f"信号生成: {symbol} {signal_data.side.value} 价格: {signal_data.price:.2f} ATR: {signal_data.atr:.2f}")
            
            return signal_data

        except Exception as e:
            self.error_handler.handle_error(e, f"处理 {symbol}")
            return None
    
    async def close_all_positions(self):
        """平仓所有持仓"""
        self.logger.info("开始平仓所有持仓")
        
        try:
            # 获取当前持仓
            positions = await self.exchange.fetch_positions()
            
            for pos in positions:
                if float(pos.get('contracts', 0)) > 0:
                    symbol = pos['symbol']
                    side = pos['side']
                    size = float(pos['contracts'])
                    
                    # 创建平仓订单
                    close_side = 'sell' if side == 'long' else 'buy'
                    order_params = {'reduceOnly': True}
                    if Config.HEDGE_MODE:
                        order_params['positionSide'] = 'LONG' if side == 'long' else 'SHORT'
                    
                    result = await self.exchange.create_order(
                        symbol, 'market', close_side, size, None, order_params
                    )
                    
                    if result.success:
                        self.logger.info(f"平仓成功: {symbol} {side} {size}")
                        # 从活跃持仓中移除
                        if symbol in self.active_positions:
                            del self.active_positions[symbol]
                            self.state_manager.set_state('active_positions', self.active_positions)
                    else:
                        self.logger.error(f"平仓失败: {symbol} - {result.error}")
                        
        except Exception as e:
            self.logger.error(f"平仓过程中发生错误: {e}")

    async def check_positions(self):
        """检查当前持仓状态"""
        current_time = time.time()
        if current_time - self.last_position_check < self.position_check_interval:
            return
            
        self.last_position_check = current_time
        
        try:
            # 获取交易所持仓信息
            positions = await self.exchange.fetch_positions()
            exchange_positions = {}
            
            # 修复: 正确解析持仓方向
            for pos in positions:
                # 兼容 ccxt 常见字段：contracts(张数)、side('long'/'short')、entryPrice(开仓均价)
                contracts = float(pos.get('contracts', 0) or pos.get('info', {}).get('positionAmt', 0))
                if contracts != 0:
                    symbol = pos.get('symbol')
                    side_str = (pos.get('side') or 'long' if contracts > 0 else 'short').lower()
                    side = PositionSide.LONG if side_str == 'long' else PositionSide.SHORT
                    entry_price = float(pos.get('entryPrice') or 0)
                    exchange_positions[symbol] = {
                        'side': side,
                        'size': contracts,
                        'entry_price': entry_price
                    }
            
            # 检查我们的记录与交易所是否一致
            for symbol in list(self.active_positions.keys()):
                if symbol not in exchange_positions:
                    self.logger.warning(f"持仓 {symbol} 在交易所中不存在，从记录中移除")
                    del self.active_positions[symbol]
                    self.state_manager.set_state('active_positions', self.active_positions)
                    
        except Exception as e:
            self.error_handler.handle_error(e, "检查持仓状态")
    
    async def health_check(self):
        """健康检查"""
        current_time = time.time()
        if current_time - self.last_health_check < Config.HEALTH_CHECK_INTERVAL:
            return
            
        self.last_health_check = current_time
        
        # 获取健康状态
        health_status = self.websocket_handler.get_health_status()
        
        # 记录健康状态
        self.logger.info(
            f"健康检查: {health_status.connected_symbols}/{health_status.total_symbols} "
            f"连接正常, 错误计数: {health_status.error_count}"
        )
        
        # 发送健康状态到Telegram
        self.risk_manager.alert_system.send_health_status(health_status)

    async def run(self):
        """主循环"""
        self.logger.info(f"🚀 启动增强版交易机器人，模式: {Config.MODE}, 对冲: {Config.HEDGE_MODE}, 杠杆: {Config.LEVERAGE}")

        # 启动WebSocket连接
        asyncio.create_task(self.websocket_handler.start())
        
        while self.running:
            try:
                # 获取余额
                balance_info = await self.exchange.fetch_balance()
                free_usdt = balance_info.free
                self.logger.debug(f"账户余额: total={balance_info.total}, free={balance_info.free}, used={balance_info.used}")

                # 获取持仓信息用于风险计算
                positions = await self.exchange.fetch_positions()

                # 检查风险限制
                if not await self.risk_manager.check_risk_limits(balance_info.total, positions):
                    self.logger.critical("风险限制触发，停止交易")
                    await self.close_all_positions()
                    break

                # 检查持仓状态
                await self.check_positions()

                # 健康检查
                await self.health_check()

                # 处理所有交易对，而不仅仅是WebSocket触发的
                for symbol in Config.SYMBOLS:
                    # 处理信号生成和交易执行
                    signal = await self.process_symbol(symbol)
                    
                    if signal:
                        # 风控：限制最大持仓数
                        if len(self.active_positions) >= Config.MAX_POSITIONS:
                            self.logger.warning(f"持仓已满({Config.MAX_POSITIONS})，跳过 {signal.symbol}")
                            continue

                        # 如果已有同一方向持仓，跳过
                        if signal.symbol in self.active_positions:
                            existing_signal = self.active_positions[signal.symbol]
                            if existing_signal.side == signal.side:
                                self.logger.debug(f"{signal.symbol} 已有同方向持仓，跳过新信号")
                                continue

                        # 执行交易
                        success, executed_sig = await self.executor.execute_signal(signal, free_usdt)
                        if success and executed_sig:
                            self.active_positions[signal.symbol] = executed_sig
                            self.state_manager.set_state('active_positions', self.active_positions)
                            
                # 定期保存状态
                current_time = time.time()
                if current_time - self.last_state_save >= Config.STATE_SAVE_INTERVAL:
                    self.state_manager.save_state(force=True)
                    self.last_state_save = current_time
                
                await asyncio.sleep(Config.POLL_INTERVAL)  # 使用配置的轮询间隔

            except Exception as e:
                self.error_handler.handle_error(e, "主循环")
                await asyncio.sleep(5)

    def stop(self, *args):
        """优雅退出"""
        self.logger.info("🛑 收到停止信号，正在退出...")
        self.running = False
        self.state_manager.save_state(force=True)
        # 关闭WebSocket连接（兼容是否仍在运行事件循环）
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.websocket_handler.stop())
        except RuntimeError:
            # 没有运行中的事件循环，直接同步关闭
            try:
                asyncio.run(self.websocket_handler.stop())
            except RuntimeError:
                # 如果已经在另一个 loop 上下文，忽略即可
                pass

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
