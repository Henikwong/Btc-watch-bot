# autotrader_enhanced.py
"""
增强版生产级多币种量化交易机器人 - 修复版
集成贝叶斯优化、WebSocket支持、高级风控和状态持久化
"""

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
import optuna  # 贝叶斯优化
import uuid
import hashlib

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
    RISK_RATIO = float(os.getenv("RISK_RATIO", "0.05"))  # 从 0.15 降到 0.05
    LEVERAGE = int(os.getenv("LEVERAGE", "5"))  # 从 10 降到 5
    SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "2.0"))
    TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "3.0"))
    RISK_ATR_MULT = float(os.getenv("RISK_ATR_MULT", "1.5"))
    PARTIAL_TP_RATIO = float(os.getenv("PARTIAL_TP_RATIO", "0.3"))
    MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "5"))
    
    # 时间参数
    POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))
    MARGIN_COOLDOWN = int(os.getenv("MARGIN_COOLDOWN", "3600"))
    SUMMARY_INTERVAL = int(os.getenv("SUMMARY_INTERVAL", "3600"))
    OHLCV_LIMIT = int(os.getenv("OHLCV_LIMIT", "200"))
    MACD_FILTER_TIMEFRAME = os.getenv("MACD_FILTER_TIMEFRAME", "4h")
    
    # API配置
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
    
    # 性能配置
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY = float(os.getenv("RETRY_DELAY", "2.0"))
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "5"))
    CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))
    VOLUME_FILTER_MULTIPLIER = float(os.getenv("VOLUME_FILTER_MULTIPLIER", "0.8"))
    
    # 风控参数
    MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", "0.2"))
    DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", "0.1"))
    ORDER_TIMEOUT = int(os.getenv("ORDER_TIMEOUT", "30"))
    
    # 新增配置
    USE_WEBSOCKET = os.getenv("USE_WEBSOCKET", "false").lower() == "true"
    BAYESIAN_OPTIMIZATION = os.getenv("BAYESIAN_OPTIMIZATION", "false").lower() == "true"
    CROSS_VALIDATION_FOLDS = int(os.getenv("CROSS_VALIDATION_FOLDS", "3"))
    SLIPPAGE_RATIO = float(os.getenv("SLIPPAGE_RATIO", "0.0005"))  # 0.05%滑点
    COMMISSION_RATE = float(os.getenv("COMMISSION_RATE", "0.001"))  # 0.1%手续费
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
    OPTUNA_STORAGE = os.getenv("OPTUNA_STORAGE", "sqlite:///optuna.db")
    OPTUNA_N_TRIALS = int(os.getenv("OPTUNA_N_TRIALS", "100"))
    
    # 新增: 状态保存间隔(秒)
    STATE_SAVE_INTERVAL = int(os.getenv("STATE_SAVE_INTERVAL", "300"))

# ================== 日志系统 ==================
class AdvancedLogger:
    """高级日志系统"""
    
    def __init__(self):
        self.setup_logging()
    
    def setup_logging(self):
        """配置日志"""
        log_level = logging.DEBUG if Config.MODE == Mode.BACKTEST else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'trading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
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
    
    async def get_historical_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """异步获取历史数据"""
        for attempt in range(Config.MAX_RETRIES):
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
                if attempt == Config.MAX_RETRIES - 1:
                    self.logger.error(f"获取历史数据失败 {symbol}: {e}")
                    raise
                await asyncio.sleep(Config.RETRY_DELAY * (attempt + 1))
    
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
    
    async def fetch_positions(self) -> List[Dict]:
        """异步获取持仓信息"""
        try:
            return await asyncio.to_thread(self.exchange.fetch_positions)
        except Exception as e:
            self.logger.error(f"获取持仓失败: {e}")
            return []
    
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
    """WebSocket实时数据处理器"""
    
    def __init__(self, exchange: ExchangeInterface, logger: AdvancedLogger, symbols: List[str]):
        self.exchange = exchange
        self.logger = logger
        self.symbols = symbols
        self.data_queue = asyncio.Queue()
        self.running = False
        self.ohlcv_data = {}
        
    async def start(self):
        """启动WebSocket连接"""
        self.running = True
        # 暂时禁用WebSocket，专注于轮询模式
        await self._start_polling()
    
    async def _start_polling(self):
        """启动轮询模式"""
        self.logger.info("使用REST API轮询模式")
        while self.running:
            try:
                for symbol in self.symbols:
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
        
        return df.dropna()
    
    def generate_signal(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame, symbol: str) -> Optional[TradeSignal]:
        if df_1h.empty or df_4h.empty:
            return None
        
        current_1h = df_1h.iloc[-1]
        current_4h = df_4h.iloc[-1]
        
        # 检查是否有NaN值
        if (pd.isna(current_1h.get('volume_ma', 0)) or 
            pd.isna(current_1h.get('volume', 0))):
            return None
        
        # 动态成交量过滤
        vol_threshold = current_1h.get('volume_ma', 0) * Config.VOLUME_FILTER_MULTIPLIER
        if current_1h['volume'] < vol_threshold:
            return None
        
        # 信号逻辑
        price = current_1h['close']
        atr = current_1h['atr']
        
        bullish_conditions = all([
            current_1h['macd'] > current_1h['macd_signal'],
            current_1h['ema_12'] > current_1h['ema_26'],
            40 < current_1h['rsi'] < 70,
            current_4h['ema_12'] > current_4h['ema_26']
        ])
        
        bearish_conditions = all([
            current_1h['macd'] < current_1h['macd_signal'],
            current_1h['ema_12'] < current_1h['ema_26'],
            30 < current_1h['rsi'] < 60,
            current_4h['ema_12'] < current_4h['ema_26']
        ])
        
        if bullish_conditions:
            return TradeSignal(
                symbol=symbol,
                side=OrderSide.BUY,
                price=price,
                atr=atr,
                quantity=0,  # 将在执行时计算
                timestamp=datetime.now()
            )
        elif bearish_conditions:
            return TradeSignal(
                symbol=symbol,
                side=OrderSide.SELL,
                price=price,
                atr=atr,
                quantity=0,
                timestamp=datetime.now()
            )
        
        return None

# ================== 交易执行器 ==================
class TradeExecutor:
    """完整的交易执行器"""
    
    def __init__(self, exchange: ExchangeInterface, logger: AdvancedLogger):
        self.exchange = exchange
        self.logger = logger
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
    
    def calculate_position_size(self, balance: float, price: float, atr: float) -> float:
        """
        根据账户余额、价格和ATR计算仓位大小
        - balance: 可用余额（USDT）
        - price: 当前标的价格
        - atr: 平均真实波幅，用于估算风险
        """
        try:
            if atr <= 0 or price <= 0:
                return 0.0
            
            # 账户风险资金
            risk_amount = balance * Config.RISK_RATIO
            
            # 每份仓位的风险（假设止损距离 = ATR * SL倍数）
            risk_per_unit = atr * Config.SL_ATR_MULT
            if risk_per_unit <= 0:
                return 0.0
            
            # 理论仓位数量（币的数量）
            position_size = risk_amount / risk_per_unit
            
            # 考虑杠杆的最大允许仓位（超出余额会被强制缩小）
            max_notional = balance * Config.LEVERAGE
            max_position = max_notional / price
            position_size = min(position_size, max_position)
            
            return max(0.0, position_size)
        
        except Exception:
            return 0.0
    
    async def execute_signal(self, signal: TradeSignal, free_balance: float) -> Tuple[bool, Optional[TradeSignal]]:
        try:
            # 计算理论仓位
            raw_qty = self.calculate_position_size(free_balance, signal.price, signal.atr)
            if raw_qty <= 0:
                self.logger.warning(f"仓位计算为0或负数: {signal.symbol}")
                return False, None

            # 先按交易规则/精度/最小名义价值修正
            qty_rules = self._apply_exchange_filters(signal.symbol, raw_qty, signal.price)
            if qty_rules <= 0:
                self.logger.warning(f"{signal.symbol} 数量在交易规则收敛后为0（可能余额过低或低于最小名义价值）")
                return False, None

            # 再按可用保证金收敛
            qty_cap = self._cap_by_available_margin(signal.symbol, qty_rules, signal.price, free_balance)
            if qty_cap <= 0:
                self.logger.error(f"{signal.symbol
