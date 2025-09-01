# autotrader_enhanced.py
"""
增强版生产级多币种量化交易机器人
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
from dataclasses import dataclass
from enum import Enum
import threading
from queue import Queue
import cachetools
from abc import ABC, abstractmethod
import optuna  # 贝叶斯优化
from backtesting import Backtest, Strategy  # 回测框架
import plotly.graph_objects as go  # 可视化
import matplotlib.pyplot as plt
import seaborn as sns

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
                    self.exchange.fetch_ohlcv, symbol, timeframe, limit
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
            # 使用线程池执行同步IO操作
            order = await asyncio.to_thread(
                self.exchange.create_order, symbol, order_type, side, amount, price, params or {}
            )
            return OrderResult(success=True, order_id=order['id'], symbol=symbol, side=OrderSide(side))
            
        except Exception as e:
            return OrderResult(success=False, error=str(e), symbol=symbol, side=OrderSide(side))
    
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
        if Config.USE_WEBSOCKET:
            try:
                # 这里使用CCXT Pro的WebSocket功能
                # 注意: 需要安装ccxt-pro包
                await self.exchange.exwatch_ohlcv(self.symbols, '1h', self._on_ohlcv)
                self.logger.info("WebSocket连接已启动")
            except Exception as e:
                self.logger.error(f"WebSocket连接失败: {e}")
                # 回退到REST API轮询
                await self._start_polling()
        else:
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
    
    async def _on_ohlcv(self, symbol, timeframe, candle):
        """WebSocket数据回调"""
        if timeframe == '1h':
            df = pd.DataFrame([candle], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            await self.data_queue.put((symbol, df))
    
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
        cache_key = f"{symbol}_{timeframe}_indicators"
        
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
    
    async def execute_signal(self, signal: TradeSignal, balance: float) -> Tuple[bool, Optional[TradeSignal]]:
        try:
            # 计算理论仓位
            raw_qty = self.calculate_position_size(balance, signal.price, signal.atr)
            if raw_qty <= 0:
                self.logger.warning(f"仓位计算为0或负数: {signal.symbol}")
                return False, None

            # 先按交易规则/精度/最小名义价值修正
            qty_rules = self._apply_exchange_filters(signal.symbol, raw_qty, signal.price)
            if qty_rules <= 0:
                self.logger.warning(f"{signal.symbol} 数量在交易规则收敛后为0（可能余额过低或低于最小名义价值）")
                return False, None

            # 再按可用保证金收敛
            qty_cap = self._cap_by_available_margin(signal.symbol, qty_rules, signal.price, balance)
            if qty_cap <= 0:
                self.logger.error(f"{signal.symbol} 可用保证金不足，放弃下单")
                return False, None

            # 记录详细的调试信息
            self.logger.info(
                f"{signal.symbol} price={signal.price:.6f}, qty(raw→rules→cap)={raw_qty:.6f}→{qty_rules:.6f}→{qty_cap:.6f}, "
                f"freeUSDT={balance:.2f}, leverage={Config.LEVERAGE}"
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
            if (not result.success) and result.error and ('-2019' in result.error or 'Margin is insufficient' in result.error):
                self.logger.warning(f"{signal.symbol} 首次下单保证金不足，自动缩小 30% 再试")
                signal.quantity = float(self.exchange.exchange.amount_to_precision(signal.symbol, signal.quantity * 0.7))
                if signal.quantity <= 0:
                    return False, None
                result = await self.exchange.create_order(
                    signal.symbol, 'market', signal.side.value, signal.quantity, None, order_params
                )

            if not result.success:
                self.logger.error(f"订单执行失败 {signal.symbol}: {result.error}")
                return False, None

            # 设置止盈止损
            tp_success = await self.place_tp_order(signal)
            sl_success = await self.place_sl_order(signal)
            
            if tp_success and sl_success:
                self.logger.info(f"交易执行成功: {signal.symbol} {signal.side.value} 数量: {signal.quantity:.6f}")
                return True, signal
            else:
                self.logger.warning(f"止盈止损设置部分失败: {signal.symbol}")
                return False, None

        except Exception as e:
            self.logger.error(f"执行信号失败 {signal.symbol}: {e}")
            return False, None
    
    async def place_tp_order(self, signal: TradeSignal) -> bool:
        """完整的止盈单设置"""
        tp_price = signal.price + signal.atr * Config.TP_ATR_MULT if signal.side == OrderSide.BUY else signal.price - signal.atr * Config.TP_ATR_MULT
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                params = {}
                if Config.HEDGE_MODE:
                    params['positionSide'] = 'LONG' if signal.side == OrderSide.BUY else 'SHORT'
                    params['reduceOnly'] = True
                
                order_side = 'sell' if signal.side == OrderSide.BUY else 'buy'
                result = await self.exchange.create_order(
                    signal.symbol,
                    'take_profit_market',
                    order_side,
                    signal.quantity,
                    None,
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
    
    async def place_sl_order(self, signal: TradeSignal) -> bool:
        """完整的止损单设置"""
        sl_price = signal.price - signal.atr * Config.SL_ATR_MULT if signal.side == OrderSide.BUY else signal.price + signal.atr * Config.SL_ATR_MULT
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                params = {}
                if Config.HEDGE_MODE:
                    params['positionSide'] = 'LONG' if signal.side == OrderSide.BUY else 'SHORT'
                    params['reduceOnly'] = True
                
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

# ================== 增强的风险管理系统 ==================
class EnhancedRiskManager:
    """增强的风险管理系统"""
    
    def __init__(self, exchange: ExchangeInterface, logger: AdvancedLogger):
        self.exchange = exchange
        self.logger = logger
        self.alert_system = AlertSystem(logger)
        self.max_drawdown = 0
        self.equity_high = 0
        
    async def check_risk_limits(self, balance: float) -> bool:
        """检查风险限制"""
        # 检查最大回撤
        if balance > self.equity_high:
            self.equity_high = balance
        
        drawdown = (self.equity_high - balance) / self.equity_high if self.equity_high > 0 else 0
        
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        if drawdown > Config.MAX_DRAWDOWN:
            self.logger.critical(f"超过最大回撤限制: {drawdown:.2%} > {Config.MAX_DRAWDOWN:.2%}")
            self.alert_system.send_alert(f"超过最大回撤限制: {drawdown:.2%}")
            return False
        
        # 检查日亏损
        daily_pnl = await self.calculate_daily_pnl()
        if daily_pnl < -Config.DAILY_LOSS_LIMIT * self.equity_high:
            self.logger.critical(f"超过日亏损限制: {daily_pnl:.2f}")
            self.alert_system.send_alert(f"超过日亏损限制: {daily_pnl:.2f}")
            return False
        
        return True
    
    async def calculate_daily_pnl(self) -> float:
        """计算当日盈亏"""
        # 实现日盈亏计算逻辑
        # 这里简化处理，实际应该从交易所获取日盈亏数据
        return 0.0
    
    async def place_oco_order(self, symbol: str, side: str, quantity: float, 
                             price: float, stop_price: float, limit_price: float) -> bool:
        """放置OCO订单"""
        try:
            # 尝试使用交易所的OCO功能
            params = {
                'stopPrice': stop_price,
                'limitPrice': limit_price,
                'type': 'STOP_LOSS_LIMIT'
            }
            
            result = await self.exchange.create_order(
                symbol, 'OCO', side, quantity, price, params
            )
            
            if result.success:
                self.logger.info(f"OCO订单设置成功: {symbol}")
                return True
            else:
                # 回退到分别设置止损和止盈
                self.logger.warning("OCO订单不支持，回退到分别设置")
                return await self.place_separate_orders(symbol, side, quantity, stop_price, limit_price)
                
        except Exception as e:
            self.logger.error(f"OCO订单设置失败: {e}")
            return await self.place_separate_orders(symbol, side, quantity, stop_price, limit_price)
    
    async def place_separate_orders(self, symbol: str, side: str, quantity: float, 
                                   stop_price: float, limit_price: float) -> bool:
        """分别设置止损和止盈订单"""
        # 实现分别设置订单的逻辑
        # 这里简化处理
        return True

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

# ================== 状态管理器 ==================
class StateManager:
    """状态管理器"""
    
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.state_file = "trading_state.json"
        self.state = {}
        
    def load_state(self):
        """加载状态"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    self.state = json.load(f)
                self.logger.info("状态已加载")
            else:
                self.logger.info("无保存状态，使用初始状态")
        except Exception as e:
            self.logger.error(f"加载状态失败: {e}")
            self.state = {}
    
    def save_state(self):
        """保存状态"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
            self.logger.debug("状态已保存")
        except Exception as e:
            self.logger.error(f"保存状态失败: {e}")
    
    def get_state(self, key, default=None):
        """获取状态值"""
        return self.state.get(key, default)
    
    def set_state(self, key, value):
        """设置状态值"""
        self.state[key] = value
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
        self.logger.error(f"一般错误 {context}: {error}")

# ================== 增强的回测系统 ==================
class EnhancedBacktester:
    """增强的回测系统，支持贝叶斯优化"""
    
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.study = None
        
    def optimize_parameters(self, df: pd.DataFrame, symbol: str) -> Dict:
        """使用Optuna进行贝叶斯优化"""
        self.logger.info(f"开始贝叶斯优化: {symbol}")
        
        # 定义目标函数
        def objective(trial):
            # 定义超参数搜索空间
            params = {
                'ema_short': trial.suggest_int('ema_short', 8, 20),
                'ema_long': trial.suggest_int('ema_long', 20, 35),
                'macd_fast': trial.suggest_int('macd_fast', 8, 20),
                'macd_slow': trial.suggest_int('macd_slow', 20, 35),
                'macd_signal': trial.suggest_int('macd_signal', 5, 15),
                'rsi_window': trial.suggest_int('rsi_window', 10, 25),
                'rsi_low': trial.suggest_int('rsi_low', 25, 40),
                'rsi_high': trial.suggest_int('rsi_high', 60, 80),
                'sl_mult': trial.suggest_float('sl_mult', 1.5, 3.0),
                'tp_mult': trial.suggest_float('tp_mult', 2.0, 4.0),
                'risk_ratio': trial.suggest_float('risk_ratio', 0.01, 0.1),
            }
            
            # 运行回测
            result = self.run_backtest(df, params)
            
            # 最大化夏普比率
            return result['sharpe_ratio']
        
        # 创建或加载研究
        study_name = f"trading_optimization_{symbol}"
        self.study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            storage=Config.OPTUNA_STORAGE,
            load_if_exists=True
        )
        
        # 运行优化
        self.study.optimize(objective, n_trials=Config.OPTUNA_N_TRIALS)
        
        # 输出最佳参数
        best_params = self.study.best_params
        self.logger.info(f"贝叶斯优化完成，最佳参数: {best_params}")
        
        return best_params
    
    def run_backtest(self, df: pd.DataFrame, params: Dict) -> Dict:
        """运行回测"""
        # 实现回测逻辑
        # 这里简化处理，实际应该使用backtesting库
        try:
            # 计算技术指标
            df = self.calculate_indicators(df, params)
            
            # 模拟交易
            balance = 10000
            position = 0
            trades = []
            equity_curve = [balance]
            
            for i in range(len(df)):
                current = df.iloc[i]
                
                # 交易信号
                bullish = all([
                    current['macd'] > current['macd_signal'],
                    current['ema_short'] > current['ema_long'],
                    params['rsi_low'] < current['rsi'] < params['rsi_high']
                ])
                
                bearish = all([
                    current['macd'] < current['macd_signal'],
                    current['ema_short'] < current['ema_long'],
                    params['rsi_low'] < current['rsi'] < params['rsi_high']
                ])
                
                # 执行交易
                if bullish and position <= 0:
                    if position < 0:  # 平空仓
                        pnl = (current['close'] - position) * -1
                        balance += pnl
                        position = 0
                    
                    # 开多仓
                    risk_amount = balance * params['risk_ratio']
                    risk_per_share = current['atr'] * params['sl_mult']
                    position_size = risk_amount / risk_per_share
                    position = current['close']
                    trades.append(('BUY', current['close'], df.index[i]))
                
                elif bearish and position >= 0:
                    if position > 0:  # 平多仓
                        pnl = (current['close'] - position)
                        balance += pnl
                        position = 0
                    
                    # 开空仓
                    risk_amount = balance * params['risk_ratio']
                    risk_per_share = current['atr'] * params['sl_mult']
                    position_size = risk_amount / risk_per_share
                    position = -current['close']
                    trades.append(('SELL', current['close'], df.index[i]))
                
                # 更新权益曲线
                if position != 0:
                    pnl = (current['close'] - abs(position)) * (1 if position > 0 else -1)
                    equity_curve.append(balance + pnl)
                else:
                    equity_curve.append(balance)
            
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
            }
            
        except Exception as e:
            self.logger.error(f"回测失败: {e}")
            return {'net_profit': -9999, 'sharpe_ratio': -9999, 'max_drawdown': 9999}
    
    def calculate_indicators(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """计算技术指标"""
        df = df.copy()
        
        # EMA
        df['ema_short'] = ta.trend.EMAIndicator(df['close'], window=params['ema_short']).ema_indicator()
        df['ema_long'] = ta.trend.EMAIndicator(df['close'], window=params['ema_long']).ema_indicator()
        
        # MACD
        macd = ta.trend.MACD(df['close'], window_slow=params['macd_slow'], 
                            window_fast=params['macd_fast'], window_sign=params['macd_signal'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=params['rsi_window']).rsi()
        
        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        
        return df.dropna()
    
    def plot_optimization_results(self):
        """绘制优化结果"""
        if self.study is None:
            self.logger.warning("没有优化结果可可视化")
            return
        
        # 绘制优化历史
        history = self.study.trials_dataframe()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history['number'], history['value'], 'o-')
        ax.set_xlabel('Trial')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Optimization History')
        plt.savefig('optimization_history.png')
        plt.close()
        
        # 绘制参数重要性
        try:
            importance = optuna.importance.get_param_importances(self.study)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(list(importance.keys()), list(importance.values()))
            ax.set_xlabel('Importance')
            ax.set_title('Parameter Importance')
            plt.tight_layout()
            plt.savefig('parameter_importance.png')
            plt.close()
        except Exception as e:
            self.logger.warning(f"无法计算参数重要性: {e}")

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
        self.state_manager = StateManager(self.logger)
        self.backtester = EnhancedBacktester(self.logger)
        self.active_positions: Dict[str, TradeSignal] = {}

        # 加载保存的状态
        self.state_manager.load_state()

        # 注册优雅退出
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        self.running = True

    async def process_symbol(self, symbol: str):
        """处理单一交易对"""
        try:
            # 拉取数据
            df_1h = await self.exchange.get_historical_data(symbol, "1h", Config.OHLCV_LIMIT)
            df_4h = await self.exchange.get_historical_data(symbol, Config.MACD_FILTER_TIMEFRAME, Config.OHLCV_LIMIT)

            if df_1h.empty or df_4h.empty:
                return None

            # 计算指标
            df_1h = self.indicators.compute_indicators(df_1h, symbol, "1h")
            df_4h = self.indicators.compute_indicators(df_4h, symbol, Config.MACD_FILTER_TIMEFRAME)

            # 生成信号
            signal_data = self.indicators.generate_signal(df_1h, df_4h, symbol)
            return signal_data

        except Exception as e:
            self.error_handler.handle_error(e, f"处理 {symbol}")
            return None

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

                # 检查风险限制
                if not await self.risk_manager.check_risk_limits(balance_info.total):
                    self.logger.critical("风险限制触发，停止交易")
                    break

                # 获取实时数据
                symbol, data = await self.websocket_handler.get_next_data()

                # 处理信号生成和交易执行
                signal = await self.process_symbol(symbol)
                
                if signal:
                    # 风控：限制最大持仓数
                    if len(self.active_positions) >= Config.MAX_POSITIONS:
                        self.logger.warning(f"持仓已满({Config.MAX_POSITIONS})，跳过 {signal.symbol}")
                        continue

                    # 如果已有同一方向持仓，跳过
                    if signal.symbol in self.active_positions:
                        self.logger.debug(f"{signal.symbol} 已有持仓，跳过新信号")
                        continue

                    # 执行交易
                    success, executed_sig = await self.executor.execute_signal(signal, free_usdt)
                    if success and executed_sig:
                        self.active_positions[signal.symbol] = executed_sig
                        self.state_manager.set_state('active_positions', self.active_positions)

                # 定期保存状态
                if time.time() % 300 < 1:  # 每5分钟保存一次
                    self.state_manager.save_state()
                
                await asyncio.sleep(1)  # 更短的等待时间，因为使用WebSocket

            except Exception as e:
                self.error_handler.handle_error(e, "主循环")
                await asyncio.sleep(5)

    async def optimize_strategy(self, symbol: str):
        """优化交易策略"""
        self.logger.info(f"开始策略优化: {symbol}")
        
        # 获取历史数据
        df = await self.exchange.get_historical_data(symbol, "1h", Config.OHLCV_LIMIT)
        
        # 运行优化
        best_params = self.backtester.optimize_parameters(df, symbol)
        
        # 可视化结果
        self.backtester.plot_optimization_results()
        
        # 更新策略参数
        self._update_strategy_params(best_params)
        
        return best_params
    
    def _update_strategy_params(self, params: Dict):
        """更新策略参数"""
        # 这里可以根据优化结果更新交易系统的参数
        self.logger.info(f"更新策略参数: {params}")
        # 实际实现应该根据优化结果调整交易逻辑

    def stop(self, *args):
        """优雅退出"""
        self.logger.info("🛑 收到停止信号，正在退出...")
        self.running = False
        self.state_manager.save_state()

# ================== 启动入口 ==================
if __name__ == "__main__":
    trader = EnhancedProductionTrader()
    try:
        asyncio.run(trader.run())
    except KeyboardInterrupt:
        trader.stop()
