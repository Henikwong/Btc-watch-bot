# autotrader_final.py
"""
最终版生产级多币种量化交易机器人 - 修复所有问题
支持异步IO、完整风控、实时余额获取
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
    MAX_DRAWDOWN = 0.2
    DAILY_LOSS_LIMIT = 0.1
    ORDER_TIMEOUT = 30

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

        # --- 新增：为每个交易对设置杠杆与保证金模式（忽略失败） ---
        mode = 'CROSS'  # 或 'ISOLATED'
        for sym in Config.SYMBOLS:
            try:
                # ccxt 统一方法（新版本支持）；旧版可以用 self.exchange.fapiPrivate_post_leverage
                if hasattr(self.exchange, 'set_leverage'):
                    self.exchange.set_leverage(Config.LEVERAGE, sym)
                # 保证金模式（有的 ccxt 版本方法名是 set_margin_mode / set_margin_mode）
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

    # --- 新增：读取市场规则并按精度/步进/最小名义价值修正数量 ---
    def _apply_exchange_filters(self, symbol: str, qty: float, price: float) -> float:
        ex = self.exchange.exchange  # 取到 ccxt 实例
        market = ex.market(symbol)
        # 名义价值(合约用)：qty * price 可能还需要 * contractSize，部分永续为 1
        contract_size = market.get('contractSize', 1) or 1
        notional = qty * price * contract_size

        # 交易所限制
        limits = market.get('limits', {})
        min_qty  = (limits.get('amount', {}) or {}).get('min', None)
        min_cost = (limits.get('cost',  {}) or {}).get('min', None)
        step     = (market.get('precision', {}) or {}).get('amount', None)  # 精度位数
        
        # 先按最小数量抬一档
        if min_qty and qty < min_qty:
            qty = min_qty

        # 再按名义价值抬到 min_notional
        if min_cost and notional < min_cost:
            # 抬到最小名义价值
            target_qty = (min_cost / (price * contract_size)) * 1.02  # +2% 缓冲
            qty = max(qty, target_qty)

        # 按精度对齐
        qty = float(ex.amount_to_precision(symbol, qty))
        return max(0.0, qty)

    # --- 新增：保证金预检，基于可用余额二次收敛 ---
    def _cap_by_available_margin(self, symbol: str, qty: float, price: float, free_usdt: float) -> float:
        leverage = Config.LEVERAGE
        ex = self.exchange.exchange
        market = ex.market(symbol)
        contract_size = market.get('contractSize', 1) or 1

        # 预估初始保证金 = 名义价值/leverage，再加 2% 费用/波动缓冲
        notional = qty * price * contract_size
        init_margin = (notional / leverage) * 1.02
        if init_margin <= 0:
            return 0.0

        if init_margin <= free_usdt:
            return qty

        # 超出余额 -> 按比例缩小
        scale = max(0.0, (free_usdt / init_margin) * 0.98)  # 留 2% 余量
        capped = qty * scale
        capped = float(ex.amount_to_precision(symbol, capped))
        return max(0.0, capped)
    
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

            # --- 首次尝试下单 ---
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
    
    def calculate_position_size(self, balance: float, price: float, atr: float) -> float:
        risk_amount = balance * Config.RISK_RATIO
        risk_per_share = atr * Config.SL_ATR_MULT
        position_size = risk_amount / risk_per_share
        
        # 计算最大可用仓位（考虑杠杆）
        max_position = (balance * Config.LEVERAGE) / price
        
        # 取两者中较小的值
        calculated_size = min(position_size, max_position)
        
        self.logger.debug(f"仓位计算: 风险金额={risk_amount:.2f}, 每份风险={risk_per_share:.2f}, "
                         f"计算数量={calculated_size:.6f}, 最大数量={max_position:.6f}")
        
        return calculated_size
    
    async def place_tp_order(self, signal: TradeSignal) -> bool:
        """完整的止盈单设置"""
        tp_price = signal.price + signal.atr * Config.TP_ATR_MULT if signal.side == OrderSide.BUY else signal.price - signal.atr * Config.TP_ATR_MULT
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                params = {}
                if Config.HEDGE_MODE:
                    params['positionSide'] = 'LONG' if signal.side == OrderSide.BUY else 'SHORT'
                    params['reduceOnly'] = True
                # 非对冲模式不使用reduceOnly
                
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
                # 非对冲模式不使用reduceOnly
                
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

# ================== 主交易机器人 ==================
class ProductionTrader:
    """生产环境交易机器人"""
    
    def __init__(self):
        self.logger = AdvancedLogger()
        self.cache = TimedCache()
        self.exchange = BinanceExchange(self.logger)
        self.indicator_system = IndicatorSystem(self.cache)
        self.trade_executor = TradeExecutor(self.exchange, self.logger)
        self.running = False
        self.last_balance = BalanceInfo(total=0, free=0, used=0)
    
    async def run(self):
        """主运行循环"""
        self.logger.info("🚀 启动生产环境交易机器人")
        self.running = True
        
        try:
            while self.running:
                start_time = time.time()
                
                # 获取实时余额
                self.last_balance = await self.exchange.fetch_balance()
                if self.last_balance.free <= 0:
                    self.logger.error("账户余额不足，停止交易")
                    break
                
                # 异步处理所有交易对
                tasks = [self.process_symbol(symbol) for symbol in Config.SYMBOLS]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 处理结果 - 只统计成功执行的信号
                successful_signals = 0
                for result in results:
                    if isinstance(result, tuple) and result[0]:  # (success, signal)
                        successful_signals += 1
                
                processing_time = time.time() - start_time
                self.logger.info(f"本轮处理完成: {successful_signals}个成功信号, 耗时: {processing_time:.2f}s")
                
                # 精确控制轮询间隔，记录超时情况
                sleep_time = max(0, Config.POLL_INTERVAL - processing_time)
                if sleep_time == 0:
                    self.logger.warning(f"处理超时: 实际耗时{processing_time:.2f}s > 轮询间隔{Config.POLL_INTERVAL}s")
                
                await asyncio.sleep(sleep_time)
                
        except Exception as e:
            self.logger.critical(f"主循环异常: {e}")
        finally:
            await self.shutdown()
    
    async def process_symbol(self, symbol: str) -> Tuple[bool, Optional[TradeSignal]]:
        """处理单个交易对，返回执行结果和信号"""
        try:
            # 获取数据
            df_1h = await self.exchange.get_historical_data(symbol, '1h', Config.OHLCV_LIMIT)
            df_4h = await self.exchange.get_historical_data(symbol, Config.MACD_FILTER_TIMEFRAME, Config.OHLCV_LIMIT)
            
            if df_1h.empty or df_4h.empty:
                self.logger.warning(f"数据为空: {symbol}")
                return False, None
            
            # 计算指标
            df_1h = self.indicator_system.compute_indicators(df_1h, symbol, '1h')
            df_4h = self.indicator_system.compute_indicators(df_4h, symbol, '4h')
            
            # 生成信号
            signal = self.indicator_system.generate_signal(df_1h, df_4h, symbol)
            
            if signal:
                self.logger.info(f"发现信号: {symbol} {signal.side.value} 价格: {signal.price:.2f}")
                
                if Config.MODE == Mode.LIVE:
                    # 使用实时余额执行交易
                    success, executed_signal = await self.trade_executor.execute_signal(signal, self.last_balance.free)
                    return success, executed_signal if success else None
            else:
                self.logger.debug(f"无交易信号: {symbol}")
                
            return False, signal
            
        except Exception as e:
            self.logger.error(f"处理交易对 {symbol} 失败: {e}")
            return False, None
    
    async def shutdown(self):
        """安全关闭"""
        self.logger.info("正在安全关闭交易机器人...")
        self.running = False

# ================== 主程序入口 ==================
async def main():
    trader = ProductionTrader()
    
    # 信号处理
    def signal_handler(signum, frame):
        asyncio.create_task(trader.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await trader.run()
    except KeyboardInterrupt:
        await trader.shutdown()
    except Exception as e:
        trader.logger.critical(f"程序崩溃: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
