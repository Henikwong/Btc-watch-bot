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

# ================== 修复WebSocket导入问题 ==================
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
# 清除任何现有的日志处理器
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Railway特定的日志格式化器
class RailwayLogFormatter(logging.Formatter):
    def format(self, record):
        # 简化日志格式以适应云环境
        if IS_RAILWAY or IS_DOCKER:
            return f"{record.levelname}: {record.getMessage()}"
        return super().format(record)

# 配置根日志记录器
log_level = logging.INFO
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s' if not IS_RAILWAY else '%(levelname)s: %(message)s'

handler = logging.StreamHandler(sys.stdout)
formatter = RailwayLogFormatter(log_format)
handler.setFormatter(formatter)

logging.basicConfig(
    level=log_level,
    handlers=[handler],
    format=log_format if not IS_RAILWAY else None
)

# 禁用过于详细的库日志
logging.getLogger("ccxt").setLevel(logging.WARNING)
if WEBSOCKETS_AVAILABLE:
    logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# ================== 配置参数 ==================
# 双开马丁策略参数
MAX_MARTINGALE_LAYERS = int(os.getenv("MAX_MARTINGALE_LAYERS", "3"))
MARTINGALE_MULTIPLIER = float(os.getenv("MARTINGALE_MULTIPLIER", "2.0"))
MARTINGALE_TRIGGER_LOSS = float(os.getenv("MARTINGALE_TRIGGER_LOSS", "0.05"))  # 新增：马丁触发比例
INITIAL_RISK_PERCENT = float(os.getenv("INITIAL_RISK_PERCENT", "0.01"))
MAX_NOTIONAL_PER_SYMBOL = float(os.getenv("MAX_NOTIONAL_PER_SYMBOL", "500"))
DUAL_OPEN_ENABLED = os.getenv("DUAL_OPEN_ENABLED", "true").lower() == "true"
TREND_FILTER_ENABLED = os.getenv("TREND_FILTER_ENABLED", "true").lower() == "true"
PARTIAL_TP_RATIO = float(os.getenv("PARTIAL_TP_RATIO", "0.3"))
HEDGE_MODE = os.getenv("HEDGE_MODE", "false").lower() == "true"

# 交易所API配置
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
TESTNET = os.getenv("TESTNET", "true").lower() == "true"

# 交易参数
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT").split(",")]
TIMEFRAMES = ["1h", "4h"]
UPDATE_INTERVAL = int(os.getenv("UPDATE_INTERVAL", "300"))  # 默认5分钟

# 风险管理参数
MAX_DRAWDOWN_PERCENT = float(os.getenv("MAX_DRAWDOWN_PERCENT", "10.0"))
DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", "5.0"))

# ================== 数据类型定义 ==================
class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"

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
    price: Optional[float] = None
    quantity: Optional[float] = None

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

@dataclass
class PositionInfo:
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    unrealized_pnl: float
    leverage: int
    timestamp: datetime

@dataclass
class MartingaleLayer:
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    layer: int
    timestamp: datetime
    stop_loss: float
    take_profit: float

# ================== 交易所接口实现 ==================
class BinanceExchange:
    """币安交易所实现（优化版）"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.exchange = None
        self.logger = logging.getLogger("BinanceExchange")
        self.rate_limiter = asyncio.Semaphore(10)  # 限制并发请求
        
    async def initialize(self):
        """初始化交易所连接"""
        try:
            exchange_class = getattr(ccxt, 'binance')
            self.exchange = exchange_class({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True,
                },
                'timeout': 30000,
            })
            
            if self.testnet:
                self.exchange.set_sandbox_mode(True)
                self.logger.info("币安测试网模式已启用")
            
            # 异步加载市场信息
            await self._run_in_thread(self.exchange.load_markets)
            self.logger.info("交易所初始化成功")
            
        except Exception as e:
            self.logger.error(f"交易所初始化失败: {e}")
            raise
    
    async def _run_in_thread(self, func, *args, **kwargs):
        """在线程池中运行同步函数"""
        loop = asyncio.get_event_loop()
        async with self.rate_limiter:
            return await loop.run_in_executor(None, func, *args, **kwargs)
    
    async def get_balance(self) -> BalanceInfo:
        """获取余额信息"""
        try:
            balance = await self._run_in_thread(self.exchange.fetch_balance)
            total = float(balance['total'].get('USDT', 0))
            free = float(balance['free'].get('USDT', 0))
            used = float(balance['used'].get('USDT', 0))
            return BalanceInfo(total=total, free=free, used=used)
        except Exception as e:
            self.logger.error(f"获取余额失败: {e}")
            return BalanceInfo(total=0, free=0, used=0)
    
    async def create_order(self, symbol: str, order_type: str, side: OrderSide, 
                          quantity: float, price: Optional[float] = None) -> OrderResult:
        """创建订单"""
        try:
            order_side = side.value
            order = await self._run_in_thread(
                self.exchange.create_order,
                symbol, order_type, order_side, quantity, price
            )
            
            return OrderResult(
                success=True,
                order_id=order['id'],
                symbol=symbol,
                side=side,
                price=float(order.get('price', 0)),
                quantity=float(order['amount'])
            )
        except Exception as e:
            self.logger.error(f"创建订单失败: {e}")
            return OrderResult(success=False, error=str(e))
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[PositionInfo]:
        """获取仓位信息"""
        try:
            positions = await self._run_in_thread(self.exchange.fetch_positions, [symbol] if symbol else None)
            result = []
            
            for pos in positions:
                if symbol and pos['symbol'] != symbol:
                    continue
                
                contracts = float(pos.get('contracts', 0))
                if contracts > 0:
                    position_side = PositionSide.LONG if pos['side'] == 'long' else PositionSide.SHORT
                    result.append(PositionInfo(
                        symbol=pos['symbol'],
                        side=position_side,
                        size=contracts,
                        entry_price=float(pos.get('entryPrice', 0)),
                        unrealized_pnl=float(pos.get('unrealizedPnl', 0)),
                        leverage=int(pos.get('leverage', 1)),
                        timestamp=datetime.now()
                    ))
            
            return result
        except Exception as e:
            self.logger.error(f"获取仓位失败: {e}")
            return []
    
    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """获取K线数据"""
        try:
            ohlcv = await self._run_in_thread(
                self.exchange.fetch_ohlcv, symbol, timeframe, limit=limit
            )
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            self.logger.error(f"获取K线数据失败: {e}")
            return pd.DataFrame()
    
    async def close_position(self, symbol: str, side: PositionSide) -> OrderResult:
        """平仓"""
        try:
            positions = await self.get_positions(symbol)
            position = next((p for p in positions if p.side == side), None)
            
            if not position or position.size == 0:
                return OrderResult(success=False, error="没有找到对应仓位")
            
            close_side = OrderSide.SELL if side == PositionSide.LONG else OrderSide.BUY
            return await self.create_order(symbol, 'market', close_side, position.size)
        except Exception as e:
            self.logger.error(f"平仓失败: {e}")
            return OrderResult(success=False, error=str(e))

# ================== 指标系统 ==================
class IndicatorSystem:
    """完整的指标计算系统"""
    
    def __init__(self):
        self.logger = logging.getLogger("IndicatorSystem")
        self.cache = {}
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """计算ATR指标"""
        try:
            if len(df) < period:
                return 0.0
                
            atr_indicator = ta.volatility.AverageTrueRange(
                high=df['high'], 
                low=df['low'], 
                close=df['close'], 
                window=period
            )
            return float(atr_indicator.average_true_range().iloc[-1])
        except Exception as e:
            self.logger.error(f"计算ATR失败: {e}")
            return 0.0
    
    def calculate_ema(self, df: pd.DataFrame, period: int) -> float:
        """计算EMA指标"""
        try:
            if len(df) < period:
                return float(df['close'].iloc[-1])
                
            ema = ta.trend.EMAIndicator(df['close'], window=period)
            return float(ema.ema_indicator().iloc[-1])
        except Exception as e:
            self.logger.error(f"计算EMA失败: {e}")
            return float(df['close'].iloc[-1]) if not df.empty else 0.0
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """计算RSI指标"""
        try:
            if len(df) < period:
                return 50.0
                
            rsi = ta.momentum.RSIIndicator(df['close'], window=period)
            return float(rsi.rsi().iloc[-1])
        except Exception as e:
            self.logger.error(f"计算RSI失败: {e}")
            return 50.0
    
    def get_trend_direction(self, df: pd.DataFrame) -> str:
        """判断趋势方向"""
        try:
            if len(df) < 50:
                return "neutral"
                
            ema_fast = self.calculate_ema(df, 20)
            ema_slow = self.calculate_ema(df, 50)
            current_price = float(df['close'].iloc[-1])
            
            if current_price > ema_fast > ema_slow:
                return "bullish"
            elif current_price < ema_fast < ema_slow:
                return "bearish"
            else:
                return "neutral"
        except Exception as e:
            self.logger.error(f"判断趋势失败: {e}")
            return "neutral"

# ================== 交易执行器 ==================
class TradeExecutor:
    """优化的交易执行器"""
    
    def __init__(self, exchange: BinanceExchange):
        self.exchange = exchange
        self.logger = logging.getLogger("TradeExecutor")
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.last_balance = 0.0
    
    async def execute_signal(self, signal: TradeSignal) -> OrderResult:
        """执行交易信号"""
        try:
            balance = await self.exchange.get_balance()
            risk_amount = balance.total * INITIAL_RISK_PERCENT
            quantity = risk_amount / signal.price
            
            # 确保最小交易量
            quantity = max(quantity, 0.001)  # 最小交易量
            
            return await self.exchange.create_order(
                symbol=signal.symbol,
                order_type='market',
                side=signal.side,
                quantity=quantity
            )
        except Exception as e:
            self.logger.error(f"执行信号失败: {e}")
            return OrderResult(success=False, error=str(e))
    
    async def set_stop_loss_take_profit(self, symbol: str, entry_price: float, 
                                       atr: float, side: OrderSide) -> Tuple[float, float]:
        """设置止损和止盈价格"""
        if atr == 0:
            atr = entry_price * 0.02  # 默认2%的ATR
            
        if side == OrderSide.BUY:
            stop_loss = entry_price - (atr * 2)
            take_profit = entry_price + (atr * 3)
        else:
            stop_loss = entry_price + (atr * 2)
            take_profit = entry_price - (atr * 3)
        
        return stop_loss, take_profit
    
    async def check_risk_limits(self) -> bool:
        """检查风险限制"""
        try:
            balance = await self.exchange.get_balance()
            
            # 检查每日亏损限制
            if self.daily_pnl < -DAILY_LOSS_LIMIT:
                self.logger.warning(f"达到每日亏损限制: {self.daily_pnl:.2f}%")
                return False
            
            # 检查最大回撤
            if balance.total < self.last_balance:
                drawdown = (self.last_balance - balance.total) / self.last_balance * 100
                self.max_drawdown = max(self.max_drawdown, drawdown)
                
                if self.max_drawdown > MAX_DRAWDOWN_PERCENT:
                    self.logger.warning(f"达到最大回撤限制: {self.max_drawdown:.2f}%")
                    return False
            
            self.last_balance = balance.total
            return True
            
        except Exception as e:
            self.logger.error(f"检查风险限制失败: {e}")
            return True

# ================== 双开马丁策略管理器 ==================
class DualSideManager:
    """管理单个symbol的双向仓位与受控马丁加仓"""
    
    def __init__(self, exchange: BinanceExchange, executor: TradeExecutor, 
                 indicators: IndicatorSystem, symbol: str):
        self.exchange = exchange
        self.executor = executor
        self.indicators = indicators
        self.symbol = symbol
        self.logger = logging.getLogger(f"DualManager.{symbol.replace('/', '')}")
        
        # 马丁加仓层记录
        self.martingale_layers: Dict[PositionSide, List[MartingaleLayer]] = {
            PositionSide.LONG: [],
            PositionSide.SHORT: []
        }
        
        # 状态跟踪
        self.last_check_time = datetime.now()
        self.is_trend_filter_active = TREND_FILTER_ENABLED
        self.initial_opened = False
    
    async def open_initial_pair(self) -> bool:
        """开初始双向仓位"""
        if not DUAL_OPEN_ENABLED:
            self.logger.info("双开功能已禁用")
            return False
            
        if self.initial_opened:
            self.logger.info("初始仓位已开立")
            return True
            
        try:
            # 检查风险限制
            if not await self.executor.check_risk_limits():
                self.logger.warning("风险限制检查未通过，暂停开仓")
                return False
            
            df = await self.exchange.get_ohlcv(self.symbol, "1h", 100)
            if df.empty:
                self.logger.error("无法获取K线数据")
                return False
            
            current_price = float(df['close'].iloc[-1])
            atr = self.indicators.calculate_atr(df)
            
            if atr == 0:
                atr = current_price * 0.02
            
            balance = await self.exchange.get_balance()
            risk_amount = balance.total * INITIAL_RISK_PERCENT
            quantity = risk_amount / current_price
            
            # 风控检查
            total_notional = quantity * current_price * 2
            if total_notional > MAX_NOTIONAL_PER_SYMBOL:
                quantity = MAX_NOTIONAL_PER_SYMBOL / (current_price * 2)
                self.logger.warning(f"调整仓位大小以符合风控限制: {quantity:.6f}")
            
            # 创建交易信号
            buy_signal = TradeSignal(
                symbol=self.symbol,
                side=OrderSide.BUY,
                price=current_price,
                atr=atr,
                quantity=quantity,
                timestamp=datetime.now()
            )
            
            sell_signal = TradeSignal(
                symbol=self.symbol,
                side=OrderSide.SELL,
                price=current_price,
                atr=atr,
                quantity=quantity,
                timestamp=datetime.now()
            )
            
            # 执行订单
            buy_result = await self.executor.execute_signal(buy_signal)
            await asyncio.sleep(1)  # 避免频繁请求
            sell_result = await self.executor.execute_signal(sell_signal)
            
            if buy_result.success and sell_result.success:
                self.logger.info(f"✅ 成功开立双向仓位")
                
                # 记录初始层
                buy_stop_loss, buy_take_profit = await self.executor.set_stop_loss_take_profit(
                    self.symbol, current_price, atr, OrderSide.BUY
                )
                
                sell_stop_loss, sell_take_profit = await self.executor.set_stop_loss_take_profit(
                    self.symbol, current_price, atr, OrderSide.SELL
                )
                
                self.martingale_layers[PositionSide.LONG].append(MartingaleLayer(
                    symbol=self.symbol,
                    side=PositionSide.LONG,
                    size=quantity,
                    entry_price=current_price,
                    layer=0,
                    timestamp=datetime.now(),
                    stop_loss=buy_stop_loss,
                    take_profit=buy_take_profit
                ))
                
                self.martingale_layers[PositionSide.SHORT].append(MartingaleLayer(
                    symbol=self.symbol,
                    side=PositionSide.SHORT,
                    size=quantity,
                    entry_price=current_price,
                    layer=0,
                    timestamp=datetime.now(),
                    stop_loss=sell_stop_loss,
                    take_profit=sell_take_profit
                ))
                
                self.initial_opened = True
                return True
            else:
                errors = []
                if not buy_result.success:
                    errors.append(f"买: {buy_result.error}")
                if not sell_result.success:
                    errors.append(f"卖: {sell_result.error}")
                self.logger.error(f"开立双向仓位失败: {', '.join(errors)}")
                
                # 清理已成功的订单
                if buy_result.success:
                    await self.exchange.close_position(self.symbol, PositionSide.LONG)
                if sell_result.success:
                    await self.exchange.close_position(self.symbol, PositionSide.SHORT)
                    
                return False
                
        except Exception as e:
            self.logger.error(f"开立初始双向仓位失败: {e}")
            return False
    
    async def monitor_and_martingale(self):
        """监控仓位并执行马丁加仓逻辑"""
        try:
            current_time = datetime.now()
            if (current_time - self.last_check_time).total_seconds() < UPDATE_INTERVAL:
                return
            
            self.last_check_time = current_time
            
            # 检查风险限制
            if not await self.executor.check_risk_limits():
                self.logger.warning("风险限制检查未通过，暂停操作")
                return
            
            # 获取当前市场数据
            df = await self.exchange.get_ohlcv(self.symbol, "1h", 100)
            if df.empty:
                return
            
            current_price = float(df['close'].iloc[-1])
            atr = self.indicators.calculate_atr(df)
            
            # 检查趋势过滤
            if self.is_trend_filter_active:
                trend = self.indicators.get_trend_direction(df)
                if trend == "bullish" and len(self.martingale_layers[PositionSide.SHORT]) > 0:
                    self.logger.info("趋势看涨，暂停空头加仓")
                elif trend == "bearish" and len(self.martingale_layers[PositionSide.LONG]) > 0:
                    self.logger.info("趋势看跌，暂停多头加仓")
            
            # 检查是否需要加仓
            await self._check_martingale_opportunity(PositionSide.LONG, current_price, atr)
            await self._check_martingale_opportunity(PositionSide.SHORT, current_price, atr)
            
            # 检查止盈止损
            await self._check_take_profit_stop_loss(current_price)
            
        except Exception as e:
            self.logger.error(f"监控马丁加仓失败: {e}")
    
    async def _check_martingale_opportunity(self, side: PositionSide, current_price: float, atr: float):
        """检查马丁加仓机会"""
        layers = self.martingale_layers[side]
        if not layers:
            return
        
        # 获取当前仓位信息
        positions = await self.exchange.get_positions(self.symbol)
        position = next((p for p in positions if p.side == side), None)
        
        if not position or position.size == 0:
            return
        
        # 计算当前亏损比例
        unrealized_pnl_percent = abs(position.unrealized_pnl) / (position.entry_price * position.size)
        
        # 如果亏损达到设定比例且还有加仓层数可用
        if unrealized_pnl_percent >= MARTINGALE_TRIGGER_LOSS and len(layers) < MAX_MARTINGALE_LAYERS + 1:
            self.logger.info(f"📈 检测到{side.value}浮亏 {unrealized_pnl_percent:.2%}，达到触发条件 {MARTINGALE_TRIGGER_LOSS:.2%}，执行马丁加仓")
            
            # 计算加仓数量
            last_layer = layers[-1]
            new_size = last_layer.size * MARTINGALE_MULTIPLIER
            
            # 检查总仓位限制
            total_notional = sum(layer.size * layer.entry_price for layer in layers) + (new_size * current_price)
            if total_notional > MAX_NOTIONAL_PER_SYMBOL:
                self.logger.warning(f"达到最大仓位限制 {MAX_NOTIONAL_PER_SYMBOL} USDT，停止加仓")
                return
            
            # 执行加仓
            order_side = OrderSide.BUY if side == PositionSide.LONG else OrderSide.SELL
            order_result = await self.exchange.create_order(
                self.symbol, 'market', order_side, new_size
            )
            
            if order_result.success:
                # 记录新层
                stop_loss, take_profit = await self.executor.set_stop_loss_take_profit(
                    self.symbol, current_price, atr, order_side
                )
                
                new_layer = MartingaleLayer(
                    symbol=self.symbol,
                    side=side,
                    size=new_size,
                    entry_price=current_price,
                    layer=len(layers),
                    timestamp=datetime.now(),
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                layers.append(new_layer)
                self.logger.info(f"✅ 马丁加仓成功: 第{len(layers)}层，数量={new_size:.6f}")
            else:
                self.logger.error(f"❌ 马丁加仓失败: {order_result.error}")
    
    async def _check_take_profit_stop_loss(self, current_price: float):
        """检查止盈止损条件"""
        for side, layers in self.martingale_layers.items():
            if not layers:
                continue

            # 计算加权平均开仓价
            total_size = sum(layer.size for layer in layers)
            if total_size == 0:
                continue

            # 取最后一层的止损和止盈（动态调整）
            last_layer = layers[-1]
            stop_loss = last_layer.stop_loss
            take_profit = last_layer.take_profit

            # 多头检查
            if side == PositionSide.LONG:
                if current_price <= stop_loss:
                    self.logger.warning(f"⚠️ {self.symbol} 多头触发止损，平仓")
                    await self.exchange.close_position(self.symbol, PositionSide.LONG)
                    self.martingale_layers[side].clear()
                elif current_price >= take_profit:
                    self.logger.info(f"✅ {self.symbol} 多头止盈，平仓")
                    await self.exchange.close_position(self.symbol, PositionSide.LONG)
                    self.martingale_layers[side].clear()

            # 空头检查
            elif side == PositionSide.SHORT:
                if current_price >= stop_loss:
                    self.logger.warning(f"⚠️ {self.symbol} 空头触发止损，平仓")
                    await self.exchange.close_position(self.symbol, PositionSide.SHORT)
                    self.martingale_layers[side].clear()
                elif current_price <= take_profit:
                    self.logger.info(f"✅ {self.symbol} 空头止盈，平仓")
                    await self.exchange.close_position(self.symbol, PositionSide.SHORT)
                    self.martingale_layers[side].clear()
    
    async def _close_all_layers(self, side: PositionSide):
        """平掉所有指定方向的仓位"""
        try:
            await self.exchange.close_position(self.symbol, side)
            self.martingale_layers[side] = []
            self.logger.info(f"已平仓所有{side.value}仓位")
        except Exception as e:
            self.logger.error(f"平仓失败: {e}")

# ================== 主交易机器人 ==================
class EnhancedProductionTrader:
    """增强的生产环境交易机器人"""
    
    def __init__(self):
        self.exchange = None
        self.executor = None
        self.indicators = IndicatorSystem()
        self.dual_managers: Dict[str, DualSideManager] = {}
        self.logger = logging.getLogger("EnhancedProductionTrader")
        self.is_running = False
    
    async def initialize(self):
        """初始化交易机器人"""
        try:
            # 检查API密钥
            if not BINANCE_API_KEY or not BINANCE_API_SECRET:
                self.logger.error("请设置 BINANCE_API_KEY 和 BINANCE_API_SECRET 环境变量")
                return False
            
            # 初始化交易所
            self.exchange = BinanceExchange(BINANCE_API_KEY, BINANCE_API_SECRET, TESTNET)
            await self.exchange.initialize()
            
            # 初始化交易执行器
            self.executor = TradeExecutor(self.exchange)
            
            # 初始化双开管理器
            for symbol in SYMBOLS:
                self.dual_managers[symbol] = DualSideManager(
                    self.exchange, self.executor, self.indicators, symbol
                )
            
            # 显示当前配置
            self.logger.info(f"📋 策略配置:")
            self.logger.info(f"   - 马丁触发比例: {MARTINGALE_TRIGGER_LOSS:.2%}")
            self.logger.info(f"   - 马丁乘数: {MARTINGALE_MULTIPLIER}")
            self.logger.info(f"   - 最大马丁层数: {MAX_MARTINGALE_LAYERS}")
            self.logger.info(f"   - 初始风险比例: {INITIAL_RISK_PERCENT:.2%}")
            self.logger.info(f"   - 单币种最大仓位: {MAX_NOTIONAL_PER_SYMBOL} USDT")
            
            self.logger.info("✅ 交易机器人初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"交易机器人初始化失败: {e}")
            return False
    
    async def run(self):
        """运行交易机器人"""
        if not await self.initialize():
            self.logger.error("初始化失败，程序退出")
            return
        
        self.is_running = True
        self.logger.info("🚀 启动增强版交易机器人")
        
        try:
            # 初始开立双向仓位
            if DUAL_OPEN_ENABLED:
                for symbol, manager in self.dual_managers.items():
                    success = await manager.open_initial_pair()
                    if success:
                        self.logger.info(f"✅ 成功为 {symbol} 开立初始双向仓位")
                    else:
                        self.logger.error(f"❌ 为 {symbol} 开立初始双向仓位失败")
            
            # 主循环
            while self.is_running:
                try:
                    tasks = []
                    for symbol, manager in self.dual_managers.items():
                        tasks.append(manager.monitor_and_martingale())
                    
                    # 并行执行所有监控任务
                    await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # 显示状态信息
                    await self.display_status()
                    
                    # 等待下一次检查
                    await asyncio.sleep(UPDATE_INTERVAL)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"主循环错误: {e}")
                    await asyncio.sleep(30)  # 出错后等待30秒再继续
                    
        except asyncio.CancelledError:
            self.logger.info("交易机器人已停止")
        except Exception as e:
            self.logger.error(f"交易循环异常: {e}")
        finally:
            self.is_running = False
    
    async def display_status(self):
        """显示当前状态信息"""
        try:
            balance = await self.exchange.get_balance()
            self.logger.info(f"💰 账户余额: 总={balance.total:.2f} USDT, 可用={balance.free:.2f} USDT")
            
            for symbol in SYMBOLS:
                positions = await self.exchange.get_positions(symbol)
                for pos in positions:
                    if pos.size > 0:
                        pnl_percent = (pos.unrealized_pnl / (pos.entry_price * pos.size)) * 100
                        self.logger.info(
                            f"📊 {symbol} {pos.side.value}: 大小={pos.size:.4f}, "
                            f"入场价={pos.entry_price:.2f}, 未实现盈亏={pnl_percent:.2f}%"
                        )
                        
                # 显示马丁层信息
                manager = self.dual_managers[symbol]
                for side, layers in manager.martingale_layers.items():
                    if layers:
                        total_size = sum(layer.size for layer in layers)
                        avg_price = sum(layer.size * layer.entry_price for layer in layers) / total_size
                        self.logger.info(
                            f"   {side.value}马丁层: {len(layers)}层, 总大小={total_size:.4f}, "
                            f"均价={avg_price:.2f}"
                        )
        except Exception as e:
            self.logger.error(f"显示状态失败: {e}")
    
    def stop(self):
        """安全停止"""
        self.is_running = False
        self.logger.info("⏹️ 交易机器人正在停止...")

# ================== 程序入口 ==================
async def main():
    """主函数"""
    trader = EnhancedProductionTrader()
    
    # 设置信号处理
    def signal_handler(sig, frame):
        logging.info("收到停止信号，正在关闭...")
        trader.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await trader.run()
    except KeyboardInterrupt:
        trader.stop()
    except Exception as e:
        logging.critical(f"未处理的异常: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # 检查必要的环境变量
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        logging.error("请设置 BINANCE_API_KEY 和 BINANCE_API_SECRET 环境变量")
        sys.exit(1)
    
    # 启动机器人
    asyncio.run(main())
