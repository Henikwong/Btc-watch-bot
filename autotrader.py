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

# ================== 配置参数 ==================
# 双开马丁策略参数
MAX_MARTINGALE_LAYERS = int(os.getenv("MAX_MARTINGALE_LAYERS", "3"))
MARTINGALE_MULTIPLIER = float(os.getenv("MARTINGALE_MULTIPLIER", "2.0"))
MARTINGALE_TRIGGER_LOSS = float(os.getenv("MARTINGALE_TRIGGER_LOSS", "0.05"))
INITIAL_RISK_PERCENT = float(os.getenv("INITIAL_RISK_PERCENT", "0.01"))
MAX_NOTIONAL_PER_SYMBOL = float(os.getenv("MAX_NOTIONAL_PER_SYMBOL", "500"))
DUAL_OPEN_ENABLED = os.getenv("DUAL_OPEN_ENABLED", "true").lower() == "true"
TREND_FILTER_ENABLED = os.getenv("TREND_FILTER_ENABLED", "true").lower() == "true"
PARTIAL_TP_RATIO = float(os.getenv("PARTIAL_TP_RATIO", "0.3"))
HEDGE_MODE = os.getenv("HEDGE_MODE", "false").lower() == "true"

# 交易所API配置 - 主网合约模式
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
CONTRACT_TYPE = os.getenv("CONTRACT_TYPE", "future")  # future: U本位, delivery: 币本位

# 交易所初始化重试配置
EXCHANGE_INIT_RETRIES = int(os.getenv("EXCHANGE_INIT_RETRIES", "5"))
EXCHANGE_INIT_RETRY_DELAY = int(os.getenv("EXCHANGE_INIT_RETRY_DELAY", "3"))

# 交易参数
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT").split(",")]
TIMEFRAMES = ["1h", "4h"]
UPDATE_INTERVAL = int(os.getenv("UPDATE_INTERVAL", "300"))

# 风险管理参数 - 主网需要更严格的风控
MAX_DRAWDOWN_PERCENT = float(os.getenv("MAX_DRAWDOWN_PERCENT", "5.0"))
DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", "2.0"))

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

@dataclass
class DualMartingaleStatus:
    symbol: str
    long_layers: int
    short_layers: int
    long_exposure: float
    short_exposure: float
    long_avg_price: float
    short_avg_price: float
    net_exposure: float

# ================== 交易所接口实现（主网合约模式） ==================
class BinanceExchange:
    """币安交易所实现（主网合约模式）"""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.exchange = None
        self.initialized = False
        self.logger = logging.getLogger(__name__)
    
    def initialize_with_retry(self, max_retries: int = EXCHANGE_INIT_RETRIES, 
                             retry_delay: int = EXCHANGE_INIT_RETRY_DELAY) -> bool:
        """带重试机制的交易所初始化 - 主网合约模式"""
        for attempt in range(1, max_retries + 1):
            try:
                self.logger.info(f"尝试初始化币安合约交易所 (第 {attempt} 次尝试，最多 {max_retries} 次)")
                
                # 创建交易所实例 - 直接主网合约
                exchange_config = {
                    'apiKey': self.api_key,
                    'secret': self.api_secret,
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': CONTRACT_TYPE,  # future 或 delivery
                        'adjustForTimeDifference': True,
                    }
                }
                
                exchange = ccxt.binance(exchange_config)
                
                # 测试连接和权限
                exchange.load_markets()
                
                # 检查合约账户余额
                try:
                    balance = exchange.fetch_balance()
                    usdt_balance = balance.get('USDT', {})
                    total = float(usdt_balance.get('total', 0))
                    free = float(usdt_balance.get('free', 0))
                    
                    self.logger.info(f"✅ 成功连接到币安{CONTRACT_TYPE}合约主网")
                    self.logger.info(f"📊 加载了 {len(exchange.markets)} 个交易对")
                    self.logger.info(f"💰 合约账户余额: 总额={total:.2f} USDT, 可用={free:.2f} USDT")
                    
                except Exception as e:
                    self.logger.error(f"获取合约余额失败，请检查API权限: {str(e)}")
                    return False
                
                self.exchange = exchange
                self.initialized = True
                
                # 主网模式下的重要警告
                self.logger.warning("⚠️ ⚠️ ⚠️ 重要警告 ⚠️ ⚠️ ⚠️")
                self.logger.warning("当前运行在币安合约主网模式")
                self.logger.warning("所有交易都是真实交易，请确保：")
                self.logger.warning("1. API密钥已开启期货交易权限")
                self.logger.warning("2. 了解合约交易的风险")
                self.logger.warning("3. 资金安全由您自己负责")
                self.logger.warning("4. 建议先用小资金测试")
                
                return True
                
            except ccxt.AuthenticationError as e:
                self.logger.error(f"API认证失败 (尝试 {attempt}/{max_retries}): {str(e)}")
                self.logger.error("请检查: 1. API密钥是否正确 2. 是否开启期货权限 3. IP白名单设置")
                return False
                
            except ccxt.PermissionDenied as e:
                self.logger.error(f"权限不足 (尝试 {attempt}/{max_retries}): {str(e)}")
                self.logger.error("请到币安官网开启期货交易权限")
                return False
                
            except ccxt.NetworkError as e:
                self.logger.warning(f"网络错误 (尝试 {attempt}/{max_retries}): {str(e)}")
                if attempt < max_retries:
                    self.logger.info(f"{retry_delay}秒后重试...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"交易所初始化失败，已达到最大重试次数: {str(e)}")
                    return False
                    
            except ccxt.ExchangeError as e:
                error_msg = str(e)
                self.logger.error(f"交易所错误 (尝试 {attempt}/{max_retries}): {error_msg}")
                
                # 特殊错误处理
                if "-2015" in error_msg:
                    self.logger.error("错误代码 -2015: API密钥没有期货交易权限")
                    self.logger.error("请到币安官网 → 管理API → 启用期货交易")
                elif "-1021" in error_msg:
                    self.logger.error("错误代码 -1021: 时间戳错误，请检查系统时间")
                elif "-1003" in error_msg:
                    self.logger.error("错误代码 -1003: 请求过于频繁，请降低频率")
                    
                return False
                
            except Exception as e:
                self.logger.error(f"未知错误 (尝试 {attempt}/{max_retries}): {str(e)}")
                if attempt < max_retries:
                    self.logger.info(f"{retry_delay}秒后重试...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"交易所初始化失败，已达到最大重试次数: {str(e)}")
                    return False
        
        return False

    def is_initialized(self) -> bool:
        return self.initialized and self.exchange is not None

    def get_balance(self) -> BalanceInfo:
        if not self.is_initialized():
            raise Exception("交易所未初始化")
        try:
            balance = self.exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {})
            total = float(usdt_balance.get('total', 0))
            free = float(usdt_balance.get('free', 0))
            used = float(usdt_balance.get('used', 0))
            
            self.logger.info(f"账户余额 - 总额: {total:.2f} USDT, 可用: {free:.2f} USDT, 已用: {used:.2f} USDT")
            return BalanceInfo(total=total, free=free, used=used)
            
        except Exception as e:
            self.logger.error(f"获取余额失败: {str(e)}")
            raise

    def create_order(self, symbol: str, side: str, quantity: float, price: Optional[float] = None) -> OrderResult:
        if not self.is_initialized():
            return OrderResult(success=False, error="交易所未初始化")
        
        try:
            order_type = 'limit' if price else 'market'
            
            # 检查最小交易量
            market = self.exchange.market(symbol)
            min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
            if quantity < min_amount:
                return OrderResult(success=False, error=f"交易量低于最小值: {min_amount}")
            
            # 主网合约订单 - 记录详细信息
            order_value = quantity * (price if price else self.get_current_price(symbol))
            self.logger.warning(f"🚀 准备执行主网合约订单: {symbol} {side} {quantity:.6f} 价值: {order_value:.2f} USDT")
            
            order = self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=quantity,
                price=price
            )
            
            self.logger.warning(f"✅ 主网合约订单执行成功: {symbol} {side} {quantity:.6f} 订单ID: {order['id']}")
            
            return OrderResult(
                success=True,
                order_id=order['id'],
                symbol=symbol,
                side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
                price=float(order['price']),
                quantity=float(order['amount'])
            )
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"创建订单失败: {error_msg}")
            
            # 主网模式下的特殊错误处理
            if "insufficient balance" in error_msg.lower():
                self.logger.critical("💥 余额不足！请立即充值")
            elif "margin" in error_msg.lower():
                self.logger.critical("💥 保证金不足！请调整仓位大小")
            elif "position" in error_msg.lower():
                self.logger.critical("💥 仓位限制！请检查现有仓位")
            
            return OrderResult(success=False, error=error_msg)

    def get_current_price(self, symbol: str) -> float:
        """获取当前价格"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker['last'])
        except:
            return 0.0

    def get_positions(self) -> List[PositionInfo]:
        if not self.is_initialized():
            return []
        
        try:
            positions = self.exchange.fetch_positions()
            result = []
            for pos in positions:
                contracts = float(pos.get('contracts', 0))
                if contracts > 0:
                    result.append(PositionInfo(
                        symbol=pos['symbol'],
                        side=PositionSide.LONG if pos['side'] == 'long' else PositionSide.SHORT,
                        size=contracts,
                        entry_price=float(pos.get('entryPrice', 0)),
                        unrealized_pnl=float(pos.get('unrealizedPnl', 0)),
                        leverage=int(pos.get('leverage', 1)),
                        timestamp=datetime.now()
                    ))
            
            # 记录持仓信息
            if result:
                for pos in result:
                    pnl_color = "🟢" if pos.unrealized_pnl >= 0 else "🔴"
                    self.logger.info(f"{pnl_color} 持仓: {pos.symbol} {pos.side.value} {pos.size:.6f} 盈亏: {pos.unrealized_pnl:.2f} USDT")
            
            return result
        except Exception as e:
            self.logger.error(f"获取仓位失败: {str(e)}")
            return []

    def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Optional[pd.DataFrame]:
        """获取K线数据"""
        if not self.is_initialized():
            return None
            
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            return df
        except Exception as e:
            self.logger.error(f"获取K线数据失败 {symbol}: {str(e)}")
            return None

# ================== 指标系统 ==================
class IndicatorSystem:
    """完整的指标计算系统"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        if df is None or df.empty:
            return df
            
        df = df.copy()
        
        # EMA指标
        df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        
        # MACD指标
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # RSI指标
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # ATR指标
        df['atr'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], window=14
        ).average_true_range()
        
        # 布林带
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # 成交量指标
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        return df.dropna()

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[TradeSignal]:
        """生成交易信号"""
        if df is None or df.empty or len(df) < 50:
            return None
            
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 检查是否有足够的指标数据
        if any(pd.isna(current[col]) for col in ['ema_12', 'ema_26', 'macd', 'rsi', 'atr']):
            return None
        
        # 趋势判断
        trend_bullish = current['ema_12'] > current['ema_26']
        trend_bearish = current['ema_12'] < current['ema_26']
        
        # MACD信号
        macd_bullish = current['macd'] > current['macd_signal'] and prev['macd'] <= prev['macd_signal']
        macd_bearish = current['macd'] < current['macd_signal'] and prev['macd'] >= prev['macd_signal']
        
        # RSI信号
        rsi_overbought = current['rsi'] > 70
        rsi_oversold = current['rsi'] < 30
        
        # 生成信号
        price = float(current['close'])
        atr = float(current['atr'])
        
        if trend_bullish and macd_bullish and not rsi_overbought:
            return TradeSignal(
                symbol=symbol,
                side=OrderSide.BUY,
                price=price,
                atr=atr,
                quantity=0,
                timestamp=datetime.now(),
                confidence=0.8
            )
        elif trend_bearish and macd_bearish and not rsi_oversold:
            return TradeSignal(
                symbol=symbol,
                side=OrderSide.SELL,
                price=price,
                atr=atr,
                quantity=0,
                timestamp=datetime.now(),
                confidence=0.8
            )
        
        return None

# ================== 交易执行器 ==================
class TradeExecutor:
    """优化的交易执行器"""
    
    def __init__(self, exchange: BinanceExchange):
        self.exchange = exchange
        self.logger = logging.getLogger(__name__)
    
    def calculate_position_size(self, balance: float, price: float, atr: float, risk_percent: float = INITIAL_RISK_PERCENT) -> float:
        """计算仓位大小"""
        if price <= 0 or atr <= 0:
            return 0
            
        # 风险金额
        risk_amount = balance * risk_percent
        
        # 每单位风险（基于ATR）
        risk_per_unit = atr
        
        # 计算仓位大小
        position_size = risk_amount / risk_per_unit
        
        # 确保最小交易量
        min_size = 0.001
        return max(position_size, min_size)
    
    async def execute_order(self, signal: TradeSignal, balance: float) -> OrderResult:
        """执行交易订单"""
        if not self.exchange.is_initialized():
            return OrderResult(success=False, error="交易所未初始化")
        
        try:
            # 计算仓位大小
            position_size = self.calculate_position_size(balance, signal.price, signal.atr)
            if position_size <= 0:
                return OrderResult(success=False, error="仓位计算错误")
            
            signal.quantity = position_size
            
            # 主网模式下的额外确认
            order_value = position_size * signal.price
            self.logger.warning(f"⚠️ 准备执行主网合约订单: {signal.symbol} {signal.side.value} {position_size:.6f} 价值: {order_value:.2f} USDT")
            
            # 创建订单
            result = self.exchange.create_order(
                symbol=signal.symbol,
                side=signal.side.value,
                quantity=position_size,
                price=None  # 市价单
            )
            
            if result.success:
                self.logger.info(f"订单执行成功: {signal.symbol} {signal.side.value} {position_size:.6f}")
            else:
                self.logger.error(f"订单执行失败: {result.error}")
                
            return result
            
        except Exception as e:
            self.logger.error(f"执行订单失败: {str(e)}")
            return OrderResult(success=False, error=str(e))

# ================== 双开马丁策略管理器 ==================
class DualMartingaleManager:
    """管理单个symbol的双向仓位与受控马丁加仓"""
    
    def __init__(self, symbol: str, exchange: BinanceExchange, executor: TradeExecutor):
        self.symbol = symbol
        self.exchange = exchange
        self.executor = executor
        self.logger = logging.getLogger(__name__)
        
        # 马丁层管理
        self.long_layers: List[MartingaleLayer] = []
        self.short_layers: List[MartingaleLayer] = []
        
        # 状态跟踪
        self.last_check_time = datetime.now()
        self.consecutive_losses = 0
    
    def get_status(self) -> DualMartingaleStatus:
        """获取当前状态"""
        long_exposure = sum(layer.size for layer in self.long_layers)
        short_exposure = sum(layer.size for layer in self.short_layers)
        
        long_avg = (sum(layer.entry_price * layer.size for layer in self.long_layers) / long_exposure 
                   if long_exposure > 0 else 0)
        short_avg = (sum(layer.entry_price * layer.size for layer in self.short_layers) / short_exposure 
                    if short_exposure > 0 else 0)
        
        return DualMartingaleStatus(
            symbol=self.symbol,
            long_layers=len(self.long_layers),
            short_layers=len(self.short_layers),
            long_exposure=long_exposure,
            short_exposure=short_exposure,
            long_avg_price=long_avg,
            short_avg_price=short_avg,
            net_exposure=long_exposure - short_exposure
        )
    
    def should_add_long_layer(self, current_price: float) -> bool:
        """检查是否应该加多仓"""
        if len(self.long_layers) >= MAX_MARTINGALE_LAYERS:
            return False
            
        if not self.long_layers:
            return True
            
        # 计算平均入场价和当前亏损
        avg_entry = self._calculate_avg_entry_price(self.long_layers)
        if avg_entry <= 0:
            return True
            
        current_loss = (avg_entry - current_price) / avg_entry
        return current_loss >= MARTINGALE_TRIGGER_LOSS
    
    def should_add_short_layer(self, current_price: float) -> bool:
        """检查是否应该加空仓"""
        if len(self.short_layers) >= MAX_MARTINGALE_LAYERS:
            return False
            
        if not self.short_layers:
            return True
            
        # 计算平均入场价和当前亏损
        avg_entry = self._calculate_avg_entry_price(self.short_layers)
        if avg_entry <= 0:
            return True
            
        current_loss = (current_price - avg_entry) / avg_entry
        return current_loss >= MARTINGALE_TRIGGER_LOSS
    
    def _calculate_avg_entry_price(self, layers: List[MartingaleLayer]) -> float:
        """计算平均入场价"""
        if not layers:
            return 0
            
        total_size = sum(layer.size for layer in layers)
        total_value = sum(layer.entry_price * layer.size for layer in layers)
        return total_value / total_size if total_size > 0 else 0
    
    def calculate_layer_size(self, side: PositionSide, balance: float, current_price: float) -> float:
        """计算马丁加仓的仓位大小"""
        current_layers = self.long_layers if side == PositionSide.LONG else self.short_layers
        layer_number = len(current_layers) + 1
        
        # 基础风险计算
        base_risk = balance * INITIAL_RISK_PERCENT
        layer_multiplier = MARTINGALE_MULTIPLIER ** (layer_number - 1)
        risk_amount = base_risk * layer_multiplier
        
        # 使用ATR计算风险
        risk_per_unit = current_price * 0.02
        
        position_size = risk_amount / risk_per_unit
        
        # 检查最大名义价值限制
        notional_value = position_size * current_price
        if notional_value > MAX_NOTIONAL_PER_SYMBOL:
            position_size = MAX_NOTIONAL_PER_SYMBOL / current_price
            
        return position_size
    
    async def add_martingale_layer(self, side: PositionSide, current_price: float, balance: float) -> Optional[MartingaleLayer]:
        """添加马丁加仓层"""
        should_add = (self.should_add_long_layer(current_price) if side == PositionSide.LONG 
                     else self.should_add_short_layer(current_price))
        
        if not should_add:
            return None
            
        layer_size = self.calculate_layer_size(side, balance, current_price)
        if layer_size <= 0:
            return None
            
        layer_number = len(self.long_layers if side == PositionSide.LONG else self.short_layers) + 1
        
        # 计算止损和止盈
        if side == PositionSide.LONG:
            stop_loss = current_price * (1 - 0.03)
            take_profit = current_price * (1 + 0.06)
        else:
            stop_loss = current_price * (1 + 0.03)
            take_profit = current_price * (1 - 0.06)
            
        layer = MartingaleLayer(
            symbol=self.symbol,
            side=side,
            size=layer_size,
            entry_price=current_price,
            layer=layer_number,
            timestamp=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        if side == PositionSide.LONG:
            self.long_layers.append(layer)
        else:
            self.short_layers.append(layer)
            
        self.logger.info(f"{self.symbol} {side.value} 第{layer_number}层马丁加仓，大小: {layer_size:.6f}")
        
        return layer
    
    async def check_take_profit(self, current_price: float) -> bool:
        """检查止盈条件"""
        status = self.get_status()
        
        # 检查多仓止盈
        if status.long_exposure > 0 and current_price >= status.long_avg_price * 1.03:
            self.logger.info(f"{self.symbol} 多仓达到止盈条件")
            return True
            
        # 检查空仓止盈
        if status.short_exposure > 0 and current_price <= status.short_avg_price * 0.97:
            self.logger.info(f"{self.symbol} 空仓达到止盈条件")
            return True
            
        return False
    
    async def close_all_positions(self):
        """平掉所有仓位"""
        self.logger.info(f"开始平仓 {self.symbol}")
        
        # 平多仓
        if self.long_layers:
            total_long = sum(layer.size for layer in self.long_layers)
            if total_long > 0:
                result = self.exchange.create_order(
                    symbol=self.symbol,
                    side='sell',
                    quantity=total_long,
                    price=None
                )
                if result.success:
                    self.long_layers.clear()
                    self.logger.info(f"平多仓成功: {total_long:.6f}")
        
        # 平空仓
        if self.short_layers:
            total_short = sum(layer.size for layer in self.short_layers)
            if total_short > 0:
                result = self.exchange.create_order(
                    symbol=self.symbol,
                    side='buy',
                    quantity=total_short,
                    price=None
                )
                if result.success:
                    self.short_layers.clear()
                    self.logger.info(f"平空仓成功: {total_short:.6f}")

# ================== 主交易机器人（主网合约模式） ==================
class EnhancedProductionTrader:
    """增强的生产环境交易机器人（主网合约模式）"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.exchange = None
        self.indicators = IndicatorSystem()
        self.executor = None
        self.martingale_managers: Dict[str, DualMartingaleManager] = {}
        self.initialized = False
        
        # 设置信号处理器
        signal.signal(signal.SIGINT, self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)
    
    def handle_exit(self, signum, frame):
        """处理退出信号"""
        self.logger.info(f"收到信号 {signum}，正在优雅退出...")
        # 尝试平掉所有仓位
        if self.initialized:
            asyncio.run(self.close_all_positions())
        sys.exit(0)
    
    async def close_all_positions(self):
        """平掉所有仓位"""
        for symbol, manager in self.martingale_managers.items():
            await manager.close_all_positions()
    
    def initialize_exchange(self) -> bool:
        """初始化交易所连接"""
        try:
            self.logger.info("开始初始化币安合约交易所连接...")
            
            if not BINANCE_API_KEY or not BINANCE_API_SECRET:
                self.logger.error("请设置 BINANCE_API_KEY 和 BINANCE_API_SECRET 环境变量")
                return False
            
            # 创建交易所实例 - 主网合约模式
            exchange = BinanceExchange(
                api_key=BINANCE_API_KEY,
                api_secret=BINANCE_API_SECRET
            )
            
            # 使用重试机制初始化
            if exchange.initialize_with_retry():
                self.exchange = exchange
                self.executor = TradeExecutor(exchange)
                
                # 为每个交易对创建马丁管理器
                for symbol in SYMBOLS:
                    self.martingale_managers[symbol] = DualMartingaleManager(
                        symbol, exchange, self.executor
                    )
                
                self.initialized = True
                self.logger.info("币安合约交易所初始化成功")
                return True
            else:
                self.logger.error("交易所初始化失败，请检查网络连接或API密钥")
                return False
                
        except Exception as e:
            self.logger.error(f"交易所初始化过程中发生未知错误: {str(e)}")
            return False
    
    def initialize(self) -> bool:
        """完整的初始化过程"""
        try:
            self.logger.info("开始初始化交易机器人...")
            
            # 第一步：初始化交易所
            if not self.initialize_exchange():
                return False
            
            # 第二步：检查余额
            try:
                balance = self.exchange.get_balance()
                self.logger.info(f"合约账户余额: 总额={balance.total:.2f} USDT, 可用={balance.free:.2f} USDT")
                
                # 资金警告
                if balance.free < 50:
                    self.logger.warning("⚠️ 可用资金较少，建议充值")
                    
            except Exception as e:
                self.logger.warning(f"获取余额失败: {str(e)}")
            
            # 第三步：检查支持的交易对
            self.logger.info(f"配置的交易对: {SYMBOLS}")
            
            self.logger.info("交易机器人初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"交易机器人初始化失败: {str(e)}")
            return False
    
    async def run_trading_cycle(self):
        """运行交易周期"""
        if not self.initialized:
            self.logger.error("交易机器人未初始化，无法运行")
            return
        
        try:
            # 获取当前余额
            balance_info = self.exchange.get_balance()
            free_balance = balance_info.free
            
            for symbol in SYMBOLS:
                await self.process_symbol(symbol, free_balance)
                
        except Exception as e:
            self.logger.error(f"交易周期执行错误: {str(e)}")
    
    async def process_symbol(self, symbol: str, balance: float):
        """处理单个交易对"""
        try:
            # 获取K线数据
            df = self.exchange.get_ohlcv(symbol, '1h', 100)
            if df is None or df.empty:
                return
            
            # 计算指标
            df_with_indicators = self.indicators.calculate_indicators(df)
            
            # 生成交易信号
            signal = self.indicators.generate_signal(df_with_indicators, symbol)
            
            # 获取当前价格
            current_price = float(df_with_indicators.iloc[-1]['close'])
            
            # 获取马丁管理器
            martingale_manager = self.martingale_managers[symbol]
            
            # 检查止盈条件
            if await martingale_manager.check_take_profit(current_price):
                await martingale_manager.close_all_positions()
                return
            
            # 处理交易信号
            if signal:
                await self.handle_trading_signal(signal, martingale_manager, current_price, balance)
            
            # 检查马丁加仓条件
            await self.check_martingale_layers(martingale_manager, current_price, balance)
            
            # 记录状态
            status = martingale_manager.get_status()
            self.logger.info(
                f"{symbol} 状态: 多{status.long_layers}层({status.long_exposure:.6f}), "
                f"空{status.short_layers}层({status.short_exposure:.6f}), "
                f"净暴露: {status.net_exposure:.6f}"
            )
            
        except Exception as e:
            self.logger.error(f"处理交易对 {symbol} 失败: {str(e)}")
    
    async def handle_trading_signal(self, signal: TradeSignal, manager: DualMartingaleManager, 
                                  current_price: float, balance: float):
        """处理交易信号"""
        status = manager.get_status()
        
        # 如果已经有相反方向的仓位，先平仓
        if (signal.side == OrderSide.BUY and status.short_exposure > 0):
            self.logger.info(f"{signal.symbol} 有相反方向仓位，先平空仓")
            await manager.close_all_positions()
        elif (signal.side == OrderSide.SELL and status.long_exposure > 0):
            self.logger.info(f"{signal.symbol} 有相反方向仓位，先平多仓")
            await manager.close_all_positions()
        
        # 执行新信号
        result = await self.executor.execute_order(signal, balance)
        if result.success:
            # 创建马丁层
            side = PositionSide.LONG if signal.side == OrderSide.BUY else PositionSide.SHORT
            await manager.add_martingale_layer(side, current_price, balance)
    
    async def check_martingale_layers(self, manager: DualMartingaleManager, 
                                    current_price: float, balance: float):
        """检查马丁加仓条件"""
        status = manager.get_status()
        
        # 检查多仓加仓
        if status.long_layers > 0 and status.long_layers < MAX_MARTINGALE_LAYERS:
            await manager.add_martingale_layer(PositionSide.LONG, current_price, balance)
        
        # 检查空仓加仓
        if status.short_layers > 0 and status.short_layers < MAX_MARTINGALE_LAYERS:
            await manager.add_martingale_layer(PositionSide.SHORT, current_price, balance)
    
    async def run(self):
        """运行交易机器人"""
        if not self.initialized:
            self.logger.error("交易机器人未初始化，无法运行")
            return
        
        self.logger.info("开始运行交易机器人...")
        
        try:
            while True:
                try:
                    await self.run_trading_cycle()
                    await asyncio.sleep(UPDATE_INTERVAL)
                    
                except Exception as e:
                    self.logger.error(f"交易周期执行错误: {str(e)}")
                    await asyncio.sleep(60)
                    
        except asyncio.CancelledError:
            self.logger.info("交易任务被取消")
        except Exception as e:
            self.logger.error(f"交易机器人运行错误: {str(e)}")

# ================== 程序入口 ==================
async def main():
    """主函数"""
    trader = EnhancedProductionTrader()
    
    # 初始化交易机器人
    if not trader.initialize():
        logging.error("初始化失败，程序退出")
        sys.exit(1)
    
    # 运行交易机器人
    try:
        await trader.run()
    except KeyboardInterrupt:
        logging.info("用户中断程序")
    except Exception as e:
        logging.error(f"程序运行错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # 检查必要的环境变量
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        logging.error("请设置 BINANCE_API_KEY 和 BINANCE_API_SECRET 环境变量")
        sys.exit(1)
    
    # 运行主程序
    asyncio.run(main())
