import os
import sys
import time
import ccxt
import pandas as pd
import numpy as np
import ta
import logging
import asyncio
import signal
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

# ================== 配置参数 ==================
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT").split(",") if s.strip()]
TIMEFRAME = os.getenv("MACD_FILTER_TIMEFRAME", "4h")
LEVERAGE = int(os.getenv("LEVERAGE", "15"))
RISK_RATIO = float(os.getenv("RISK_RATIO", "0.15"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))

# 马丁策略参数
MAX_LAYERS = int(os.getenv("MAX_MARTINGALE_LAYERS", "4"))
MARTINGALE_MULTIPLIER = float(os.getenv("MARTINGALE_MULTIPLIER", "2.0"))
LAYER_TRIGGER = float(os.getenv("MARTINGALE_TRIGGER_LOSS", "0.05"))
INITIAL_RISK = float(os.getenv("INITIAL_RISK_PERCENT", "0.02"))
MIN_LAYER_INTERVAL = int(os.getenv("MIN_LAYER_INTERVAL_MINUTES", "240"))  # 加仓最小间隔时间(分钟)

# 指标参数
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# ================== 日志设置 ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('trading_bot.log')]
)
logger = logging.getLogger("HedgeMartingaleBot")

# ================== 数据模型 ==================
class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"

class TradeSignal:
    def __init__(self, symbol: str, signal_type: SignalType, price: float, confidence: float, indicators: dict):
        self.symbol = symbol
        self.type = signal_type
        self.price = price
        self.confidence = confidence
        self.indicators = indicators
        self.timestamp = datetime.now()

    def __str__(self):
        return f"{self.symbol} {self.type.value}@{self.price:.2f} (Conf: {self.confidence:.2f})"

# ================== 交易所接口 ==================
class BinanceFutureAPI:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.exchange = None

    def initialize(self) -> bool:
        try:
            self.exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'options': {'defaultType': 'future', 'hedgeMode': True},
                'enableRateLimit': True
            })
            for symbol in SYMBOLS:
                try:
                    self.exchange.set_leverage(LEVERAGE, symbol)
                    logger.info(f"设置杠杆 {symbol} {LEVERAGE}x")
                except Exception as e:
                    logger.warning(f"设置杠杆失败 {symbol}: {e}")
            logger.info("交易所初始化成功")
            return True
        except Exception as e:
            logger.error(f"交易所初始化失败: {e}")
            return False

    def get_balance(self) -> float:
        try:
            balance = self.exchange.fetch_balance()
            return float(balance['USDT']['free'])
        except Exception as e:
            logger.error(f"获取余额失败: {e}")
            return 0.0

    def get_ohlcv_data(self, symbol: str, timeframe: str, limit: int = 100) -> Optional[pd.DataFrame]:
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"K线获取失败 {symbol}: {e}")
            return None

    def execute_market_order(self, symbol: str, side: str, amount: float) -> bool:
        try:
            market = self.exchange.market(symbol)
            min_amount = market['limits']['amount']['min']
            amount = max(amount, min_amount)

            position_side = "LONG" if side.lower() == "buy" else "SHORT"

            order = self.exchange.create_market_order(
                symbol,
                side.lower(),
                amount,
                params={"positionSide": position_side}
            )
            logger.info(f"订单成功 {symbol} {side} {amount:.6f} ({position_side}) - 订单ID: {order['id']}")
            return True
        except Exception as e:
            logger.error(f"下单失败 {symbol} {side}: {e}")
            return False

# ================== 技术指标分析 ==================
class TechnicalAnalyzer:
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> dict:
        if len(df) < 50: return {}
        macd_indicator = ta.trend.MACD(df['close'], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL)
        macd_line = macd_indicator.macd()
        macd_signal = macd_indicator.macd_signal()
        macd_hist = macd_indicator.macd_diff()
        rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        ema_12 = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        ema_26 = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        boll = ta.volatility.BollingerBands(df['close'])
        return {
            'macd': macd_line.iloc[-1],
            'macd_signal': macd_signal.iloc[-1],
            'macd_histogram': macd_hist.iloc[-1],
            'rsi': rsi.iloc[-1],
            'ema_12': ema_12.iloc[-1],
            'ema_26': ema_26.iloc[-1],
            'atr': atr.iloc[-1],
            'bb_upper': boll.bollinger_hband().iloc[-1],
            'bb_lower': boll.bollinger_lband().iloc[-1],
            'bb_middle': boll.bollinger_mavg().iloc[-1],
            'price': df['close'].iloc[-1],
            'volume': df['volume'].iloc[-1]
        }

    @staticmethod
    def generate_signal(symbol: str, indicators: dict) -> Optional[TradeSignal]:
        if not indicators: return None
        price = indicators['price']
        confidence = 0.5
        macd_bull = indicators['macd'] > indicators['macd_signal']
        macd_bear = indicators['macd'] < indicators['macd_signal']
        trend_bull = indicators['ema_12'] > indicators['ema_26']
        trend_bear = indicators['ema_12'] < indicators['ema_26']
        rsi_over = indicators['rsi'] > RSI_OVERBOUGHT
        rsi_under = indicators['rsi'] < RSI_OVERSOLD
        above_bb = price > indicators['bb_middle']
        below_bb = price < indicators['bb_middle']

        buy_signals = sum([macd_bull, trend_bull, rsi_under, above_bb])
        sell_signals = sum([macd_bear, trend_bear, rsi_over, below_bb])
        confidence += 0.1 * max(buy_signals, sell_signals)
        if buy_signals >= 3 and buy_signals > sell_signals:
            return TradeSignal(symbol, SignalType.BUY, price, min(confidence, 0.9), indicators)
        elif sell_signals >= 3 and sell_signals > buy_signals:
            return TradeSignal(symbol, SignalType.SELL, price, min(confidence, 0.9), indicators)
        return None

# ================== 马丁策略管理 ==================
class MartingaleManager:
    def __init__(self):
        self.positions: Dict[str, List[dict]] = {}
        self.last_layer_time: Dict[str, datetime] = {}  # 记录每个交易对最后一次加仓时间

    def add_position(self, symbol: str, side: str, size: float, price: float):
        if symbol not in self.positions:
            self.positions[symbol] = []
        layer = len(self.positions[symbol]) + 1
        self.positions[symbol].append({
            'side': side,
            'size': size,
            'entry_price': price,
            'timestamp': datetime.now(),
            'layer': layer
        })
        self.last_layer_time[symbol] = datetime.now()
        logger.info(f"📊 {symbol} 第{layer}层仓位: {side} {size:.6f} @ {price:.2f}")

    def should_add_layer(self, symbol: str, current_price: float) -> bool:
        if symbol not in self.positions or not self.positions[symbol]:
            return False
            
        # 检查是否已达到最大层数
        if len(self.positions[symbol]) >= MAX_LAYERS:
            logger.info(f"⚠️ {symbol} 已达到最大层数 {MAX_LAYERS}")
            return False
            
        # 检查加仓时间间隔
        last_time = self.last_layer_time.get(symbol)
        if last_time and (datetime.now() - last_time) < timedelta(minutes=MIN_LAYER_INTERVAL):
            logger.info(f"⏰ {symbol} 加仓间隔时间不足，跳过加仓")
            return False
            
        positions = self.positions[symbol]
        total_size = sum(p['size'] for p in positions)
        total_value = sum(p['size'] * p['entry_price'] for p in positions)
        avg_price = total_value / total_size
        
        # 计算当前盈亏百分比
        if positions[0]['side'] == 'buy':
            pnl_pct = (current_price - avg_price) / avg_price
        else:  # sell
            pnl_pct = (avg_price - current_price) / avg_price
            
        logger.info(f"📈 {symbol} 当前盈亏: {pnl_pct*100:.2f}%, 触发阈值: {-LAYER_TRIGGER*100:.2f}%")
        
        # 只有当亏损达到触发阈值时才加仓
        return pnl_pct <= -LAYER_TRIGGER

    def calculate_layer_size(self, symbol: str, balance: float, current_price: float, atr: float) -> float:
        if symbol not in self.positions:
            return self.calculate_initial_size(balance, current_price, atr)
        layer = len(self.positions[symbol]) + 1
        base_size = (balance * INITIAL_RISK) / (atr * float(os.getenv("RISK_ATR_MULT", "1.5")))
        martingale_size = base_size * (MARTINGALE_MULTIPLIER ** (layer - 1))
        
        # 限制最大仓位大小不超过余额的50%
        max_size = balance * 0.5 / current_price
        final_size = min(martingale_size, max_size)
        
        logger.info(f"📏 {symbol} 第{layer}层计算仓位: 基础={base_size:.6f}, 马丁={martingale_size:.6f}, 最终={final_size:.6f}")
        return final_size

    def calculate_initial_size(self, balance: float, current_price: float, atr: float) -> float:
        risk_amount = balance * INITIAL_RISK
        size = risk_amount / (atr * float(os.getenv("RISK_ATR_MULT", "1.5")))
        
        # 限制最大仓位大小不超过余额的10%
        max_size = balance * 0.1 / current_price
        return min(size, max_size)
        
    def should_close_all(self, symbol: str, current_price: float) -> bool:
        """检查是否应该平掉所有仓位（止损或止盈）"""
        if symbol not in self.positions or not self.positions[symbol]:
            return False
            
        positions = self.positions[symbol]
        total_size = sum(p['size'] for p in positions)
        total_value = sum(p['size'] * p['entry_price'] for p in positions)
        avg_price = total_value / total_size
        
        # 计算当前盈亏百分比
        if positions[0]['side'] == 'buy':
            pnl_pct = (current_price - avg_price) / avg_price
        else:  # sell
            pnl_pct = (avg_price - current_price) / avg_price
            
        # 如果亏损超过30%，强制平仓
        if pnl_pct <= -0.3:
            logger.warning(f"🚨 {symbol} 亏损超过30%，强制平仓")
            return True
            
        # 如果盈利超过20%，止盈
        if pnl_pct >= 0.2:
            logger.info(f"🎯 {symbol} 盈利超过20%，止盈平仓")
            return True
            
        return False

# ================== 主交易机器人 ==================
class SignalBasedTradingBot:
    def __init__(self):
        self.api = BinanceFutureAPI(BINANCE_API_KEY, BINANCE_API_SECRET)
        self.analyzer = TechnicalAnalyzer()
        self.martingale = MartingaleManager()
        self.running = True
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def shutdown(self, signum, frame):
        logger.info("收到关闭信号，停止交易...")
        self.running = False

    async def run(self):
        if not self.api.initialize():
            logger.error("交易所初始化失败，程序退出")
            return
        logger.info("🚀 开始自动交易...")
        while self.running:
            try:
                balance = self.api.get_balance()
                logger.info(f"当前余额: {balance:.2f} USDT")
                for symbol in SYMBOLS:
                    await self.process_symbol(symbol, balance)
                await asyncio.sleep(POLL_INTERVAL)
            except Exception as e:
                logger.error(f"交易循环错误: {e}")
                await asyncio.sleep(10)

    async def process_symbol(self, symbol: str, balance: float):
        df = self.api.get_ohlcv_data(symbol, TIMEFRAME, 100)
        if df is None or df.empty:
            return
            
        indicators = self.analyzer.calculate_indicators(df)
        if not indicators:
            return
            
        current_price = indicators['price']
        
        # 检查是否需要平仓（止损或止盈）
        if self.martingale.should_close_all(symbol, current_price):
            await self.close_all_positions(symbol)
            return
            
        # 检查是否需要加仓
        if self.martingale.should_add_layer(symbol, current_price):
            await self.add_martingale_layer(symbol, balance, indicators)
            
        # 检查是否有新信号
        signal = self.analyzer.generate_signal(symbol, indicators)
        if signal:
            logger.info(f"🎯 发现交易信号: {signal}")
            await self.execute_signal(signal, balance)

    async def execute_signal(self, signal: TradeSignal, balance: float):
        # 如果有现有仓位且方向相反，先平仓
        if signal.symbol in self.martingale.positions and self.martingale.positions[signal.symbol]:
            current_side = self.martingale.positions[signal.symbol][0]['side']
            new_side = "buy" if signal.type == SignalType.BUY else "sell"
            
            if current_side != new_side:
                logger.info(f"🔄 {signal.symbol} 发现反向信号，先平仓再开新仓")
                await self.close_all_positions(signal.symbol)
                
        # 开新仓
        position_size = self.martingale.calculate_initial_size(balance, signal.price, signal.indicators['atr'])
        if position_size <= 0:
            return
            
        side = "buy" if signal.type == SignalType.BUY else "sell"
        success = self.api.execute_market_order(signal.symbol, side, position_size)
        if success:
            self.martingale.add_position(signal.symbol, side, position_size, signal.price)

    async def add_martingale_layer(self, symbol: str, balance: float, indicators: dict):
        positions = self.martingale.positions.get(symbol, [])
        if not positions:
            return
            
        side = positions[0]['side']
        layer_size = self.martingale.calculate_layer_size(symbol, balance, indicators['price'], indicators['atr'])
        
        logger.info(f"📈 {symbol} 准备加仓第{len(positions)+1}层，方向: {side}, 大小: {layer_size:.6f}")
        
        success = self.api.execute_market_order(symbol, side, layer_size)
        if success:
            self.martingale.add_position(symbol, side, layer_size, indicators['price'])

    async def close_all_positions(self, symbol: str):
        if symbol not in self.martingale.positions or not self.martingale.positions[symbol]:
            return
            
        positions = self.martingale.positions[symbol]
        side = positions[0]['side']
        total_size = sum(p['size'] for p in positions)
        
        # 平仓方向与开仓方向相反
        close_side = "sell" if side == "buy" else "buy"
        
        logger.info(f"📤 {symbol} 平仓所有仓位，方向: {close_side}, 大小: {total_size:.6f}")
        
        success = self.api.execute_market_order(symbol, close_side, total_size)
        if success:
            self.martingale.positions[symbol] = []
            logger.info(f"✅ {symbol} 所有仓位已平仓")

# ================== 启动程序 ==================
async def main():
    bot = SignalBasedTradingBot()
    try:
        await bot.run()
    except KeyboardInterrupt:
        logger.info("用户中断程序")
    except Exception as e:
        logger.error(f"程序错误: {e}")
    finally:
        logger.info("交易程序结束")

if __name__ == "__main__":
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        print("错误: 请设置 BINANCE_API_KEY 和 BINANCE_API_SECRET 环境变量")
        sys.exit(1)
        
    if not SYMBOLS:
        print("错误: 请设置 SYMBOLS 环境变量，例如: BTC/USDT,ETH/USDT")
        sys.exit(1)
        
    asyncio.run(main())
