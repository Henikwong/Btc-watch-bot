import os
import time
import ccxt
import pandas as pd
import numpy as np
import ta
import logging
import asyncio
from datetime import datetime
from enum import Enum

# ================== 配置参数 ==================
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
SYMBOLS = os.getenv("SYMBOLS", "BTC/USDT").split(",")
TIMEFRAME = os.getenv("MACD_FILTER_TIMEFRAME", "4h")
LEVERAGE = int(os.getenv("LEVERAGE", "15"))
RISK_RATIO = float(os.getenv("RISK_RATIO", "0.15"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))

# 马丁参数
MAX_LAYERS = int(os.getenv("MAX_MARTINGALE_LAYERS", "4"))
MARTINGALE_MULTIPLIER = float(os.getenv("MARTINGALE_MULTIPLIER", "2.0"))
LAYER_TRIGGER = float(os.getenv("MARTINGALE_TRIGGER_LOSS", "0.05"))
INITIAL_RISK = float(os.getenv("INITIAL_RISK_PERCENT", "0.02"))

# 技术指标参数
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

PARTIAL_TP_RATIO = float(os.getenv("PARTIAL_TP_RATIO", "0.3"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "2.0"))
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "3.0"))

# ================== 日志 ==================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AutoTrader")

# ================== Enum ==================
class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"

# ================== 交易信号 ==================
class TradeSignal:
    def __init__(self, symbol, signal_type: SignalType, price, confidence, indicators):
        self.symbol = symbol
        self.type = signal_type
        self.price = price
        self.confidence = confidence
        self.indicators = indicators
        self.timestamp = datetime.now()

    def __str__(self):
        return f"{self.symbol} {self.type.value}@{self.price:.2f} (Conf: {self.confidence:.2f})"

# ================== Binance 接口 ==================
class BinanceFutureAPI:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.exchange = None

    def initialize(self):
        try:
            self.exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'options': {'defaultType': 'future'},
                'enableRateLimit': True
            })
            for symbol in SYMBOLS:
                try:
                    self.exchange.set_leverage(LEVERAGE, symbol)
                    logger.info(f"设置杠杆 {symbol} {LEVERAGE}x")
                except Exception as e:
                    logger.warning(f"设置杠杆失败 {symbol}: {e}")
            return True
        except Exception as e:
            logger.error(f"交易所初始化失败: {e}")
            return False

    def get_balance(self):
        try:
            balance = self.exchange.fetch_balance()
            return float(balance['USDT']['free'])
        except:
            return 0.0

    def get_ohlcv_data(self, symbol, timeframe, limit=200):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"{symbol} K线获取失败: {e}")
            return None

    def execute_market_order(self, symbol, side, amount):
        try:
            market = self.exchange.market(symbol)
            min_amount = market['limits']['amount']['min']
            amount = max(amount, min_amount)
            order = self.exchange.create_market_order(symbol, side, amount)
            logger.info(f"订单成功 {symbol} {side} {amount} - ID: {order['id']}")
            return True
        except Exception as e:
            logger.error(f"{symbol} 下单失败 {side}: {e}")
            return False

# ================== 技术指标分析 ==================
class TechnicalAnalyzer:
    @staticmethod
    def calculate_indicators(df):
        if len(df) < 50: return {}
        macd_indicator = ta.trend.MACD(df['close'], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL)
        rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        ema_12 = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        ema_26 = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        bollinger = ta.volatility.BollingerBands(df['close'])
        return {
            'macd': macd_indicator.macd().iloc[-1],
            'macd_signal': macd_indicator.macd_signal().iloc[-1],
            'macd_histogram': macd_indicator.macd_diff().iloc[-1],
            'rsi': rsi.iloc[-1],
            'ema_12': ema_12.iloc[-1],
            'ema_26': ema_26.iloc[-1],
            'atr': atr.iloc[-1],
            'bb_upper': bollinger.bollinger_hband().iloc[-1],
            'bb_lower': bollinger.bollinger_lband().iloc[-1],
            'bb_middle': bollinger.bollinger_mavg().iloc[-1],
            'price': df['close'].iloc[-1],
            'volume': df['volume'].iloc[-1]
        }

    @staticmethod
    def generate_signal(symbol, indicators):
        if not indicators: return None
        price = indicators['price']
        confidence = 0.5

        buy_signals = sum([
            indicators['macd'] > indicators['macd_signal'],
            indicators['ema_12'] > indicators['ema_26'],
            indicators['rsi'] < RSI_OVERSOLD,
            price > indicators['bb_middle']
        ])
        sell_signals = sum([
            indicators['macd'] < indicators['macd_signal'],
            indicators['ema_12'] < indicators['ema_26'],
            indicators['rsi'] > RSI_OVERBOUGHT,
            price < indicators['bb_middle']
        ])

        confidence += 0.1 * (buy_signals + sell_signals)
        if buy_signals >= 3 and buy_signals > sell_signals:
            return TradeSignal(symbol, SignalType.BUY, price, min(confidence, 0.9), indicators)
        elif sell_signals >= 3 and sell_signals > buy_signals:
            return TradeSignal(symbol, SignalType.SELL, price, min(confidence, 0.9), indicators)
        return None

# ================== 马丁管理 ==================
class MartingaleManager:
    def __init__(self):
        self.positions = {}

    def add_position(self, symbol, side, size, price):
        if symbol not in self.positions: self.positions[symbol] = []
        self.positions[symbol].append({'side': side, 'size': size, 'entry_price': price})
        logger.info(f"{symbol} 新仓位: {side} {size:.6f} @ {price:.2f}")

    def should_add_layer(self, symbol, current_price):
        if symbol not in self.positions or not self.positions[symbol]: return False
        positions = self.positions[symbol]
        if len(positions) >= MAX_LAYERS: return False
        total_size = sum(p['size'] for p in positions)
        total_value = sum(p['size'] * p['entry_price'] for p in positions)
        avg_price = total_value / total_size
        pnl_pct = (current_price - avg_price)/avg_price if positions[0]['side']=='buy' else (avg_price - current_price)/avg_price
        return pnl_pct <= -LAYER_TRIGGER

    def calculate_layer_size(self, symbol, balance, atr):
        layer = len(self.positions.get(symbol, [])) + 1
        base_size = (balance * INITIAL_RISK) / (atr * float(os.getenv("RISK_ATR_MULT", "1.5")))
        return base_size * (MARTINGALE_MULTIPLIER ** (layer - 1))

# ================== 交易机器人 ==================
class SignalBasedTradingBot:
    def __init__(self):
        self.api = BinanceFutureAPI(BINANCE_API_KEY, BINANCE_API_SECRET)
        self.analyzer = TechnicalAnalyzer()
        self.martingale = MartingaleManager()
        self.running = True

    async def run(self):
        if not self.api.initialize(): return
        while self.running:
            balance = self.api.get_balance()
            for symbol in SYMBOLS:
                await self.process_symbol(symbol, balance)
            await asyncio.sleep(POLL_INTERVAL)

    async def process_symbol(self, symbol, balance):
        df = self.api.get_ohlcv_data(symbol, TIMEFRAME)
        if df is None or df.empty: return
        indicators = self.analyzer.calculate_indicators(df)
        signal = self.analyzer.generate_signal(symbol, indicators)
        if signal: await self.execute_signal(signal, balance)
        # 检查马丁加仓
        if self.martingale.should_add_layer(symbol, indicators['price']):
            side = self.martingale.positions[symbol][0]['side']
            size = self.martingale.calculate_layer_size(symbol, balance, indicators['atr'])
            if self.api.execute_market_order(symbol, side, size):
                self.martingale.add_position(symbol, side, size, indicators['price'])

    async def execute_signal(self, signal, balance):
        size = self.martingale.calculate_layer_size(signal.symbol, balance, signal.indicators['atr'])
        side = "buy" if signal.type == SignalType.BUY else "sell"
        if self.api.execute_market_order(signal.symbol, side, size):
            self.martingale.add_position(signal.symbol, side, size, signal.price)

# ================== 主函数 ==================
async def main():
    bot = SignalBasedTradingBot()
    try:
        await bot.run()
    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"程序异常: {e}")

if __name__ == "__main__":
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        logger.error("请设置 API Key/Secret")
    else:
        asyncio.run(main())
