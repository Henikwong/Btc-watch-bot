import os
import time
import ccxt
import pandas as pd
import numpy as np
import ta
import logging
import asyncio
from datetime import datetime, timedelta
from enum import Enum
import sys

# ================== 配置 ==================
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
SYMBOLS = [s for s in os.getenv("SYMBOLS", "").split(",") if s]
TIMEFRAME = os.getenv("MACD_FILTER_TIMEFRAME", "4h")
LEVERAGE = int(os.getenv("LEVERAGE", "15"))
RISK_RATIO = float(os.getenv("RISK_RATIO", "0.15"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))
MAX_LAYERS = int(os.getenv("MAX_MARTINGALE_LAYERS", "4"))
MARTINGALE_MULTIPLIER = float(os.getenv("MARTINGALE_MULTIPLIER", "2.0"))
LAYER_TRIGGER = float(os.getenv("MARTINGALE_TRIGGER_LOSS", "0.05"))
INITIAL_RISK = float(os.getenv("INITIAL_RISK_PERCENT", "0.02"))
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("HedgeMartingaleBot")

# ================== 信号类型 ==================
class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"

class TradeSignal:
    def __init__(self, symbol, signal_type, price, confidence, indicators):
        self.symbol = symbol
        self.type = signal_type
        self.price = price
        self.confidence = confidence
        self.indicators = indicators
        self.timestamp = datetime.now()

# ================== 交易所接口 ==================
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
                'options': {'defaultType': 'future', 'defaultPositionSide': 'HEDGE'},
                'enableRateLimit': True
            })
            for symbol in SYMBOLS:
                self.exchange.set_leverage(LEVERAGE, symbol)
                logger.info(f"设置杠杆 {symbol} {LEVERAGE}x")
            logger.info("交易所初始化成功")
            return True
        except Exception as e:
            logger.error(f"交易所初始化失败: {e}")
            return False

    def get_balance(self):
        try:
            balance = self.exchange.fetch_balance()
            return float(balance['USDT']['free'])
        except Exception as e:
            logger.error(f"获取余额失败: {e}")
            return 0.0

    def get_ohlcv_data(self, symbol, timeframe, limit=100):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"K线获取失败: {symbol} {e}")
            return None

    def execute_market_order(self, symbol, side, amount, position_side):
        try:
            params = {'positionSide': position_side}
            order = self.exchange.create_market_order(symbol, side, amount, params=params)
            logger.info(f"订单成功 {symbol} {side} {amount:.6f} ({position_side}) - 订单ID: {order['id']}")
            return True
        except Exception as e:
            logger.error(f"下单失败 {symbol} {side}: {e}")
            return False

# ================== 技术指标分析 ==================
class TechnicalAnalyzer:
    @staticmethod
    def calculate_indicators(df):
        if len(df) < 50:
            return {}
        macd_ind = ta.trend.MACD(df['close'], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        macd_line = macd_ind.macd().iloc[-1]
        macd_signal = macd_ind.macd_signal().iloc[-1]
        macd_hist = macd_ind.macd_diff().iloc[-1]
        rsi = ta.momentum.RSIIndicator(df['close'], 14).rsi().iloc[-1]
        ema_12 = ta.trend.EMAIndicator(df['close'], 12).ema_indicator().iloc[-1]
        ema_26 = ta.trend.EMAIndicator(df['close'], 26).ema_indicator().iloc[-1]
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range().iloc[-1]
        bb = ta.volatility.BollingerBands(df['close'])
        return {
            'macd': macd_line,
            'macd_signal': macd_signal,
            'macd_hist': macd_hist,
            'rsi': rsi,
            'ema_12': ema_12,
            'ema_26': ema_26,
            'atr': atr,
            'bb_upper': bb.bollinger_hband().iloc[-1],
            'bb_lower': bb.bollinger_lband().iloc[-1],
            'bb_middle': bb.bollinger_mavg().iloc[-1],
            'price': df['close'].iloc[-1]
        }

    @staticmethod
    def generate_signal(symbol, ind):
        price = ind['price']
        conf = 0.5
        buy_signals = sum([ind['macd']>ind['macd_signal'], ind['ema_12']>ind['ema_26'], ind['rsi']<RSI_OVERSOLD, price>ind['bb_middle']])
        sell_signals = sum([ind['macd']<ind['macd_signal'], ind['ema_12']<ind['ema_26'], ind['rsi']>RSI_OVERBOUGHT, price<ind['bb_middle']])
        conf += 0.1 * max(buy_signals, sell_signals)
        if buy_signals >=3 and buy_signals>sell_signals:
            return TradeSignal(symbol, SignalType.BUY, price, min(conf,0.9), ind)
        elif sell_signals>=3 and sell_signals>buy_signals:
            return TradeSignal(symbol, SignalType.SELL, price, min(conf,0.9), ind)
        return None

# ================== 马丁策略管理 ==================
class MartingaleManager:
    def __init__(self):
        self.positions = {}
        self.last_layer_time = {}

    def add_position(self, symbol, side, size, price):
        if symbol not in self.positions:
            self.positions[symbol] = []
        self.positions[symbol].append({'side':side,'size':size,'price':price,'time':datetime.now()})
        logger.info(f"📊 {symbol} 第{len(self.positions[symbol])}层仓位: {side} {size:.6f} @ {price:.2f}")

    def should_add_layer(self, symbol, current_price):
        if symbol not in self.positions or not self.positions[symbol]:
            return False
        last_time = self.last_layer_time.get(symbol, datetime.min)
        if datetime.now() - last_time < timedelta(hours=24):
            return False  # 24小时内不加仓
        positions = self.positions[symbol]
        avg_price = sum(p['price']*p['size'] for p in positions)/sum(p['size'] for p in positions)
        side = positions[0]['side']
        pnl_pct = (current_price-avg_price)/avg_price if side=='buy' else (avg_price-current_price)/avg_price
        return pnl_pct <= -LAYER_TRIGGER

    def calculate_size(self, balance, atr, symbol):
        layer = len(self.positions.get(symbol, [])) +1
        base_size = (balance*INITIAL_RISK)/(atr*1.5)
        return base_size * (MARTINGALE_MULTIPLIER**(layer-1))

# ================== 主交易机器人 ==================
class HedgeMartingaleBot:
    def __init__(self):
        self.api = BinanceFutureAPI(BINANCE_API_KEY,BINANCE_API_SECRET)
        self.analyzer = TechnicalAnalyzer()
        self.martingale = MartingaleManager()
        self.running = True

    async def process_symbol(self, symbol, balance):
        df = self.api.get_ohlcv_data(symbol,TIMEFRAME)
        if df is None or df.empty:
            return
        ind = self.analyzer.calculate_indicators(df)
        signal = self.analyzer.generate_signal(symbol, ind)
        if not signal:
            return

        # 下单逻辑
        size = self.martingale.calculate_size(balance, ind['atr'], symbol)
        if signal.type == SignalType.BUY:
            self.api.execute_market_order(symbol, 'buy', size, 'LONG')
            self.martingale.add_position(symbol, 'buy', size, signal.price)
            self.martingale.last_layer_time[symbol] = datetime.now()
        elif signal.type == SignalType.SELL:
            self.api.execute_market_order(symbol, 'sell', size, 'SHORT')
            self.martingale.add_position(symbol, 'sell', size, signal.price)
            self.martingale.last_layer_time[symbol] = datetime.now()

# ================== 启动入口 ==================
async def main():
    bot = HedgeMartingaleBot()
    await bot.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 手动停止")
