import os
import sys
import time
import asyncio
import logging
import ccxt
import pandas as pd
import ta
from datetime import datetime
from enum import Enum

# ================== 配置参数 ==================
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
SYMBOLS = [s for s in os.getenv("SYMBOLS", "BTC/USDT").split(",") if s]
TIMEFRAME = os.getenv("MACD_FILTER_TIMEFRAME", "4h")
LEVERAGE = int(os.getenv("LEVERAGE", "15"))
RISK_RATIO = float(os.getenv("RISK_RATIO", "0.15"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))

# 马丁策略
MAX_LAYERS = int(os.getenv("MAX_MARTINGALE_LAYERS", "4"))
MARTINGALE_MULTIPLIER = float(os.getenv("MARTINGALE_MULTIPLIER", "2.0"))
LAYER_TRIGGER = float(os.getenv("MARTINGALE_TRIGGER_LOSS", "0.05"))  # 亏损5%加仓
INITIAL_RISK = float(os.getenv("INITIAL_RISK_PERCENT", "0.02"))

# 技术指标
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# 日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.StreamHandler(sys.stdout),
    logging.FileHandler("trading_bot.log")
])
logger = logging.getLogger("HedgeMartingaleBot")

# ================== 交易所接口 ==================
class BinanceFutureAPI:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.exchange = None

    def initialize(self):
        try:
            self.exchange = ccxt.binance({
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "future",
                    "hedgeMode": True
                }
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
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"K线获取失败: {symbol}: {e}")
            return None

    def execute_market_order(self, symbol, side, amount):
        try:
            market = self.exchange.market(symbol)
            min_amount = market['limits']['amount']['min']
            amount = max(amount, min_amount)
            order = self.exchange.create_market_order(symbol, side, amount, params={"positionSide": "BOTH"})
            logger.info(f"订单成功 {symbol} {side} {amount:.6f} - ID: {order['id']}")
            return True
        except Exception as e:
            logger.error(f"{symbol} 下单失败 {side}: {e}")
            return False

# ================== 技术指标分析 ==================
class TechnicalAnalyzer:
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> dict:
        if len(df) < 50:
            return {}
        macd = ta.trend.MACD(df['close'], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL)
        rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        ema12 = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        ema26 = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        bb = ta.volatility.BollingerBands(df['close'])
        return {
            'macd': macd.macd().iloc[-1],
            'macd_signal': macd.macd_signal().iloc[-1],
            'rsi': rsi.iloc[-1],
            'ema12': ema12.iloc[-1],
            'ema26': ema26.iloc[-1],
            'atr': atr.iloc[-1],
            'bb_middle': bb.bollinger_mavg().iloc[-1],
            'price': df['close'].iloc[-1]
        }

# ================== 马丁策略 ==================
class MartingaleManager:
    def __init__(self):
        self.positions = {}

    def add_position(self, symbol, side, size, price):
        if symbol not in self.positions:
            self.positions[symbol] = []
        self.positions[symbol].append({
            'side': side,
            'size': size,
            'entry_price': price,
            'layer': len(self.positions[symbol]) + 1
        })
        logger.info(f"📊 {symbol} 第{len(self.positions[symbol])}层仓位: {side} {size:.6f} @ {price:.2f}")

    def should_add_layer(self, symbol, current_price):
        if symbol not in self.positions or not self.positions[symbol]:
            return False
        positions = self.positions[symbol]
        if len(positions) >= MAX_LAYERS:
            return False
        total_size = sum(p['size'] for p in positions)
        total_value = sum(p['size'] * p['entry_price'] for p in positions)
        avg_price = total_value / total_size
        if positions[0]['side'] == 'buy':
            pnl_pct = (current_price - avg_price) / avg_price
            return pnl_pct <= -LAYER_TRIGGER
        else:
            pnl_pct = (avg_price - current_price) / avg_price
            return pnl_pct <= -LAYER_TRIGGER

    def calculate_layer_size(self, symbol, balance, atr):
        if symbol not in self.positions:
            return self.calculate_initial_size(balance, atr)
        layer = len(self.positions[symbol]) + 1
        base_size = (balance * INITIAL_RISK) / (atr * float(os.getenv("RISK_ATR_MULT", "1.5")))
        return base_size * (MARTINGALE_MULTIPLIER ** (layer - 1))

    def calculate_initial_size(self, balance, atr):
        risk_amount = balance * INITIAL_RISK
        return risk_amount / (atr * float(os.getenv("RISK_ATR_MULT", "1.5")))

# ================== 主机器人 ==================
class HedgeMartingaleBot:
    def __init__(self):
        self.api = BinanceFutureAPI(BINANCE_API_KEY, BINANCE_API_SECRET)
        self.analyzer = TechnicalAnalyzer()
        self.martingale = MartingaleManager()
        self.running = True

    async def run(self):
        if not self.api.initialize():
            logger.error("交易所初始化失败，程序退出")
            return
        while self.running:
            balance = self.api.get_balance()
            logger.info(f"当前余额: {balance:.2f} USDT")
            for symbol in SYMBOLS:
                await self.process_symbol(symbol, balance)
            await asyncio.sleep(POLL_INTERVAL)

    async def process_symbol(self, symbol, balance):
        df = self.api.get_ohlcv_data(symbol, TIMEFRAME, 100)
        if df is None or df.empty:
            return
        indicators = self.analyzer.calculate_indicators(df)
        if not indicators:
            return

        price = indicators['price']
        # 简单信号示例：MACD上穿买，MACD下穿卖
        side = None
        if indicators['macd'] > indicators['macd_signal']:
            side = "buy"
        elif indicators['macd'] < indicators['macd_signal']:
            side = "sell"

        if side:
            size = self.martingale.calculate_initial_size(balance, indicators['atr'])
            if size > 0:
                success = self.api.execute_market_order(symbol, side, size)
                if success:
                    self.martingale.add_position(symbol, side, size, price)

        # 检查马丁加仓
        if self.martingale.should_add_layer(symbol, price):
            positions = self.martingale.positions.get(symbol, [])
            if positions:
                layer_side = positions[0]['side']
                layer_size = self.martingale.calculate_layer_size(symbol, balance, indicators['atr'])
                success = self.api.execute_market_order(symbol, layer_side, layer_size)
                if success:
                    self.martingale.add_position(symbol, layer_side, layer_size, price)

# ================== 启动 ==================
async def main():
    bot = HedgeMartingaleBot()
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
        logger.error("请设置 BINANCE_API_KEY 和 BINANCE_API_SECRET 环境变量")
        sys.exit(1)
    asyncio.run(main())
