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
import json
import math
import requests
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ================== 配置参数 ==================
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
SYMBOLS_CONFIG = [s.strip() for s in os.getenv("SYMBOLS", "LTC/USDT,DOGE/USDT,XRP/USDT,ADA/USDT,LINK/USDT").split(",") if s.strip()]
TIMEFRAME = os.getenv("MACD_FILTER_TIMEFRAME", "4h")
LEVERAGE = int(os.getenv("LEVERAGE", "15"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))
BASE_TRADE_SIZE = float(os.getenv("BASE_TRADE_SIZE", "8"))  # 基础交易大小改为8 USDT

# 从环境变量读取加仓触发百分比
position_sizes_str = os.getenv("POSITION_SIZES", "2.678,5,6,7,8,9,10,13,14")
POSITION_SIZES = [float(size.strip()) for size in position_sizes_str.split(',')]
MAX_LAYERS = len(POSITION_SIZES)  # 最大层数等于仓位比例的数量

# 从环境变量读取止盈比例
TP_PERCENT = float(os.getenv("TP_PERCENT", "1.5")) / 100

# 从环境变量读取止损设置
STOP_LOSS = float(os.getenv("STOP_LOSS", "-100"))

# 从环境变量读取趋势捕捉和马丁设置
ENABLE_TREND_CATCH = os.getenv("ENABLE_TREND_CATCH", "true").lower() == "true"
ENABLE_MARTINGALE = os.getenv("ENABLE_MARTINGALE", "true").lower() == "true"

# 加仓间隔配置
INITIAL_ADD_INTERVAL = int(os.getenv("INITIAL_ADD_INTERVAL", "1"))  # 前3层加仓间隔(小时)
LATER_ADD_INTERVAL = int(os.getenv("LATER_ADD_INTERVAL", "12"))  # 3层后加仓间隔(小时)

# 趋势捕捉加仓配置
TREND_CATCH_LAYERS = 2  # 捕捉行情时额外加仓层数
TREND_CATCH_SIZES = [5, 7]  # 额外加仓的仓位大小
TREND_SIGNAL_STRENGTH = 0.7  # 趋势信号强度阈值
TREND_COOLDOWN_HOURS = 6  # 趋势加仓冷却时间

# 止损配置
STOP_LOSS_PER_SYMBOL = -1000  # 单币种亏损1000USDT时止损

# Telegram 配置
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# 重试参数
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.0"))

# 币安最小名义价值要求（USDT）
MIN_NOTIONAL = {
    "LTC/USDT": 20,
    "XRP/USDT": 5,
    "ADA/USDT": 5,
    "DOGE/USDT": 20,
    "LINK/USDT": 20,
    "BTC/USDT": 10,
    "ETH/USDT": 10,
    "BNB/USDT": 10,
    "SOL/USDT": 10,
    "DOT/USDT": 10,
    "AVAX/USDT": 10,
    "MATIC/USDT": 10,
    "UNI/USDT": 10,
    "SUI/USDT": 10,
}

# ================== 日志设置 ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('cointech2u_bot.log')]
)
logger = logging.getLogger("CoinTech2uBot")

# ================== Telegram 通知类 ==================
class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
    
    def send_message(self, message: str):
        """发送Telegram消息"""
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram配置不完整，无法发送消息")
            return
        
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code != 200:
                logger.error(f"Telegram消息发送失败: {response.text}")
        except Exception as e:
            logger.error(f"发送Telegram消息时出错: {e}")

# ================== 技术分析函数 ==================
def analyze_trend(df: pd.DataFrame) -> Tuple[float, str]:
    """分析趋势方向和强度，使用多时间框架确认"""
    try:
        # 计算MACD
        macd_indicator = ta.trend.MACD(df['close'])
        macd_line = macd_indicator.macd()
        macd_signal = macd_indicator.macd_signal()
        macd_histogram = macd_indicator.macd_diff()
        
        # 计算RSI
        rsi = ta.momentum.RSIIndicator(df['close']).rsi()
        
        # 计算EMA
        ema_short = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        ema_long = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        
        # 判断趋势方向
        last_macd = macd_line.iloc[-1]
        last_signal = macd_signal.iloc[-1]
        last_rsi = rsi.iloc[-1]
        last_ema_short = ema_short.iloc[-1]
        last_ema_long = ema_long.iloc[-1]
        
        # 计算趋势强度 (0-1)
        trend_strength = min(abs(macd_histogram.iloc[-1]) / (df['close'].iloc[-1] * 0.01), 1.0)
        
        # 判断趋势方向
        if last_macd > last_signal and last_ema_short > last_ema_long:
            return trend_strength, "bullish"
        elif last_macd < last_signal and last_ema_short < last_ema_long:
            return trend_strength, "bearish"
        else:
            return trend_strength, "neutral"
            
    except Exception as e:
        logger.error(f"分析趋势时出错: {e}")
        return 0.0, "neutral"

# ================== 工具函数 ==================
def quantize_amount(amount: float, market) -> float:
    """量化交易量到交易所允许的精度"""
    try:
        # 尝试从filters获取stepSize
        step = None
        for f in market['info'].get('filters', []):
            if f.get('filterType') == 'LOT_SIZE':
                step = float(f.get('stepSize'))
                break
        
        if step is None:
            # 如果没有找到stepSize，使用默认精度
            precision = market['precision']['amount']
            if isinstance(precision, int):
                step = 1.0 / (10 ** precision)
            else:
                step = precision
        
        # 量化到步长的整数倍
        return math.floor(amount / step) * step
    except Exception as e:
        logger.error(f"量化数量时出错: {e}")
        return amount

def quantize_price(price: float, market) -> float:
    """量化价格到交易所允许的精度"""
    try:
        # 尝试从filters获取tickSize
        tick = None
        for f in market['info'].get('filters', []):
            if f.get('filterType') == 'PRICE_FILTER':
                tick = float(f.get('tickSize'))
                break
        
        if tick is None:
            # 如果没有找到tickSize，使用默认精度
            precision = market['precision']['price']
            if isinstance(precision, int):
                tick = 1.0 / (10 ** precision)
            else:
                tick = precision
        
        # 量化到tickSize的整数倍
        return math.floor(price / tick) * tick
    except Exception as e:
        logger.error(f"量化价格时出错: {e}")
        return price

# ================== 交易所接口 ==================
class BinanceFutureAPI:
    def __init__(self, api_key: str, api_secret: str, symbols: List[str]):
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbols = symbols
        self.exchange = None
        self.symbol_info = {}  # 缓存交易对信息
        self.initialize_exchange()
    
    def initialize_exchange(self):
        """初始化交易所连接"""
        try:
            self.exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                }
            })
            
            # 加载市场信息
            self.exchange.load_markets()
            
            # 为每个交易对设置杠杆
            for symbol in self.symbols:
                try:
                    self.exchange.set_leverage(LEVERAGE, symbol)
                    logger.info(f"为 {symbol} 设置杠杆 {LEVERAGE}x")
                    
                    # 缓存交易对信息
                    self.symbol_info[symbol] = self.exchange.market(symbol)
                except Exception as e:
                    logger.error(f"设置 {symbol} 杠杆时出错: {e}")
            
            logger.info("交易所初始化成功")
        except Exception as e:
            logger.error(f"初始化交易所时出错: {e}")
            raise
    
    def get_balance(self):
        """获取账户余额"""
        try:
            balance = self.exchange.fetch_balance()
            return balance['USDT']['free']
        except Exception as e:
            logger.error(f"获取余额时出错: {e}")
            return None
    
    def get_ohlcv(self, symbol: str, timeframe: str = '4h', limit: int = 100):
        """获取K线数据"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"获取 {symbol} K线数据时出错: {e}")
            return None
    
    def get_positions(self):
        """获取所有仓位"""
        try:
            positions = self.exchange.fetch_positions()
            return {p['symbol']: p for p in positions if p['symbol'] in self.symbols and float(p['contracts']) > 0}
        except Exception as e:
            logger.error(f"获取仓位时出错: {e}")
            return {}
    
    def get_position(self, symbol: str):
        """获取特定交易对的仓位"""
        try:
            positions = self.exchange.fetch_positions([symbol])
            for p in positions:
                if p['symbol'] == symbol and float(p['contracts']) > 0:
                    return p
            return None
        except Exception as e:
            logger.error(f"获取 {symbol} 仓位时出错: {e}")
            return None
    
    def get_ticker(self, symbol: str):
        """获取交易对价格"""
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"获取 {symbol} 价格时出错: {e}")
            return None
    
    def place_order(self, symbol: str, side: str, amount: float, type: str = 'market', price: float = None):
        """下单"""
        try:
            # 量化数量
            market = self.symbol_info[symbol]
            quantized_amount = quantize_amount(amount, market)
            
            # 检查最小名义价值
            ticker = self.get_ticker(symbol)
            if ticker and quantized_amount * ticker['last'] < MIN_NOTIONAL.get(symbol, 10):
                logger.warning(f"订单名义价值 {quantized_amount * ticker['last']} 低于最小要求 {MIN_NOTIONAL.get(symbol, 10)}")
                return None
            
            # 下单
            order_params = {
                'symbol': symbol,
                'type': type,
                'side': side,
                'amount': quantized_amount,
            }
            
            if price:
                order_params['price'] = quantize_price(price, market)
            
            order = self.exchange.create_order(**order_params)
            logger.info(f"下单成功: {symbol} {side} {quantized_amount} @ {ticker['last'] if ticker else 'N/A'}")
            return order
        except Exception as e:
            logger.error(f"下单时出错: {e}")
            return None
    
    def close_position(self, symbol: str, side: str, amount: float):
        """平仓"""
        try:
            close_side = 'sell' if side == 'long' else 'buy'
            return self.place_order(symbol, close_side, amount)
        except Exception as e:
            logger.error(f"平仓时出错: {e}")
            return None

# ================== 双仓马丁策略管理 ==================
class DualMartingaleManager:
    def __init__(self, telegram_notifier: TelegramNotifier = None):
        # 仓位结构: {symbol: {'long': [], 'short': []}}
        self.positions: Dict[str, Dict[str, List[dict]]] = {}
        
        # 当前层级: {symbol: {'long': int, 'short': int}}
        self.current_layers: Dict[str, Dict[str, int]] = {}
        
        # 最后加仓时间: {symbol: {'long': datetime, 'short': datetime}}
        self.last_layer_time: Dict[str, Dict[str, datetime]] = {}
        
        # 趋势捕捉加仓时间: {symbol: {'long': datetime, 'short': datetime}}
        self.last_trend_catch_time: Dict[str, Dict[str, datetime]] = {}
        
        # 趋势捕捉加仓计数: {symbol: {'long': int, 'short': int}}
        self.trend_catch_count: Dict[str, Dict[str, int]] = {}
        
        # 仓位状态文件
        self.positions_file = "positions.json"
        
        # Telegram 通知器
        self.telegram = telegram_notifier
        
        # 加载保存的仓位
        self.load_positions()
    
    def load_positions(self):
        """从文件加载仓位状态"""
        try:
            if os.path.exists(self.positions_file):
                with open(self.positions_file, 'r') as f:
                    data = json.load(f)
                    self.positions = data.get('positions', {})
                    self.current_layers = data.get('current_layers', {})
                    self.last_layer_time = {
                        symbol: {
                            side: datetime.fromisoformat(time_str) if time_str else None
                            for side, time_str in times.items()
                        }
                        for symbol, times in data.get('last_layer_time', {}).items()
                    }
                    self.last_trend_catch_time = {
                        symbol: {
                            side: datetime.fromisoformat(time_str) if time_str else None
                            for side, time_str in times.items()
                        }
                        for symbol, times in data.get('last_trend_catch_time', {}).items()
                    }
                    self.trend_catch_count = data.get('trend_catch_count', {})
                logger.info("仓位状态加载成功")
        except Exception as e:
            logger.error(f"加载仓位状态时出错: {e}")
    
    def save_positions(self):
        """保存仓位状态到文件"""
        try:
            data = {
                'positions': self.positions,
                'current_layers': self.current_layers,
                'last_layer_time': {
                    symbol: {
                        side: time.isoformat() if time else None
                        for side, time in times.items()
                    }
                    for symbol, times in self.last_layer_time.items()
                },
                'last_trend_catch_time': {
                    symbol: {
                        side: time.isoformat() if time else None
                        for side, time in times.items()
                    }
                    for symbol, times in self.last_trend_catch_time.items()
                },
                'trend_catch_count': self.trend_catch_count
            }
            with open(self.positions_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"保存仓位状态时出错: {e}")
    
    def get_add_interval(self, current_layer: int) -> int:
        """根据当前层级动态计算加仓间隔"""
        if current_layer <= 3:  # 前3层较短间隔
            return INITIAL_ADD_INTERVAL
        else:
            return LATER_ADD_INTERVAL
    
    def should_add_layer(self, symbol: str, side: str, unrealized_pnl_percent: float) -> bool:
        """检查是否应该加仓"""
        if symbol not in self.current_layers:
            self.current_layers[symbol] = {"long": 0, "short": 0}
        
        current_layer = self.current_layers[symbol][side]
        
        # 如果已经达到最大层数，不再加仓
        if current_layer >= MAX_LAYERS:
            return False
        
        # 检查浮亏是否达到加仓阈值
        threshold = POSITION_SIZES[current_layer]
        return unrealized_pnl_percent <= -threshold
    
    def add_layer(self, symbol: str, side: str, current_price: float, api: BinanceFutureAPI):
        """执行加仓操作"""
        if symbol not in self.current_layers:
            self.current_layers[symbol] = {"long": 0, "short": 0}
        
        current_layer = self.current_layers[symbol][side]
        
        # 计算加仓大小（起始8U，每次翻倍）
        layer_size = BASE_TRADE_SIZE * (2 ** current_layer)
        
        # 计算购买数量
        amount = layer_size / current_price
        
        # 执行加仓操作
        order_side = "buy" if side == "long" else "sell"
        order = api.place_order(symbol, order_side, amount)
        
        if order:
            # 更新层级计数
            self.current_layers[symbol][side] += 1
            
            # 记录加仓时间
            current_time = datetime.now()
            if symbol not in self.last_layer_time:
                self.last_layer_time[symbol] = {"long": current_time, "short": current_time}
            else:
                self.last_layer_time[symbol][side] = current_time
            
            # 发送通知
            message = f"✅ {symbol} {side}方向第{current_layer+1}层加仓\n"
            message += f"加仓金额: {layer_size} USDT\n"
            message += f"当前价格: {current_price}"
            if self.telegram:
                self.telegram.send_message(message)
            
            # 保存仓位状态
            self.save_positions()
            
            return True
        return False
    
    def check_add_condition(self, symbol: str, positions: dict, current_price: float, api: BinanceFutureAPI):
        """检查是否满足加仓条件"""
        for side in ["long", "short"]:
            if side in positions and positions[side]:
                # 计算平均开仓价格和总数量
                total_amount = sum(float(pos["positionAmt"]) for pos in positions[side])
                if total_amount == 0:
                    continue
                
                avg_price = sum(float(pos["positionAmt"]) * float(pos["entryPrice"]) for pos in positions[side]) / total_amount
                
                # 计算浮亏百分比
                if side == "long":
                    unrealized_pnl_percent = (current_price - avg_price) / avg_price * 100 * LEVERAGE
                else:  # short
                    unrealized_pnl_percent = (avg_price - current_price) / avg_price * 100 * LEVERAGE
                
                # 检查是否应该加仓
                if self.should_add_layer(symbol, side, unrealized_pnl_percent):
                    # 检查加仓冷却时间
                    current_time = datetime.now()
                    last_add_time = self.last_layer_time.get(symbol, {}).get(side)
                    
                    if last_add_time:
                        # 计算冷却时间
                        current_layer = self.current_layers.get(symbol, {}).get(side, 0)
                        cooldown_hours = self.get_add_interval(current_layer)
                        cooldown = timedelta(hours=cooldown_hours)
                        
                        if current_time - last_add_time < cooldown:
                            continue  # 还在冷却期内，不加仓
                    
                    # 执行加仓
                    self.add_layer(symbol, side, current_price, api)
    
    def check_take_profit(self, symbol: str, positions: dict, current_price: float, api: BinanceFutureAPI):
        """检查是否满足止盈条件"""
        for side in ["long", "short"]:
            if side in positions and positions[side]:
                # 计算平均开仓价格和总数量
                total_amount = sum(float(pos["positionAmt"]) for pos in positions[side])
                if total_amount == 0:
                    continue
                
                avg_price = sum(float(pos["positionAmt"]) * float(pos["entryPrice"]) for pos in positions[side]) / total_amount
                
                # 计算浮盈百分比
                if side == "long":
                    unrealized_pnl_percent = (current_price - avg_price) / avg_price * 100 * LEVERAGE
                else:  # short
                    unrealized_pnl_percent = (avg_price - current_price) / avg_price * 100 * LEVERAGE
                
                # 检查是否达到止盈点
                if unrealized_pnl_percent >= TP_PERCENT * 100:
                    # 平仓
                    order = api.close_position(symbol, side, total_amount)
                    if order:
                        # 重置层级计数
                        if symbol in self.current_layers:
                            self.current_layers[symbol][side] = 0
                        
                        # 发送通知
                        message = f"✅ {symbol} {side}方向止盈平仓\n"
                        message += f"盈利: {unrealized_pnl_percent:.2f}%\n"
                        message += f"平仓价格: {current_price}"
                        if self.telegram:
                            self.telegram.send_message(message)
                        
                        # 保存仓位状态
                        self.save_positions()
    
    def check_stop_loss(self, symbol: str, positions: dict, current_price: float, api: BinanceFutureAPI):
        """检查是否满足止损条件"""
        for side in ["long", "short"]:
            if side in positions and positions[side]:
                # 计算总亏损
                total_pnl = sum(float(pos["unrealizedPnl"]) for pos in positions[side])
                
                # 检查是否达到止损点
                if total_pnl <= STOP_LOSS_PER_SYMBOL:
                    # 计算总数量
                    total_amount = sum(float(pos["positionAmt"]) for pos in positions[side])
                    
                    # 平仓
                    order = api.close_position(symbol, side, total_amount)
                    if order:
                        # 重置层级计数
                        if symbol in self.current_layers:
                            self.current_layers[symbol][side] = 0
                        
                        # 发送通知
                        message = f"⚠️ {symbol} {side}方向止损平仓\n"
                        message += f"亏损: {total_pnl:.2f} USDT\n"
                        message += f"平仓价格: {current_price}"
                        if self.telegram:
                            self.telegram.send_message(message)
                        
                        # 保存仓位状态
                        self.save_positions()

# ================== 主交易机器人 ==================
class CoinTech2uBot:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.api = BinanceFutureAPI(BINANCE_API_KEY, BINANCE_API_SECRET, symbols)
        self.telegram = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID) if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID else None
        self.martingale_manager = DualMartingaleManager(self.telegram)
        self.running = True
        
        # 注册信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """处理终止信号"""
        logger.info("收到终止信号，正在关闭...")
        self.running = False
    
    def run(self):
        """主循环"""
        logger.info("交易机器人启动")
        
        if self.telegram:
            self.telegram.send_message("🚀 交易机器人启动")
        
        while self.running:
            try:
                # 检查每个交易对
                for symbol in self.symbols:
                    self.check_symbol(symbol)
                
                # 等待下一次检查
                time.sleep(POLL_INTERVAL)
            except Exception as e:
                logger.error(f"主循环出错: {e}")
                time.sleep(POLL_INTERVAL)
        
        logger.info("交易机器人停止")
        
        if self.telegram:
            self.telegram.send_message("🛑 交易机器人停止")
    
    def check_symbol(self, symbol: str):
        """检查单个交易对"""
        try:
            # 获取当前价格
            ticker = self.api.get_ticker(symbol)
            if not ticker:
                return
            
            current_price = ticker['last']
            
            # 获取仓位信息
            position = self.api.get_position(symbol)
            
            # 组织仓位数据
            positions = {"long": [], "short": []}
            if position:
                side = "long" if float(position["positionAmt"]) > 0 else "short"
                positions[side].append(position)
            
            # 检查加仓条件
            self.martingale_manager.check_add_condition(symbol, positions, current_price, self.api)
            
            # 检查止盈条件
            self.martingale_manager.check_take_profit(symbol, positions, current_price, self.api)
            
            # 检查止损条件
            self.martingale_manager.check_stop_loss(symbol, positions, current_price, self.api)
            
            # 如果有趋势捕捉功能，检查趋势
            if ENABLE_TREND_CATCH:
                self.check_trend_catch(symbol, positions, current_price)
        
        except Exception as e:
            logger.error(f"检查 {symbol} 时出错: {e}")
    
    def check_trend_catch(self, symbol: str, positions: dict, current_price: float):
        """检查趋势捕捉加仓条件"""
        try:
            # 获取K线数据
            df = self.api.get_ohlcv(symbol, TIMEFRAME)
            if df is None or len(df) < 50:
                return
            
            # 分析趋势
            trend_strength, trend_direction = analyze_trend(df)
            
            # 检查是否有强趋势信号
            if trend_strength > TREND_SIGNAL_STRENGTH:
                for side in ["long", "short"]:
                    # 只对与趋势方向一致的仓位进行趋势加仓
                    if (side == "long" and trend_direction == "bullish") or (side == "short" and trend_direction == "bearish"):
                        if side in positions and positions[side]:
                            # 检查冷却时间
                            current_time = datetime.now()
                            last_trend_catch_time = self.martingale_manager.last_trend_catch_time.get(symbol, {}).get(side)
                            
                            if last_trend_catch_time:
                                cooldown = timedelta(hours=TREND_COOLDOWN_HOURS)
                                if current_time - last_trend_catch_time < cooldown:
                                    continue  # 还在冷却期内
                            
                            # 检查趋势加仓次数
                            if symbol not in self.martingale_manager.trend_catch_count:
                                self.martingale_manager.trend_catch_count[symbol] = {"long": 0, "short": 0}
                            
                            if self.martingale_manager.trend_catch_count[symbol][side] < TREND_CATCH_LAYERS:
                                # 执行趋势加仓
                                layer_size = BASE_TRADE_SIZE * TREND_CATCH_SIZES[self.martingale_manager.trend_catch_count[symbol][side]]
                                amount = layer_size / current_price
                                
                                order_side = "buy" if side == "long" else "sell"
                                order = self.api.place_order(symbol, order_side, amount)
                                
                                if order:
                                    # 更新趋势加仓计数
                                    self.martingale_manager.trend_catch_count[symbol][side] += 1
                                    
                                    # 记录趋势加仓时间
                                    if symbol not in self.martingale_manager.last_trend_catch_time:
                                        self.martingale_manager.last_trend_catch_time[symbol] = {"long": current_time, "short": current_time}
                                    else:
                                        self.martingale_manager.last_trend_catch_time[symbol][side] = current_time
                                    
                                    # 发送通知
                                    message = f"🎯 {symbol} {side}方向趋势捕捉加仓\n"
                                    message += f"加仓金额: {layer_size} USDT\n"
                                    message += f"趋势强度: {trend_strength:.2f}\n"
                                    message += f"当前价格: {current_price}"
                                    if self.telegram:
                                        self.telegram.send_message(message)
                                    
                                    # 保存仓位状态
                                    self.martingale_manager.save_positions()
        
        except Exception as e:
            logger.error(f"检查 {symbol} 趋势捕捉时出错: {e}")

# ================== 启动程序 ==================
def main():
    bot = CoinTech2uBot(SYMBOLS_CONFIG)
    try:
        bot.run()
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
    
    main()
