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
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ================== 配置参数 ==================
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
SYMBOLS_CONFIG = [s.strip() for s in os.getenv("SYMBOLS", "ETH/USDT,LTC/USDT,BNB/USDT,DOGE/USDT,XRP/USDT,SOL/USDT,AVAX/USDT,ADA/USDT,LINK/USDT,UNI/USDT,SUI/USDT").split(",") if s.strip()]
TIMEFRAME = os.getenv("MACD_FILTER_TIMEFRAME", "4h")
LEVERAGE = int(os.getenv("LEVERAGE", "15"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))

# 每一层的触发百分比（负数代表下跌触发）
LAYER_TRIGGER_PCTS = [-0.02678, -0.05, -0.06, -0.07, -0.08, -0.09, -0.10, -0.13, -0.14]
MAX_LAYERS = len(LAYER_TRIGGER_PCTS)  # 最大层数等于触发百分比的数量

# 基础开仓金额和翻倍逻辑
BASE_LAYER_SIZE = float(os.getenv("BASE_LAYER_SIZE", "6"))  # 第一层的金额(USDT)

# 止损止盈参数
STOP_LOSS = float(os.getenv("STOP_LOSS", "-100"))  # 固定金额止损
TP_PERCENT = float(os.getenv("TP_PERCENT", "1.5").strip('%')) / 100  # 百分比止盈

# 加仓间隔时间(分钟)
MIN_LAYER_INTERVAL = int(os.getenv("MIN_LAYER_INTERVAL", "240"))

# 冷静期参数
MAX_DAILY_LAYERS = int(os.getenv("MAX_DAILY_LAYERS", "3"))  # 每天最大加仓次数
COOLDOWN_HOURS = int(os.getenv("COOLDOWN_HOURS", "24"))  # 冷静期小时数

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
            # 回退到精度
            prec = market.get('precision', {}).get('amount')
            if isinstance(prec, int):
                return round(amount, prec)
            # 默认精度
            return float(Decimal(amount).quantize(Decimal('0.000001'), rounding=ROUND_DOWN))
        
        # 使用Decimal进行精确计算
        step_dec = Decimal(str(step))
        amount_dec = Decimal(str(amount))
        # 向下取整到step的倍数
        quantized = (amount_dec // step_dec) * step_dec
        return float(quantized)
    except Exception as e:
        logger.error(f"量化数量失败: {e}")
        # 回退到简单舍入
        return round(amount, 6)

# ================== 交易所接口 ==================
class BinanceFutureAPI:
    def __init__(self, api_key: str, api_secret: str, symbols: List[str]):
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbols = symbols
        self.exchange = None
        self.symbol_info = {}  # 缓存交易对信息

    def initialize(self) -> bool:
        try:
            self.exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'options': {'defaultType': 'future'},
                'enableRateLimit': True
            })
            
            # 加载所有交易对信息
            markets = self.exchange.load_markets()
            valid_symbols = []
            
            for symbol in self.symbols:
                if symbol in markets:
                    self.symbol_info[symbol] = markets[symbol]
                    try:
                        # 设置杠杆
                        self.exchange.set_leverage(LEVERAGE, symbol)
                        logger.info(f"设置杠杆 {symbol} {LEVERAGE}x")
                        
                        # 设置对冲模式
                        self.exchange.set_position_mode(True, symbol)
                        logger.info(f"设置对冲模式 {symbol}")
                        
                        valid_symbols.append(symbol)
                    except Exception as e:
                        logger.warning(f"设置杠杆或对冲模式失败 {symbol}: {e}")
                        # 即使设置失败，也继续使用该交易对
                        valid_symbols.append(symbol)
                else:
                    logger.warning(f"交易对 {symbol} 不存在，跳过")
            
            # 更新有效的交易对列表
            self.symbols = valid_symbols
            
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

    def get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker['last'])
        except Exception as e:
            logger.error(f"获取价格失败 {symbol}: {e}")
            return None

    def get_positions(self, symbol: str) -> Dict[str, dict]:
        """获取当前持仓信息"""
        try:
            positions = self.exchange.fetch_positions([symbol])
            result = {}
            for pos in positions:
                if float(pos['contracts']) > 0:
                    side = pos['side'].lower()
                    result[side] = {
                        'size': float(pos['contracts']),
                        'entry_price': float(pos['entryPrice']),
                        'side': pos['side'],
                        'position_side': pos['positionSide']
                    }
            return result
        except Exception as e:
            logger.error(f"获取持仓失败 {symbol}: {e}")
            return {}

    def create_order_with_retry(self, symbol: str, side: str, contract_size: float, position_side: str):
        """创建订单，带重试机制"""
        for attempt in range(MAX_RETRIES):
            try:
                # 尝试带positionSide下单
                params = {"positionSide": position_side}
                order = self.exchange.create_order(
                    symbol,
                    'market',
                    side.lower(),
                    contract_size,
                    None,
                    params
                )
                return order
            except Exception as e:
                err_msg = str(e)
                logger.error(f"下单失败 (尝试 {attempt+1}/{MAX_RETRIES}): {e}")
                
                # 如果是position side不匹配的错误，尝试不带positionSide下单
                if "-4061" in err_msg or "position side does not match" in err_msg.lower():
                    logger.warning(f"positionSide与账户设置不符，尝试不带positionSide重试")
                    try:
                        order = self.exchange.create_order(
                            symbol,
                            'market',
                            side.lower(),
                            contract_size
                        )
                        return order
                    except Exception as e2:
                        logger.error(f"重试不带positionSide失败: {e2}")
                        if attempt == MAX_RETRIES - 1:
                            return None
                else:
                    if attempt == MAX_RETRIES - 1:
                        return None
            
            # 等待一段时间后重试
            time.sleep(RETRY_DELAY * (2 ** attempt))
        
        return None

    def execute_market_order(self, symbol: str, side: str, amount: float, position_side: str) -> bool:
        try:
            # 获取交易对信息
            market = self.symbol_info.get(symbol)
            if not market:
                logger.error(f"找不到交易对信息: {symbol}")
                return False
                
            # 获取当前价格
            current_price = self.get_current_price(symbol)
            if current_price is None:
                logger.error(f"无法获取 {symbol} 的价格")
                return False
                
            # 计算合约数量
            contract_size = amount / current_price
            
            # 量化到交易所精度
            contract_size = quantize_amount(contract_size, market)
            
            # 确保不低于最小交易量
            min_amount = market['limits']['amount']['min']
            if contract_size < min_amount:
                contract_size = min_amount
                logger.warning(f"交易量低于最小值，使用最小值: {min_amount}")

            # 检查最小名义价值
            min_notional = MIN_NOTIONAL.get(symbol, 20)  # 默认20 USDT
            notional_value = contract_size * current_price
            if notional_value < min_notional:
                # 调整合约数量以满足最小名义价值要求
                contract_size = min_notional / current_price
                contract_size = quantize_amount(contract_size, market)
                logger.warning(f"名义价值 {notional_value:.2f} USDT 低于最小值 {min_notional} USDT，调整合约数量为 {contract_size:.6f}")
            
            # 创建订单
            order = self.create_order_with_retry(symbol, side, contract_size, position_side)
            if order:
                logger.info(f"订单成功 {symbol} {side} {contract_size:.6f} ({position_side}) - 订单ID: {order['id']}")
                return True
            else:
                logger.error(f"下单失败 {symbol} {side}: 所有重试均失败")
                return False
                
        except Exception as e:
            logger.error(f"下单失败 {symbol} {side}: {e}")
            return False

# ================== 双仓马丁策略管理 ==================
class DualMartingaleManager:
    def __init__(self, api):
        self.api = api
        # 仓位结构: {symbol: {'long': [], 'short': []}}
        self.positions: Dict[str, Dict[str, List[dict]]] = {}
        # 最后加仓时间: {symbol: {'long': datetime, 'short': datetime}}
        self.last_layer_time: Dict[str, Dict[str, datetime]] = {}
        # 每日加仓次数: {symbol: {'long': count, 'short': count}}
        self.daily_layer_count: Dict[str, Dict[str, int]] = {}
        # 冷静期开始时间: {symbol: {'long': datetime, 'short': datetime}}
        self.cooldown_start: Dict[str, Dict[str, datetime]] = {}
        # 仓位状态文件
        self.positions_file = "positions.json"
        # 加载保存的仓位
        self.load_positions()

    def initialize_symbol(self, symbol: str):
        """初始化交易对仓位结构"""
        if symbol not in self.positions:
            self.positions[symbol] = {'long': [], 'short': []}
        if symbol not in self.last_layer_time:
            self.last_layer_time[symbol] = {'long': None, 'short': None}
        if symbol not in self.daily_layer_count:
            self.daily_layer_count[symbol] = {'long': 0, 'short': 0}
        if symbol not in self.cooldown_start:
            self.cooldown_start[symbol] = {'long': None, 'short': None}

    def calculate_safe_position_size(self, symbol: str, target_size_usdt: float, current_price: float) -> float:
        """计算安全的仓位大小，确保不低于最小名义价值"""
        # 获取该交易对的最小名义价值要求
        min_notional = MIN_NOTIONAL.get(symbol, 20)  # 默认20 USDT
        
        # 如果目标USDT价值小于最小名义价值，使用最小名义价值
        if target_size_usdt < min_notional:
            logger.warning(f"⚠️ 目标仓位 {target_size_usdt:.2f} USDT 低于最小值 {min_notional} USDT，使用最小值")
            safe_size_usdt = min_notional
        else:
            safe_size_usdt = target_size_usdt
        
        # 转换为币的数量
        size = safe_size_usdt / current_price
        
        logger.info(f"📏 安全仓位计算: USDT价值={safe_size_usdt:.2f}, 数量={size:.6f}")
        return size

    def add_position(self, symbol: str, side: str, size: float, price: float):
        """添加仓位到对应方向"""
        self.initialize_symbol(symbol)
        position_side = 'long' if side.lower() == 'buy' else 'short'
        layer = len(self.positions[symbol][position_side]) + 1
        
        # 如果是加仓（不是初始开仓），增加每日加仓计数
        if layer > 1:
            self.daily_layer_count[symbol][position_side] += 1
            logger.info(f"📊 {symbol} {position_side.upper()} 今日已加仓 {self.daily_layer_count[symbol][position_side]} 次")
            
            # 检查是否达到每日加仓上限
            if self.daily_layer_count[symbol][position_side] >= MAX_DAILY_LAYERS:
                logger.warning(f"⏳ {symbol} {position_side.upper()} 今日加仓已达 {MAX_DAILY_LAYERS} 次上限，进入 {COOLDOWN_HOURS} 小时冷静期")
                self.cooldown_start[symbol][position_side] = datetime.now()
        
        self.positions[symbol][position_side].append({
            'side': side,
            'size': size,
            'entry_price': price,
            'timestamp': datetime.now(),
            'layer': layer
        })
        
        self.last_layer_time[symbol][position_side] = datetime.now()
        logger.info(f"📊 {symbol} {position_side.upper()} 第{layer}层仓位: {side} {size:.6f} @ {price:.2f}")
        
        # 保存仓位状态
        self.save_positions()

    def is_in_cooldown(self, symbol: str, position_side: str) -> bool:
        """检查是否处于冷静期"""
        self.initialize_symbol(symbol)
        
        cooldown_start = self.cooldown_start[symbol][position_side]
        if cooldown_start is None:
            return False
            
        # 检查冷静期是否已过
        if datetime.now() - cooldown_start >= timedelta(hours=COOLDOWN_HOURS):
            # 冷静期结束，重置计数和冷静期
            self.daily_layer_count[symbol][position_side] = 0
            self.cooldown_start[symbol][position_side] = None
            logger.info(f"✅ {symbol} {position_side.upper()} 冷静期结束，可以重新加仓")
            return False
            
        # 仍在冷静期中
        remaining_time = cooldown_start + timedelta(hours=COOLDOWN_HOURS) - datetime.now()
        logger.info(f"⏳ {symbol} {position_side.upper()} 处于冷静期，剩余时间: {remaining_time}")
        return True

    def should_add_layer(self, symbol: str, position_side: str, current_price: float) -> bool:
        """检查是否应该加仓"""
        self.initialize_symbol(symbol)
        
        # 检查是否处于冷静期
        if self.is_in_cooldown(symbol, position_side):
            return False
            
        # 检查是否已达到最大层数
        positions = self.positions[symbol][position_side]
        if len(positions) >= MAX_LAYERS:
            logger.info(f"⚠️ {symbol} {position_side.upper()} 已达到最大层数 {MAX_LAYERS}")
            return False
            
        # 检查加仓时间间隔
        last_time = self.last_layer_time[symbol][position_side]
        if last_time and (datetime.now() - last_time) < timedelta(minutes=MIN_LAYER_INTERVAL):
            logger.info(f"⏰ {symbol} {position_side.upper()} 加仓间隔时间不足，跳过加仓")
            return False
            
        if not positions:
            return False
            
        total_size = sum(p['size'] for p in positions)
        total_value = sum(p['size'] * p['entry_price'] for p in positions)
        avg_price = total_value / total_size
        
        # 计算当前盈亏百分比
        if position_side == 'long':
            pnl_pct = (current_price - avg_price) / avg_price
        else:  # short
            pnl_pct = (avg_price - current_price) / avg_price
        
        # 获取当前层数对应的触发百分比
        current_layer = len(positions)
        if current_layer >= len(LAYER_TRIGGER_PCTS):
            return False
            
        trigger_pct = LAYER_TRIGGER_PCTS[current_layer]
        
        logger.info(f"📈 {symbol} {position_side.upper()} 当前盈亏: {pnl_pct*100:.2f}%, 触发阈值: {trigger_pct*100:.2f}%")
        
        # 只有当亏损达到触发阈值时才加仓
        return pnl_pct <= trigger_pct

    def calculate_layer_size(self, symbol: str, position_side: str, current_price: float) -> float:
        """计算加仓大小 - 使用翻倍逻辑"""
        self.initialize_symbol(symbol)
        layer = len(self.positions[symbol][position_side]) + 1
        
        # 资金翻倍逻辑
        size_in_usdt = BASE_LAYER_SIZE * (2 ** (layer - 1))
        
        logger.info(f"📏 {symbol} {position_side.upper()} 第{layer}层仓位金额: {size_in_usdt} USDT")
        
        # 使用安全仓位计算，确保不低于最小名义价值
        return self.calculate_safe_position_size(symbol, size_in_usdt, current_price)

    def calculate_initial_size(self, symbol: str, current_price: float) -> float:
        """计算初始仓位大小 - 使用基础金额"""
        logger.info(f"📏 初始仓位计算: 固定USDT价值={BASE_LAYER_SIZE:.2f}")
        
        # 使用安全仓位计算，确保不低于最小名义价值
        return self.calculate_safe_position_size(symbol, BASE_LAYER_SIZE, current_price)
        
    def should_close_position(self, symbol: str, position_side: str, current_price: float) -> bool:
        """检查是否应该平仓（固定金额止损和百分比止盈）"""
        self.initialize_symbol(symbol)
        if not self.positions[symbol][position_side]:
            return False
            
        positions = self.positions[symbol][position_side]
        total_size = sum(p['size'] for p in positions)
        total_value = sum(p['size'] * p['entry_price'] for p in positions)
        avg_price = total_value / total_size
        
        # 计算当前盈亏金额（USDT）
        if position_side == 'long':
            pnl_amount = (current_price - avg_price) * total_size
        else:  # short
            pnl_amount = (avg_price - current_price) * total_size
            
        # 如果亏损超过100 USDT，强制平仓
        if pnl_amount <= STOP_LOSS:
            logger.warning(f"🚨 {symbol} {position_side.upper()} 亏损超过{abs(STOP_LOSS)} USDT，强制平仓")
            return True
            
        # 如果盈利超过止盈点，止盈平仓
        # 这里计算从第一层入场价到当前价的盈利百分比
        first_entry_price = positions[0]['entry_price']
        if position_side == 'long':
            profit_pct = (current_price - first_entry_price) / first_entry_price
        else:  # short
            profit_pct = (first_entry_price - current_price) / first_entry_price
            
        if profit_pct >= TP_PERCENT:
            logger.info(f"🎯 {symbol} {position_side.upper()} 盈利超过{TP_PERCENT*100:.2f}%，止盈平仓")
            return True
            
        return False

    def get_position_size(self, symbol: str, position_side: str) -> float:
        """获取某个方向的仓位总大小"""
        self.initialize_symbol(symbol)
        return sum(p['size'] for p in self.positions[symbol][position_side])
    
    def clear_positions(self, symbol: str, position_side: str):
        """清空某个方向的仓位记录"""
        self.initialize_symbol(symbol)
        self.positions[symbol][position_side] = []
        logger.info(f"🔄 {symbol} {position_side.upper()} 仓位记录已清空")
        # 保存仓位状态
        self.save_positions()
        
    def has_open_positions(self, symbol: str) -> bool:
        """检查是否有任何方向的仓位"""
        self.initialize_symbol(symbol)
        return len(self.positions[symbol]['long']) > 0 or len(self.positions[symbol]['short']) > 0
    
    def sync_with_exchange(self, symbol: str):
        """同步交易所仓位和本地记录"""
        try:
            # 获取交易所的实际仓位
            exchange_positions = self.api.get_positions(symbol)
            
            # 检查多仓
            long_position = exchange_positions.get('long')
            local_long_size = self.get_position_size(symbol, 'long')
            
            # 如果交易所没有多仓但本地记录有，清空本地记录
            if (not long_position or long_position['size'] == 0) and local_long_size > 0:
                logger.info(f"🔄 {symbol} 交易所多仓位已平，清空本地记录")
                self.clear_positions(symbol, 'long')
            
            # 检查空仓
            short_position = exchange_positions.get('short')
            local_short_size = self.get_position_size(symbol, 'short')
            
            # 如果交易所没有空仓但本地记录有，清空本地记录
            if (not short_position or short_position['size'] == 0) and local_short_size > 0:
                logger.info(f"🔄 {symbol} 交易所空仓位已平，清空本地记录")
                self.clear_positions(symbol, 'short')
                
        except Exception as e:
            logger.error(f"同步仓位失败 {symbol}: {e}")
    
    def save_positions(self):
        """保存仓位状态到文件"""
        try:
            # 转换为可序列化的格式
            serializable_data = {
                'positions': {},
                'daily_layer_count': self.daily_layer_count,
                'cooldown_start': {}
            }
            
            for symbol, sides in self.positions.items():
                serializable_data['positions'][symbol] = {}
                for side, positions in sides.items():
                    serializable_data['positions'][symbol][side] = []
                    for pos in positions:
                        serializable_data['positions'][symbol][side].append({
                            'side': pos['side'],
                            'size': pos['size'],
                            'entry_price': pos['entry_price'],
                            'timestamp': pos['timestamp'].isoformat(),
                            'layer': pos['layer']
                        })
            
            # 转换cooldown_start
            for symbol, sides in self.cooldown_start.items():
                serializable_data['cooldown_start'][symbol] = {}
                for side, start_time in sides.items():
                    serializable_data['cooldown_start'][symbol][side] = start_time.isoformat() if start_time else None
            
            with open(self.positions_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
        except Exception as e:
            logger.error(f"保存仓位状态失败: {e}")
    
    def load_positions(self):
        """从文件加载仓位状态"""
        try:
            if os.path.exists(self.positions_file):
                with open(self.positions_file, 'r') as f:
                    serializable_data = json.load(f)
                
                # 转换回原始格式
                if 'positions' in serializable_data:
                    for symbol, sides in serializable_data['positions'].items():
                        self.positions[symbol] = {}
                        for side, positions in sides.items():
                            self.positions[symbol][side] = []
                            for pos in positions:
                                self.positions[symbol][side].append({
                                    'side': pos['side'],
                                    'size': pos['size'],
                                    'entry_price': pos['entry_price'],
                                    'timestamp': datetime.fromisoformat(pos['timestamp']),
                                    'layer': pos['layer']
                                })
                
                # 加载每日加仓计数
                if 'daily_layer_count' in serializable_data:
                    for symbol, sides in serializable_data['daily_layer_count'].items():
                        self.daily_layer_count[symbol] = sides
                
                # 加载冷静期开始时间
                if 'cooldown_start' in serializable_data:
                    for symbol, sides in serializable_data['cooldown_start'].items():
                        self.cooldown_start[symbol] = {}
                        for side, start_time_str in sides.items():
                            self.cooldown_start[symbol][side] = datetime.fromisoformat(start_time_str) if start_time_str else None
                
                logger.info("仓位状态已从文件加载")
        except Exception as e:
            logger.error(f"加载仓位状态失败: {e}")

# ================== 主交易机器人 ==================
class CoinTech2uBot:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.api = BinanceFutureAPI(BINANCE_API_KEY, BINANCE_API_SECRET, symbols)
        self.martingale = DualMartingaleManager(self.api)
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
            
        logger.info("🚀 开始CoinTech2u策略交易...")
        
        # 程序启动时立即对所有币对开双仓
        logger.info("🔄 程序启动时对所有币对开双仓")
        for symbol in self.symbols:
            await self.open_immediate_hedge(symbol)
        
        while self.running:
            try:
                balance = self.api.get_balance()
                logger.info(f"当前余额: {balance:.2f} USDT")
                
                for symbol in self.symbols:
                    await self.process_symbol(symbol)
                    
                await asyncio.sleep(POLL_INTERVAL)
            except Exception as e:
                logger.error(f"交易循环错误: {e}")
                await asyncio.sleep(10)

    async def open_immediate_hedge(self, symbol: str):
        """程序启动时立即开双仓"""
        # 先同步交易所仓位状态
        self.martingale.sync_with_exchange(symbol)
        
        # 如果已经有仓位，不需要再开
        if self.martingale.has_open_positions(symbol):
            logger.info(f"⏩ {symbol} 已有仓位，跳过开仓")
            return
        
        # 获取当前价格
        current_price = self.api.get_current_price(symbol)
        if current_price is None:
            logger.error(f"无法获取 {symbol} 的价格，跳过")
            return
        
        # 计算初始仓位大小
        position_size = self.martingale.calculate_initial_size(symbol, current_price)
        if position_size <= 0:
            logger.error(f"{symbol} 仓位大小计算错误，跳过")
            return
        
        logger.info(f"📊 {symbol} 准备开双仓，价格: {current_price:.2f}, 大小: {position_size:.6f}")
        
        # 同时开多仓和空仓
        long_success = self.api.execute_market_order(symbol, "buy", position_size, "LONG")
        short_success = self.api.execute_market_order(symbol, "sell", position_size, "SHORT")
        
        if long_success and short_success:
            logger.info(f"✅ {symbol} 已同时开多空仓位: 多单 {position_size:.6f} | 空单 {position_size:.6f}")
            # 记录仓位
            self.martingale.add_position(symbol, "buy", position_size, current_price)
            self.martingale.add_position(symbol, "sell", position_size, current_price)
        else:
            logger.error(f"❌ {symbol} 开仓失败，需要手动检查")

    async def process_symbol(self, symbol: str):
        """处理单个交易对的交易逻辑"""
        # 先同步交易所仓位状态
        self.martingale.sync_with_exchange(symbol)
        
        # 如果没有仓位，重新开仓
        if not self.martingale.has_open_positions(symbol):
            logger.info(f"🔄 {symbol} 检测到无仓位，重新开双仓")
            await self.open_immediate_hedge(symbol)
            return
        
        # 获取当前价格
        current_price = self.api.get_current_price(symbol)
        if current_price is None:
            return
        
        # 检查是否需要平仓（止损或止盈）
        for position_side in ['long', 'short']:
            if self.martingale.should_close_position(symbol, position_side, current_price):
                await self.close_and_reopen_position(symbol, position_side, current_price)
        
        # 检查是否需要加仓
        for position_side in ['long', 'short']:
            if self.martingale.should_add_layer(symbol, position_side, current_price):
                await self.add_martingale_layer(symbol, position_side, current_price)

    async def add_martingale_layer(self, symbol: str, position_side: str, current_price: float):
        """为指定方向加仓"""
        positions = self.martingale.positions[symbol][position_side]
        if not positions:
            return
            
        side = "buy" if position_side == "long" else "sell"
        position_side_param = "LONG" if position_side == "long" else "SHORT"
        layer_size = self.martingale.calculate_layer_size(symbol, position_side, current_price)
        
        logger.info(f"📈 {symbol} {position_side.upper()} 准备加仓第{len(positions)+1}层，方向: {side}, 大小: {layer_size:.6f}")
        
        success = self.api.execute_market_order(symbol, side, layer_size, position_side_param)
        if success:
            self.martingale.add_position(symbol, side, layer_size, current_price)

    async def close_and_reopen_position(self, symbol: str, position_side: str, current_price: float):
        """平掉指定方向的所有仓位并立即重新开仓"""
        position_size = self.martingale.get_position_size(symbol, position_side)
        if position_size <= 0:
            return
            
        # 平仓方向与开仓方向相反
        if position_side == "long":
            close_side = "sell"
            position_side_param = "LONG"
            reopen_side = "buy"
        else:  # short
            close_side = "buy"
            position_side_param = "SHORT"
            reopen_side = "sell"
        
        logger.info(f"📤 {symbol} {position_side.upper()} 平仓，方向: {close_side}, 大小: {position_size:.6f}")
        
        # 平仓
        success = self.api.execute_market_order(symbol, close_side, position_size, position_side_param)
        if success:
            self.martingale.clear_positions(symbol, position_side)
            logger.info(f"✅ {symbol} {position_side.upper()} 所有仓位已平仓")
            
            # 等待一下再开新仓
            await asyncio.sleep(2)
            
            # 重新开仓
            new_position_size = self.martingale.calculate_initial_size(symbol, current_price)
            logger.info(f"🔄 {symbol} {position_side.upper()} 重新开仓，大小: {new_position_size:.6f}")
            
            reopen_success = self.api.execute_market_order(symbol, reopen_side, new_position_size, position_side_param)
            if reopen_success:
                self.martingale.add_position(symbol, reopen_side, new_position_size, current_price)
                logger.info(f"✅ {symbol} {position_side.upper()} 已重新开仓")
            else:
                logger.error(f"❌ {symbol} {position_side.upper()} 重新开仓失败")

# ================== 启动程序 ==================
async def main():
    bot = CoinTech2uBot(SYMBOLS_CONFIG)
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
        
    if not SYMBOLS_CONFIG:
        print("错误: 请设置 SYMBOLS 环境变量，例如: BTC/USDT,ETH/USDT")
        sys.exit(1)
        
    # 检查是否安装了必要的库
    try:
        import ccxt
        import pandas
        import numpy
        import ta
        import dotenv
    except ImportError as e:
        print(f"错误: 缺少必要的Python库: {e}")
        print("请运行: pip install ccxt pandas numpy ta python-dotenv")
        sys.exit(1)
        
    asyncio.run(main())
