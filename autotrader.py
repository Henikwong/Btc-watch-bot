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
BASE_TRADE_SIZE = float(os.getenv("BASE_TRADE_SIZE", "10"))  # 基础交易大小(USDT)

# 策略参数 - CoinTech2u核心逻辑
TAKE_PROFIT_PCT = 0.015  # 1.5%止盈
ADD_INTERVAL_HOURS = 12  # 加仓间隔12小时
MAX_LAYERS = 9  # 最大9层仓位

# cointech2u加仓倍数配置
MARTINGALE_MULTIPLIERS = [2.678, 5, 6, 7, 8, 9, 10, 13, 14]

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

    async def initialize(self) -> bool:
        """异步初始化交易所连接"""
        try:
            self.exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'options': {'defaultType': 'future'},
                'enableRateLimit': True
            })
            
            # 加载所有交易对信息
            markets = await self.exchange.load_markets()
            valid_symbols = []
            
            for symbol in self.symbols:
                if symbol in markets:
                    self.symbol_info[symbol] = markets[symbol]
                    try:
                        await self.exchange.set_leverage(LEVERAGE, symbol)
                        logger.info(f"设置杠杆 {symbol} {LEVERAGE}x")
                        valid_symbols.append(symbol)
                    except Exception as e:
                        logger.warning(f"设置杠杆失败 {symbol}: {e}")
                else:
                    logger.warning(f"交易对 {symbol} 不存在，跳过")
            
            # 更新有效的交易对列表
            self.symbols = valid_symbols
            
            logger.info("交易所初始化成功")
            return True
        except Exception as e:
            logger.error(f"交易所初始化失败: {e}")
            return False

    async def get_balance(self) -> float:
        """获取账户余额"""
        try:
            balance = await self.exchange.fetch_balance()
            return float(balance['USDT']['free'])
        except Exception as e:
            logger.error(f"获取余额失败: {e}")
            return 0.0

    async def get_ohlcv_data(self, symbol: str, timeframe: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """获取K线数据"""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"K线获取失败 {symbol}: {e}")
            return None

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格"""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return float(ticker['last'])
        except Exception as e:
            logger.error(f"获取价格失败 {symbol}: {e}")
            return None

    async def get_positions(self, symbol: str) -> Dict[str, dict]:
        """获取当前持仓信息"""
        try:
            positions = await self.exchange.fetch_positions([symbol])
            result = {}
            for pos in positions:
                if float(pos['contracts']) > 0:
                    # 使用side作为键，而不是positionSide
                    side = pos['side'].lower()
                    result[side] = {
                        'size': float(pos['contracts']),
                        'entry_price': float(pos['entryPrice']),
                        'side': pos['side'],
                    }
            return result
        except Exception as e:
            logger.error(f"获取持仓失败 {symbol}: {e}")
            return {}

    async def create_order_with_fallback(self, symbol: str, side: str, contract_size: float, position_side: str):
        """创建订单，如果失败则尝试回退到单向模式"""
        for attempt in range(MAX_RETRIES):
            try:
                # 尝试带positionSide下单
                params = {"positionSide": position_side}
                order = await self.exchange.create_order(
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
                # 如果是position side不匹配的错误，尝试不带positionSide下单
                if "-4061" in err_msg or "position side does not match" in err_msg.lower():
                    logger.warning(f"positionSide与账户设置不符，尝试不带positionSide重试")
                    try:
                        order = await self.exchange.create_order(
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
                    logger.error(f"下单失败: {e}")
                    if attempt == MAX_RETRIES - 1:
                        return None
            
            # 等待一段时间后重试
            await asyncio.sleep(RETRY_DELAY * (2 ** attempt))
        
        return None

    async def execute_market_order(self, symbol: str, side: str, amount: float, position_side: str) -> bool:
        """执行市价订单"""
        try:
            # 获取交易对信息
            market = self.symbol_info.get(symbol)
            if not market:
                logger.error(f"找不到交易对信息: {symbol}")
                return False
                
            # 获取当前价格
            current_price = await self.get_current_price(symbol)
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
            min_notional = MIN_NOTIONAL.get(symbol, 10)  # 默认10 USDT
            notional_value = contract_size * current_price
            if notional_value < min_notional:
                # 调整合约数量以满足最小名义价值要求
                contract_size = min_notional / current_price
                contract_size = quantize_amount(contract_size, market)
                logger.warning(f"名义价值 {notional_value:.2f} USDT 低于最小值 {min_notional} USDT，调整合约数量为 {contract_size:.6f}")
            
            # 创建订单
            order = await self.create_order_with_fallback(symbol, side, contract_size, position_side)
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
    def __init__(self):
        # 仓位结构: {symbol: {'long': [], 'short': []}}
        self.positions: Dict[str, Dict[str, List[dict]]] = {}
        # 最后加仓时间: {symbol: {'long': datetime, 'short': datetime}}
        self.last_layer_time: Dict[str, Dict[str, datetime]] = {}
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

    def add_position(self, symbol: str, side: str, size: float, price: float):
        """添加仓位到对应方向"""
        self.initialize_symbol(symbol)
        position_side = 'long' if side.lower() == 'buy' else 'short'
        layer = len(self.positions[symbol][position_side]) + 1
        
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

    def should_add_layer(self, symbol: str, position_side: str, current_price: float) -> bool:
        """检查是否应该加仓"""
        self.initialize_symbol(symbol)
        
        # 检查是否已达到最大层数
        if len(self.positions[symbol][position_side]) >= MAX_LAYERS:
            logger.info(f"⚠️ {symbol} {position_side.upper()} 已达到最大层数 {MAX_LAYERS}")
            return False
            
        # 检查加仓时间间隔
        last_time = self.last_layer_time[symbol][position_side]
        if last_time and (datetime.now() - last_time) < timedelta(hours=ADD_INTERVAL_HOURS):
            logger.info(f"⏰ {symbol} {position_side.upper()} 加仓间隔时间不足，跳过加仓")
            return False
            
        positions = self.positions[symbol][position_side]
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
            
        logger.info(f"📈 {symbol} {position_side.upper()} 当前盈亏: {pnl_pct*100:.2f}%")
        
        # 只有当亏损达到触发阈值时才加仓
        return pnl_pct <= -0.05  # 5%亏损触发加仓

    def calculate_layer_size(self, symbol: str, position_side: str, current_price: float) -> float:
        """计算加仓大小 - 使用cointech2u的层级配置"""
        self.initialize_symbol(symbol)
        layer = len(self.positions[symbol][position_side]) + 1
        
        # 使用cointech2u的层级配置
        if layer <= len(MARTINGALE_MULTIPLIERS):
            size_in_usdt = MARTINGALE_MULTIPLIERS[layer - 1]
        else:
            # 如果层级超过配置，使用最后一层的值
            size_in_usdt = MARTINGALE_MULTIPLIERS[-1]
            
        size = size_in_usdt / current_price
        
        logger.info(f"📏 {symbol} {position_side.upper()} 第{layer}层计算仓位: USDT价值={size_in_usdt:.3f}, 数量={size:.6f}")
        return size

    def calculate_initial_size(self, current_price: float) -> float:
        """计算初始仓位大小 - 使用cointech2u的初始配置"""
        # 使用cointech2u的初始配置
        size = BASE_TRADE_SIZE / current_price
        
        logger.info(f"📏 初始仓位计算: USDT价值={BASE_TRADE_SIZE:.3f}, 数量={size:.6f}")
        return size
        
    def should_close_position(self, symbol: str, position_side: str, current_price: float) -> bool:
        """检查是否应该平仓（止盈）"""
        self.initialize_symbol(symbol)
        if not self.positions[symbol][position_side]:
            return False
            
        positions = self.positions[symbol][position_side]
        total_size = sum(p['size'] for p in positions)
        total_value = sum(p['size'] * p['entry_price'] for p in positions)
        avg_price = total_value / total_size
        
        # 计算当前盈亏百分比
        if position_side == 'long':
            pnl_pct = (current_price - avg_price) / avg_price
        else:  # short
            pnl_pct = (avg_price - current_price) / avg_price
            
        # 如果盈利超过止盈点，止盈平仓
        if pnl_pct >= TAKE_PROFIT_PCT:
            logger.info(f"🎯 {symbol} {position_side.upper()} 盈利超过{TAKE_PROFIT_PCT*100:.2f}%，止盈平仓")
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
    
    def save_positions(self):
        """保存仓位状态到文件"""
        try:
            # 转换为可序列化的格式
            serializable_positions = {}
            for symbol, sides in self.positions.items():
                serializable_positions[symbol] = {}
                for side, positions in sides.items():
                    serializable_positions[symbol][side] = []
                    for pos in positions:
                        serializable_positions[symbol][side].append({
                            'side': pos['side'],
                            'size': pos['size'],
                            'entry_price': pos['entry_price'],
                            'timestamp': pos['timestamp'].isoformat(),
                            'layer': pos['layer']
                        })
            
            with open(self.positions_file, 'w') as f:
                json.dump(serializable_positions, f, indent=2)
        except Exception as e:
            logger.error(f"保存仓位状态失败: {e}")
    
    def load_positions(self):
        """从文件加载仓位状态"""
        try:
            if os.path.exists(self.positions_file):
                with open(self.positions_file, 'r') as f:
                    serializable_positions = json.load(f)
                
                # 转换回原始格式
                for symbol, sides in serializable_positions.items():
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
                
                logger.info("仓位状态已从文件加载")
        except Exception as e:
            logger.error(f"加载仓位状态失败: {e}")
            
    async def check_and_fill_base_position(self, api: BinanceFutureAPI, symbol: str):
        """检查并填充基础仓位 - 核心功能：一测试到没有仓位就补上"""
        try:
            # 获取交易所当前仓位
            exchange_positions = await api.get_positions(symbol)
            has_long = exchange_positions.get('long') and exchange_positions['long']['size'] > 0
            has_short = exchange_positions.get('short') and exchange_positions['short']['size'] > 0
            
            # 检查本地记录
            self.initialize_symbol(symbol)
            local_has_long = len(self.positions[symbol]['long']) > 0
            local_has_short = len(self.positions[symbol]['short']) > 0
            
            # 如果交易所和本地记录不一致，以交易所为准
            if has_long != local_has_long or has_short != local_has_short:
                logger.warning(f"⚠️ {symbol} 本地与交易所仓位记录不一致，同步中...")
                # 清空本地记录
                self.positions[symbol]['long'] = []
                self.positions[symbol]['short'] = []
                
                # 重新记录仓位
                if has_long:
                    self.add_position(symbol, "buy", exchange_positions['long']['size'], exchange_positions['long']['entry_price'])
                if has_short:
                    self.add_position(symbol, "sell", exchange_positions['short']['size'], exchange_positions['short']['entry_price'])
            
            # 检查是否需要补仓
            if not has_long or not has_short:
                logger.info(f"🔄 {symbol} 检测到仓位不完整，准备补仓")
                
                # 获取当前价格
                current_price = await api.get_current_price(symbol)
                if current_price is None:
                    logger.error(f"无法获取 {symbol} 的价格，跳过补仓")
                    return
                
                # 计算初始仓位大小
                position_size = self.calculate_initial_size(current_price)
                if position_size <= 0:
                    logger.error(f"{symbol} 仓位大小计算错误，跳过补仓")
                    return
                
                # 补多仓
                if not has_long:
                    logger.info(f"📈 {symbol} 补多仓，大小: {position_size:.6f}")
                    success = await api.execute_market_order(symbol, "buy", position_size, "LONG")
                    if success:
                        self.add_position(symbol, "buy", position_size, current_price)
                        logger.info(f"✅ {symbol} 多仓补充成功")
                    else:
                        logger.error(f"❌ {symbol} 多仓补充失败")
                
                # 补空仓
                if not has_short:
                    logger.info(f"📉 {symbol} 补空仓，大小: {position_size:.6f}")
                    success = await api.execute_market_order(symbol, "sell", position_size, "SHORT")
                    if success:
                        self.add_position(symbol, "sell", position_size, current_price)
                        logger.info(f"✅ {symbol} 空仓补充成功")
                    else:
                        logger.error(f"❌ {symbol} 空仓补充失败")
        except Exception as e:
            logger.error(f"检查并填充基础仓位错误 {symbol}: {e}")

# ================== 主交易机器人 ==================
class CoinTech2uBot:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.api = BinanceFutureAPI(BINANCE_API_KEY, BINANCE_API_SECRET, symbols)
        self.martingale = DualMartingaleManager()
        self.running = True
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def shutdown(self, signum, frame):
        logger.info("收到关闭信号，停止交易...")
        self.running = False
        self.martingale.save_positions()

    async def run(self):
        if not await self.api.initialize():
            logger.error("交易所初始化失败，程序退出")
            return
            
        logger.info("🚀 开始CoinTech2u策略交易...")
        
        # 程序启动时立即对所有币对开双仓
        logger.info("🔄 程序启动时对所有币对开双仓")
        for symbol in self.symbols:
            await self.open_immediate_hedge(symbol)
        
        while self.running:
            try:
                balance = await self.api.get_balance()
                logger.info(f"当前余额: {balance:.2f} USDT")
                
                for symbol in self.symbols:
                    # 检查并填充基础仓位 - 核心功能：一测试到没有仓位就补上
                    await self.martingale.check_and_fill_base_position(self.api, symbol)
                    # 处理交易逻辑
                    await self.process_symbol(symbol)
                    
                await asyncio.sleep(POLL_INTERVAL)
            except Exception as e:
                logger.error(f"交易循环错误: {e}")
                await asyncio.sleep(10)

    async def open_immediate_hedge(self, symbol: str):
        """程序启动时立即开双仓"""
        # 检查交易所是否已有仓位
        exchange_positions = await self.api.get_positions(symbol)
        has_long = exchange_positions.get('long') and exchange_positions['long']['size'] > 0
        has_short = exchange_positions.get('short') and exchange_positions['short']['size'] > 0
        
        if has_long or has_short:
            logger.info(f"⏩ {symbol} 交易所已有仓位，跳过开仓")
            # 同步本地记录
            if has_long:
                self.martingale.add_position(symbol, "buy", exchange_positions['long']['size'], exchange_positions['long']['entry_price'])
            if has_short:
                self.martingale.add_position(symbol, "sell", exchange_positions['short']['size'], exchange_positions['short']['entry_price'])
            return
        
        # 获取当前价格
        current_price = await self.api.get_current_price(symbol)
        if current_price is None:
            logger.error(f"无法获取 {symbol} 的价格，跳过")
            return
        
        # 计算初始仓位大小
        position_size = self.martingale.calculate_initial_size(current_price)
        if position_size <= 0:
            logger.error(f"{symbol} 仓位大小计算错误，跳过")
            return
        
        logger.info(f"📊 {symbol} 准备开双仓，价格: {current_price:.2f}, 大小: {position_size:.6f}")
        
        # 同时开多仓和空仓
        long_success = await self.api.execute_market_order(symbol, "buy", position_size, "LONG")
        short_success = await self.api.execute_market_order(symbol, "sell", position_size, "SHORT")
        
        if long_success and short_success:
            logger.info(f"✅ {symbol} 已同时开多空仓位: 多单 {position_size:.6f} | 空单 {position_size:.6f}")
            # 记录仓位
            self.martingale.add_position(symbol, "buy", position_size, current_price)
            self.martingale.add_position(symbol, "sell", position_size, current_price)
        else:
            logger.error(f"❌ {symbol} 开仓失败，需要手动检查")

    async def process_symbol(self, symbol: str):
        """处理单个交易对的交易逻辑"""
        # 获取当前价格
        current_price = await self.api.get_current_price(symbol)
        if current_price is None:
            return
        
        # 检查是否需要止盈
        for position_side in ['long', 'short']:
            if self.martingale.should_close_position(symbol, position_side, current_price):
                await self.close_profitable_position(symbol, position_side, current_price)
        
        # 检查是否需要加仓
        for position_side in ['long', 'short']:
            if self.martingale.should_add_layer(symbol, position_side, current_price):
                await self.add_martingale_layer(symbol, position_side, current_price)

    async def close_profitable_position(self, symbol: str, position_side: str, current_price: float):
        """平掉盈利的仓位"""
        position_size = self.martingale.get_position_size(symbol, position_side)
        if position_size <= 0:
            return
            
        # 平仓方向与开仓方向相反
        if position_side == "long":
            close_side = "sell"
            position_side_param = "LONG"
        else:  # short
            close_side = "buy"
            position_side_param = "SHORT"
        
        logger.info(f"📤 {symbol} {position_side.upper()} 止盈平仓，方向: {close_side}, 大小: {position_size:.6f}")
        
        success = await self.api.execute_market_order(symbol, close_side, position_size, position_side_param)
        if success:
            self.martingale.clear_positions(symbol, position_side)
            logger.info(f"✅ {symbol} {position_side.upper()} 所有仓位已平仓")
            
            # 平仓后重新开仓
            await asyncio.sleep(1)  # 等待一下再开新仓
            new_position_size = self.martingale.calculate_initial_size(current_price)
            open_side = "buy" if position_side == "long" else "sell"
            open_success = await self.api.execute_market_order(symbol, open_side, new_position_size, position_side_param)
            
            if open_success:
                self.martingale.add_position(symbol, open_side, new_position_size, current_price)
                logger.info(f"🔄 {symbol} {position_side.upper()} 已重新开仓")

    async def add_martingale_layer(self, symbol: str, position_side: str, current_price: float):
        """为指定方向加仓"""
        positions = self.martingale.positions[symbol][position_side]
        if not positions:
            return
            
        side = "buy" if position_side == "long" else "sell"
        position_side_param = "LONG" if position_side == "long" else "SHORT"
        layer_size = self.martingale.calculate_layer_size(symbol, position_side, current_price)
        
        logger.info(f"📈 {symbol} {position_side.upper()} 准备加仓第{len(positions)+1}层，方向: {side}, 大小: {layer_size:.6f}")
        
        success = await self.api.execute_market_order(symbol, side, layer_size, position_side_param)
        if success:
            self.martingale.add_position(symbol, side, layer_size, current_price)

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
        print("错误: 请设置 SYMBOLS 环境变量，例如: LTC/USDT,DOGE/USDT,XRP/USDT,ADA/USDT,LINK/USDT")
        sys.exit(1)
        
    asyncio.run(main())
