TP_PERCENT = float(os.getenv("TP_PERCENT", "1.5").replace('%', '')) / 100

# 从环境变量读取止损设置
STOP_LOSS = float(os.getenv("STOP_LOSS", "-100"))

# 从环境变量读取趋势捕捉和马丁设置
ENABLE_TREND_CATCH = os.getenv("ENABLE_TREND_CATCH", "true").lower() == "true"
ENABLE_MARTINGALE = os.getenv("ENABLE_MARTINGALE", "true").lower() == "true"

# 加仓间隔 - 已删除冷静期，此参数不再使用
MAX_LAYERS = len(POSITION_SIZES)  # 最大层数等于仓位比例的数量

# 趋势捕捉加仓配置
TREND_CATCH_LAYERS = 2  # 捕捉行情时额外加仓层数
TREND_CATCH_SIZES = [5, 7]  # 额外加仓的仓位大小
TREND_SIGNAL_STRENGTH = 0.7  # 趋势信号强度阈值
# 已删除趋势加仓冷却时间

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
    "ADA/USDT": 8,
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
        
    def send_message(self, message: str) -> bool:
        """发送消息到Telegram"""
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram 配置未设置，跳过发送消息")
            return False
            
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info("Telegram 消息发送成功")
                return True
            else:
                logger.error(f"Telegram 消息发送失败: {response.text}")
                return False
        except Exception as e:
            logger.error(f"发送 Telegram 消息时出错: {e}")
            return False

# ================== 技术分析函数 ==================
def analyze_trend(df: pd.DataFrame) -> Tuple[float, str]:
    """分析趋势方向和强度，使用多时间框架确认
    
    Returns:
        Tuple[float, str]: (趋势强度, 趋势方向) 方向为 'long', 'short' 或 'neutral'
    """
    try:
        # 计算多时间框架EMA指标
        ema_fast = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        ema_slow = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        
        # 计算RSI指标
        rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # 计算MACD
        macd = ta.trend.MACD(df['close'])
        macd_line = macd.macd()
        macd_signal = macd.macd_signal()
        
        # 计算ADX趋势强度
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
        
        # 计算成交量指标
        volume = df['volume']
        volume_ma = volume.rolling(window=20).mean()
        
        # 获取最新值
        latest_ema_fast = ema_fast.iloc[-1]
        latest_ema_slow = ema_slow.iloc[-1]
        latest_rsi = rsi.iloc[-1]
        latest_macd_line = macd_line.iloc[-1]
        latest_macd_signal = macd_signal.iloc[-1]
        latest_adx = adx.iloc[-1]
        latest_volume = volume.iloc[-1]
        latest_volume_ma = volume_ma.iloc[-1]
        
        # 判断趋势方向
        trend_direction = "neutral"
        if latest_ema_fast > latest_ema_slow and latest_macd_line > latest_macd_signal:
            trend_direction = "long"
        elif latest_ema_fast < latest_ema_slow and latest_macd_line < latest_macd_signal:
            trend_direction = "short"
            
        # 计算综合趋势强度 (0-1之间)
        trend_strength = min(latest_adx / 100, 1.0)  # ADX归一化
        trend_strength = max(trend_strength, 0)
        
        # 考虑RSI极端值
        if (trend_direction == "long" and latest_rsi > 70) or (trend_direction == "short" and latest_rsi < 30):
            trend_strength *= 0.7  # 在超买超卖区域减弱信号强度
            
        # 成交量确认：如果成交量没有放大，减弱信号强度
        if latest_volume < latest_volume_ma * 1.2:
            trend_strength *= 0.8
            
        return trend_strength, trend_direction
        
    except Exception as e:
        logger.error(f"趋势分析错误: {e}")
        return 0, "neutral"

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
        """同步初始化交易所连接"""
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
                        self.exchange.set_leverage(LEVERAGE, symbol)
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

    def get_balance(self) -> float:
        """获取账户余额"""
        try:
            balance = self.exchange.fetch_balance()
            return float(balance['USDT']['free'])
        except Exception as e:
            logger.error(f"获取余额失败: {e}")
            return 0.0

    def get_ohlcv_data(self, symbol: str, timeframe: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """获取K线数据"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"K线获取失败 {symbol}: {e}")
            return None

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

    def create_order_with_fallback(self, symbol: str, side: str, contract_size: float, position_side: str):
        """创建订单，如果失败则尝试回退到单向模式"""
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
                    logger.error(f"下单失败: {e}")
                    if attempt == MAX_RETRIES - 1:
                        return None
            
            # 等待一段时间后重试
            time.sleep(RETRY_DELAY * (2 ** attempt))
        
        return None

    def execute_market_order(self, symbol: str, side: str, amount: float, position_side: str) -> bool:
        """执行市价订单"""
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
            min_notional = MIN_NOTIONAL.get(symbol, 10)  # 默认10 USDT
            notional_value = contract_size * current_price
            
            # 如果名义价值不足，调整合约数量
            if notional_value < min_notional:
                # 计算需要的最小合约数量
                min_contract_size = min_notional / current_price
                contract_size = max(contract_size, min_contract_size)
                
                # 重新量化到交易所精度
                contract_size = quantize_amount(contract_size, market)
                
                # 重新计算名义价值
                notional_value = contract_size * current_price
                
                # 如果仍然不足，继续增加直到满足要求
                step = 0.001  # 默认步长
                for f in market['info'].get('filters', []):
                    if f.get('filterType') == 'LOT_SIZE':
                        step = float(f.get('stepSize'))
                        break
                
                while notional_value < min_notional:
                    contract_size += step
                    contract_size = quantize_amount(contract_size, market)
                    notional_value = contract_size * current_price
                    
                    # 安全保护，避免无限循环
                    if contract_size > min_contract_size * 10:
                        logger.error(f"无法满足最小名义价值要求: {notional_value:.2f} < {min_notional}")
                        return False
                
                logger.warning(f"调整合约数量以满足最小名义价值: {contract_size:.6f}, 名义价值: {notional_value:.2f} USDT")
            
            # 创建订单
            order = self.create_order_with_fallback(symbol, side, contract_size, position_side)
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
    def __init__(self, telegram_notifier: TelegramNotifier = None):
        # 仓位结构: {symbol: {'long': [], 'short': []}}
        self.positions: Dict[str, Dict[str, List[dict]]] = {}
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

    def initialize_symbol(self, symbol: str):
        """初始化交易对仓位结构"""
        if symbol not in self.positions:
            self.positions[symbol] = {'long': [], 'short': []}
        if symbol not in self.last_layer_time:
            self.last_layer_time[symbol] = {'long': None, 'short': None}
        if symbol not in self.last_trend_catch_time:
            self.last_trend_catch_time[symbol] = {'long': None, 'short': None}
        if symbol not in self.trend_catch_count:
            self.trend_catch_count[symbol] = {'long': 0, 'short': 0}

    def add_position(self, symbol: str, side: str, size: float, price: float, is_trend_catch: bool = False):
        """添加仓位到对应方向"""
        self.initialize_symbol(symbol)
        position_side = 'long' if side.lower() == 'buy' else 'short'
        layer = len(self.positions[symbol][position_side]) + 1
        
        self.positions[symbol][position_side].append({
            'side': side,
            'size': size,
            'entry_price': price,
            'timestamp': datetime.now(),
            'layer': layer,
            'is_trend_catch': is_trend_catch
        })
        
        if is_trend_catch:
            self.last_trend_catch_time[symbol][position_side] = datetime.now()
            self.trend_catch_count[symbol][position_side] += 1
        else:
            self.last_layer_time[symbol][position_side] = datetime.now()
        
        # 记录日志
        log_msg = f"📊 {symbol} {position_side.upper()} 第{layer}层仓位: {side} {size:.6f} @ {price:.2f}"
        if is_trend_catch:
            log_msg += " (趋势捕捉)"
        logger.info(log_msg)
        
        # 发送 Telegram 通知
        if self.telegram:
            if is_trend_catch:
                telegram_msg = f"<b>🎯 趋势捕捉加仓</b>\n{symbol} {position_side.upper()} 第{layer}层\n操作: {side.upper()}\n数量: {size:.6f}\n价格: ${price:.2f}\n趋势加仓次数: {self.trend_catch_count[symbol][position_side]}/{TREND_CATCH_LAYERS}"
            else:
                telegram_msg = f"<b>🔄 常规加仓</b>\n{symbol} {position_side.upper()} 第{layer}层\n操作: {side.upper()}\n数量: {size:.6f}\n价格: ${price:.2f}"
            self.telegram.send_message(telegram_msg)
        
        # 保存仓位状态
        self.save_positions()

    def should_add_trend_catch_layer(self, symbol: str, position_side: str, trend_strength: float) -> Tuple[bool, int]:
        """检查是否应该进行趋势捕捉加仓"""
        if not ENABLE_TREND_CATCH:
            return False, 0
            
        self.initialize_symbol(symbol)
        
        # 检查是否有持仓
        if not self.positions[symbol][position_side]:
            return False, 0
            
        # 检查趋势强度
        if trend_strength < TREND_SIGNAL_STRENGTH:
            return False, 0
            
        # 检查是否已达到最大趋势加仓次数
        if self.trend_catch_count[symbol][position_side] >= TREND_CATCH_LAYERS:
            return False, 0
            
        # 已删除趋势加仓冷却期检查
        
        # 获取当前仓位层数
        current_layers = len(self.positions[symbol][position_side])
        next_layer = current_layers + 1
        
        return True, next_layer

    def should_add_layer(self, symbol: str, position_side: str, current_price: float) -> bool:
        """检查是否应该加仓 - 使用环境变量中的配置"""
        if not ENABLE_MARTINGALE:
            return False
            
        self.initialize_symbol(symbol)
        
        # 检查是否已达到最大层数
        current_layers = len(self.positions[symbol][position_side])
        if current_layers >= MAX_LAYERS:
            logger.info(f"⚠️ {symbol} {position_side.upper()} 已达到最大层数 {MAX_LAYERS}")
            return False
            
        # 已删除加仓时间间隔检查
        
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
            
        logger.info(f"📈 {symbol} {position_side.upper()} 第{current_layers}层仓位 当前盈亏: {pnl_pct*100:.2f}%")
        
        # 检查止损条件
        unrealized_pnl = total_size * (current_price - avg_price) if position_side == 'long' else total_size * (avg_price - current_price)
        if unrealized_pnl <= STOP_LOSS:
            logger.warning(f"🚨 {symbol} {position_side.upper()} 达到止损条件: {unrealized_pnl:.2f} USDT <= {STOP_LOSS} USDT")
            return False
        
        # 检查是否达到加仓阈值 - 使用环境变量中的配置
        if current_layers < len(POSITION_SIZES):
            threshold = POSITION_SIZES[current_layers]  # 当前层对应的亏损阈值
        else:
            # 如果层级超过配置，使用最后一层的阈值
            threshold = POSITION_SIZES[-1]
            
        # 只有当亏损达到触发阈值时才加仓
        should_add = pnl_pct <= -threshold
        if should_add:
            logger.info(f"✅ {symbol} {position_side.upper()} 达到加仓条件: 亏损{abs(pnl_pct)*100:.2f}% >= 阈值{threshold*100:.2f}%")
        else:
            logger.info(f"❌ {symbol} {position_side.upper()} 未达到加仓条件: 亏损{abs(pnl_pct)*100:.2f}% < 阈值{threshold*100:.2f}%")
            
        return should_add

    def calculate_layer_size(self, symbol: str, position_side: str, current_price: float, is_trend_catch: bool = False) -> float:
        """计算加仓大小 - 使用环境变量中的配置"""
        self.initialize_symbol(symbol)
        layer = len(self.positions[symbol][position_side]) + 1
        
        if is_trend_catch:
            # 使用趋势捕捉加仓配置
            if layer <= len(TREND_CATCH_SIZES):
                size_in_usdt = TREND_CATCH_SIZES[layer - 1]
            else:
                # 如果层级超过配置，使用最后一层的值
                size_in_usdt = TREND_CATCH_SIZES[-1]
        else:
            # 使用环境变量中的仓位比例
            if layer <= len(POSITION_SIZES):
                # 直接使用对应的百分比计算仓位大小
                size_in_usdt = BASE_TRADE_SIZE * POSITION_SIZES[layer - 1]
            else:
                # 如果层级超过配置，使用最后一层的值
                size_in_usdt = BASE_TRADE_SIZE * POSITION_SIZES[-1]
        
        size = size_in_usdt / current_price
        
        logger.info(f"📏 {symbol} {position_side.upper()} 第{layer}层计算仓位: USDT价值={size_in_usdt:.3f}, 数量={size:.6f}")
        return size

    def calculate_initial_size(self, current_price: float) -> float:
        """计算初始仓位大小 - 使用环境变量中的配置"""
        # 使用环境变量中的第一层仓位比例
        size_in_usdt = BASE_TRADE_SIZE * POSITION_SIZES[0]
        size = size_in_usdt / current_price
        
        logger.info(f"📏 初始仓位计算: USDT价值={size_in_usdt:.3f}, 数量={size:.6f}")
        return size
        
    def should_close_position(self, symbol: str, position_side: str, current_price: float) -> bool:
        """检查是否应该平仓（止盈） - 使用环境变量中的配置"""
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
        if pnl_pct >= TP_PERCENT:
            current_layers = len(positions)
            logger.info(f"🎯 {symbol} {position_side.upper()} 第{current_layers}层仓位 盈利超过{TP_PERCENT*100:.2f}%，止盈平仓")
            
            # 发送 Telegram 通知
            if self.telegram:
                profit_usdt = total_size * (current_price - avg_price) if position_side == 'long' else total_size * (avg_price - current_price)
                telegram_msg = f"<b>🎯 止盈触发</b>\n{symbol} {position_side.upper()} 第{current_layers}层\n盈利: {pnl_pct*100:.2f}%\n收益: ${profit_usdt:.2f}\n平均成本: ${avg_price:.2f}\n当前价格: ${current_price:.2f}"
                self.telegram.send_message(telegram_msg)
                
            return True
            
        return False

    def check_stop_loss(self, symbol: str, position_side: str, current_price: float, api: BinanceFutureAPI) -> bool:
        """检查并执行止损操作"""
        self.initialize_symbol(symbol)
        positions = self.positions[symbol][position_side]
        if not positions:
            return False
        
        # 计算总仓位大小和平均入场价格
        total_size = sum(p['size'] for p in positions)
        total_value = sum(p['size'] * p['entry_price'] for p in positions)
        avg_price = total_value / total_size
        
        # 计算当前盈亏
        if position_side == 'long':
            unrealized_pnl = total_size * (current_price - avg_price)
        else:  # short
            unrealized_pnl = total_size * (avg_price - current_price)
        
        # 检查是否达到止损条件
        if unrealized_pnl <= STOP_LOSS:  # STOP_LOSS是负值，所以用<=
            logger.warning(f"🚨 {symbol} {position_side.upper()} 达到止损条件: {unrealized_pnl:.2f} USDT <= {STOP_LOSS} USDT")
            
            # 平仓方向与开仓方向相反
            if position_side == 'long':
                close_side = 'sell'
                position_side_param = 'LONG'
            else:
                close_side = 'buy'
                position_side_param = 'SHORT'
            
            # 执行平仓
            success = api.execute_market_order(symbol, close_side, total_size, position_side_param)
            if success:
                # 清空本地记录
                self.clear_positions(symbol, position_side)
                logger.info(f"✅ {symbol} {position_side.upper()} 已止损平仓")
                
                # 发送通知
                if self.telegram:
                    telegram_msg = f"<b>🚨 止损触发</b>\n{symbol} {position_side.upper()}\n亏损: {unrealized_pnl:.2f} USDT\n价格: ${current_price:.2f}"
                    self.telegram.send_message(telegram_msg)
                
                return True
            else:
                logger.error(f"❌ {symbol} {position_side.upper()} 止损平仓失败")
                return False
        
        return False

    def get_position_size(self, symbol: str, position_side: str) -> float:
        """获取某个方向的仓位总大小"""
        self.initialize_symbol(symbol)
        return sum(p['size'] for p in self.positions[symbol][position_side])
    
    def get_position_layers(self, symbol: str, position_side: str) -> int:
        """获取某个方向的仓位层数"""
        self.initialize_symbol(symbol)
        return len(self.positions[symbol][position_side])
    
    def clear_positions(self, symbol: str, position_side: str):
        """清空某个方向的仓位记录"""
        self.initialize_symbol(symbol)
        self.positions[symbol][position_side] = []
        self.trend_catch_count[symbol][position_side] = 0
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
                            'layer': pos['layer'],
                            'is_trend_catch': pos.get('is_trend_catch', False)
                        })
            
            # 保存趋势捕捉计数
            serializable_data = {
                'positions': serializable_positions,
                'trend_catch_count': self.trend_catch_count,
                'last_trend_catch_time': {
                    sym: {side: time.isoformat() if time else None 
                         for side, time in sides.items()}
                    for sym, sides in self.last_trend_catch_time.items()
                },
                'last_layer_time': {
                    sym: {side: time.isoformat() if time else None 
                         for side, time in sides.items()}
                    for sym, sides in self.last_layer_time.items()
                },
                'saved_at': datetime.now().isoformat()
            }
            
            with open(self.positions_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
        except Exception as e:
            logger.error(f"保存仓位状态失败: {e}")
    
    def load_positions(self):
        """从文件加载仓位状态"""
        try:
            if os.path.exists(self.positions_file):
                with open(self.positions_file, 'r') as f:
                    data = json.load(f)
                
                # 加载仓位数据
                serializable_positions = data.get('positions', {})
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
                                'layer': pos['layer'],
                                'is_trend_catch': pos.get('is_trend_catch', False)
                            })
                
                # 加载趋势捕捉计数
                self.trend_catch_count = data.get('trend_catch_count', {})
                
                # 加载时间数据
                self.last_trend_catch_time = {}
                for sym, sides in data.get('last_trend_catch_time', {}).items():
                    self.last_trend_catch_time[sym] = {}
                    for side, time_str in sides.items():
                        self.last_trend_catch_time[sym][side] = datetime.fromisoformat(time_str) if time_str else None
                
                self.last_layer_time = {}
                for sym, sides in data.get('last_layer_time', {}).items():
                    self.last_layer_time[sym] = {}
                    for side, time_str in sides.items():
                        self.last_layer_time[sym][side] = datetime.fromisoformat(time_str) if time_str else None
                
                logger.info("仓位状态已从文件加载")
        except Exception as e:
            logger.error(f"加载仓位状态失败: {e}")
            
    def check_and_fill_base_position(self, api: BinanceFutureAPI, symbol: str):
        """检查并填充基础仓位 - 核心功能：一测试到没有仓位就补上"""
        try:
            # 获取交易所当前仓位
            exchange_positions = api.get_positions(symbol)
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
                current_price = api.get_current_price(symbol)
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
                    success = api.execute_market_order(symbol, "buy", position_size, "LONG")
                    if success:
                        self.add_position(symbol, "buy", position_size, current_price)
                        logger.info(f"✅ {symbol} 多仓补充成功")
                    else:
                        logger.error(f"❌ {symbol} 多仓补充失败")
                
                # 补空仓
                if not has_short:
                    logger.info(f"📉 {symbol} 补空仓，大小: {position_size:.6f}")
                    success = api.execute_market_order(symbol, "sell", position_size, "SHORT")
                    if success:
                        self.add_position(symbol, "sell", position_size, current_price)
                        logger.info(f"✅ {symbol} 空仓补充成功")
                    else:
                        logger.error(f"❌ {symbol} 空仓补充失败")
        except Exception as e:
            logger.error(f"检查并填充基础仓位错误 {symbol}: {e}")

    def get_position_summary(self, symbol: str) -> str:
        """获取仓位摘要信息"""
        self.initialize_symbol(symbol)
        long_layers = len(self.positions[symbol]['long'])
        short_layers = len(self.positions[symbol]['short'])
        
        if long_layers == 0 and short_layers == 0:
            return f"{symbol}: 无仓位"
        
        long_size = sum(p['size'] for p in self.positions[symbol]['long'])
        short_size = sum(p['size'] for p in self.positions[symbol]['short'])
        
        return f"{symbol}: 多仓{long_layers}层({long_size:.6f}) | 空仓{short_layers}层({short_size:.6f})"

# ================== 主交易机器人 ==================
class CoinTech2uBot:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.api = BinanceFutureAPI(BINANCE_API_KEY, BINANCE_API_SECRET, symbols)
        
        # 初始化 Telegram 通知器
        self.telegram = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.martingale = DualMartingaleManager(self.telegram)
        
        self.running = True
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def shutdown(self, signum, frame):
        logger.info("收到关闭信号，停止交易...")
        self.running = False
        self.martingale.save_positions()
        
        # 发送关闭通知
        if self.telegram:
            self.telegram.send_message("<b>🛑 交易机器人已停止</b>")

    def run(self):
        if not self.api.initialize():
            logger.error("交易所初始化失败，程序退出")
            # 发送错误通知
            if self.telegram:
                self.telegram.send_message("<b>❌ 交易所初始化失败，程序退出</b>")
            return
            
        logger.info("🚀 开始CoinTech2u策略交易...")
        
        # 发送启动通知
        if self.telegram:
            self.telegram.send_message(f"<b>🚀 CoinTech2u交易机器人已启动</b>\n交易对: {', '.join(self.symbols)}\n杠杆: {LEVERAGE}x\n基础仓位: ${BASE_TRADE_SIZE}")
        
        # 程序启动时立即对所有币对开双仓
        logger.info("🔄 程序启动时对所有币对开双仓")
        for symbol in self.symbols:
            self.open_immediate_hedge(symbol)
        
        while self.running:
            try:
                balance = self.api.get_balance()
                logger.info(f"当前余额: {balance:.2f} USDT")
                
                # 打印所有币种的仓位摘要
                self.print_position_summary()
                
                for symbol in self.symbols:
                    # 检查并填充基础仓位 - 核心功能：一测试到没有仓位就补上
                    self.martingale.check_and_fill_base_position(self.api, symbol)
                    # 处理交易逻辑
                    self.process_symbol(symbol)
                    
                time.sleep(POLL_INTERVAL)
            except Exception as e:
                logger.error(f"交易循环错误: {e}")
                # 发送错误通知
                if self.telegram:
                    self.telegram.send_message(f"<b>❌ 交易循环错误</b>\n{str(e)}")
                time.sleep(10)

    def print_position_summary(self):
        """打印所有币种的仓位摘要"""
        logger.info("📋 仓位摘要:")
        for symbol in self.symbols:
            summary = self.martingale.get_position_summary(symbol)
            logger.info(f"   {summary}")

    def open_immediate_hedge(self, symbol: str):
        """程序启动时立即开双仓"""
        # 检查交易所是否已有仓位
        exchange_positions = self.api.get_positions(symbol)
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
        current_price = self.api.get_current_price(symbol)
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
        long_success = self.api.execute_market_order(symbol, "buy", position_size, "LONG")
        short_success = self.api.execute_market_order(symbol, "sell", position_size, "SHORT")
        
        if long_success and short_success:
            logger.info(f"✅ {symbol} 已同时开多空仓位: 多单 {position_size:.6f} | 空单 {position_size:.6f}")
            # 记录仓位
            self.martingale.add_position(symbol, "buy", position_size, current_price)
            self.martingale.add_position(symbol, "sell", position_size, current_price)
        else:
            logger.error(f"❌ {symbol} 开仓失败，需要手动检查")
            # 发送错误通知
            if self.telegram:
                self.telegram.send_message(f"<b>❌ {symbol} 开仓失败</b>\n需要手动检查")

    def process_symbol(self, symbol: str):
        """处理单个交易对的交易逻辑"""
        # 获取当前价格
        current_price = self.api.get_current_price(symbol)
        if current_price is None:
            return
        
        # 检查止损
        for position_side in ['long', 'short']:
            self.martingale.check_stop_loss(symbol, position_side, current_price, self.api)
        
        # 获取K线数据用于趋势分析
        df = self.api.get_ohlcv_data(symbol, TIMEFRAME, 100)
        if df is not None:
            # 分析趋势
            trend_strength, trend_direction = analyze_trend(df)
            logger.info(f"📊 {symbol} 趋势分析: 方向={trend_direction}, 强度={trend_strength:.2f}")
            
            # 检查趋势捕捉加仓
            if ENABLE_TREND_CATCH:
                for position_side in ['long', 'short']:
                    if trend_direction == position_side and trend_strength >= TREND_SIGNAL_STRENGTH:
                        should_add, next_layer = self.martingale.should_add_trend_catch_layer(symbol, position_side, trend_strength)
                        if should_add:
                            self.add_trend_catch_layer(symbol, position_side, current_price)
        
        # 检查是否需要止盈
        for position_side in ['long', 'short']:
            if self.martingale.should_close_position(symbol, position_side, current_price):
                self.close_profitable_position(symbol, position_side, current_price)
        
        # 检查是否需要加仓
        if ENABLE_MARTINGALE:
            for position_side in ['long', 'short']:
                if self.martingale.should_add_layer(symbol, position_side, current_price):
                    self.add_martingale_layer(symbol, position_side, current_price)

    def add_trend_catch_layer(self, symbol: str, position_side: str, current_price: float):
        """为指定方向添加趋势捕捉加仓"""
        positions = self.martingale.positions[symbol][position_side]
        if not positions:
            return
            
        side = "buy" if position_side == "long" else "sell"
        position_side_param = "LONG" if position_side == "long" else "SHORT"
        layer_size = self.martingale.calculate_layer_size(symbol, position_side, current_price, True)
        
        current_layers = len(positions)
        logger.info(f"🎯 {symbol} {position_side.upper()} 第{current_layers}层仓位 趋势捕捉加仓第{current_layers+1}层，方向: {side}, 大小: {layer_size:.6f}")
        
        success = self.api.execute_market_order(symbol, side, layer_size, position_side_param)
        if success:
            self.martingale.add_position(symbol, side, layer_size, current_price, True)
        else:
            # 发送错误通知
            if self.telegram:
                self.telegram.send_message(f"<b>❌ {symbol} {position_side.upper()} 趋势捕捉加仓失败</b>")

    def close_profitable_position(self, symbol: str, position_side: str, current_price: float):
        """平掉盈利的仓位"""
        position_size = self.martingale.get_position_size(symbol, position_side)
        if position_size <= 0:
            return
            
        # 获取当前层数
        current_layers = self.martingale.get_position_layers(symbol, position_side)
            
        # 平仓方向与开仓方向相反
        if position_side == "long":
            close_side = "sell"
            position_side_param = "LONG"
        else:  # short
            close_side = "buy"
            position_side_param = "SHORT"
        
        logger.info(f"📤 {symbol} {position_side.upper()} 第{current_layers}层仓位 止盈平仓，方向: {close_side}, 大小: {position_size:.6f}")
        
        success = self.api.execute_market_order(symbol, close_side, position_size, position_side_param)
        if success:
            self.martingale.clear_positions(symbol, position_side)
            logger.info(f"✅ {symbol} {position_side.upper()} 所有仓位已平仓")
            
            # 平仓后重新开仓
            time.sleep(1)  # 等待一下再开新仓
            new_position_size = self.martingale.calculate_initial_size(current_price)
            open_side = "buy" if position_side == "long" else "sell"
            open_success = self.api.execute_market_order(symbol, open_side, new_position_size, position_side_param)
            
            if open_success:
                self.martingale.add_position(symbol, open_side, new_position_size, current_price)
                logger.info(f"🔄 {symbol} {position_side.upper()} 已重新开仓")
        else:
            # 发送错误通知
            if self.telegram:
                self.telegram.send_message(f"<b>❌ {symbol} {position_side.upper()} 止盈平仓失败</b>")

    def add_martingale_layer(self, symbol: str, position_side: str, current_price: float):
        """为指定方向加仓"""
        positions = self.martingale.positions[symbol][position_side]
        if not positions:
            return
            
        side = "buy" if position_side == "long" else "sell"
        position_side_param = "LONG" if position_side == "long" else "SHORT"
        layer_size = self.martingale.calculate_layer_size(symbol, position_side, current_price, False)
        
        current_layers = len(positions)
        logger.info(f"📈 {symbol} {position_side.upper()} 第{current_layers}层仓位 准备加仓第{current_layers+1}层，方向: {side}, 大小: {layer_size:.6f}")
        
        success = self.api.execute_market_order(symbol, side, layer_size, position_side_param)
        if success:
            self.martingale.add_position(symbol, side, layer_size, current_price, False)
        else:
            # 发送错误通知
            if self.telegram:
                self.telegram.send_message(f"<b>❌ {symbol} {position_side.upper()} 加仓失败</b>")

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
        
    if not SYMBOLS_CONFIG:
        print("错误: 请设置 SYMBOLS 环境变量，例如: LTC/USDT,DOGE/USDT,XRP/USDT,ADA/USDT,LINK/USDT")
        sys.exit(1)
        
    main()
