      self.initialize_symbol(symbol)
        
        # 检查是否已达到最大层数
        if len(self.positions[symbol][position_side]) >= MAX_LAYERS:
            logger.info(f"⚠️ {symbol} {position_side.upper()} 已达到最大层数 {MAX_LAYERS}")
            return False
            
        # 检查加仓时间间隔
        last_time = self.last_layer_time[symbol][position_side]
        if last_time and (datetime.now() - last_time) < timedelta(minutes=MIN_LAYER_INTERVAL):
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
            
        logger.info(f"📈 {symbol} {position_side.upper()} 当前盈亏: {pnl_pct*100:.2f}%, 触发阈值: {-LAYER_TRIGGER*100:.2f}%")
        
        # 只有当亏损达到触发阈值时才加仓
        return pnl_pct <= -LAYER_TRIGGER

    def calculate_layer_size(self, symbol: str, position_side: str, balance: float, current_price: float) -> float:
        """计算加仓大小"""
        self.initialize_symbol(symbol)
        layer = len(self.positions[symbol][position_side]) + 1
        
        # 使用固定USDT价值或基于风险的动态计算
        if TRADE_SIZE > 0:
            # 使用固定USDT价值
            size_in_usdt = TRADE_SIZE * (MARTINGALE_MULTIPLIER ** (layer - 1))
            size = size_in_usdt / current_price
        else:
            # 基于风险的动态计算
            base_size = (balance * INITIAL_RISK) / current_price
            size = base_size * (MARTINGALE_MULTIPLIER ** (layer - 1))
        
        # 限制最大仓位大小不超过余额的20%
        max_size = balance * 0.2 / current_price
        final_size = min(size, max_size)
        
        logger.info(f"📏 {symbol} {position_side.upper()} 第{layer}层计算仓位: 基础={size:.6f}, 最终={final_size:.6f}")
        return final_size

    def calculate_initial_size(self, balance: float, current_price: float) -> float:
        """计算初始仓位大小"""
        # 使用固定USDT价值或基于风险的动态计算
        if TRADE_SIZE > 0:
            # 使用固定USDT价值
            size = TRADE_SIZE / current_price
        else:
            # 基于风险的动态计算
            size = (balance * INITIAL_RISK) / current_price
        
        # 限制最大仓位大小不超过余额的10%
        max_size = balance * 0.1 / current_price
        return min(size, max_size)
        
    def should_close_position(self, symbol: str, position_side: str, current_price: float) -> bool:
        """检查是否应该平仓（止损或止盈）"""
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
            
        # 如果亏损超过止损点，强制平仓
        if pnl_pct <= -STOP_LOSS_PCT:
            logger.warning(f"🚨 {symbol} {position_side.upper()} 亏损超过{STOP_LOSS_PCT*100:.0f}%，强制平仓")
            return True
            
        # 如果盈利超过止盈点，止盈平仓
        if pnl_pct >= TAKE_PROFIT_PCT:
            logger.info(f"🎯 {symbol} {position_side.upper()} 盈利超过{TAKE_PROFIT_PCT*100:.0f}%，止盈平仓")
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

# ================== 主交易机器人 ==================
class HedgeMartingaleBot:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols  # 使用传入的symbols而不是全局变量
        self.api = BinanceFutureAPI(BINANCE_API_KEY, BINANCE_API_SECRET, symbols)
        self.analyzer = TechnicalAnalyzer()  # 现在这个类已经定义了
        self.martingale = DualMartingaleManager()
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
            
        logger.info("🚀 开始双仓马丁对冲交易...")
        
        # 程序启动时立即对所有币对开双仓
        logger.info("🔄 程序启动时对所有币对开双仓")
        balance = self.api.get_balance()
        for symbol in self.symbols:
            await self.open_immediate_hedge(symbol, balance)
        
        while self.running:
            try:
                balance = self.api.get_balance()
                logger.info(f"当前余额: {balance:.2f} USDT")
                
                for symbol in self.symbols:
                    await self.process_symbol(symbol, balance)
                    
                await asyncio.sleep(POLL_INTERVAL)
            except Exception as e:
                logger.error(f"交易循环错误: {e}")
                await asyncio.sleep(10)

    async def open_immediate_hedge(self, symbol: str, balance: float):
        """程序启动时立即开双仓"""
        # 获取当前价格
        current_price = self.api.get_current_price(symbol)
        if current_price is None:
            logger.error(f"无法获取 {symbol} 的价格，跳过")
            return
        
        # 计算初始仓位大小
        position_size = self.martingale.calculate_initial_size(balance, current_price)
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

    async def process_symbol(self, symbol: str, balance: float):
        """处理单个交易对的交易逻辑"""
        # 获取当前价格
        current_price = self.api.get_current_price(symbol)
        if current_price is None:
            return
        
        # 检查是否需要平仓（止损或止盈）
        for position_side in ['long', 'short']:
            if self.martingale.should_close_position(symbol, position_side, current_price):
                await self.close_position(symbol, position_side)
        
        # 检查是否需要加仓
        for position_side in ['long', 'short']:
            if self.martingale.should_add_layer(symbol, position_side, current_price):
                await self.add_martingale_layer(symbol, position_side, balance, current_price)
                
        # 检查是否有新信号
        df = self.api.get_ohlcv_data(symbol, TIMEFRAME, 100)
        if df is None or df.empty:
            return
            
        indicators = self.analyzer.calculate_indicators(df)
        if not indicators:
            return
            
        signal = self.analyzer.generate_signal(symbol, indicators)
        if signal:
            logger.info(f"🎯 发现交易信号: {signal}")
            # 对于双仓策略，我们通常不根据信号开仓，而是始终保持双仓
            # 这里可以添加额外的逻辑，比如根据信号调整仓位大小

    async def add_martingale_layer(self, symbol: str, position_side: str, balance: float, current_price: float):
        """为指定方向加仓"""
        positions = self.martingale.positions[symbol][position_side]
        if not positions:
            return
            
        side = "buy" if position_side == "long" else "sell"
        position_side_param = "LONG" if position_side == "long" else "SHORT"
        layer_size = self.martingale.calculate_layer_size(symbol, position_side, balance, current_price)
        
        logger.info(f"📈 {symbol} {position_side.upper()} 准备加仓第{len(positions)+1}层，方向: {side}, 大小: {layer_size:.6f}")
        
        success = self.api.execute_market_order(symbol, side, layer_size, position_side_param)
        if success:
            self.martingale.add_position(symbol, side, layer_size, current_price)

    async def close_position(self, symbol: str, position_side: str):
        """平掉指定方向的所有仓位"""
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
        
        logger.info(f"📤 {symbol} {position_side.upper()} 平仓，方向: {close_side}, 大小: {position_size:.6f}")
        
        success = self.api.execute_market_order(symbol, close_side, position_size, position_side_param)
        if success:
            self.martingale.clear_positions(symbol, position_side)
            logger.info(f"✅ {symbol} {position_side.upper()} 所有仓位已平仓")

# ================== 启动程序 ==================
async def main():
    bot = HedgeMartingaleBot(SYMBOLS_CONFIG)  # 传入配置的symbols
    try:
        await bot.run()  # 🔥 启动交易主循环
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
        
    asyncio.run(main())
