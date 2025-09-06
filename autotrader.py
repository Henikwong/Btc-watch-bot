zable_data['positions'][symbol][side].append({
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
            new_position_size = self.martingale.calculate_initial_size(current_price)
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
