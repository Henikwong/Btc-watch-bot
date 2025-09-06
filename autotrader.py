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

    def run(self):
        if not self.api.initialize():
            logger.error("交易所初始化失败，程序退出")
            return
            
        logger.info("🚀 开始CoinTech2u策略交易...")
        
        # 程序启动时立即对所有币对开双仓
        logger.info("🔄 程序启动时对所有币对开双仓")
        for symbol in self.symbols:
            self.open_immediate_hedge(symbol)
        
        while self.running:
            try:
                balance = self.api.get_balance()
                logger.info(f"当前余额: {balance:.2f} USDT")
                
                for symbol in self.symbols:
                    # 检查并填充基础仓位 - 核心功能：一测试到没有仓位就补上
                    self.martingale.check_and_fill_base_position(self.api, symbol)
                    # 处理交易逻辑
                    self.process_symbol(symbol)
                    
                time.sleep(POLL_INTERVAL)
            except Exception as e:
                logger.error(f"交易循环错误: {e}")
                time.sleep(10)

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

    def process_symbol(self, symbol: str):
        """处理单个交易对的交易逻辑"""
        # 获取当前价格
        current_price = self.api.get_current_price(symbol)
        if current_price is None:
            return
        
        # 检查是否需要止盈
        for position_side in ['long', 'short']:
            if self.martingale.should_close_position(symbol, position_side, current_price):
                self.close_profitable_position(symbol, position_side, current_price)
        
        # 检查是否需要加仓
        for position_side in ['long', 'short']:
            if self.martingale.should_add_layer(symbol, position_side, current_price):
                self.add_martingale_layer(symbol, position_side, current_price)

    def close_profitable_position(self, symbol: str, position_side: str, current_price: float):
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

    def add_martingale_layer(self, symbol: str, position_side: str, current_price: float):
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
