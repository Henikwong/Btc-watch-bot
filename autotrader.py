elegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
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
