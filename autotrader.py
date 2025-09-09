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
    
    def get_all_positions_summary(self) -> str:
        """获取所有交易对的仓位摘要"""
        summary = "📊 <b>仓位摘要</b>\n\n"
        for symbol in self.symbols:
            self.initialize_symbol(symbol)
            long_layers = len(self.positions[symbol]['long'])
            short_layers = len(self.positions[symbol]['short'])
            
            if long_layers > 0 or short_layers > 0:
                long_size = sum(p['size'] for p in self.positions[symbol]['long'])
                short_size = sum(p['size'] for p in self.positions[symbol]['short'])
                
                # 计算平均入场价格
                long_avg_price = 0
                if long_layers > 0:
                    long_total_value = sum(p['size'] * p['entry_price'] for p in self.positions[symbol]['long'])
                    long_avg_price = long_total_value / long_size
                
                short_avg_price = 0
                if short_layers > 0:
                    short_total_value = sum(p['size'] * p['entry_price'] for p in self.positions[symbol]['short'])
                    short_avg_price = short_total_value / short_size
                
                summary += f"<b>{symbol}</b>\n"
                summary += f"  多仓: {long_layers}层, 数量: {long_size:.6f}, 均价: ${long_avg_price:.4f}\n"
                summary += f"  空仓: {short_layers}层, 数量: {short_size:.6f}, 均价: ${short_avg_price:.4f}\n\n"
        
        if summary == "📊 <b>仓位摘要</b>\n\n":
            summary += "暂无持仓"
            
        return summary

# ================== 主交易机器人 ==================
class CoinTech2uBot:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.api = BinanceFutureAPI(BINANCE_API_KEY, BINANCE_API_SECRET, symbols)
        
        # 初始化风险管理器
        self.risk_manager = RiskManager(self.api)
        
        # 初始化 Telegram 通知器
        self.telegram = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        
        # 初始化策略管理器（传入风险管理器）
        self.martingale = DualMartingaleManager(self.telegram, symbols, self.risk_manager)
        
        # 初始化精准加仓监控系统
        self.layer_monitor = PrecisionLayerMonitor(self.martingale, self.api, self.telegram)
        
        # 上次发送摘要的时间
        self.last_summary_time = 0
        
        self.running = True
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        
        # 初始化层级配置
        self.martingale.initialize()

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
        
        # 初始化风险管理
        self.risk_manager.initialize()
            
        logger.info("🚀 开始CoinTech2u策略交易...")
        
        # 发送启动通知
        if self.telegram:
            # 获取第一层配置
            first_layer_config = self.martingale.layer_config.layers[1]
            telegram_msg = (f"<b>🚀 CoinTech2u交易机器人已启动</b>\n"
                           f"交易对: {', '.join(self.symbols)}\n"
                           f"杠杆: {LEVERAGE}x\n"
                           f"基础仓位: ${first_layer_config['base_size']}\n"
                           f"最大层数: {self.martingale.layer_config.max_layers}\n"
                           f"风险管理: 单日最大亏损{DAILY_LOSS_LIMIT*100}%, 单币种最大风险{MAX_SYMBOL_RISK*100}%")
            self.telegram.send_message(telegram_msg)
        
        # 程序启动时立即对所有币对开双仓
        logger.info("🔄 程序启动时对所有币对开双仓")
        for symbol in self.symbols:
            self.open_immediate_hedge(symbol)
        
        # 记录启动时间
        start_time = time.time()
        self.last_summary_time = start_time
        
        while self.running:
            try:
                # 重置每日余额（如果需要）
                self.risk_manager.reset_daily_balance()
                
                # 检查是否允许交易
                if not self.risk_manager.should_trade(""):
                    logger.warning("⚠️ 交易被风险管理阻止")
                    # 发送警告通知
                    if self.telegram:
                        self.telegram.send_message("<b>⚠️ 交易被风险管理阻止</b>\n已达到风险限制，暂停交易")
                    time.sleep(POLL_INTERVAL)
                    continue
                
                balance = self.api.get_balance()
                logger.info(f"当前余额: {balance:.2f} USDT")
                
                # 监控所有仓位状态
                self.layer_monitor.monitor_all_symbols()
                
                # 打印所有币种的仓位摘要
                self.print_position_summary()
                
                # 检查是否需要发送Telegram摘要
                current_time = time.time()
                if current_time - self.last_summary_time >= TELEGRAM_SUMMARY_INTERVAL:
                    self.send_telegram_summary(balance)
                    self.last_summary_time = current_time
                
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

    def send_telegram_summary(self, balance: float):
        """发送仓位摘要到Telegram"""
        if not self.telegram:
            return
            
        summary = self.martingale.get_all_positions_summary()
        summary += f"\n💰 <b>账户余额</b>: ${balance:.2f} USDT"
        summary += f"\n⏰ <b>更新时间</b>: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        self.telegram.send_message(summary)

    def print_position_summary(self):
        """打印所有币种的仓位摘要"""
        logger.info("📋 仓位摘要:")
        for symbol in self.symbols:
            summary = self.martingale.get_position_summary(symbol)
            logger.info(f"   {summary}")

    def open_immediate_hedge(self, symbol: str):
        """程序启动时立即开双仓（增加风险管理）"""
        # 检查是否允许交易
        if not self.risk_manager.should_trade(symbol):
            logger.warning(f"⚠️ {symbol} 开仓被风险管理阻止")
            return
            
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
        
        # 计算初始仓位大小（使用风险管理）
        position_size = self.martingale.calculate_initial_size(current_price, symbol)
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
            self.martingale.add_position(symbol, "sell", position_size, current
