 current_balance - self.daily_start_equity

# ================== 警报系统 ==================
class AlertSystem:
    """警报系统"""
    
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
    
    def send_alert(self, message: str):
        """发送警报"""
        self.logger.critical(f"警报: {message}")
        
        # 发送到Telegram
        if Config.TELEGRAM_BOT_TOKEN and Config.TELEGRAM_CHAT_ID:
            try:
                url = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendMessage"
                payload = {
                    "chat_id": Config.TELEGRAM_CHAT_ID,
                    "text": f"交易警报: {message}",
                    "parse_mode": "HTML"
                }
                response = requests.post(url, json=payload, timeout=10)
                if response.status_code != 200:
                    self.logger.error(f"Telegram消息发送失败: {response.text}")
            except Exception as e:
                self.logger.error(f"发送Telegram警报失败: {e}")

# ================== 状态管理器 ==================
class StateManager:
    """状态管理器"""
    
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.state_file = "trading_state.json"
        self.state = {}
        self.last_save_time = 0
        
    def load_state(self):
        """加载状态"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    self.state = json.load(f)
                
                # 恢复活跃持仓
                if 'active_positions' in self.state:
                    active_positions = {}
                    for symbol, pos_data in self.state['active_positions'].items():
                        try:
                            active_positions[symbol] = TradeSignal.from_dict(pos_data)
                        except Exception as e:
                            self.logger.error(f"恢复持仓状态失败 {symbol}: {e}")
                    self.state['active_positions'] = active_positions
                
                self.logger.info("状态已加载")
            else:
                self.logger.info("无保存状态，使用初始状态")
        except Exception as e:
            self.logger.error(f"加载状态失败: {e}")
            self.state = {}
    
    def save_state(self, force: bool = False):
        """保存状态"""
        current_time = time.time()
        if not force and current_time - self.last_save_time < Config.STATE_SAVE_INTERVAL:
            return
            
        try:
            # 转换活跃持仓为可序列化格式
            if 'active_positions' in self.state:
                self.state['active_positions'] = {
                    k: v.to_dict() for k, v in self.state['active_positions'].items()
                }
                
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2, default=str)
                
            self.last_save_time = current_time
            self.logger.debug("状态已保存")
        except Exception as e:
            self.logger.error(f"保存状态失败: {e}")
    
    def get_state(self, key, default=None):
        """获取状态值"""
        return self.state.get(key, default)
    
    def set_state(self, key, value):
        """设置状态值"""
        self.state[key] = value
        self.save_state()

# ================== 增强的错误处理 ==================
class EnhancedErrorHandler:
    """增强的错误处理"""
    
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.error_counts = {}
        self.last_alert_time = {}
        
    def handle_error(self, error: Exception, context: str = ""):
        """处理错误"""
        error_type = type(error).__name__
        error_key = f"{error_type}_{context}"
        
        # 计数错误
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # 分类处理错误
        if "Network" in error_type or "Connection" in error_type:
            self.handle_network_error(error, context)
        elif "Insufficient" in error_type or "Balance" in error_type:
            self.handle_balance_error(error, context)
        elif "RateLimit" in error_type:
            self.handle_rate_limit_error(error, context)
        else:
            self.handle_general_error(error, context)
        
        # 如果错误频繁发生，发送警报
        if self.error_counts[error_key] > 5:
            current_time = time.time()
            last_alert = self.last_alert_time.get(error_key, 0)
            
            if current_time - last_alert > 3600:  # 每小时最多报警一次
                self.logger.critical(f"频繁错误警报: {error_key} (count: {self.error_counts[error_key]})")
                self.last_alert_time[error_key] = current_time
    
    def handle_network_error(self, error: Exception, context: str):
        """处理网络错误"""
        self.logger.warning(f"网络错误 {context}: {error}")
        # 实现指数退避重试逻辑
    
    def handle_balance_error(self, error: Exception, context: str):
        """处理余额不足错误"""
        self.logger.error(f"余额不足 {context}: {error}")
        # 可能需要停止交易或调整仓位大小
    
    def handle_rate_limit_error(self, error: Exception, context: str):
        """处理速率限制错误"""
        self.logger.warning(f"速率限制 {context}: {error}")
        # 实现适当的等待和重试逻辑
    
    def handle_general_error(self, error: Exception, context: str):
        """处理一般错误"""
        self.logger.error(f"一般错误 {context}: {error}")

# ================== 主交易机器人 ==================
class EnhancedProductionTrader:
    """增强的生产环境交易机器人"""
    
    def __init__(self):
        self.logger = AdvancedLogger()
        self.cache = TimedCache()
        self.exchange = BinanceExchange(self.logger)
        self.indicators = IndicatorSystem(self.cache)
        self.executor = TradeExecutor(self.exchange, self.logger)
        self.websocket_handler = WebSocketDataHandler(self.exchange, self.logger, Config.SYMBOLS)
        self.risk_manager = EnhancedRiskManager(self.exchange, self.logger)
        self.error_handler = EnhancedErrorHandler(self.logger)
        self.state_manager = StateManager(self.logger)
        self.active_positions: Dict[str, TradeSignal] = {}
        self.last_state_save = 0

        # 加载保存的状态
        self.state_manager.load_state()
        self.active_positions = self.state_manager.get_state('active_positions', {})

        # 注册优雅退出
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        self.running = True

    async def process_symbol(self, symbol: str):
        """处理单一交易对"""
        try:
            # 拉取数据
            df_1h = await self.exchange.get_historical_data(symbol, "1h", Config.OHLCV_LIMIT)
            df_4h = await self.exchange.get_historical_data(symbol, Config.MACD_FILTER_TIMEFRAME, Config.OHLCV_LIMIT)

            if df_1h.empty or df_4h.empty:
                return None

            # 计算指标
            df_1h = self.indicators.compute_indicators(df_1h, symbol, "1h")
            df_4h = self.indicators.compute_indicators(df_4h, symbol, Config.MACD_FILTER_TIMEFRAME)

            # 生成信号
            signal_data = self.indicators.generate_signal(df_1h, df_4h, symbol)
            return signal_data

        except Exception as e:
            self.error_handler.handle_error(e, f"处理 {symbol}")
            return None

    async def run(self):
        """主循环"""
        self.logger.info(f"🚀 启动增强版交易机器人，模式: {Config.MODE}, 对冲: {Config.HEDGE_MODE}, 杠杆: {Config.LEVERAGE}")

        # 启动WebSocket连接
        asyncio.create_task(self.websocket_handler.start())
        
        while self.running:
            try:
                # 获取余额
                balance_info = await self.exchange.fetch_balance()
                free_usdt = balance_info.free
                self.logger.debug(f"账户余额: total={balance_info.total}, free={balance_info.free}, used={balance_info.used}")

                # 检查风险限制
                if not await self.risk_manager.check_risk_limits(balance_info.total):
                    self.logger.critical("风险限制触发，停止交易")
                    break

                # 获取实时数据
                symbol, data = await self.websocket_handler.get_next_data()

                # 处理信号生成和交易执行
                signal = await self.process_symbol(symbol)
                
                if signal:
                    # 风控：限制最大持仓数
                    if len(self.active_positions) >= Config.MAX_POSITIONS:
                        self.logger.warning(f"持仓已满({Config.MAX_POSITIONS})，跳过 {signal.symbol}")
                        continue

                    # 如果已有同一方向持仓，跳过
                    if signal.symbol in self.active_positions:
                        existing_signal = self.active_positions[signal.symbol]
                        if existing_signal.side == signal.side:
                            self.logger.debug(f"{signal.symbol} 已有同方向持仓，跳过新信号")
                            continue

                    # 执行交易
                    success, executed_sig = await self.executor.execute_signal(signal, free_usdt)
                    if success and executed_sig:
                        self.active_positions[signal.symbol] = executed_sig
                        self.state_manager.set_state('active_positions', self.active_positions)

                # 定期保存状态
                current_time = time.time()
                if current_time - self.last_state_save >= Config.STATE_SAVE_INTERVAL:
                    self.state_manager.save_state(force=True)
                    self.last_state_save = current_time
                
                await asyncio.sleep(1)  # 更短的等待时间，因为使用WebSocket

            except Exception as e:
                self.error_handler.handle_error(e, "主循环")
                await asyncio.sleep(5)

    def stop(self, *args):
        """优雅退出"""
        self.logger.info("🛑 收到停止信号，正在退出...")
        self.running = False
        self.state_manager.save_state(force=True)
        # 关闭WebSocket连接
        asyncio.create_task(self.websocket_handler.stop())

# ================== 启动入口 ==================
if __name__ == "__main__":
    trader = EnhancedProductionTrader()
    try:
        asyncio.run(trader.run())
    except KeyboardInterrupt:
        trader.stop()
