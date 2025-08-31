ma_26'],
            30 < current_1h['rsi'] < 60,
            current_4h['ema_12'] < current_4h['ema_26']
        ])
        
        if bullish_conditions:
            return TradeSignal(
                symbol=symbol,
                side=OrderSide.BUY,
                price=price,
                atr=atr,
                quantity=0,  # 将在执行时计算
                timestamp=datetime.now()
            )
        elif bearish_conditions:
            return TradeSignal(
                symbol=symbol,
                side=OrderSide.SELL,
                price=price,
                atr=atr,
                quantity=0,
                timestamp=datetime.now()
            )
        
        return None

# ================== 交易执行器 ==================
class TradeExecutor:
    """完整的交易执行器"""
    
    def __init__(self, exchange: ExchangeInterface, logger: AdvancedLogger):
        self.exchange = exchange
        self.logger = logger
    
    async def execute_signal(self, signal: TradeSignal, balance: float) -> Tuple[bool, Optional[TradeSignal]]:
        """执行交易信号，返回执行结果和信号"""
        try:
            # 计算仓位大小
            quantity = self.calculate_position_size(balance, signal.price, signal.atr)
            if quantity <= 0:
                self.logger.warning(f"仓位计算为0或负数: {signal.symbol}")
                return False, None
            
            signal.quantity = quantity
            
            # 执行主订单
            order_params = {}
            if Config.HEDGE_MODE:
                order_params['positionSide'] = 'LONG' if signal.side == OrderSide.BUY else 'SHORT'
            
            order_result = await self.exchange.create_order(
                signal.symbol,
                'market',
                signal.side.value,
                quantity,
                None,
                order_params
            )
            
            if not order_result.success:
                self.logger.error(f"订单执行失败 {signal.symbol}: {order_result.error}")
                return False, None
            
            # 设置止盈止损
            tp_success = await self.place_tp_order(signal)
            sl_success = await self.place_sl_order(signal)
            
            if tp_success and sl_success:
                self.logger.info(f"交易执行成功: {signal.symbol} {signal.side.value}")
                return True, signal
            else:
                self.logger.warning(f"止盈止损设置部分失败: {signal.symbol}")
                return False, None
                
        except Exception as e:
            self.logger.error(f"执行信号失败 {signal.symbol}: {e}")
            return False, None
    
    def calculate_position_size(self, balance: float, price: float, atr: float) -> float:
        risk_amount = balance * Config.RISK_RATIO
        risk_per_share = atr * Config.SL_ATR_MULT
        position_size = risk_amount / risk_per_share
        max_position = (balance * Config.LEVERAGE) / price
        return min(position_size, max_position)
    
    async def place_tp_order(self, signal: TradeSignal) -> bool:
        """完整的止盈单设置"""
        tp_price = signal.price + signal.atr * Config.TP_ATR_MULT if signal.side == OrderSide.BUY else signal.price - signal.atr * Config.TP_ATR_MULT
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                params = {}
                if Config.HEDGE_MODE:
                    params['positionSide'] = 'LONG' if signal.side == OrderSide.BUY else 'SHORT'
                    params['reduceOnly'] = True
                # 非对冲模式不使用reduceOnly
                
                order_side = 'sell' if signal.side == OrderSide.BUY else 'buy'
                result = await self.exchange.create_order(
                    signal.symbol,
                    'take_profit_market',
                    order_side,
                    signal.quantity,
                    None,
                    params
                )
                
                if result.success:
                    self.logger.info(f"止盈单设置成功: {signal.symbol} @ {tp_price:.2f}")
                    return True
                else:
                    raise Exception(result.error)
                    
            except Exception as e:
                self.logger.warning(f"止盈单设置失败(尝试{attempt+1}): {e}")
                if attempt == Config.MAX_RETRIES - 1:
                    self.logger.error(f"止盈单设置失败: {signal.symbol} - {e}")
                    return False
                await asyncio.sleep(Config.RETRY_DELAY)
        
        return False
    
    async def place_sl_order(self, signal: TradeSignal) -> bool:
        """完整的止损单设置"""
        sl_price = signal.price - signal.atr * Config.SL_ATR_MULT if signal.side == OrderSide.BUY else signal.price + signal.atr * Config.SL_ATR_MULT
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                params = {}
                if Config.HEDGE_MODE:
                    params['positionSide'] = 'LONG' if signal.side == OrderSide.BUY else 'SHORT'
                    params['reduceOnly'] = True
                # 非对冲模式不使用reduceOnly
                
                order_side = 'sell' if signal.side == OrderSide.BUY else 'buy'
                result = await self.exchange.create_order(
                    signal.symbol,
                    'stop_market',
                    order_side,
                    signal.quantity,
                    None,
                    params
                )
                
                if result.success:
                    self.logger.info(f"止损单设置成功: {signal.symbol} @ {sl_price:.2f}")
                    return True
                else:
                    raise Exception(result.error)
                    
            except Exception as e:
                self.logger.warning(f"止损单设置失败(尝试{attempt+1}): {e}")
                if attempt == Config.MAX_RETRIES - 1:
                    self.logger.error(f"止损单设置失败: {signal.symbol} - {e}")
                    return False
                await asyncio.sleep(Config.RETRY_DELAY)
        
        return False

# ================== 主交易机器人 ==================
class ProductionTrader:
    """生产环境交易机器人"""
    
    def __init__(self):
        self.logger = AdvancedLogger()
        self.cache = TimedCache()
        self.exchange = BinanceExchange(self.logger)
        self.indicator_system = IndicatorSystem(self.cache)
        self.trade_executor = TradeExecutor(self.exchange, self.logger)
        self.running = False
        self.last_balance = BalanceInfo(total=0, free=0, used=0)
    
    async def run(self):
        """主运行循环"""
        self.logger.info("🚀 启动生产环境交易机器人")
        self.running = True
        
        try:
            while self.running:
                start_time = time.time()
                
                # 获取实时余额
                self.last_balance = await self.exchange.fetch_balance()
                if self.last_balance.free <= 0:
                    self.logger.error("账户余额不足，停止交易")
                    break
                
                # 异步处理所有交易对
                tasks = [self.process_symbol(symbol) for symbol in Config.SYMBOLS]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 处理结果 - 只统计成功执行的信号
                successful_signals = 0
                for result in results:
                    if isinstance(result, tuple) and result[0]:  # (success, signal)
                        successful_signals += 1
                
                processing_time = time.time() - start_time
                self.logger.info(f"本轮处理完成: {successful_signals}个成功信号, 耗时: {processing_time:.2f}s")
                
                # 精确控制轮询间隔，记录超时情况
                sleep_time = max(0, Config.POLL_INTERVAL - processing_time)
                if sleep_time == 0:
                    self.logger.warning(f"处理超时: 实际耗时{processing_time:.2f}s > 轮询间隔{Config.POLL_INTERVAL}s")
                
                await asyncio.sleep(sleep_time)
                
        except Exception as e:
            self.logger.critical(f"主循环异常: {e}")
        finally:
            await self.shutdown()
    
    async def process_symbol(self, symbol: str) -> Tuple[bool, Optional[TradeSignal]]:
        """处理单个交易对，返回执行结果和信号"""
        try:
            # 获取数据
            df_1h = await self.exchange.get_historical_data(symbol, '1h', Config.OHLCV_LIMIT)
            df_4h = await self.exchange.get_historical_data(symbol, Config.MACD_FILTER_TIMEFRAME, Config.OHLCV_LIMIT)
            
            if df_1h.empty or df_4h.empty:
                return False, None
            
            # 计算指标
            df_1h = self.indicator_system.compute_indicators(df_1h, symbol, '1h')
            df_4h = self.indicator_system.compute_indicators(df_4h, symbol, '4h')
            
            # 生成信号
            signal = self.indicator_system.generate_signal(df_1h, df_4h, symbol)
            
            if signal and Config.MODE == Mode.LIVE:
                # 使用实时余额执行交易
                success, executed_signal = await self.trade_executor.execute_signal(signal, self.last_balance.free)
                return success, executed_signal if success else None
            
            return False, signal
            
        except Exception as e:
            self.logger.error(f"处理交易对 {symbol} 失败: {e}")
            return False, None
    
    async def shutdown(self):
        """安全关闭"""
        self.logger.info("正在安全关闭交易机器人...")
        self.running = False

# ================== 主程序入口 ==================
async def main():
    trader = ProductionTrader()
    
    # 信号处理
    def signal_handler(signum, frame):
        asyncio.create_task(trader.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await trader.run()
    except KeyboardInterrupt:
        await trader.shutdown()
    except Exception as e:
        trader.logger.critical(f"程序崩溃: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
