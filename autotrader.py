lance, current_price)
                
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
