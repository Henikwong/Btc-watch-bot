import os
import time
import ccxt
import pandas as pd
import ta

# =========================
# 环境变量
# =========================
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
SYMBOLS = os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT").split(",")
MODE = os.getenv("MODE", "live")  # live / backtest
HEDGE_MODE = os.getenv("HEDGE_MODE", "false").lower() == "true"

# =========================
# 初始化交易所
# =========================
exchange = ccxt.binance({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "enableRateLimit": True,
    "options": {"defaultType": "future"},
})

print(f"🚀 启动实盘交易... (模式: {'双向' if HEDGE_MODE else '单向'})")

# =========================
# 技术指标计算
# =========================
def add_indicators(df: pd.DataFrame):
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=12).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=26).ema_indicator()
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["vol_ma"] = df["volume"].rolling(20).mean()
    return df.dropna()

# =========================
# 信号生成（严格金叉/死叉）
# =========================
def signal_from_indicators(df, trend_df):
    prev_macd = df["macd"].iloc[-2]
    prev_signal = df["macd_signal"].iloc[-2]
    curr_macd = df["macd"].iloc[-1]
    curr_signal = df["macd_signal"].iloc[-1]

    ema_fast = df["ema_fast"].iloc[-1]
    ema_slow = df["ema_slow"].iloc[-1]
    rsi = df["rsi"].iloc[-1]
    vol = df["volume"].iloc[-1]
    vol_ma = df["vol_ma"].iloc[-1]

    trend_macd = trend_df["macd"].iloc[-1]
    trend_signal = trend_df["macd_signal"].iloc[-1]
    trend_ema_fast = trend_df["ema_fast"].iloc[-1]
    trend_ema_slow = trend_df["ema_slow"].iloc[-1]

    # 金叉 = 从下往上穿越
    if prev_macd <= prev_signal and curr_macd > curr_signal:
        if ema_fast > ema_slow and rsi > 50 and vol > vol_ma:
            if trend_macd > trend_signal and trend_ema_fast > trend_ema_slow:
                print(f"🔔 {df.index[-1]} {df['symbol'].iloc[-1]} 检测到【金叉信号】 → 开多")
                return "buy"

    # 死叉 = 从上往下穿越
    if prev_macd >= prev_signal and curr_macd < curr_signal:
        if ema_fast < ema_slow and rsi < 50 and vol > vol_ma:
            if trend_macd < trend_signal and trend_ema_fast < trend_ema_slow:
                print(f"🔔 {df.index[-1]} {df['symbol'].iloc[-1]} 检测到【死叉信号】 → 开空")
                return "sell"

    return "hold"

# =========================
# 下单函数
# =========================
def place_order(symbol, side, amount):
    try:
        order = exchange.create_market_order(symbol, side, amount)
        print(f"✅ {symbol} 下单成功: {side} {amount}")
        return order
    except Exception as e:
        print(f"❌ 下单失败 {symbol}: {e}")
        return None

# =========================
# 主循环
# =========================
def run_trading():
    while True:
        for symbol in SYMBOLS:
            try:
                # 获取 1h 数据
                ohlcv = exchange.fetch_ohlcv(symbol, "1h", limit=100)
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                df = add_indicators(df)
                df["symbol"] = symbol

                # 获取 4h 数据作为趋势
                ohlcv_trend = exchange.fetch_ohlcv(symbol, "4h", limit=100)
                trend_df = pd.DataFrame(ohlcv_trend, columns=["timestamp", "open", "high", "low", "close", "volume"])
                trend_df["timestamp"] = pd.to_datetime(trend_df["timestamp"], unit="ms")
                trend_df.set_index("timestamp", inplace=True)
                trend_df = add_indicators(trend_df)

                # 生成信号
                signal = signal_from_indicators(df, trend_df)
                if signal == "buy":
                    place_order(symbol, "buy", 0.01)  # ⚠️ 下单数量请根据资金调整
                elif signal == "sell":
                    place_order(symbol, "sell", 0.01)

            except Exception as e:
                print(f"⚠️ 主循环异常 {symbol}: {e}")

        time.sleep(60)  # 每分钟循环一次

# =========================
# 入口
# =========================
if __name__ == "__main__":
    if MODE == "live":
        run_trading()
    else:
        print("⚠️ 当前是回测模式，还没写回测逻辑。")
