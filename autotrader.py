# ==========================================================
# 导入所需的库
# ==========================================================
import os
import time
import ccxt
import ssl

# ==========================================================
# 从环境变量安全地获取API密钥和配置
# ==========================================================
# 密钥和配置应从环境变量中安全读取，而不是硬编码在代码中 [1]。
# 请在 Railway 的“Variables”面板中设置这些变量。
try:
    API_KEY = os.environ.get('HUOBI_API_KEY')
    SECRET_KEY = os.environ.get('HUOBI_SECRET_KEY')
    LIVE_TRADE = int(os.environ.get('LIVE_TRADE', '0')) # '0'表示纸上交易，'1'表示实盘交易
    
    if not API_KEY or not SECRET_KEY:
        raise ValueError("API keys are not set in environment variables.")

except (ValueError, TypeError) as e:
    print(f"Error: {e}")
    # 缺少关键配置，退出程序。
    exit()

# ==========================================================
# 初始化交易所
# ==========================================================
# CCXT 库提供了一个统一的接口来与加密货币交易所交互 [2, 3]。
exchange = ccxt.huobi({
    'apiKey': API_KEY,
    'secret': SECRET_KEY,
})

print("交易脚本已启动...")

# ==========================================================
# 核心交易逻辑循环
# ==========================================================
# 交易脚本作为一个“后台工作者”需要持续运行 [4]。
# `while True` 循环可以防止脚本在执行一次后立即退出。
while True:
    try:
        print("正在获取最新市场数据...")
        
        # 在这里添加您自己的交易逻辑。
        # 以下是获取 BTC/USDT 最新价格的示例。
        ticker = exchange.fetch_ticker('BTC/USDT')
        price = ticker['last']
        
        print(f"BTC/USDT 最新价格：{price}")

        # 根据您的策略执行交易。
        if LIVE_TRADE == 1:
            print("当前模式：实盘交易")
            # 在这里放置您的实盘交易代码
            # 例如: exchange.create_order(...)
        else:
            print("当前模式：纸上交易 (Dry-run)")
            # 在这里放置您的模拟交易代码
        
        # 重要的：添加延迟以避免过于频繁的API调用，并遵守交易所的API限速。
        time.sleep(60) # 每60秒运行一次
        
    except Exception as e:
        # 捕获任何可能发生的错误，并打印出来以便在 Railway 日志中查看。
        print(f"发生错误：{e}")
        # 如果出现错误，等待一段时间后重试，避免连续失败。
        time.sleep(300)

