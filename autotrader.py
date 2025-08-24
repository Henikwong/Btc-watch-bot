import ccxt
import math

exchange = ccxt.binanceusdm({
    "apiKey": "你的API_KEY",
    "secret": "你的API_SECRET",
    "enableRateLimit": True
})

# 自动预算函数
def calculate_order_size(symbol, balance, risk_pct=0.01, leverage=10, atr=50, price=50000):
    risk_amount = balance * risk_pct  # 账户风险资金
    contract_size = (risk_amount * leverage) / (atr)  # ATR 风控法
    usdt_value = contract_size * atr
    qty = usdt_value / price
    return round(qty, 3)

# 示例：下单
def place_order(symbol, side, qty, price=None):
    params = {"type": "MARKET"}
    if side == "buy":
        order = exchange.create_market_buy_order(symbol, qty, params)
    else:
        order = exchange.create_market_sell_order(symbol, qty, params)
    return order

# 测试
balance = exchange.fetch_balance()["total"]["USDT"]
atr = 100  # 这里假设 ATR 已经算好
price = exchange.fetch_ticker("BTC/USDT:USDT")["last"]

qty = calculate_order_size("BTC/USDT:USDT", balance, risk_pct=0.01, leverage=10, atr=atr, price=price)
print("下单数量:", qty)

place_order("BTC/USDT:USDT", "buy", qty)
