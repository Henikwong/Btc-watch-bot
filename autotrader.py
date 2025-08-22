import os
import time
import requests
import certifi
from dotenv import load_dotenv

# 载入 .env 环境变量
load_dotenv()

# ========== 配置 ==========
TG_TOKEN = os.getenv("TG_TOKEN")       # Telegram Bot Token
TG_CHAT_ID = os.getenv("TG_CHAT_ID")   # Telegram Chat ID
EXCHANGE = os.getenv("EXCHANGE", "huobi")  # 默认交易所
MARKET = os.getenv("MARKET", "spot")       # 默认市场类型
LIVE = int(os.getenv("LIVE", 0))           # 0=纸面测试, 1=实盘
SLEEP = int(os.getenv("SLEEP", 30))        # 循环间隔秒

# API Endpoint (Huobi 合约 API)
HUOBI_SWAP_URL = "https://api.hbdm.com/linear-swap-api/v1/swap_contract_info?business_type=all"


# ========== 工具函数 ==========
def tg_send(msg: str):
    """发送消息到 Telegram"""
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        payload = {"chat_id": TG_CHAT_ID, "text": msg}
        requests.post(url, json=payload, timeout=10, verify=certifi.where())
    except Exception as e:
        print("❌ Telegram 发送失败:", e)


def get_huobi_contracts():
    """获取 Huobi 合约信息"""
    try:
        resp = requests.get(HUOBI_SWAP_URL, timeout=10, verify=certifi.where())
        resp.raise_for_status()
        data = resp.json()
        return data
    except Exception as e:
        tg_send(f"⚠️ Huobi API 请求失败: {e}")
        return None


# ========== 主循环 ==========
def run_bot():
    tg_send(f"🤖 Bot启动 {EXCHANGE}/{MARKET} 模式={'实盘' if LIVE else '纸面'}")

    while True:
        try:
            contracts = get_huobi_contracts()
            if contracts:
                tg_send(f"✅ 成功获取 {len(contracts.get('data', []))} 个合约信息")
            else:
                tg_send("⚠️ 没有拿到合约数据")

        except Exception as e:
            tg_send(f"循环异常: {e}")

        time.sleep(SLEEP)


if __name__ == "__main__":
    run_bot()
