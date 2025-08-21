import os
import requests
import base64
import hashlib
import hmac
import datetime
from dotenv import load_dotenv
from urllib.parse import urlencode

# 读取 API key
load_dotenv()
API_KEY = os.getenv("HUOBI_API_KEY")
SECRET_KEY = os.getenv("HUOBI_SECRET_KEY")

BASE_URL = "https://api.huobi.pro"

# 签名函数
def create_signature(params, method, host, path, secret_key):
    sorted_params = sorted(params.items(), key=lambda d: d[0], reverse=False)
    encode_params = urlencode(sorted_params)
    payload = "\n".join([method, host, path, encode_params])
    digest = hmac.new(secret_key.encode("utf-8"),
                      payload.encode("utf-8"),
                      digestmod=hashlib.sha256).digest()
    signature = base64.b64encode(digest).decode()
    return signature

# 测试账户信息
def get_account_info():
    path = "/v1/account/accounts"
    method = "GET"
    host = "api.huobi.pro"
    url = BASE_URL + path

    params = {
        "AccessKeyId": API_KEY,
        "SignatureMethod": "HmacSHA256",
        "SignatureVersion": "2",
        "Timestamp": datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
    }

    params["Signature"] = create_signature(params, method, host, path, SECRET_KEY)

    resp = requests.get(url, params=params)
    return resp.json()

if __name__ == "__main__":
    print("测试 Huobi API ...")
    result = get_account_info()
    print(result)
