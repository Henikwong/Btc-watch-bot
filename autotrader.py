#!/usr/bin/env python3
# diagnose.py — 快速诊断 Binance + Telegram + 环境 问题
import os, sys, time, traceback, json
import requests
import ccxt
from datetime import datetime
try:
    import certifi
except Exception:
    certifi = None

def now(): return datetime.utcnow().isoformat() + "Z"
def mask(s):
    if not s: return None
    s = str(s)
    if len(s) <= 8: return "****"
    return s[:4] + "..." + s[-4:]

print("=== DIAGNOSE START:", now(), "===\n")

# 1) 环境变量
env_vars = ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "API_KEY", "API_SECRET",
            "EXCHANGE", "MARKET_TYPE", "SYMBOLS", "BASE_USDT", "LEVERAGE", "LIVE_TRADE"]
print("1) Environment variables (masked):")
for v in env_vars:
    print(f"  {v} = {mask(os.getenv(v))}  present={v in os.environ}")
print("")

# 2) certifi / SSL
print("2) SSL / certifi:")
if certifi:
    print("  certifi available:", certifi.where())
else:
    print("  certifi NOT installed")
print("  requests version:", getattr(requests, '__version__', 'unknown'))
print("")

# 3) network test -> Telegram & Binance endpoints
def try_get(url, name):
    print(f"  -> testing {name}: {url}")
    try:
        r = requests.get(url, timeout=10, verify=(certifi.where() if certifi else True))
        print(f"     status: {r.status_code}  len={len(r.content)}")
        # try to show trimmed json
        try:
            j = r.json()
            print("     json keys:", list(j.keys())[:10])
        except Exception:
            pass
    except Exception as e:
        print("     ERROR:", type(e).__name__, str(e))

tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
if tg_token:
    try_get(f"https://api.telegram.org/bot{tg_token}/getMe", "Telegram getMe")
else:
    print("  Telegram token not set — skipping")
try_get("https://api.binance.com/api/v3/ping", "Binance REST (public)")
try_get("https://fapi.binance.com/fapi/v1/ping", "Binance Futures REST (public)")
print("")

# 4) ccxt exchange & load_markets
print("3) ccxt / exchange tests:")
EXCHANGE = (os.getenv("EXCHANGE", "binance") or "binance").lower()
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
MARKET_TYPE = (os.getenv("MARKET_TYPE") or "future").lower()

print(f"  target exchange: {EXCHANGE}  market_type: {MARKET_TYPE}")
try:
    if EXCHANGE == "binance":
        ex = ccxt.binance({
            "apiKey": API_KEY, "secret": API_SECRET, "enableRateLimit": True,
            "options": {"defaultType": "future" if MARKET_TYPE=="future" else "spot"}
        })
    else:
        ex = getattr(ccxt, EXCHANGE)({"apiKey": API_KEY, "secret": API_SECRET, "enableRateLimit": True})
    print("  created ccxt exchange object:", type(ex).__name__)
except Exception as e:
    print("  FAILED to create exchange:", type(e).__name__, e)
    print(traceback.format_exc())
    sys.exit(1)

# try load_markets
try:
    ex.load_markets()
    print("  load_markets OK, markets count:", len(ex.markets))
except Exception as e:
    print("  load_markets FAILED:", type(e).__name__, e)
    print(traceback.format_exc())
    # don't exit — still try some tests below

# 5) SYMBOLS and small fetch tests
SYM_RAW = os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,LTC/USDT,BNB/USDT,DOGE/USDT")
SYM_RAW = SYM_RAW.replace("，", ",")
SYMBOLS = [s.strip() for s in SYM_RAW.split(",") if s.strip()]
print("\n4) Symbols to test:", SYMBOLS)

for s in SYMBOLS:
    print(f"\n--- testing symbol {s} ---")
    try:
        market = ex.market(s)
        print("  market id:", market.get("id"))
        print("  precision.amount:", market.get("precision", {}).get("amount"))
        limits = market.get("limits", {}).get("amount")
        print("  limits.amount:", limits)
        info = market.get("info")
        if info and "filters" in info:
            for f in info["filters"]:
                if f.get("filterType") in ("LOT_SIZE","MARKET_LOT_SIZE"):
                    print("   filter LOT_SIZE:", {k:f[k] for k in ("minQty","maxQty","stepSize") if k in f})
    except Exception as e:
        print("  market() failed:", type(e).__name__, e)

    # fetch ticker
    try:
        tk = ex.fetch_ticker(s)
        last = tk.get("last")
        print("  ticker last:", last)
    except Exception as e:
        print("  fetch_ticker failed:", type(e).__name__, e)

    # fetch ohlcv (1h, small)
    try:
        o = ex.fetch_ohlcv(s, timeframe="1h", limit=10)
        print("  fetch_ohlcv ok, rows:", len(o))
    except Exception as e:
        print("  fetch_ohlcv failed:", type(e).__name__, e)

    # compute suggested minimum BASE_USDT for this symbol (if minQty available)
    try:
        amt_prec = market.get("precision", {}).get("amount")
        min_qty = (market.get("limits") or {}).get("amount", {}).get("min")
        last_price = tk.get("last") if 'tk' in locals() else None
        if min_qty and last_price:
            suggested_base = float(min_qty) * float(last_price)
            print(f"  min_qty={min_qty}, last={last_price} => suggested BASE_USDT >= {suggested_base:.4f} to meet min order")
        else:
            print("  cannot compute suggested BASE_USDT (missing min_qty or last_price)")
    except Exception as e:
        print("  compute suggested failed:", type(e).__name__, e)

# 6) Telegram send test (do not spam if no token)
print("\n5) Telegram send test (sends one lightweight message if token present):")
if tg_token:
    try:
        r = requests.get(f"https://api.telegram.org/bot{tg_token}/sendMessage",
                         params={"chat_id": os.getenv("TELEGRAM_CHAT_ID"), "text": "Diag ping from diagnose.py"},
                         timeout=10, verify=(certifi.where() if certifi else True))
        print("  Telegram send response:", r.status_code, r.json() if r and r.content else "no content")
    except Exception as e:
        print("  Telegram send failed:", type(e).__name__, e)

print("\n=== DIAGNOSE END:", now(), "===\n")
