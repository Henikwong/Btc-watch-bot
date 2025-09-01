import asyncio import ccxt import logging import os import sys import json import random import time import uuid from datetime import datetime, timedelta from typing import Dict, List, Optional, Tuple, Any from enum import Enum from dataclasses import dataclass

import pandas as pd import numpy as np

=====================

Configuration

=====================

class Config: EXCHANGE = os.getenv("EXCHANGE", "binance") SYMBOLS = os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT").split(",") TIMEFRAMES = os.getenv("TIMEFRAMES", "1h,4h").split(",") INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", 10000))

RISK_RATIO = float(os.getenv("RISK_RATIO", 0.05))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", 2.0))
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", 3.0))
MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", 0.25))
DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", 0.1))

LEVERAGE = int(os.getenv("LEVERAGE", 10))
MARGIN_TYPE = os.getenv("MARGIN_TYPE", "ISOLATED")

COMMISSION_RATE = float(os.getenv("COMMISSION_RATE", 0.0004))
SLIPPAGE_RATIO = float(os.getenv("SLIPPAGE_RATIO", 0.0002))
PARTIAL_TP_RATIO = float(os.getenv("PARTIAL_TP_RATIO", 0.5))

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 5))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 5))

STATE_FILE = "trading_state.json"

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

HEDGE_MODE = os.getenv("HEDGE_MODE", "true").lower() == "true"

# Telegram、Optuna 略...

=====================

Logger

=====================

def setup_logger(name: str) -> logging.Logger: logger = logging.getLogger(name) logger.setLevel(logging.DEBUG) fh = logging.FileHandler(os.path.join(Config.LOG_DIR, f"{name}.log")) fh.setLevel(logging.DEBUG) ch = logging.StreamHandler(sys.stdout) ch.setLevel(logging.INFO) formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s") fh.setFormatter(formatter) ch.setFormatter(formatter) logger.addHandler(fh) logger.addHandler(ch) return logger

logger = setup_logger("autotrader")

=====================

Enums & Data Classes

=====================

class OrderSide(str, Enum): BUY = "buy" SELL = "sell"

@dataclass class TradeSignal: symbol: str side: OrderSide price: float atr: float quantity: float timestamp: datetime confidence: float = 1.0 timeframe: str = "1h"

def to_dict(self):
    return {
        "symbol": self.symbol,
        "side": self.side.value,
        "price": self.price,
        "atr": self.atr,
        "quantity": self.quantity,
        "timestamp": self.timestamp.isoformat(),
        "confidence": self.confidence,
        "timeframe": self.timeframe,
    }

@staticmethod
def from_dict(d: Dict[str, Any]):
    return TradeSignal(
        symbol=d["symbol"],
        side=OrderSide(d["side"]),
        price=d["price"],
        atr=d["atr"],
        quantity=d["quantity"],
        timestamp=datetime.fromisoformat(d["timestamp"]),
        confidence=d.get("confidence", 1.0),
        timeframe=d.get("timeframe", "1h"),
    )

=====================

Exchange Wrapper

=====================

class ExchangeInterface: def init(self, exchange_id: str): self.exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True}) self.logger = setup_logger("exchange")

async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    for attempt in range(Config.MAX_RETRIES):
        try:
            ohlcv = await asyncio.to_thread(
                self.exchange.fetch_ohlcv, symbol, timeframe, None, limit
            )
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).tz_convert(None)
            df.set_index("datetime", inplace=True)
            return df
        except Exception as e:
            if attempt == Config.MAX_RETRIES - 1:
                self.logger.error(f"获取K线失败 {symbol}: {e}")
                raise
            await asyncio.sleep(Config.RETRY_DELAY * (attempt + 1))

async def create_order(
    self, symbol: str, order_type: str, side: str, qty: float, price=None, params=None
) -> Dict[str, Any]:
    params = params or {}
    params["newClientOrderId"] = f"bot-{uuid.uuid4().hex[:16]}"
    return await asyncio.to_thread(
        self.exchange.create_order, symbol, order_type, side, qty, price, params
    )

=====================

Trade Executor

=====================

class TradeExecutor: def init(self, exchange: ExchangeInterface): self.exchange = exchange self.logger = setup_logger("trade_executor")

async def place_tp_order(self, signal: TradeSignal) -> bool:
    try:
        tp_price = signal.price + signal.atr * Config.TP_ATR_MULT if signal.side == OrderSide.BUY else signal.price - signal.atr * Config.TP_ATR_MULT
        params = {
            "stopPrice": float(self.exchange.exchange.price_to_precision(signal.symbol, tp_price)),
            "reduceOnly": True,
        }
        if Config.HEDGE_MODE:
            params["positionSide"] = "LONG" if signal.side == OrderSide.BUY else "SHORT"
        order_side = "sell" if signal.side == OrderSide.BUY else "buy"
        await self.exchange.create_order(signal.symbol, "take_profit_market", order_side, signal.quantity, None, params)
        return True
    except Exception as e:
        self.logger.error(f"止盈单失败 {signal.symbol}: {e}")
        return False

async def place_sl_order(self, signal: TradeSignal) -> bool:
    try:
        sl_price = signal.price - signal.atr * Config.SL_ATR_MULT if signal.side == OrderSide.BUY else signal.price + signal.atr * Config.SL_ATR_MULT
        params = {
            "stopPrice": float(self.exchange.exchange.price_to_precision(signal.symbol, sl_price)),
            "reduceOnly": True,
        }
        if Config.HEDGE_MODE:
            params["positionSide"] = "LONG" if signal.side == OrderSide.BUY else "SHORT"
        order_side = "sell" if signal.side == OrderSide.BUY else "buy"
        await self.exchange.create_order(signal.symbol, "stop_market", order_side, signal.quantity, None, params)
        return True
    except Exception as e:
        self.logger.error(f"止损单失败 {signal.symbol}: {e}")
        return False

def calculate_position_size(self, balance: float, price: float, atr: float) -> float:
    try:
        if atr <= 0 or price <= 0:
            return 0.0
        risk_amount = balance * Config.RISK_RATIO
        risk_per_unit = atr * Config.SL_ATR_MULT
        if risk_per_unit <= 0:
            return 0.0
        position_size = risk_amount / risk_per_unit
        max_notional = balance * Config.LEVERAGE
        max_position = max_notional / price
        position_size = min(position_size, max_position)
        return max(0.0, position_size)
    except Exception:
        return 0.0

=====================

State Manager

=====================

class StateManager: def init(self, state_file: str): self.state_file = state_file self.state: Dict[str, Any] = {} self._load_state()

def _load_state(self):
    if os.path.exists(self.state_file):
        try:
            with open(self.state_file, "r") as f:
                self.state = json.load(f)
        except Exception:
            self.state = {}

def save_state(self):
    try:
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)
    except Exception as e:
        logger.error(f"保存状态失败: {e}")

def set_state(self, key: str, value: Any):
    self.state[key] = value
    self.save_state()

def get_state(self, key: str, default: Any = None) -> Any:
    return self.state.get(key, default)

=====================

Main Entrypoint (demo)

=====================

async def main(): exchange = ExchangeInterface(Config.EXCHANGE) state_manager = StateManager(Config.STATE_FILE) executor = TradeExecutor(exchange)

# demo: fetch K线, 生成信号, 下单
df = await exchange.fetch_ohlcv("BTC/USDT", "1h", 200)
if not df.empty:
    last_price = df["close"].iloc[-1]
    atr = df["high"].rolling(14).max().iloc[-1] - df["low"].rolling(14).min().iloc[-1]
    sig = TradeSignal(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        price=last_price,
        atr=atr,
        quantity=0.001,
        timestamp=datetime.utcnow(),
    )
    await executor.place_tp_order(sig)
    await executor.place_sl_order(sig)

if name == "main": asyncio.run(main())
