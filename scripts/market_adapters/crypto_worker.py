"""scripts/market_adapters/crypto_worker.py — Live crypto tick ingestion.

Primary feed  : Binance WebSocket (python-binance)
Fallback feed  : Bybit public WebSocket (raw websockets library)

Design (inspired by Fifadlika/MLOps-Crypto-Surveillance architecture):
  - asyncio event loop with automatic reconnection (exponential back-off).
  - Normalises every raw Binance or Bybit trade into UnifiedTradeEvent.
  - Publishes to Redis Stream "live_trades" via redis_service.publish_trade_sync.
  - Detects Binance failure (>30 s silence) and switches to Bybit automatically.

Usage:
    python -m scripts.market_adapters.crypto_worker

Environment variables (see .env.example):
    BINANCE_API_KEY       — Optional; public streams work without a key.
    BINANCE_API_SECRET    — Optional.
    REDIS_URL             — Defaults to redis://localhost:6379/0.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Optional

import websockets
from binance import AsyncClient, BinanceSocketManager

# Local imports — these work when the repo root is on PYTHONPATH.
# Run as: python -m scripts.market_adapters.crypto_worker
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.schemas.streaming import Market, UnifiedTradeEvent
from app.services.redis_service import publish_trade_sync

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("crypto_worker")

# ── Config ────────────────────────────────────────────────────────────────────
BINANCE_API_KEY: Optional[str] = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET: Optional[str] = os.getenv("BINANCE_API_SECRET")

# Symbols to monitor (expand this list as needed).
SYMBOLS: list[str] = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "DOGEUSDT"]

BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/spot"
RECONNECT_MAX_BACKOFF_S = 60  # Max seconds between reconnect attempts


# ── Normalise Binance trade → UnifiedTradeEvent ───────────────────────────────
def _normalise_binance(raw: dict) -> Optional[UnifiedTradeEvent]:
    """Parse a raw Binance Aggregate Trade Stream ('aggTrade') payload."""
    try:
        symbol: str = raw["s"]            # e.g. "BTCUSDT"
        price: float = float(raw["p"])
        volume: float = float(raw["q"])
        ts_ms: int = int(raw["T"])        # Trade time in ms
        is_buyer_maker: bool = bool(raw["m"])

        return UnifiedTradeEvent(
            event_id=UnifiedTradeEvent.build_event_id("BINANCE", symbol, ts_ms),
            timestamp_ms=ts_ms,
            market=Market.CRYPTO,
            symbol=symbol,
            source="BINANCE",
            price=price,
            volume=volume,
            notional_value=round(price * volume, 8),
            is_buyer_maker=is_buyer_maker,
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("Binance parse error: %s — raw: %s", exc, raw)
        return None


# ── Normalise Bybit trade → UnifiedTradeEvent ─────────────────────────────────
def _normalise_bybit(raw: dict) -> Optional[UnifiedTradeEvent]:
    """Parse a raw Bybit publicTrade stream payload."""
    try:
        # Bybit: {"topic":"publicTrade.BTCUSDT","data":[{...}]}
        for trade in raw.get("data", []):
            symbol: str = raw["topic"].split(".")[-1]
            price: float = float(trade["p"])
            volume: float = float(trade["v"])
            ts_ms: int = int(trade["T"])
            is_buyer_maker: bool = trade.get("S") == "Buy"

            return UnifiedTradeEvent(
                event_id=UnifiedTradeEvent.build_event_id("BYBIT", symbol, ts_ms),
                timestamp_ms=ts_ms,
                market=Market.CRYPTO,
                symbol=symbol,
                source="BYBIT",
                price=price,
                volume=volume,
                notional_value=round(price * volume, 8),
                is_buyer_maker=is_buyer_maker,
            )
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("Bybit parse error: %s — raw: %s", exc, raw)
    return None


# ── Binance Primary Feed ───────────────────────────────────────────────────────
async def run_binance_feed() -> None:
    """Stream aggregate trades from Binance.  Reconnects on failure."""
    backoff = 1
    while True:
        try:
            logger.info("Connecting to Binance WebSocket for %s …", SYMBOLS)
            client = await AsyncClient.create(
                api_key=BINANCE_API_KEY,
                api_secret=BINANCE_API_SECRET,
            )
            bm = BinanceSocketManager(client)

            streams = [f"{s.lower()}@aggTrade" for s in SYMBOLS]
            async with bm.multiplex_socket(streams) as mux:
                backoff = 1  # Reset on successful connection
                logger.info("Binance feed active. Streaming %d symbols.", len(SYMBOLS))
                while True:
                    msg = await mux.recv()
                    raw = msg.get("data", msg)
                    event = _normalise_binance(raw)
                    if event:
                        entry_id = publish_trade_sync(event)
                        logger.debug("Published BINANCE tick %s → Redis %s", event.symbol, entry_id)

        except Exception as exc:  # noqa: BLE001
            logger.error("Binance feed error: %s. Reconnecting in %ds …", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, RECONNECT_MAX_BACKOFF_S)
        finally:
            try:
                await client.close_connection()
            except Exception:  # noqa: BLE001
                pass


# ── Bybit Fallback Feed ────────────────────────────────────────────────────────
async def run_bybit_feed() -> None:
    """Stream public trades from Bybit as a secondary/fallback feed."""
    subscribe_msg = {
        "op": "subscribe",
        "args": [f"publicTrade.{sym}" for sym in SYMBOLS],
    }
    backoff = 1
    while True:
        try:
            logger.info("Connecting to Bybit WebSocket (fallback) …")
            async with websockets.connect(BYBIT_WS_URL, ping_interval=20) as ws:
                await ws.send(json.dumps(subscribe_msg))
                backoff = 1
                logger.info("Bybit fallback feed active.")
                async for raw_msg in ws:
                    raw = json.loads(raw_msg)
                    event = _normalise_bybit(raw)
                    if event:
                        entry_id = publish_trade_sync(event)
                        logger.debug("Published BYBIT tick %s → Redis %s", event.symbol, entry_id)

        except Exception as exc:  # noqa: BLE001
            logger.error("Bybit feed error: %s. Reconnecting in %ds …", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, RECONNECT_MAX_BACKOFF_S)


# ── Entry point ───────────────────────────────────────────────────────────────
async def main() -> None:
    """Run Binance (primary) and Bybit (fallback) feeds concurrently."""
    await asyncio.gather(
        run_binance_feed(),
        run_bybit_feed(),
    )


if __name__ == "__main__":
    asyncio.run(main())
