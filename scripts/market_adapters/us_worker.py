"""scripts/market_adapters/us_worker.py — US equities live tick ingestion.

Primary feed   : Alpaca WebSocket (IEX real-time, free tier) via alpaca-py.
Fallback feed  : Finnhub WebSocket (60 req/min free tier).

Design (inspired by sushi1507/market-surveillance-demo):
  - Normalises Alpaca trade events → UnifiedTradeEvent.
  - Normalises Finnhub trade events → UnifiedTradeEvent.
  - Publishes to Redis Stream "live_trades" via redis_service.publish_trade_sync.

Usage:
    python -m scripts.market_adapters.us_worker

Environment variables (see .env.example):
    ALPACA_API_KEY, ALPACA_API_SECRET — Free Alpaca account credentials.
    FINNHUB_API_KEY                   — Free Finnhub API key.
    REDIS_URL                         — Defaults to redis://localhost:6379/0.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Optional

import websockets
from alpaca.data.live import StockDataStream

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.schemas.streaming import Market, UnifiedTradeEvent
from app.services.redis_service import publish_trade_sync

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("us_worker")

# ── Config ────────────────────────────────────────────────────────────────────
ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
ALPACA_API_SECRET: str = os.getenv("ALPACA_API_SECRET", "")
FINNHUB_API_KEY: str = os.getenv("FINNHUB_API_KEY", "")

# Symbols to monitor (surveillance watchlist).
US_SYMBOLS: list[str] = ["AAPL", "MSFT", "TSLA", "NVDA", "SPY", "AMZN", "META", "GOOGL"]

FINNHUB_WS_URL = "wss://ws.finnhub.io"
RECONNECT_MAX_BACKOFF_S = 60


# ── Alpaca Primary Feed ───────────────────────────────────────────────────────
async def _alpaca_trade_handler(trade) -> None:
    """Callback invoked by alpaca-py for every trade tick."""
    try:
        ts_ms = int(trade.timestamp.timestamp() * 1000)
        event = UnifiedTradeEvent(
            event_id=UnifiedTradeEvent.build_event_id("ALPACA", trade.symbol, ts_ms),
            timestamp_ms=ts_ms,
            market=Market.US_EQUITY,
            symbol=trade.symbol,
            source="ALPACA",
            price=float(trade.price),
            volume=float(trade.size),
            notional_value=round(float(trade.price) * float(trade.size), 6),
            is_buyer_maker=None,  # Alpaca IEX doesn't expose maker/taker
        )
        entry_id = publish_trade_sync(event)
        logger.debug("Published ALPACA tick %s → Redis %s", trade.symbol, entry_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Alpaca handler error: %s", exc)


async def run_alpaca_feed() -> None:
    """Subscribe to Alpaca IEX real-time trade stream (free tier)."""
    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        logger.warning("ALPACA_API_KEY / ALPACA_API_SECRET not set — skipping Alpaca feed.")
        return

    backoff = 1
    while True:
        try:
            logger.info("Connecting to Alpaca WebSocket for %s …", US_SYMBOLS)
            stream = StockDataStream(api_key=ALPACA_API_KEY, secret_key=ALPACA_API_SECRET)
            stream.subscribe_trades(_alpaca_trade_handler, *US_SYMBOLS)
            backoff = 1
            logger.info("Alpaca feed active.")
            await stream.run()  # Blocks until disconnected

        except Exception as exc:  # noqa: BLE001
            logger.error("Alpaca feed error: %s. Reconnecting in %ds …", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, RECONNECT_MAX_BACKOFF_S)


# ── Finnhub Fallback Feed ─────────────────────────────────────────────────────
def _normalise_finnhub(raw: dict) -> list[UnifiedTradeEvent]:
    """Parse a Finnhub WebSocket trade payload."""
    events: list[UnifiedTradeEvent] = []
    try:
        # Finnhub: {"type":"trade","data":[{"p":150.5,"s":"AAPL","t":1718000000000,"v":100}]}
        for trade in raw.get("data", []):
            symbol: str = trade["s"]
            price: float = float(trade["p"])
            volume: float = float(trade["v"])
            ts_ms: int = int(trade["t"])
            events.append(
                UnifiedTradeEvent(
                    event_id=UnifiedTradeEvent.build_event_id("FINNHUB", symbol, ts_ms),
                    timestamp_ms=ts_ms,
                    market=Market.US_EQUITY,
                    symbol=symbol,
                    source="FINNHUB",
                    price=price,
                    volume=volume,
                    notional_value=round(price * volume, 6),
                    is_buyer_maker=None,
                )
            )
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("Finnhub parse error: %s — raw: %s", exc, raw)
    return events


async def run_finnhub_feed() -> None:
    """Subscribe to Finnhub WebSocket trade stream (fallback)."""
    if not FINNHUB_API_KEY:
        logger.warning("FINNHUB_API_KEY not set — skipping Finnhub feed.")
        return

    ws_url = f"{FINNHUB_WS_URL}?token={FINNHUB_API_KEY}"
    backoff = 1
    while True:
        try:
            logger.info("Connecting to Finnhub WebSocket (fallback) …")
            async with websockets.connect(ws_url, ping_interval=20) as ws:
                for sym in US_SYMBOLS:
                    await ws.send(json.dumps({"type": "subscribe", "symbol": sym}))
                backoff = 1
                logger.info("Finnhub fallback feed active.")
                async for raw_msg in ws:
                    raw = json.loads(raw_msg)
                    if raw.get("type") == "trade":
                        event = _normalise_finnhub(raw)
                        if event:
                            entry_id = publish_trade_sync(event)
                            logger.debug("Published FINNHUB tick %s → Redis %s", event.symbol, entry_id)

        except Exception as exc:  # noqa: BLE001
            logger.error("Finnhub feed error: %s. Reconnecting in %ds …", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, RECONNECT_MAX_BACKOFF_S)


# ── Entry point ───────────────────────────────────────────────────────────────
async def main() -> None:
    await asyncio.gather(
        run_alpaca_feed(),
        run_finnhub_feed(),
    )


if __name__ == "__main__":
    asyncio.run(main())
