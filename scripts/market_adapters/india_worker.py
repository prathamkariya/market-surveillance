"""scripts/market_adapters/india_worker.py — Indian equities live tick ingestion.

Primary feed   : Upstox WebSocket (free broker API) via upstox-python-sdk.
Fallback feed  : yfinance polling (30-second candles as backup).

Design (inspired by aryan1078/indian-equities-market-surveillance-platform):
  - Upstox uses an OAuth2 access token (stored in UPSTOX_ACCESS_TOKEN env var).
    See https://upstox.com/trading-api/ for setup instructions.
  - yfinance fallback is used outside market hours or if Upstox is unavailable.
  - Normalises every tick → UnifiedTradeEvent → Redis "live_trades" stream.

Usage:
    python -m scripts.market_adapters.india_worker

Environment variables (see .env.example):
    UPSTOX_ACCESS_TOKEN — Short-lived OAuth token from Upstox.
    REDIS_URL           — Defaults to redis://localhost:6379/0.

Note: NSE market hours are Mon–Fri 09:15–15:30 IST.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.schemas.streaming import Market, UnifiedTradeEvent
from app.services.redis_service import publish_trade_sync

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("india_worker")

# ── Config ────────────────────────────────────────────────────────────────────
UPSTOX_ACCESS_TOKEN: str = os.getenv("UPSTOX_ACCESS_TOKEN", "")

# NSE symbols to monitor (Upstox format: NSE_EQ|RELIANCE, etc.)
INDIA_SYMBOLS_UPSTOX: list[str] = [
    "NSE_EQ|INE002A01018",  # Reliance
    "NSE_EQ|INE040A01034",  # HDFC Bank
    "NSE_EQ|INE009A01021",  # Infosys
    "NSE_EQ|INE467B01029",  # TCS
    "NSE_EQ|INE030A01027",  # ICICI Bank
]

# yfinance symbols (for fallback polling)
INDIA_SYMBOLS_YF: list[str] = [
    "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "ICICIBANK.NS",
]

RECONNECT_MAX_BACKOFF_S = 60

# Track primary feed health (grace period at boot to avoid flooding)
last_seen_primary: float = time.time()


# ── Upstox Primary Feed ───────────────────────────────────────────────────────
async def run_upstox_feed() -> None:
    """Connect to Upstox WebSocket for live NSE/BSE ticks.
    
    Upstox provides a market data WebSocket at wss://api.upstox.com/v2/feed/market-data-feed
    with the access token passed as a header.
    """
    if not UPSTOX_ACCESS_TOKEN:
        logger.warning(
            "UPSTOX_ACCESS_TOKEN not set — skipping Upstox feed.\n"
            "  1. Register at https://upstox.com/trading-api/\n"
            "  2. Complete OAuth and paste the token in .env as UPSTOX_ACCESS_TOKEN."
        )
        return

    import websockets

    ws_url = "wss://api.upstox.com/v2/feed/market-data-feed"
    headers = {
        "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
        "Api-Version": "2.0",
    }

    # Subscription request payload
    subscribe_payload = {
        "guid": "market_surveillance_india",
        "method": "sub",
        "data": {
            "mode": "ltpc",  # Last Traded Price + Close
            "instrumentKeys": INDIA_SYMBOLS_UPSTOX,
        },
    }

    backoff = 1
    while True:
        try:
            logger.info("Connecting to Upstox WebSocket …")
            async with websockets.connect(ws_url, extra_headers=headers) as ws:
                await ws.send(json.dumps(subscribe_payload))
                backoff = 1
                logger.info("Upstox feed active. Monitoring %d NSE symbols.", len(INDIA_SYMBOLS_UPSTOX))

                async for raw_msg in ws:
                    try:
                        data = json.loads(raw_msg)
                        feeds = data.get("feeds", {})
                        for instrument_key, feed_data in feeds.items():
                            ltpc = feed_data.get("ltpc", {})
                            if not ltpc:
                                continue

                            price = float(ltpc.get("ltp", 0))
                            ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                            # Use the last part of the instrument key as symbol
                            symbol_raw = instrument_key.split("|")[-1] if "|" in instrument_key else instrument_key

                            if price <= 0:
                                continue

                            event = UnifiedTradeEvent(
                                event_id=UnifiedTradeEvent.build_event_id("UPSTOX", symbol_raw, ts_ms),
                                timestamp_ms=ts_ms,
                                market=Market.INDIA_EQUITY,
                                symbol=symbol_raw,
                                source="UPSTOX",
                                price=price,
                                volume=float(ltpc.get("vol", 0)),
                                notional_value=round(price * float(ltpc.get("vol", 0)), 2),
                                is_buyer_maker=None,
                            )
                            
                            global last_seen_primary
                            last_seen_primary = time.time()
                            
                            entry_id = publish_trade_sync(event)
                            logger.debug("Published UPSTOX tick %s → Redis %s", event.symbol, entry_id)

                    except (KeyError, ValueError, json.JSONDecodeError) as exc:
                        logger.warning("Upstox parse error: %s", exc)

        except Exception as exc:  # noqa: BLE001
            logger.error("Upstox feed error: %s. Reconnecting in %ds …", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, RECONNECT_MAX_BACKOFF_S)


# ── yfinance Fallback Feed (polling) ─────────────────────────────────────────
async def run_yfinance_fallback(poll_interval_s: int = 30) -> None:
    """Poll yfinance every `poll_interval_s` seconds as a fallback.

    Note: This is a workaround for the lack of an Upstox OAuth refresh flow.
    It provides 30-second polling rather than real streaming when the token expires.
    """
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        logger.warning("yfinance not installed — skipping India fallback feed. Run: pip install yfinance")
        return

    logger.info("yfinance fallback polling every %ds for %s", poll_interval_s, INDIA_SYMBOLS_YF)
    while True:
        try:
            tickers = yf.download(
                tickers=INDIA_SYMBOLS_YF,
                period="1d",
                interval="1m",
                progress=False,
                auto_adjust=True,
            )

            if not tickers.empty:
                for sym in INDIA_SYMBOLS_YF:
                        try:
                            # 2.2: Stale data filtering
                            latest_idx = tickers["Close"][sym].dropna().index[-1]
                            latest_ts_s = latest_idx.timestamp()
                            if time.time() - latest_ts_s > 300: # 5 minutes stale
                                continue
                                
                            latest = tickers["Close"][sym].dropna().iloc[-1]
                            vol = tickers["Volume"][sym].dropna().iloc[-1]
                            ts_ms = int(latest_ts_s * 1000)

                            event = UnifiedTradeEvent(
                                event_id=UnifiedTradeEvent.build_event_id("YFINANCE", sym, ts_ms),
                                timestamp_ms=ts_ms,
                                market=Market.INDIA_EQUITY,
                                symbol=sym,
                                source="YFINANCE",
                                price=float(latest),
                                volume=float(vol),
                                notional_value=round(float(latest) * float(vol), 2),
                                is_buyer_maker=None,
                            )
                            
                            # 1.4: Only publish if primary feed is dead
                            if time.time() - last_seen_primary > 30:
                                entry_id = publish_trade_sync(event)
                                logger.debug("Published YFINANCE %s → Redis %s", sym, entry_id)
                        except Exception as sym_exc:  # noqa: BLE001
                            logger.warning("yfinance error for %s: %s", sym, sym_exc)

        except Exception as exc:  # noqa: BLE001
            logger.error("yfinance polling error: %s", exc)

        await asyncio.sleep(poll_interval_s)


# ── Entry point ───────────────────────────────────────────────────────────────
async def main() -> None:
    await asyncio.gather(
        run_upstox_feed(),
        run_yfinance_fallback(),
    )


if __name__ == "__main__":
    asyncio.run(main())
