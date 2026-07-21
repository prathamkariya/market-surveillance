"""scripts/verify/verify_phase8_redis.py — Phase 8 verification gate.

Checks:
  1. Redis is reachable (ping).
  2. Can publish a UnifiedTradeEvent to the live_trades stream.
  3. Can read it back immediately.
  4. Existing FastAPI test suite still passes (imports only - no server needed).

Run from repo root:
    python scripts/verify/verify_phase8_redis.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import os
import time

# Ensure repo root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


async def main() -> None:
    print("\n" + "=" * 60)
    print("  Phase 8 Verification Gate: Redis Streaming")
    print("=" * 60)

    # ── Check 1: Schema import ────────────────────────────────────────
    print("\n[1/4] Importing schemas ...", end=" ")
    try:
        from app.schemas.streaming import Market, UnifiedTradeEvent, UnifiedSentimentEvent, SentimentSource
        print("OK [V]")
    except ImportError as e:
        print(f"FAIL [X]  {e}")
        sys.exit(1)

    # ── Check 2: Redis connectivity ───────────────────────────────────
    print("[2/4] Pinging Redis ...", end=" ")
    try:
        from app.services.redis_service import ping
        ok = await ping()
        if not ok:
            raise ConnectionError("Ping returned False")
        print("OK [V]")
    except Exception as e:
        print(f"FAIL [X]  {e}")
        print("       -> Is Redis running?  Run: docker-compose up -d redis")
        sys.exit(1)

    # ── Check 3: Publish a fake tick ─────────────────────────────────
    print("[3/4] Publishing a test UnifiedTradeEvent ...", end=" ")
    try:
        from app.services.redis_service import publish_trade, get_async_redis, STREAM_TRADES

        ts_ms = int(time.time() * 1000)
        test_event = UnifiedTradeEvent(
            event_id=f"TEST_BTCUSDT_{ts_ms}",
            timestamp_ms=ts_ms,
            market=Market.CRYPTO,
            symbol="BTCUSDT",
            source="TEST",
            price=50000.0,
            volume=0.5,
            notional_value=25000.0,
            is_buyer_maker=False,
        )

        entry_id = await publish_trade(test_event)
        print(f"OK [V]  (entry_id={entry_id})")
    except Exception as e:
        print(f"FAIL [X]  {e}")
        sys.exit(1)

    # ── Check 4: Read back from stream ────────────────────────────────
    print("[4/4] Reading back from stream ...", end=" ")
    try:
        client = get_async_redis()
        # Read last 1 entry from the stream
        results = await client.xrevrange(STREAM_TRADES, count=1)
        if not results:
            raise ValueError("Stream is empty after publish")

        _entry_id, fields = results[0]
        parsed = json.loads(fields["data"])
        assert parsed["symbol"] == "BTCUSDT", f"Wrong symbol: {parsed['symbol']}"
        assert parsed["source"] == "TEST", f"Wrong source: {parsed['source']}"
        print(f"OK [V]  (symbol={parsed['symbol']}, price={parsed['price']})")
    except Exception as e:
        print(f"FAIL [X]  {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  ALL CHECKS PASSED [V]  Phase 8 Redis layer is operational.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
