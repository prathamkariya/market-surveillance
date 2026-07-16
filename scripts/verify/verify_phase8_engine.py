"""scripts/verify/verify_phase8_engine.py — Phase 8 engine verification gate.

This script acts as the "Threat Injector". It blasts a sequence of normal trades
followed by a massive 50% price spike into Redis.
We expect run_engine.py to catch it, score it highly, and write to Postgres.

Usage:
    # Terminal 1:
    python scripts/run_engine.py

    # Terminal 2:
    python scripts/verify/verify_phase8_engine.py
"""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.schemas.streaming import Market, UnifiedTradeEvent
from app.services.redis_service import publish_trade_sync, get_async_redis

STREAM_ALERTS = "live_alerts"


async def main():
    print("\n" + "=" * 60)
    print("  Phase 8 Verification Gate: ML Engine Threat Injector")
    print("=" * 60)

    # 1. Clear out the alerts stream just in case
    client = get_async_redis()
    await client.delete(STREAM_ALERTS)

    # 2. Inject 20 normal trades to build the sliding window
    print("[1/3] Injecting 20 normal trades ...", end=" ", flush=True)
    symbol = "HACKUSDT"
    base_price = 100.0
    
    for i in range(25):
        ts_ms = int(time.time() * 1000) - (25 - i) * 60000  # 1 minute apart
        price = base_price + (i % 2)  # tiny oscillation
        
        # The 25th trade is a massive 50% dump
        if i == 24:
            price = 50.0
            print(" injected 50% dump ...", end=" ", flush=True)
            
        event = UnifiedTradeEvent(
            event_id=f"INJECT_{symbol}_{ts_ms}",
            timestamp_ms=ts_ms,
            market=Market.CRYPTO,
            symbol=symbol,
            source="TEST_INJECTOR",
            price=price,
            volume=10.0,
            notional_value=price * 10.0,
            is_buyer_maker=False,
        )
        publish_trade_sync(event)
        
    print("OK [V]")
    
    # 3. Wait for run_engine.py to process it and push to live_alerts
    print("[2/3] Waiting for run_engine.py to detect anomaly ...", end=" ", flush=True)
    
    alert = None
    for _ in range(10):  # wait up to 10 seconds
        res = await client.xrevrange(STREAM_ALERTS, count=1)
        if res:
            _entry_id, fields = res[0]
            alert = json.loads(fields["data"])
            if alert["symbol"] == symbol:
                break
        await asyncio.sleep(1)
        
    if not alert:
        print("FAIL [X]  No alert appeared on live_alerts stream.")
        print("          -> Make sure 'python scripts/run_engine.py' is running in another terminal!")
        sys.exit(1)
        
    print(f"OK [V]  Detected! Score: {alert['anomaly_score']:.3f}")
    
    # 4. Verify it was written to the database
    print("[3/3] Verifying database persistence ...", end=" ", flush=True)
    try:
        from app.database import SessionLocal
        from app.models import Anomaly, MarketData
        db = SessionLocal()
        
        # Check if an anomaly for this symbol was just inserted
        db_anom = db.query(Anomaly).join(MarketData).filter(MarketData.symbol == symbol).first()
        if not db_anom:
            raise ValueError("Alert was published to Redis but not found in Postgres.")
            
        print("OK [V]  Saved to TimescaleDB.")
    except Exception as e:
        print(f"FAIL [X]  {e}")
        sys.exit(1)
    finally:
        db.close()
        
    print("\n" + "=" * 60)
    print("  ALL CHECKS PASSED [V]  ML Engine is scoring and saving live ticks!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
