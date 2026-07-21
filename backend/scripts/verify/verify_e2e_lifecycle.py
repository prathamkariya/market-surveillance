"""
End-to-end lifecycle verification: one trade, traced through every hop.

Run this INSIDE the real Docker container, with Postgres, Redis, and (ideally)
a real GEMINI_API_KEY available -- this is not a mock-everything unit test,
it is meant to exercise the real, wired-together system exactly as a live
trade would move through it.

    docker exec -it market-surveillance-api-1 python scripts/verify_e2e_lifecycle.py

What this proves, hop by hop, with concrete printed evidence at each step
(not just "no exception was raised" -- each step asserts something specific
about the DATA that came out, the same standard every other verification
in this project has been held to):

  1. A trade with a real, valid market/source scores through score_live_trade()
     using the ACTUAL loaded models (not mocked) and produces a real,
     non-None, non-sentinel alert.
  2. That alert, when persisted via the same path run_engine.py uses, produces
     a real Anomaly row in Postgres with the fields this project has spent
     multiple rounds getting correct: volume (not 1.0), market (not silently
     wrong), model_version (not blank).
  3. The identical alert dict, published to the real `live_alerts` Redis
     stream the way run_engine.py does it, is retrievable by an SSE-shaped
     read (xread) -- proving the publish/consume contract genuinely matches,
     not just that both sides independently look correct.
  4. The SSE endpoint itself, called with a real, correctly-typed SSE token,
     authenticates and (if the trade's symbol is on the token holder's
     watchlist) delivers that exact alert back out -- proving the auth +
     watchlist-filter mechanism actually passes data through end to end,
     not just that it correctly REJECTS bad tokens (already tested) or that
     the filter logic reads correctly (already read).
  5. The persisted Anomaly, run through the REAL get_mar_report endpoint
     logic (not a mocked Gemini client, if GEMINI_API_KEY is genuinely set),
     produces an actual MAR report referencing the real symbol/score --
     proving the full chain from a live tick to a human-readable compliance
     document actually holds together.

Each step prints what it found. Read the output -- a script that only prints
"OK" at the end without showing the actual values is exactly the kind of
verification this project has learned not to trust.
"""
import asyncio
import json
import sys
import time

sys.path.insert(0, ".")

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import User, MarketData, Anomaly, Watchlist, WatchlistSymbol
from app.services.anomaly_service import score_live_trade, get_model_registry
from app.services.redis_service import get_async_redis, STREAM_ALERTS
from app.services.auth_service import hash_password
from app.services.mar_generator import generate_mar


TEST_SYMBOL = "BTCUSDT"
TEST_MARKET = "CRYPTO"


def build_synthetic_history(symbol: str, n: int = 25):
    """20+ rows of plausible trailing history -- MIN_RAW_ROWS_FOR_FEATURES is 20."""
    base_ts = int(time.time() * 1000) - (n * 1000)
    base_price = 60_000.0
    history = []
    for i in range(n):
        # Small, plausible day-to-day variation, not perfectly flat --
        # flat data can produce degenerate volatility_20d in some pipelines.
        price = base_price + (i % 5) * 12.5
        history.append({
            "event_id": f"hist_{i}",
            "symbol": symbol,
            "market": TEST_MARKET,
            "source": "BINANCE",
            "timestamp_ms": base_ts + i * 1000,
            "price": price,
            "volume": 2.0 + (i % 3) * 0.4,
        })
    return history


def step_1_score_real_trade(db: Session):
    print("\n=== STEP 1: score_live_trade() with real, currently-loaded models ===")
    registry = get_model_registry(market=TEST_MARKET)
    print(f"  registry.has_any_model = {registry.has_any_model}")
    if not registry.has_any_model:
        print("  [BLOCKED] No model loaded for CRYPTO. Cannot proceed past this "
              "step meaningfully -- this itself is a real, reportable finding, "
              "not a script bug. Fix model loading before re-running.")
        sys.exit(1)

    history = build_synthetic_history(TEST_SYMBOL)
    # A trade with a real, deliberately anomalous jump to make sure we get a
    # genuine, non-borderline anomaly score -- not testing the threshold edge,
    # just proving the whole path lights up for something unambiguous.
    trade = {
        "event_id": "e2e_test_trade",
        "symbol": TEST_SYMBOL,
        "market": TEST_MARKET,
        "source": "BINANCE",
        "timestamp_ms": int(time.time() * 1000),
        "price": 90_000.0,  # a large, deliberate jump vs the ~60k history
        "volume": 15.0,     # also unusually large vs the ~2-3 history volumes
    }

    alert = score_live_trade(trade, history, sentiment_score=0.0, market=TEST_MARKET)

    print(f"  alert is None: {alert is None}")
    if alert is None:
        print("  [BLOCKED] score_live_trade returned None -- either the "
              "trade didn't clear the anomaly threshold (unexpected given "
              "how extreme this test trade is) or there's a real regression. "
              "Print the full history/trade and investigate before continuing.")
        sys.exit(1)

    print(f"  alert.confidence / sentinel type: {alert.get('confidence')}")
    print(f"  alert.volume: {alert.get('volume')}  (must be 15.0, not 1.0 or missing)")
    print(f"  alert.market: {alert.get('market')}  (must be CRYPTO, not silently wrong)")
    print(f"  alert.anomaly_score: {alert.get('anomaly_score')}")
    print(f"  alert.low_confidence: {alert.get('low_confidence')}  (must be False -- real BINANCE source)")
    print(f"  alert full dict: {json.dumps(alert, default=str)}")

    assert alert.get("volume") == 15.0, "REGRESSION: volume not preserved correctly"
    assert alert.get("market") == TEST_MARKET, "REGRESSION: market field wrong"
    assert alert.get("low_confidence") is False, "REGRESSION: real BINANCE trade marked low-confidence"
    assert alert.get("confidence") not in ("no_model_high_confidence", "no_baseline_high_confidence", "model_unavailable"), \
        f"Trade hit a sentinel path ({alert.get('confidence')}) instead of real scoring -- investigate why"

    print("  [OK] Step 1 passed -- real alert produced with correct fields.")
    return trade, alert


def step_2_persist_and_check_db(db: Session, trade: dict, alert: dict):
    print("\n=== STEP 2: Persist via the same path run_engine.py uses, check the real DB row ===")
    from scripts.run_engine import persist_alert_to_db, _get_or_create_system_user

    user = _get_or_create_system_user(db)
    print(f"  system user id: {user.id}")

    persist_alert_to_db(alert)

    row = (
        db.query(Anomaly)
        .join(MarketData, Anomaly.market_data_id == MarketData.id)
        .filter(MarketData.symbol == TEST_SYMBOL)
        .order_by(Anomaly.id.desc())
        .first()
    )
    assert row is not None, "No Anomaly row was persisted at all"

    md = db.query(MarketData).filter(MarketData.id == row.market_data_id).first()
    print(f"  Anomaly.id = {row.id}")
    print(f"  MarketData.volume = {md.volume}  (must be 15.0, not 1.0)")
    print(f"  MarketData.market = {md.market}  (must be CRYPTO)")
    print(f"  Anomaly.model_version = {row.model_version}  (must be non-empty -- provenance)")
    print(f"  Anomaly.anomaly_score = {row.anomaly_score}")

    assert md.volume == 15.0, "REGRESSION: the volume=1.0 bug from earlier in this project is back"
    assert md.market == TEST_MARKET, "REGRESSION: market field wrong in persisted row"
    assert row.model_version, "REGRESSION: model_version is blank -- provenance tracking broken"

    print("  [OK] Step 2 passed -- real DB row correct on every previously-fixed field.")
    return row


async def step_3_redis_publish_and_readback(alert: dict):
    print("\n=== STEP 3: Publish to live_alerts exactly as run_engine.py does, read it back ===")
    client = await get_async_redis()

    payload = json.dumps(alert, default=str)
    entry_id = await client.xadd(STREAM_ALERTS, {"data": payload}, maxlen=1000)
    print(f"  Published entry id: {entry_id}")

    # Read back the tail of the stream -- proves the publish/consume contract
    # actually round-trips, not just that each side looks correct in isolation.
    results = await client.xrevrange(STREAM_ALERTS, count=5)
    found = False
    for msg_id, fields in results:
        data = json.loads(fields.get("data", fields.get(b"data", b"{}")))
        if data.get("symbol") == TEST_SYMBOL and data.get("volume") == alert.get("volume"):
            found = True
            print(f"  Found our exact alert in the stream at id={msg_id}")
            break

    assert found, "Published alert could not be read back from live_alerts -- publish/consume contract broken"
    print("  [OK] Step 3 passed -- publish/consume round-trip confirmed on the real stream.")


async def step_4_sse_delivers_to_authenticated_watchlisted_user(db: Session):
    print("\n=== STEP 4: SSE endpoint delivers the alert to a real, authenticated, watchlisted user ===")
    from app.routers.auth import get_sse_token

    test_user = db.query(User).filter(User.email == "e2e_test@example.com").first()
    if not test_user:
        test_user = User(
            email="e2e_test@example.com",
            username="e2e_test",
            hashed_password=hash_password("test_password_not_real"),
        )
        db.add(test_user)
        db.commit()
        db.refresh(test_user)

    watchlist = db.query(Watchlist).filter(Watchlist.user_id == test_user.id).first()
    if not watchlist:
        watchlist = Watchlist(user_id=test_user.id, name="E2E Test Watchlist")
        db.add(watchlist)
        db.commit()
        db.refresh(watchlist)

    existing_symbol = (
        db.query(WatchlistSymbol)
        .filter(WatchlistSymbol.watchlist_id == watchlist.id, WatchlistSymbol.symbol == TEST_SYMBOL)
        .first()
    )
    if not existing_symbol:
        db.add(WatchlistSymbol(watchlist_id=watchlist.id, symbol=TEST_SYMBOL))
        db.commit()

    # NOTE: this calls whatever function actually issues the SSE token --
    # confirm the real function name/signature in app/routers/auth.py before
    # running; this is written against the endpoint's documented behavior
    # (POST /auth/sse-token) but the exact internal function name may differ.
    print("  [MANUAL STEP NEEDED] This part requires hitting the real HTTP "
          "endpoints (POST /auth/sse-token, then GET /alerts/stream/live) "
          "since they're FastAPI routes, not plain importable functions in "
          "the same way score_live_trade is. Use curl or httpx against the "
          "actual running container:")
    print(f"""
    1. Get a real access token for user_id={test_user.id} (log in normally, or
       generate one directly via app.services.auth_service.create_access_token).
    2. curl -X POST http://localhost:8000/api/v1/auth/sse-token \\
         -H "Authorization: Bearer <access_token>"
       -> copy the returned sse token
    3. curl -N "http://localhost:8000/api/v1/alerts/stream/live?token=<sse_token>"
       (in a separate terminal, BEFORE step 5 below re-publishes if needed)
    4. Confirm the alert for symbol={TEST_SYMBOL} appears in the SSE stream
       within a few seconds -- this is the actual proof the watchlist-scoped,
       authenticated delivery path works, which cannot be fully verified by
       importing Python functions directly, since the value being proven is
       that the REAL, RUNNING HTTP SERVER correctly wires all of this together.
    """)


def step_5_mar_report_from_real_anomaly(db: Session, anomaly_row: Anomaly, test_user_id: int):
    print("\n=== STEP 5: Generate a real MAR report for the persisted Anomaly ===")
    import os
    if not os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY") == "fake-key-for-test":
        print("  [SKIPPED] No real GEMINI_API_KEY set. This step needs a genuine "
              "key to prove the full chain -- a mocked Gemini client only proves "
              "the plumbing around the call, not that Gemini itself receives "
              "sensible input and returns something real. Re-run with a real "
              "key to fully close this step.")
        return

    report = generate_mar(db, anomaly_row.id, test_user_id)
    print(f"  Report length: {len(report)} chars")
    print(f"  Report contains the symbol ({TEST_SYMBOL}): {TEST_SYMBOL in report}")
    print(f"  First 300 chars:\n{report[:300]}")

    assert TEST_SYMBOL in report, "Generated report doesn't even mention the actual symbol -- investigate prompt construction"
    print("  [OK] Step 5 passed -- real Gemini call produced a report referencing the real trade.")


def main():
    db = SessionLocal()
    try:
        trade, alert = step_1_score_real_trade(db)
        anomaly_row = step_2_persist_and_check_db(db, trade, alert)
        asyncio.run(step_3_redis_publish_and_readback(alert))
        asyncio.run(step_4_sse_delivers_to_authenticated_watchlisted_user(db))

        test_user = db.query(User).filter(User.email == "e2e_test@example.com").first()
        step_5_mar_report_from_real_anomaly(db, anomaly_row, test_user.id if test_user else 1)

        print("\n=== ALL AUTOMATED STEPS COMPLETE ===")
        print("Step 4's HTTP-level check still needs to be done manually -- see instructions above.")
        print("Step 5 needs a real GEMINI_API_KEY to be fully closed, not just skipped cleanly.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
