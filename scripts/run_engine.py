"""scripts/run_engine.py — The background ML inference engine.

Design:
  - Continuously reads the "live_trades" Redis Stream.
  - Maintains an in-memory or Redis-backed sliding window of the last 20 
    trades per symbol to feed the rolling-window feature engineering.
  - Reads the latest sentiment for the symbol from Redis.
  - Passes the fused data to anomaly_service.score_live_trade().
  - If anomaly detected, writes back to Postgres/TimescaleDB and publishes
    to a "live_alerts" Redis stream for the WebSocket UI.

Usage:
    python scripts/run_engine.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from socket import gethostname

# Ensure repo root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.database import SessionLocal
from app.models import Anomaly, MarketData, User
from app.services.anomaly_service import score_live_trade
from app.services.redis_service import (
    claim_pending_trades,
    get_async_redis,
    setup_consumer_group,
    read_trades_blocking,
    STREAM_TRADES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_engine")

STREAM_ALERTS = "live_alerts"
HISTORY_PREFIX = "history:"

_last_disarmed_log_time = {}

_last_successful_sentiment = {}
_last_sentiment_warning = {}

async def get_latest_sentiment(client, symbol: str) -> float:
    """Fetch the most recent sentiment score for a symbol."""
    val = await client.hget("latest_sentiment", symbol)
    now = time.time()

    if val is not None:
        try:
            parsed = float(val)
            _last_successful_sentiment[symbol] = now
            return parsed
        except ValueError:
            pass
            
    # Cache miss or invalid value
    # Default to now if we've never seen it, so we don't instantly warn
    last_success = _last_successful_sentiment.setdefault(symbol, now)
    
    if now - last_success > 300:
        last_warn = _last_sentiment_warning.get(symbol, 0)
        if now - last_warn > 300:
            logger.warning("No sentiment data for %s in >5m; defaulting to neutral 0.0", symbol)
            _last_sentiment_warning[symbol] = now
            
    return 0.0


async def update_and_get_history(client, trade: dict) -> list[dict]:
    """Maintain the last 30 ticks for a symbol in a Redis List."""
    key = f"{HISTORY_PREFIX}{trade['symbol']}"
    trade_json = json.dumps(trade)
    
    # Push to right, keep only last 30
    await client.rpush(key, trade_json)
    await client.ltrim(key, -30, -1)
    
    # Fetch all
    raw_history = await client.lrange(key, 0, -1)
    return [json.loads(r) for r in raw_history[:-1]]  # exclude the one we just pushed


async def run():
    logger.info("Initializing ML Engine...")
    await setup_consumer_group("engine_group")
    
    client = get_async_redis()
    consumer_name = os.getenv("ENGINE_CONSUMER_NAME", f"engine_{gethostname()}_{os.getpid()}")
    
    logger.info("Engine listening for live trades...")
    last_claim_check = 0.0
    while True:
        try:
            now = time.time()
            batch = None
            
            # Check for crashed-worker pending trades periodically, not every loop
            if now - last_claim_check > 60:
                batch = await claim_pending_trades(
                    group_name="engine_group",
                    consumer_name=consumer_name,
                    min_idle_ms=60_000,
                    count=50,
                )
                last_claim_check = now
                
            if not batch:
                batch = await read_trades_blocking(
                    group_name="engine_group",
                    consumer_name=consumer_name,
                    count=50,
                    block_ms=2000,
                )
            if not batch:
                continue
            
            for entry_id, trade in batch:
                market = trade.get("market")
                if market is None:
                    logger.error(
                        "Dropping malformed tick: missing 'market' field. symbol=%s source=%s",
                        trade.get("symbol", "?"), trade.get("source", "?")
                    )
                    # Ack it so we don't get stuck in a poison pill loop
                    await client.xack(STREAM_TRADES, "engine_group", entry_id)
                    continue

                # 1. Update history
                history = await update_and_get_history(client, trade)
                
                # 2. Get sentiment
                sentiment = await get_latest_sentiment(client, trade["symbol"])
                
                # 3. Score — route to the correct per-market model registry.
                alert = score_live_trade(trade, history, sentiment, market=market)

                if alert is not None:
                    # model_unavailable sentinel: ingested but no trained model yet.
                    # Log at DEBUG (not WARNING) and skip DB/stream publish.
                    if alert.get("confidence") == "model_unavailable":
                        logger.debug(
                            "No model for market=%s symbol=%s source=%s — coverage pending",
                            alert.get("market"), alert.get("symbol"), alert.get("source"),
                        )
                    elif alert.get("confidence") == "no_model_high_confidence":
                        now = time.time()
                        mkt = alert.get("market")
                        if now - _last_disarmed_log_time.get(mkt, 0) > 300:
                            logger.warning(
                                "ENGINE DISARMED: no model loaded for market=%s — trades are being ingested but NOT scored",
                                mkt
                            )
                            _last_disarmed_log_time[mkt] = now
                    elif alert.get("confidence") == "no_baseline_high_confidence":
                        now = time.time()
                        sym = alert.get("symbol")
                        if now - _last_disarmed_log_time.get(f"baseline_{sym}", 0) > 300:
                            logger.warning(
                                "ENGINE PARTIALLY DISARMED: no valid baseline for symbol=%s — trades are being ingested but NOT scored",
                                sym
                            )
                            _last_disarmed_log_time[f"baseline_{sym}"] = now
                    elif alert.get("anomaly_score") is not None:
                        confidence_tag = " [LOW CONFIDENCE — polled data]" if alert.get("low_confidence") else ""
                        logger.warning(
                            "🚨 ANOMALY DETECTED: %s score=%.2f%s",
                            trade["symbol"], alert["anomaly_score"], confidence_tag,
                        )
                        # 4. Persist first; only publish and ack after durable storage succeeds.
                        anomaly_id = await asyncio.to_thread(persist_alert_to_db, alert)
                        alert["anomaly_id"] = anomaly_id
                        try:
                            # Use the incoming trade's entry_id as the alert's stream ID for idempotency
                            await client.xadd(STREAM_ALERTS, {"data": json.dumps(alert)}, maxlen=1000, id=entry_id)
                        except redis.exceptions.ResponseError as e:
                            if "is equal or smaller" in str(e):
                                logger.info("Alert for entry %s already exists in %s, skipping publish", entry_id, STREAM_ALERTS)
                            else:
                                raise

                # Acknowledge the processed trade
                await client.xack(STREAM_TRADES, "engine_group", entry_id)

                        
        except Exception as e:
            logger.error("Engine loop error: %s", e)
            await asyncio.sleep(1)


def _get_or_create_system_user(db) -> User:
    """Fetch or safely create a dedicated system user for streaming alerts."""
    from sqlalchemy.exc import IntegrityError
    
    SYSTEM_EMAIL = "system_surveillance@example.com"
    user = db.query(User).filter(User.email == SYSTEM_EMAIL).first()
    if user:
        return user
        
    try:
        from passlib.context import CryptContext
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        user = User(
            email=SYSTEM_EMAIL,
            username="system_surveillance",
            hashed_password=pwd_context.hash("system_password_not_used"),
            is_active=True
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    except IntegrityError:
        db.rollback()
        return db.query(User).filter(User.email == SYSTEM_EMAIL).first()

def persist_alert_to_db(alert: dict) -> int:
    """Save the anomaly to TimescaleDB for audit/MAR generation."""
    db = SessionLocal()
    try:
        user = _get_or_create_system_user(db)
        if not user:
            raise RuntimeError("Failed to resolve system surveillance user")
            
        # Create a raw MarketData record so Anomaly has a parent.
        # Note: We intentionally map point-in-time tick price to all OHLC fields
        # because this is a single tick, not an aggregated candle.
        from datetime import datetime, timezone
        alert_timestamp = datetime.fromtimestamp(alert["timestamp_ms"] / 1000.0, tz=timezone.utc)
        
        md = db.query(MarketData).filter_by(
            user_id=user.id,
            symbol=alert["symbol"],
            timestamp=alert_timestamp
        ).first()

        if not md:
            md = MarketData(
                user_id=user.id,
                symbol=alert["symbol"],
                timestamp=alert_timestamp,
                open=alert["price"],
                high=alert["price"],
                low=alert["price"],
                close=alert["price"],
                volume=alert["volume"],
                market=alert.get("market"),
            )
            db.add(md)
            db.flush()
        
        anom = db.query(Anomaly).filter_by(market_data_id=md.id).first()
        if not anom:
            anom = Anomaly(
                market_data_id=md.id,
                anomaly_score=alert["anomaly_score"],
                is_anomaly=True,
                isolation_forest_score=alert.get("isolation_forest_score"),
                multi_pattern_max_score=alert.get("multi_pattern_max_score"),
                pattern_scores=json.dumps(alert.get("pattern_scores")),
                features=json.dumps(alert.get("features")),
                model_version=alert.get("model_version"),
            )
            db.add(anom)
            db.flush()
            
        db.commit()
        db.refresh(anom)
        return anom.id
    except Exception as e:
        logger.error("Failed to persist alert to DB: %s", e)
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(run())
