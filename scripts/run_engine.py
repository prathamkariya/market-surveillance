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

# Ensure repo root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.database import SessionLocal
from app.models import Anomaly, MarketData, User
from app.services.anomaly_service import score_live_trade
from app.services.redis_service import get_async_redis, STREAM_TRADES, STREAM_SENTIMENT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_engine")

STREAM_ALERTS = "live_alerts"
HISTORY_PREFIX = "history:"

async def get_latest_sentiment(client, symbol: str) -> float:
    """Fetch the most recent sentiment score for a symbol."""
    val = await client.hget("latest_sentiment", symbol)
    if val is None:
        return 0.0
    try:
        return float(val)
    except ValueError:
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
    
    logger.info("Engine listening for live trades...")
    while True:
        try:
            batch = await read_trades_blocking(group_name="engine_group", consumer_name="engine_1", count=50, block_ms=2000)
            if not batch:
                continue
            
            for entry_id, trade in batch:
                # 1. Update history
                history = await update_and_get_history(client, trade)
                
                # 2. Get sentiment
                sentiment = await get_latest_sentiment(client, trade["symbol"])
                
                # 3. Score
                alert = score_live_trade(trade, history, sentiment)
                
                if alert:
                    logger.warning("🚨 ANOMALY DETECTED: %s score=%.2f", trade["symbol"], alert["anomaly_score"])
                    
                    # 4. Publish to alerts stream for the UI
                    await client.xadd(STREAM_ALERTS, {"data": json.dumps(alert)}, maxlen=1000)
                    
                    # 5. Persist to Postgres/TimescaleDB without blocking event loop
                    await asyncio.to_thread(persist_alert_to_db, alert)
                    
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

def persist_alert_to_db(alert: dict) -> None:
    """Save the anomaly to TimescaleDB for audit/MAR generation."""
    db = SessionLocal()
    try:
        user = _get_or_create_system_user(db)
        if not user:
            return
            
        # Create a raw MarketData record so Anomaly has a parent.
        # Note: We intentionally map point-in-time tick price to all OHLC fields
        # because this is a single tick, not an aggregated candle.
        from datetime import datetime, timezone
        md = MarketData(
            user_id=user.id,
            symbol=alert["symbol"],
            timestamp=datetime.fromtimestamp(alert["timestamp_ms"] / 1000.0, tz=timezone.utc),
            open=alert["price"],
            high=alert["price"],
            low=alert["price"],
            close=alert["price"],
            volume=alert.get("volume", 1.0),
        )
        db.add(md)
        db.commit()
        db.refresh(md)
        
        anom = Anomaly(
            market_data_id=md.id,
            anomaly_score=alert["anomaly_score"],
            is_anomaly=True,
            isolation_forest_score=alert.get("isolation_forest_score"),
            multi_pattern_max_score=alert.get("multi_pattern_max_score"),
            pattern_scores=json.dumps(alert.get("pattern_scores")),
            features=json.dumps(alert.get("features")),
        )
        db.add(anom)
        db.commit()
    except Exception as e:
        logger.error("Failed to persist alert to DB: %s", e)
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(run())
