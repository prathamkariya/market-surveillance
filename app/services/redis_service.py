"""app/services/redis_service.py — Singleton Redis client for streaming.

Provides two primitives the rest of the codebase uses:
  • publish_trade(event)     — write a UnifiedTradeEvent to "live_trades" stream.
  • publish_sentiment(event) — write a UnifiedSentimentEvent to "live_sentiment".
  • read_stream(...)         — blocking consumer for the detection engine.

Design (inspired by Fifadlika/MLOps-Crypto-Surveillance):
  - We use Redis Streams (XADD / XREAD) rather than plain Pub/Sub because
    Streams are persistent — a worker crash doesn't lose ticks.
  - A single connection pool (redis.asyncio.ConnectionPool) is created once
    at module import time and shared across all coroutines.
  - The synchronous client (redis.Redis) is also exposed for use in
    FastAPI dependencies that still run synchronously (e.g., scripts/).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

import redis
import redis.asyncio as aioredis
from redis.asyncio import ConnectionPool

from app.schemas.streaming import UnifiedSentimentEvent, UnifiedTradeEvent

logger = logging.getLogger(__name__)

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Stream names — keep as module-level constants so both publisher and
# consumer always refer to the same key.
STREAM_TRADES: str = "live_trades"
STREAM_SENTIMENT: str = "live_sentiment"

# Maximum entries to keep in each stream before Redis auto-trims.
# At ~5 000 ticks/sec (Binance burst), 50 000 entries ≈ 10 s of data in RAM.
STREAM_MAXLEN: int = 50_000

# ── Singleton async connection pool ──────────────────────────────────────────
_async_pool: Optional[ConnectionPool] = None


def _get_async_pool() -> ConnectionPool:
    global _async_pool
    if _async_pool is None:
        _async_pool = aioredis.ConnectionPool.from_url(
            REDIS_URL,
            decode_responses=True,
            max_connections=20,
        )
    return _async_pool


def get_async_redis() -> aioredis.Redis:
    """Return an async Redis client backed by the shared pool."""
    return aioredis.Redis(connection_pool=_get_async_pool())


def get_sync_redis() -> redis.Redis:
    """Return a synchronous Redis client (for scripts / workers that don't run
    inside an asyncio loop, e.g. crypto_worker.py)."""
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)


# ── Publishers ────────────────────────────────────────────────────────────────
async def publish_trade(event: UnifiedTradeEvent) -> str:
    """Async: Publish a normalised trade tick to the 'live_trades' stream.

    Returns the Redis entry ID (e.g. "1718000000000-0").
    """
    client = get_async_redis()
    payload = event.model_dump_json()
    entry_id: str = await client.xadd(
        STREAM_TRADES,
        {"data": payload},
        maxlen=STREAM_MAXLEN,
        approximate=True,
    )
    return entry_id


def publish_trade_sync(event: UnifiedTradeEvent) -> str:
    """Sync: Publish a normalised trade tick — use from non-async scripts."""
    client = get_sync_redis()
    payload = event.model_dump_json()
    entry_id: str = client.xadd(
        STREAM_TRADES,
        {"data": payload},
        maxlen=STREAM_MAXLEN,
        approximate=True,
    )
    return entry_id


async def publish_sentiment(event: UnifiedSentimentEvent) -> str:
    """Async: Publish a sentiment event to the 'live_sentiment' stream."""
    client = get_async_redis()
    payload = event.model_dump_json()
    entry_id: str = await client.xadd(
        STREAM_SENTIMENT,
        {"data": payload},
        maxlen=10_000,
        approximate=True,
    )
    await client.hset("latest_sentiment", event.symbol, event.sentiment_score)
    return entry_id

def publish_sentiment_sync(event: UnifiedSentimentEvent) -> str:
    """Sync: Publish a sentiment event and update the latest HSET."""
    client = get_sync_redis()
    payload = event.model_dump_json()
    entry_id: str = client.xadd(
        STREAM_SENTIMENT,
        {"data": payload},
        maxlen=10_000,
        approximate=True,
    )
    client.hset("latest_sentiment", event.symbol, event.sentiment_score)
    return entry_id


async def setup_consumer_group(group_name: str = "engine_group") -> None:
    """Initialize the Redis consumer group for reliable processing."""
    client = get_async_redis()
    try:
        await client.xgroup_create(STREAM_TRADES, group_name, mkstream=True)
        logger.info(f"Consumer group '{group_name}' created on '{STREAM_TRADES}'.")
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" in str(e):
            logger.debug(f"Consumer group '{group_name}' already exists.")
        else:
            raise

async def read_trades_blocking(
    group_name: str = "engine_group",
    consumer_name: str = "engine_1",
    count: int = 100,
    block_ms: int = 1000,
) -> list[tuple[str, dict]]:
    """Block until up to `count` new entries arrive in the live_trades stream via XREADGROUP.

    Returns:
        List of tuples (entry_id, event_dict).
    """
    client = get_async_redis()
    results = await client.xreadgroup(
        group_name,
        consumer_name,
        {STREAM_TRADES: ">"},
        count=count,
        block=block_ms,
    )
    events: list[tuple[str, dict]] = []
    if results:
        # results = [(stream_name, [(entry_id, {field: value}), ...])]
        for _stream, entries in results:
            for entry_id, fields in entries:
                try:
                    events.append((entry_id, json.loads(fields["data"])))
                except (KeyError, json.JSONDecodeError) as exc:
                    logger.warning("Malformed Redis entry: %s — %s", fields, exc)
    return events


async def ping() -> bool:
    """Health-check — returns True if Redis is reachable."""
    try:
        client = get_async_redis()
        return await client.ping()
    except Exception:  # noqa: BLE001
        return False
