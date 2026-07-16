"""app/schemas/streaming.py — Unified event schema for all live market adapters.

Every market adapter (Binance, Alpaca, Upstox, Reddit) must convert its raw
payload into one of these two schemas before publishing to Redis Streams.
This guarantees that anomaly_service.py and run_engine.py deal with one
canonical data structure regardless of source.

Design note (inspired by aryan1078/indian-equities-market-surveillance-platform):
  The schema is intentionally flat — no nested objects — so it can be
  serialised to a Redis Stream field-value pair with a single json.dumps().
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Market(str, Enum):
    CRYPTO = "CRYPTO"
    US_EQUITY = "US_EQUITY"
    INDIA_EQUITY = "INDIA_EQUITY"


class UnifiedTradeEvent(BaseModel):
    """A single normalised trade tick from any exchange / broker.

    Fields:
        event_id        — Globally unique: "<SOURCE>_<SYMBOL>_<timestamp_ms>"
        timestamp_ms    — Unix epoch in milliseconds (UTC).  Always integer.
        market          — Which asset class this tick belongs to.
        symbol          — Normalised ticker: "BTCUSDT", "AAPL", "RELIANCE.NS".
        source          — Raw source name: "BINANCE", "ALPACA", "UPSTOX", etc.
        price           — Last traded price (float).
        volume          — Trade quantity (float).
        notional_value  — price × volume, pre-computed so the engine doesn't
                          have to.
        is_buyer_maker  — True when the buyer was the passive (limit) side.
                          Critical for wash-trading detection.
        market_cap_usd  — Optional enrichment from CoinGecko / FMP.
    """

    event_id: str = Field(..., description="<SOURCE>_<SYMBOL>_<timestamp_ms>")
    timestamp_ms: int = Field(..., gt=0, description="Unix epoch milliseconds (UTC)")
    market: Market
    symbol: str = Field(..., min_length=1, max_length=30)
    source: str = Field(..., description="e.g. BINANCE, BYBIT, ALPACA, UPSTOX")
    price: float = Field(..., gt=0)
    volume: float = Field(..., ge=0)
    notional_value: float = Field(..., ge=0)
    is_buyer_maker: Optional[bool] = None
    market_cap_usd: Optional[float] = None

    @field_validator("symbol", mode="before")
    @classmethod
    def uppercase_symbol(cls, v: str) -> str:
        return v.upper().strip()

    @classmethod
    def build_event_id(cls, source: str, symbol: str, timestamp_ms: int) -> str:
        import uuid
        suffix = uuid.uuid4().hex[:8]
        return f"{source}_{symbol}_{timestamp_ms}_{suffix}"


class SentimentSource(str, Enum):
    FINNHUB_NEWS = "FINNHUB_NEWS"
    REDDIT = "REDDIT"
    GDELT = "GDELT"


class UnifiedSentimentEvent(BaseModel):
    """Normalised news / social sentiment tick.

    Fields:
        event_id        — Unique: "<SOURCE>_<SYMBOL>_<timestamp_ms>"
        timestamp_ms    — Unix epoch in milliseconds (UTC).
        symbol          — Ticker this event is about.
        source          — Where the signal came from.
        sentiment_score — Float in [-1.0, 1.0].  Negative = bearish.
        headline        — Optional raw headline / post title for the audit log.
    """

    event_id: str
    timestamp_ms: int = Field(..., gt=0)
    symbol: str = Field(..., min_length=1, max_length=30)
    source: SentimentSource
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    headline: Optional[str] = Field(None, max_length=500)

    @field_validator("symbol", mode="before")
    @classmethod
    def uppercase_symbol(cls, v: str) -> str:
        return v.upper().strip()
