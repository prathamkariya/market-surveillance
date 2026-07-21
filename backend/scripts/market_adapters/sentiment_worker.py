"""scripts/market_adapters/sentiment_worker.py — News and social sentiment ingestion.

Sources:
  • Finnhub News WebSocket — company news in real-time.
  • Reddit PRAW (r/wallstreetbets, r/stocks, r/IndiaInvestments) — retail sentiment.

Design (inspired by ARGUS/market-surveillance approach of fusing news with price data):
  - Publishes to Redis Stream "live_sentiment" via redis_service.publish_sentiment.
  - Sentiment score: naive keyword-based scoring (-1.0 to +1.0).
    Replace with a proper FinBERT model in Phase 3 for production quality.

Usage:
    python -m scripts.market_adapters.sentiment_worker

Environment variables:
    FINNHUB_API_KEY       — Free tier provides news.
    REDDIT_CLIENT_ID      — From https://www.reddit.com/prefs/apps
    REDDIT_CLIENT_SECRET  — From Reddit app settings.
    REDDIT_USER_AGENT     — e.g. "MarketSurveillance/1.0"
    REDIS_URL             — Defaults to redis://localhost:6379/0.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.schemas.streaming import SentimentSource, UnifiedSentimentEvent
from app.services.redis_service import publish_sentiment_sync

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sentiment_worker")

# ── Config ────────────────────────────────────────────────────────────────────
FINNHUB_API_KEY: str = os.getenv("FINNHUB_API_KEY", "")
REDDIT_CLIENT_ID: str = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET: str = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT: str = os.getenv("REDDIT_USER_AGENT", "MarketSurveillance/1.0")

# Subreddits to monitor
SUBREDDITS: list[str] = ["wallstreetbets", "stocks", "investing", "IndiaInvestments", "IndiaAlgoTrading"]

# Rough sentiment keywords (replace with FinBERT in Phase 3)
_POSITIVE_WORDS = {"bullish", "moon", "buy", "calls", "long", "surge", "rally", "breakout", "pump", "up"}
_NEGATIVE_WORDS = {"bearish", "short", "puts", "crash", "dump", "sell", "tank", "drop", "plummet", "down"}

# Symbols to watch for in text (broad list; refine to your watchlist)
WATCH_SYMBOLS = {
    "BTC", "ETH", "BTCUSDT", "AAPL", "TSLA", "NVDA", "SPY", "GME",
    "AMC", "RELIANCE", "HDFCBANK", "TCS", "INFY",
}


def _simple_sentiment(text: str) -> float:
    """Naive word-count sentiment (-1.0 to 1.0). Replace with FinBERT in Phase 3."""
    words = set(text.lower().split())
    pos = len(words & _POSITIVE_WORDS)
    neg = len(words & _NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 3)


def _extract_symbol(text: str) -> Optional[str]:
    """Return the first watched symbol mentioned in text, or None."""
    text_upper = text.upper()
    for sym in WATCH_SYMBOLS:
        if sym in text_upper:
            return sym
    return None


# (Removed inline _publish_sentiment_sync, importing from redis_service now)


# ── Finnhub News Feed ─────────────────────────────────────────────────────────
async def run_finnhub_news() -> None:
    """Poll Finnhub company news endpoint every 60 seconds."""
    if not FINNHUB_API_KEY:
        logger.warning("FINNHUB_API_KEY not set — skipping Finnhub news.")
        return

    import finnhub  # type: ignore

    client = finnhub.Client(api_key=FINNHUB_API_KEY)
    SYMBOLS_TO_POLL = ["AAPL", "TSLA", "NVDA", "SPY", "MSFT"]

    logger.info("Finnhub news polling started for %s.", SYMBOLS_TO_POLL)
    seen_ids: set[str] = set()

    while True:
        try:
            from datetime import datetime, timedelta
            from_date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
            to_date = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
            
            for sym in SYMBOLS_TO_POLL:
                news_items = client.company_news(sym, _from=from_date, to=to_date)
                for item in news_items[:5]:  # latest 5 per symbol
                    news_id = str(item.get("id", ""))
                    if news_id in seen_ids:
                        continue
                    seen_ids.add(news_id)

                    headline = item.get("headline", "")
                    score = _simple_sentiment(headline)
                    ts_ms = int(item.get("datetime", time.time()) * 1000)

                    event = UnifiedSentimentEvent(
                        event_id=f"FINNHUB_{sym}_{ts_ms}",
                        timestamp_ms=ts_ms,
                        symbol=sym,
                        source=SentimentSource.FINNHUB_NEWS,
                        sentiment_score=score,
                        headline=headline[:500],
                    )
                    publish_sentiment_sync(event)
                    logger.debug("Published Finnhub news for %s (score=%.2f)", sym, score)

        except Exception as exc:  # noqa: BLE001
            logger.error("Finnhub news error: %s", exc)

        await asyncio.sleep(60)  # Finnhub free: 60 req/min


# ── Reddit Feed ────────────────────────────────────────────────────────────────
async def run_reddit_feed() -> None:
    """Stream Reddit posts mentioning watched symbols."""
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        logger.warning("REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set — skipping Reddit feed.")
        return

    import praw  # type: ignore

    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )

    logger.info("Reddit sentiment streaming from: %s", SUBREDDITS)

    def _stream_reddit() -> None:
        """Run PRAW streaming in a thread (PRAW is not async-native)."""
        combined_sub = reddit.subreddit("+".join(SUBREDDITS))
        for submission in combined_sub.stream.submissions(skip_existing=True):
            title = submission.title or ""
            body = submission.selftext or ""
            full_text = f"{title} {body}"

            symbol = _extract_symbol(full_text)
            if not symbol:
                continue

            score = _simple_sentiment(full_text)
            ts_ms = int(time.time() * 1000)

            event = UnifiedSentimentEvent(
                event_id=f"REDDIT_{symbol}_{ts_ms}",
                timestamp_ms=ts_ms,
                symbol=symbol,
                source=SentimentSource.REDDIT,
                sentiment_score=score,
                headline=title[:500],
            )
            publish_sentiment_sync(event)
            logger.debug("Published Reddit sentiment for %s (score=%.2f)", symbol, score)

    # Run PRAW in a thread pool so it doesn't block the asyncio loop.
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _stream_reddit)


# ── Entry point ───────────────────────────────────────────────────────────────
async def main() -> None:
    await asyncio.gather(
        run_finnhub_news(),
        run_reddit_feed(),
    )


if __name__ == "__main__":
    asyncio.run(main())
