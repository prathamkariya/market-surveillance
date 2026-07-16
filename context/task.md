# Task Tracker: Enterprise Market Surveillance

## Phase 1: Infrastructure & Data Plumbing (The Ingestion Layer)
- `[x]` Update `docker-compose.yml` to include Redis and upgrade PostgreSQL to TimescaleDB.
- `[x]` Update `requirements.txt` with new data engineering dependencies.
- `[x]` Create `app/schemas/streaming.py` for `UnifiedTradeEvent` and `UnifiedSentimentEvent`.
- `[x]` Create `app/services/redis_service.py` for singleton connection pooling.
- `[x]` Create `scripts/market_adapters/crypto_worker.py` (multi-feed Binance/Bybit).
- `[x]` Create `scripts/market_adapters/sentiment_worker.py` (Finnhub News / Reddit).
- `[/]` **Verification:** Run `docker-compose up -d`, `pytest`, and verify ticks in Redis.

## Phase 2: Hybrid Detection Engine Upgrades (The ML Layer)
- `[x]` Modify `app/services/anomaly_service.py` to add Redis streaming capabilities while preserving REST compatibility.
- `[x]` Create `scripts/run_engine.py` background worker.
- `[/]` **Verification:** Inject a fake threat via script and verify the engine triggers and saves to TimescaleDB.

## Phase 3: The Enterprise UI & Auto-MAR (The Presentation Layer)
- `[x]` Modify `app/routers/alerts.py` to add SSE/WebSocket streaming endpoint.
- `[x]` Create `app/services/mar_generator.py` for Gemini Markdown/PDF generation.
- `[x]` Modify `app/routers/reports.py` to expose `/reports/mar/{alert_id}`. generate MAR PDF.

## Phase 4: CI/CD, MLOps & Hardening
- `[x]` Initialize DVC and add MLflow tracking.
- `[x]` Create `docker-compose.prod.yml` with Nginx rate limiting.
- `[x]` **Verification:** Load test (10k req/sec) and verify DVC repro.
- `[x]` Create `docker-compose.prod.yml`
- `[x]` Final Audits and Testing.pro.
