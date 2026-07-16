# Enterprise Market Surveillance: The Ultimate Implementation Plan

This is the definitive, exhaustively detailed technical roadmap for transforming our existing synchronous FastAPI application into an asynchronous, ML-driven Market Surveillance platform. 

It guarantees **zero breakage** of existing endpoints, includes **strict debugging gates** after every phase, and explicitly lists **which open-source repositories to use as reference implementations** when coding.

> [!IMPORTANT]
> **Final Review Required**
> Please review this ultimate blueprint. If you are satisfied with the extreme level of detail, click "Proceed" and I will immediately begin executing Phase 1!

---

## The Unified Event Schema (The Core Standard)

Before touching any logic, we must define our single standard schema so all adapters (Binance, Alpaca, Reddit) output the exact same JSON format.

#### [NEW] `app/schemas/streaming.py`
- Create a `UnifiedTradeEvent` Pydantic model (`event_id`, `timestamp`, `market`, `symbol`, `price`, `volume`, `is_buyer_maker`).
- Create a `UnifiedSentimentEvent` Pydantic model (`timestamp`, `symbol`, `source`, `sentiment_score`).

**Verification:** Run `pytest` to ensure these schemas initialize correctly and reject malformed data.

---

## Phase 1: Infrastructure & Data Plumbing (The Ingestion Layer)

**Goal:** Add Redis and TimescaleDB, and build the WebSocket connections with multi-feed redundancy.

**Inspiration Repositories:**
- 📖 **`aryan1078/indian-equities-market-surveillance-platform`**: Copy their architecture for cleanly separating WebSocket collectors from the main API.
- 📖 **`Fifadlika/MLOps-Crypto-Surveillance`**: Copy their exact Binance WebSocket ingestion logic.

#### [MODIFY] `docker-compose.yml`
- Add a `redis:alpine` service for high-throughput messaging.
- Upgrade the `db` image from `postgres:15` to `timescale/timescaledb:latest-pg15` (fully backwards compatible with existing Postgres tables).

#### [MODIFY] `requirements.txt`
- Add `redis`, `websockets`, `aiohttp`, `alpaca-py`, `python-binance`, `fyers-apiv3`, `tiingo`.

#### [NEW] `app/services/redis_service.py`
- Create a singleton Redis client to handle connection pooling (`publish_to_stream` and `read_stream_blocking`).

#### [NEW] `scripts/market_adapters/crypto_worker.py` (and `us_worker.py`, `india_worker.py`)
- **Multi-Feed Redundancy**: `crypto_worker.py` will connect to **Binance** (Primary) and **Bybit** (Secondary). If Binance drops, it automatically fails over.
- **Backward Compatibility**: These run as completely separate Python processes. They do not touch the FastAPI app.

#### [NEW] `scripts/market_adapters/sentiment_worker.py`
- Connects to **Finnhub News** and the **Reddit API** to publish to the `"live_sentiment"` Redis stream.

> [!WARNING]
> **Phase 1 Debugging & Verification Gate**
> 1. Run `docker-compose up -d`. Verify existing Postgres tables remain intact on TimescaleDB.
> 2. Run `pytest tests/` to ensure the existing `POST /market-data` endpoint is not broken.
> 3. Run the `crypto_worker.py` and manually check `redis-cli XREAD` to visually verify thousands of ticks streaming flawlessly without memory leaks.
> **DO NOT PROCEED TO PHASE 2 UNTIL THESE PASS.**

---

## Phase 2: Hybrid Detection Engine Upgrades (The ML Layer)

**Goal:** Shift `anomaly_service.py` from pulling slow historical data out of Postgres to reading real-time streaming data from Redis.

**Inspiration Repositories:**
- 📖 **`mayurpatil10001/ARGUS-Market-Surveillance`**: Copy their logic for fusing price data with sentiment data before passing it to the ML engine.
- 📖 **`quynhanhha/crypto-market-surveillance`**: Copy their central "Severity Scoring" logic to normalize anomaly scores from 0-100.

#### [MODIFY] `app/services/anomaly_service.py`
- **Backward Compatibility**: Keep the existing `detect_anomaly(db, market_data_id, ...)` function completely untouched so existing REST clients don't break.
- Add a *new* function `score_live_trade(unified_trade: UnifiedTradeEvent, sentiment: float)`.
- Use an in-memory Redis cache (Redis Lists) for the 20-candle trailing history instead of querying Postgres.

#### [NEW] `scripts/run_engine.py`
- A dedicated background worker script. It pulls ticks from the `"live_trades"` Redis stream, merges them with the `sentiment_score`, passes them to the existing `mkt_surveillance_ml` Isolation Forest models, and calculates SHAP values.
- If an anomaly is found (Score > 0.7), it writes the *Alert* back to TimescaleDB.

> [!WARNING]
> **Phase 2 Debugging & Verification Gate**
> 1. Run `pytest tests/test_anomaly.py` to ensure the old `detect_anomaly` still works perfectly.
> 2. Create a "Threat Injector" script that blasts a fake 50% price spike into Redis. 
> 3. Verify `run_engine.py` successfully reads the spike, triggers the Isolation Forest, and correctly writes an Alert to the database.
> **DO NOT PROCEED TO PHASE 3 UNTIL THESE PASS.**

---

## Phase 3: The Enterprise UI & Auto-MAR (The Presentation Layer)

**Goal:** Build a professional triage dashboard and automate compliance reporting via LLMs.

**Inspiration Repositories:**
- 📖 **`mayurpatil10001/ARGUS-Market-Surveillance`**: Use this as the definitive guide for prompting Google Gemini to generate SEBI/SEC-compliant PDFs.
- 📖 **`sushi1507/market-surveillance-demo`**: Look at their frontend code to understand how to build a high-performance streaming dashboard.

#### [MODIFY] `app/routers/alerts.py`
- Add a Server-Sent Events (SSE) or WebSocket endpoint (`/alerts/stream`) so the frontend UI can update in real-time without refreshing.

#### [NEW] `app/services/mar_generator.py`
- Add `generate_market_abuse_report(anomaly_id: int)`.
- Fetches the anomaly, SHAP values, and correlated news events, feeds it to the **Google Gemini API**, and returns a PDF.

> [!WARNING]
> **Phase 3 Debugging & Verification Gate**
> 1. Connect a simple WebSocket test client to `/alerts/stream`. Inject an anomaly via Redis and verify the alert pops up instantly on the client.
> 2. Trigger the `/generate-report` endpoint and manually review the generated PDF to ensure it is correctly formatted and uses real data.
> **DO NOT PROCEED TO PHASE 4 UNTIL THESE PASS.**

---

## Phase 4: CI/CD, MLOps & Hardening

**Goal:** Secure the application and implement enterprise ML lifecycle tools.

**Inspiration Repositories:**
- 📖 **`Fifadlika/MLOps-Crypto-Surveillance`**: Copy their exact `dvc.yaml` file structure for dataset locking.

#### [MODIFY] `mkt_surveillance_ml/pyproject.toml` & [NEW] `dvc.yaml`
- Initialize Data Version Control (DVC) to lock our training datasets.
- Add MLflow hooks inside `mkt_surveillance_ml/scripts/train.py` to log precision and recall metrics locally.

#### [NEW] `docker-compose.prod.yml`
- Create a production variant of the docker-compose file that includes Nginx for rate limiting and API gateway routing.

> [!WARNING]
> **Phase 4 Debugging & Verification Gate**
> 1. Run a load test (10,000 req/sec) using `wrk` or `locust` to ensure Nginx and Redis handle backpressure gracefully.
> 2. Run a full DVC pipeline test (`dvc repro`) to ensure the dataset and model hashes are correctly tracked.
> 3. Perform a final End-to-End full stack test.
