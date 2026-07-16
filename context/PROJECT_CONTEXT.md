# 🌐 Enterprise Market Surveillance & Anomaly Detection Platform

**Date:** July 2026  
**Status:** Backend Infrastructure Complete, ML Engine Operational, Streaming Layer Active  

---

## 1. Executive Summary & Project Vision

This project is a high-performance, real-time **Market Surveillance and Anomaly Detection Platform** designed for enterprise compliance and risk management. 

Its primary purpose is to ingest live market ticks across multiple asset classes (Crypto, US Equities, Indian Equities) alongside social sentiment, analyze them in real-time using trained Machine Learning models, and instantly flag suspicious trading activities such as **pump-and-dump schemes, wash trading, spoofing, and abnormal volume spikes**.

When an anomaly is detected, the system generates real-time alerts via Server-Sent Events (SSE) to a live dashboard, and leverages Generative AI (Google Gemini) to automatically draft regulatory-grade **Market Abuse Reports (MAR)**.

---

## 2. The Technology Stack

The platform is built using a modern, highly decoupled, async-first architecture.

### Core Backend
* **Language:** Python 3.10+
* **Web Framework:** FastAPI (Async-first REST and SSE APIs)
* **Data Validation:** Pydantic V2
* **ORM:** SQLAlchemy 2.0
* **Authentication:** JWT with rotating refresh tokens, Bcrypt password hashing.

### Data Ingestion & Persistence
* **Message Broker / Event Bus:** **Redis Streams** (Alpine). Chosen over standard Pub/Sub for persistence and consumer groups, ensuring no ticks are lost during ML processing.
* **Database:** **TimescaleDB** (PostgreSQL 15 extension) optimized for hyper-table time-series data storage and massive analytical queries.

### Data Sources (The Adapters)
We use independent, horizontally scalable Python background workers to connect to public exchanges via WebSockets and REST:
* **Crypto:** Binance (Primary), Bybit (Secondary Fallback).
* **US Equities:** Alpaca (Primary WebSockets), Finnhub (Secondary REST).
* **Indian Equities:** Upstox (Primary WebSockets).
* **Sentiment / News:** Finnhub News, Reddit API (PRAW).

### The ML Engine (`mkt_surveillance_ml`)
The intelligence layer is driven by custom scikit-learn models wrapped in a unified pipeline:
* **Unsupervised Learning:** `IsolationForestScratch` for detecting zero-day unknown market anomalies based on distance from normal clustering.
* **Supervised Learning:** `MultiPatternDetector` (RandomForestClassifier) trained on synthetic pump-and-dump and wash trading datasets to output specific pattern probabilities.
* **Explainability:** **SHAP** values are computed to explain *why* the model flagged an anomaly (e.g., "Volume is 12x standard deviation").

### AI Reporting Layer
* **LLM Provider:** Google Gemini 1.5 Flash (`google-generativeai`). Used to synthesize raw numeric ML features and alert contexts into human-readable compliance PDFs/Markdown documents.

---

## 3. The Architecture & How It Works

The architecture guarantees **zero blocking** of the main web server. It works as a continuous pipeline:

1. **Ingestion (The Workers):** Standalone scripts like `crypto_worker.py` and `sentiment_worker.py` connect to Binance/Finnhub. They standardize incoming ticks into a `UnifiedTradeEvent` schema and push them to the Redis `live_trades` and `live_sentiment` streams.
2. **Inference (The Engine):** A background daemon (`run_engine.py`) continuously consumes the Redis streams. It maintains a rolling, in-memory sliding window of the last 20 candles per symbol. It computes engineered features (like 20-day volatility and return), fuses them with the live sentiment score, and passes the vector to the loaded `Isolation Forest` and `Random Forest` models.
3. **Alerting & Persistence:** If the combined anomaly score exceeds `0.7`, the engine persists the alert into TimescaleDB. It simultaneously pushes an event payload to a `live_alerts` Redis stream.
4. **Presentation (FastAPI):** The FastAPI web server exposes a Server-Sent Events endpoint (`/alerts/stream/live`) that listens to `live_alerts` and pushes the notification to the frontend browser instantly.

---

## 4. Our Research & Transformation Journey

We didn't build this blindly. We extensively researched best-in-class open-source repositories to aggregate the best design patterns for market surveillance:

* **Inspiration:** `aryan1078/indian-equities-market-surveillance-platform`
  * *Lesson learned:* We adopted their decoupled architecture, separating slow Python WebSocket ingestion scripts from the core FastAPI application to prevent API latency spikes.
* **Inspiration:** `Fifadlika/MLOps-Crypto-Surveillance`
  * *Lesson learned:* We utilized their DVC (Data Version Control) and MLflow tracking philosophies for the upcoming MLOps phase, and mirrored their Binance stream ingestion handling.
* **Inspiration:** `quynhanhha/crypto-market-surveillance`
  * *Lesson learned:* We adopted their "Severity Score Normalization" (0-100) logic to fuse unsupervised Isolation Forest scores with supervised Random Forest probabilities.

### How We Evolved (The Four Phases)
Initially, this project was a synchronous REST API with "mock" mathematical formulas. We transformed it through these phases:
* **Phase 1 (Plumbing):** We injected Redis into `docker-compose.yml`, upgraded Postgres to TimescaleDB, and built the multi-feed WebSocket ingestion workers.
* **Phase 2 (The ML Engine):** We wrote `run_engine.py` to decouple inference from the API. We upgraded `anomaly_service.py` to score live Redis streams statelessly.
* **Phase 3 (Enterprise UI & Auto-MAR):** We built the SSE streaming endpoints (`/alerts/stream/live`) and the Gemini AI compliance reporter (`/reports/mar/{alert_id}`).
* **Phase 4 (Production):** We mapped out `docker-compose.prod.yml` to orchestrate Nginx rate limiting, multiple API workers, and the ML engine sidecars.

---

## 5. Where We Are Right Now

We have successfully completed all four phases of the backend infrastructure. 

* **The Codebase:** The API is stable, the streaming endpoints are active, and the ML worker successfully detects anomalies. 
* **Testing:** The automated test suite (`pytest`) verifies all backward compatibility, returning a 100% pass rate. 
* **Verification:** We created a Threat Injector script (`verify_phase8_engine.py`) that blasts a simulated 50% price dump into Redis, which successfully triggers the ML engine to flag an anomaly and generate a MAR.

## 6. What Is Next

The backend is a powerful, headless engine. To make it a fully realized platform, our next steps are:

1. **The Frontend Dashboard:** Build a high-performance React/Next.js dashboard with interactive charting (TradingView Lightweight Charts) that subscribes to our SSE endpoint to visually flash anomalies in real-time.
2. **MLOps Implementation:** Initialize **DVC** to lock our synthetic training datasets and implement **MLflow** inside our `mkt_surveillance_ml` package to track precision, recall, and model drift over time.
3. **Load Testing & Hardening:** Run `wrk` load tests (targeting 10,000+ ticks/sec) to ensure Redis and Nginx handle immense backpressure during high-volatility market events without memory leaks.
