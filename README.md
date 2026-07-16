# Market Surveillance Platform

A production-ready market anomaly detection system consisting of two tightly integrated components:

- **FastAPI backend** — REST API with JWT auth, OHLCV ingestion, anomaly detection, alerts, and watchlists
- **`mkt_surveillance_ml`** — installable Python ML package with from-scratch implementations of Isolation Forest, Multi-Pattern Detector (XGBoost), ARIMA, Prophet, and LSTM

---

## Architecture

```
market-surveillance/
├── app/                     # FastAPI backend
│   ├── core/                # Security helpers, custom exceptions
│   ├── routers/             # API route handlers (auth, market_data, anomalies, alerts, watchlists)
│   ├── services/            # Business logic (anomaly_service, auth_service, watchlist_service)
│   ├── models.py            # SQLAlchemy ORM models
│   ├── schemas.py           # Pydantic request/response schemas
│   ├── database.py          # Engine + session factory
│   ├── dependencies.py      # FastAPI dependency injection (get_current_user, etc.)
│   └── config.py            # Pydantic Settings (reads .env)
├── mkt_surveillance_ml/     # ML package (pip install -e mkt_surveillance_ml)
│   ├── src/mkt_surveillance_ml/
│   │   ├── anomaly/         # IsolationForestScratch
│   │   ├── detection/       # MultiPatternDetector, WeakLabeling
│   │   ├── time_series/     # ARIMA+Prophet, LSTM, stationarity tests
│   │   ├── models/          # Decision tree, random forest, gradient boosting, logistic regression, clustering
│   │   ├── features/        # Feature scaling and selection
│   │   ├── evaluation/      # Cross-validation, calibration, learning curves
│   │   ├── data/            # Synthetic data generation
│   │   └── serving/         # ModelRegistry — thread-safe model loading
│   ├── scripts/             # CLI entry points: train.py, forecast.py
│   └── tests/               # 311 unit tests
├── alembic/                 # Database migrations (001–004)
├── tests/                   # 161 backend integration tests (require PostgreSQL)
├── scripts/verify/          # Standalone verification scripts (SQLite, no Docker needed)
├── docs/                    # Audit reports and architecture notes
├── trained_models/          # Drop .joblib model files here (gitignored)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quick Start

### 1. Environment setup

```bash
# Copy and fill in environment variables
cp .env.example .env

# Create virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux

pip install -e mkt_surveillance_ml
pip install -r requirements.txt
```

### 2. Train models (required before running the API)

```bash
# Train both anomaly detection models and save to trained_models/
python mkt_surveillance_ml/scripts/train.py --output-dir trained_models/
```

### 3. Start with Docker Compose

```bash
docker-compose up --build
```

API available at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

### 4. Database migrations (first run)

```bash
alembic upgrade head
```

---

## Running Tests

### ML package (no database required)

```bash
.venv\Scripts\pytest mkt_surveillance_ml/tests/ -v
```

Expected: **307 passed, 4 skipped** (Prophet tests require working `cmdstanpy` binary — known Windows env issue, unrelated to ML logic).

### Backend integration tests (requires PostgreSQL)

Start the database container first:

```bash
docker-compose up -d db
```

Then run:

```bash
.venv\Scripts\pytest tests/ -v
```

Expected: **161 passed**.

### Standalone verification (no Docker, no PostgreSQL)

These scripts use SQLite + in-memory databases — useful for quick smoke-tests:

```bash
python scripts/verify/verify_anomaly_service.py
python scripts/verify/verify_full_stack.py
```

---

## API Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/auth/register` | Register a new user |
| `POST` | `/api/v1/auth/login` | Get access + refresh tokens |
| `POST` | `/api/v1/auth/refresh` | Rotate refresh token |
| `POST` | `/api/v1/auth/logout` | Revoke refresh token |
| `POST` | `/api/v1/market-data` | Ingest OHLCV candle |
| `GET` | `/api/v1/market-data` | List user's market data |
| `POST` | `/api/v1/anomalies` | Run anomaly detection on a candle |
| `GET` | `/api/v1/anomalies` | List anomaly detections |
| `POST` | `/api/v1/alerts` | Create alert from anomaly |
| `PATCH` | `/api/v1/alerts/{id}` | Update alert status |
| `GET/DELETE` | `/api/v1/alerts/{id}` | Get or delete alert |
| `POST` | `/api/v1/watchlists` | Create watchlist |
| `GET` | `/api/v1/watchlists` | List watchlists |
| `POST` | `/api/v1/watchlists/{id}/symbols` | Add symbol to watchlist |

---

## Environment Variables

See [`.env.example`](.env.example) for all required variables. Key ones:

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string |
| `SECRET_KEY` | JWT signing secret (use `openssl rand -hex 32`) |
| `MODEL_DIR` | Path to trained model files (default: `trained_models/`) |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | JWT access token TTL |

---

## Database Schema

4 Alembic migrations covering:

- **001** — `users`, `market_data`, `anomalies`, `alerts`
- **002** — `watchlists`, `watchlist_symbols`
- **003** — `refresh_tokens`
- **004** — Extended `anomalies` table: `multi_pattern_max_score`, `pattern_scores`, `model_version`
