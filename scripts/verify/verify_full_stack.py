"""
Full-stack verification harness: replicates conftest.py's client/db_session
fixture pattern (SQLite instead of Postgres, since this sandbox has neither
Docker nor a running Postgres server) and exercises the REAL FastAPI app --
real routers, real auth flow, real Pydantic validation, real SQLAlchemy
models -- through actual HTTP requests via TestClient.

What this DOES verify: the full request/response cycle for the rewritten
anomaly detection endpoint, against real trained models, matching what
tests/test_anomaly.py's updated tests assert.

What this does NOT verify: that migration 004's exact DDL matches the
SQLAlchemy models (Base.metadata.create_all() builds tables directly from
the models, not by running the migration), and nothing here has been
checked against a real PostgreSQL instance specifically.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
from datetime import datetime, timedelta, timezone

import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app
from app.config import settings
import app.services.anomaly_service as anomaly_service

# ── Train real models, point MODEL_DIR at them (mirrors conftest.py's
#    trained_models_for_tests fixture) ──
from mkt_surveillance_ml.data.synthetic import generate_synthetic_market_data
from mkt_surveillance_ml.detection.multi_pattern import MultiPatternDetector
from mkt_surveillance_ml.anomaly.isolation_forest import IsolationForestScratch
from mkt_surveillance_ml.config import BASE_FEATURE_COLUMNS
import joblib, tempfile
from pathlib import Path

model_dir = Path(tempfile.mkdtemp())
df = generate_synthetic_market_data(n_days=800, random_state=123)
detector = MultiPatternDetector(random_state=123)
detector.fit(df[BASE_FEATURE_COLUMNS], df)
joblib.dump(detector, model_dir / "multi_pattern_detector.joblib")
(model_dir / "multi_pattern_detector_metadata.json").write_text(json.dumps(
    {"trained_at_utc": datetime.now(timezone.utc).isoformat(), "feature_columns": BASE_FEATURE_COLUMNS}
))
iso_forest = IsolationForestScratch(n_estimators=100, contamination=0.05, random_state=123)
iso_forest.fit(df[BASE_FEATURE_COLUMNS].values)
joblib.dump(iso_forest, model_dir / "isolation_forest_scratch.joblib")
(model_dir / "isolation_forest_metadata.json").write_text(json.dumps(
    {"trained_at_utc": datetime.now(timezone.utc).isoformat(), "feature_columns": BASE_FEATURE_COLUMNS}
))
settings.MODEL_DIR = str(model_dir)
anomaly_service._registry = None

# ── SQLite engine + tables (mirrors test_engine/create_tables) ──
engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
Base.metadata.create_all(engine)

# ── db_session + client override (mirrors conftest.py exactly) ──
connection = engine.connect()
transaction = connection.begin()
TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=connection)
db_session = TestingSession()


def override_get_db():
    yield db_session


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app, raise_server_exceptions=True)

results = {"passed": 0, "failed": 0}


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        results["passed"] += 1
        print(f"  PASS: {name}")
    else:
        results["failed"] += 1
        print(f"  FAIL: {name}  {detail}")


print("=" * 70)
print("Real HTTP stack verification (TestClient -> FastAPI -> real router -> real service -> real models)")
print("=" * 70)

# ── Register + login (real auth flow) ──
print("\n-- Auth flow --")
r = client.post("/api/v1/auth/register", json={
    "email": "verify@example.com", "username": "verifyuser", "password": "TestPass123",
})
check("register returns 201", r.status_code == 201, r.text)

r = client.post("/api/v1/auth/login", json={"email": "verify@example.com", "password": "TestPass123"})
check("login returns 200", r.status_code == 200, r.text)
access_token = r.json()["access_token"]
auth_headers = {"Authorization": f"Bearer {access_token}"}

# ── Create 30 days of history via the REAL market-data endpoint ──
print("\n-- Creating historical market data via real endpoint --")
rng = np.random.RandomState(7)
n = 30
close = 150 + np.cumsum(rng.normal(0, 1, n))
volume = rng.lognormal(13, 0.2, n)
base_date = datetime(2024, 2, 1, 10, 0, 0)

last_market_data_id = None
for i in range(n):
    payload = {
        "symbol": "MSFT",
        "timestamp": (base_date + timedelta(days=i)).isoformat() + "Z",
        "open": float(close[i]) - 0.3, "high": float(close[i]) + 0.8,
        "low": float(close[i]) - 0.8, "close": float(close[i]), "volume": float(volume[i]),
    }
    r = client.post("/api/v1/market-data", json=payload, headers=auth_headers)
    if r.status_code != 201:
        check(f"market-data creation (day {i})", False, r.text)
        break
    last_market_data_id = r.json()["id"]
check("created 30 days of market data", last_market_data_id is not None)

# ── Real anomaly detection call through the real HTTP endpoint ──
print("\n-- POST /api/v1/anomalies (real endpoint, real models) --")
r = client.post("/api/v1/anomalies", json={"market_data_id": last_market_data_id}, headers=auth_headers)
check("detect returns 201", r.status_code == 201, r.text)
body = r.json()
print(f"  Response: {json.dumps(body, indent=2)}")

check("anomaly_score in [0,1]", 0 <= body["anomaly_score"] <= 1)
check("isolation_forest_score in [0,1]", 0 <= body["isolation_forest_score"] <= 1)
check("multi_pattern_max_score in [0,1]", 0 <= body["multi_pattern_max_score"] <= 1)
pattern_scores = json.loads(body["pattern_scores"])
check("pattern_scores has all 4 patterns", set(pattern_scores.keys()) == {"pump_and_dump", "wash_trading", "spoofing", "layering"})
features = json.loads(body["features"])
check("features has the 3 real feature keys", set(features.keys()) == {"return", "volume_ratio_20d", "volatility_20d"})
check("model_version mentions both models", "isolation_forest=" in body["model_version"] and "multi_pattern=" in body["model_version"])
expected_combined = round(0.6 * body["isolation_forest_score"] + 0.4 * body["multi_pattern_max_score"], 4)
check("anomaly_score is the weighted average", abs(body["anomaly_score"] - expected_combined) < 1e-4)

# ── Insufficient history -> 400 ──
print("\n-- Insufficient history case --")
r = client.post("/api/v1/market-data", json={
    "symbol": "AAPL", "timestamp": "2024-01-15T10:00:00Z",
    "open": 185.5, "high": 186.2, "low": 185.1, "close": 185.9, "volume": 1250000.0,
}, headers=auth_headers)
single_record_id = r.json()["id"]
r = client.post("/api/v1/anomalies", json={"market_data_id": single_record_id}, headers=auth_headers)
check("insufficient history returns 400", r.status_code == 400, r.text)
check("400 message mentions insufficient history", "Not enough historical data" in r.json().get("detail", ""))

# ── Auth guard ──
print("\n-- Auth guard --")
r = client.post("/api/v1/anomalies", json={"market_data_id": last_market_data_id})
check("no auth returns 401/403", r.status_code in (401, 403), str(r.status_code))

print()
print("=" * 70)
print(f"RESULTS: {results['passed']} passed, {results['failed']} failed")
print("=" * 70)

db_session.close()
transaction.rollback()
connection.close()

sys.exit(1 if results["failed"] > 0 else 0)
