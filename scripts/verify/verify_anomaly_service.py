"""
Verification harness for anomaly_service.py (Phase 7 rewrite), testing
the SERVICE FUNCTION DIRECTLY -- bypassing HTTP/routing entirely. Faster
and more targeted than verify_full_stack.py (which drives the real app
through real HTTP requests); the two test different layers.

Does NOT use tests/conftest.py (which requires real PostgreSQL via
testcontainers/Docker). Uses an in-memory SQLite database instead --
proves the service logic is correct, does not prove anything
Postgres-specific. Fully self-contained: trains its own throwaway
models into a temp directory, no external setup required.

Run from the repo root:
    python scripts/verification/verify_anomaly_service.py
"""
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import joblib
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi import HTTPException

from app.database import Base
from app.models import User, MarketData
import app.services.anomaly_service as anomaly_service
from app.services.auth_service import hash_password
from app.config import settings as app_settings

from ml.data.synthetic import generate_synthetic_market_data
from ml.detection.multi_pattern import MultiPatternDetector
from ml.anomaly.isolation_forest import IsolationForestScratch
from ml.config import BASE_FEATURE_COLUMNS

results = {"passed": 0, "failed": 0}


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        results["passed"] += 1
        print(f"  PASS: {name}")
    else:
        results["failed"] += 1
        print(f"  FAIL: {name}  {detail}")


# ── Train real, throwaway models into a temp directory ──
print("Training real models for verification (a few seconds)...")
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

app_settings.MODEL_DIR = str(model_dir)
anomaly_service._registry = None  # force a fresh load against model_dir

# ── In-memory SQLite DB ──
engine = create_engine("sqlite:///:memory:")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
db = Session()

print("\n" + "=" * 70)
print("TEST 1: real end-to-end scoring with both models loaded")
print("=" * 70)

user = User(email="verify@test.com", username="verifyuser", hashed_password=hash_password("testpass123"))
db.add(user)
db.commit()
db.refresh(user)

rng = np.random.RandomState(42)
n = 35
base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
close = 100 + np.cumsum(rng.normal(0, 1, n))
volume = rng.lognormal(14, 0.2, n)

records = []
for i in range(n):
    r = MarketData(
        user_id=user.id, symbol="AAPL", timestamp=base_time + timedelta(days=i),
        open=float(close[i]) - 0.5, high=float(close[i]) + 1.0, low=float(close[i]) - 1.0,
        close=float(close[i]), volume=float(volume[i]),
    )
    db.add(r)
    records.append(r)
db.commit()
for r in records:
    db.refresh(r)

last_record = records[-1]
result = anomaly_service.detect_anomaly(db, market_data_id=last_record.id, user_id=user.id, threshold=0.5)

print(f"  anomaly_score={result.anomaly_score}  is_anomaly={result.is_anomaly}")
print(f"  isolation_forest_score={result.isolation_forest_score}  multi_pattern_max_score={result.multi_pattern_max_score}")

pattern_scores = json.loads(result.pattern_scores)
features = json.loads(result.features)
check("pattern_scores has all 4 patterns", set(pattern_scores.keys()) == {"pump_and_dump", "wash_trading", "spoofing", "layering"})
check("features has the 3 real feature keys", set(features.keys()) == {"return", "volume_ratio_20d", "volatility_20d"})
check("isolation_forest_score in [0,1]", 0 <= result.isolation_forest_score <= 1)
check("multi_pattern_max_score in [0,1]", 0 <= result.multi_pattern_max_score <= 1)
check("anomaly_score in [0,1]", 0 <= result.anomaly_score <= 1)

print()
print("=" * 70)
print("TEST 2: insufficient history returns 400, not a crash or silent garbage")
print("=" * 70)

sparse_record = MarketData(
    user_id=user.id, symbol="TSLA", timestamp=base_time,
    open=200.0, high=202.0, low=199.0, close=201.0, volume=500000.0,
)
db.add(sparse_record)
db.commit()
db.refresh(sparse_record)

try:
    anomaly_service.detect_anomaly(db, market_data_id=sparse_record.id, user_id=user.id)
    check("insufficient history raises HTTPException", False, "did not raise")
except HTTPException as e:
    check("insufficient history returns 400", e.status_code == 400, f"got {e.status_code}")

print()
print("=" * 70)
print("TEST 3: no models available returns 503, not a crash")
print("=" * 70)
# settings is a module-level singleton constructed once at import
# (app/config.py: `settings = get_settings()`) -- reassigning os.environ
# AFTER that point has no effect on the already-built object. Mutate the
# attribute directly instead, which is what actually changes what
# get_model_registry() reads.
app_settings.MODEL_DIR = str(Path(tempfile.mkdtemp()))  # a real, but empty, directory
anomaly_service._registry = None

try:
    anomaly_service.detect_anomaly(db, market_data_id=last_record.id, user_id=user.id)
    check("no models raises HTTPException", False, "did not raise")
except HTTPException as e:
    check("no models available returns 503", e.status_code == 503, f"got {e.status_code}")

print()
print("=" * 70)
print(f"RESULTS: {results['passed']} passed, {results['failed']} failed")
print("=" * 70)
sys.exit(1 if results["failed"] > 0 else 0)
