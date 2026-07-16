"""
tests/conftest.py — Phase 2 test infrastructure

Architecture:
  session scope:  Start PostgreSQL container → create engine → create all tables
  function scope: Begin transaction → yield session → rollback (isolation)

The transaction rollback pattern:
  - Every test runs inside a database transaction
  - The transaction is NEVER committed
  - It's rolled back after the test — database returns to pre-test state
  - ~1ms teardown vs 100ms+ for drop/recreate or TRUNCATE

testcontainers:
  - Spins up postgres:15 Docker container once for the whole test run
  - Each run gets a fresh, empty PostgreSQL instance
  - Requires Docker to be running
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app
from app.config import settings
import app.services.anomaly_service as anomaly_service


# ══════════════════════════════════════════════════════════════
# SESSION-SCOPED: Container + Engine + Schema (run once)
# ══════════════════════════════════════════════════════════════
@pytest.fixture(scope="session")
def postgres_url():
    """
    Start a PostgreSQL 15 container for the entire test run.
    Falls back to a local test database if Docker is unavailable.

    To use testcontainers: pip install testcontainers[postgresql]
    Requires Docker to be running.
    """
    try:
        from testcontainers.postgres import PostgresContainer
        with PostgresContainer("postgres:15-alpine") as pg:
            yield pg.get_connection_url()
    except Exception:
        # Fallback: use a local PostgreSQL test DB
        # Set TEST_DATABASE_URL env var to override
        import os
        url = os.environ.get(
            "TEST_DATABASE_URL",
            "postgresql://postgres:password@localhost:5432/market_surveillance_test"
        )
        yield url


@pytest.fixture(scope="session")
def test_engine(postgres_url):
    """
    Create SQLAlchemy engine connected to the test PostgreSQL instance.
    NullPool: no connection pooling in tests (connections are managed per-test).
    """
    from sqlalchemy.pool import NullPool
    engine = create_engine(postgres_url, poolclass=NullPool, echo=False)
    yield engine
    engine.dispose()


@pytest.fixture(scope="session", autouse=True)
def create_tables(test_engine):
    """
    Create all tables once before any test runs.
    Drop all after the session ends.
    autouse=True: runs automatically without being listed in test parameters.
    """
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


# ══════════════════════════════════════════════════════════════
# FUNCTION-SCOPED: Session per test (transaction rollback)
# ══════════════════════════════════════════════════════════════
@pytest.fixture(scope="function")
def db_session(test_engine):
    """
    Provide a database session that is rolled back after each test.

    How it works:
    1. Get a connection from the engine
    2. Begin a transaction on that connection
    3. Create a Session bound to that connection
    4. Yield the session to the test
    5. After test: rollback the transaction (undo all changes)
    6. Close session and connection

    Result: each test starts with an empty database, teardown is instant.
    """
    connection = test_engine.connect()
    transaction = connection.begin()

    # Create session bound to this specific connection
    TestingSession = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=connection,
    )
    session = TestingSession()

    yield session

    # Teardown — always runs even if test failed
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="function")
def client(db_session):
    """
    Provide a FastAPI TestClient that uses the test session.

    Overrides the get_db dependency so all database operations
    inside the request handler use the SAME session as the test setup.
    This is what makes the rollback pattern work — test and app
    share one connection and one transaction.
    """
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app, raise_server_exceptions=True) as test_client:
        yield test_client

    app.dependency_overrides.clear()


# ══════════════════════════════════════════════════════════════
# HELPER FIXTURES
# ══════════════════════════════════════════════════════════════
@pytest.fixture(scope="function")
def registered_user(client) -> dict:
    """
    Register and return a test user dict.
    Returns: {"id": int, "email": str, "username": str, ...}
    """
    payload = {
        "email": "test@example.com",
        "username": "testuser",
        "password": "SecurePass1",
    }
    response = client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 201, f"Registration failed: {response.text}"
    return response.json()


@pytest.fixture(scope="function")
def auth_tokens(client, registered_user) -> dict:
    """
    Log in the test user and return both tokens.
    Returns: {"access_token": str, "refresh_token": str, ...}
    """
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "test@example.com", "password": "SecurePass1"},
    )
    assert response.status_code == 200, f"Login failed: {response.text}"
    return response.json()


@pytest.fixture(scope="function")
def auth_headers(auth_tokens) -> dict:
    """
    Return Authorization headers dict ready to pass to client requests.
    Usage: client.get("/some/endpoint", headers=auth_headers)
    """
    return {"Authorization": f"Bearer {auth_tokens['access_token']}"}


@pytest.fixture(scope="function")
def sample_market_data(client, auth_headers) -> dict:
    """
    Create one OHLCV record and return the response body.

    NOTE: only enough for CRUD-style tests. Anomaly-detection tests that
    need a real score require rolling-window features (20+ trailing
    days) -- use sample_market_data_with_history for those. This
    fixture deliberately stays single-record so
    test_insufficient_history_returns_400 has something realistic to
    exercise.
    """
    payload = {
        "symbol": "AAPL",
        "timestamp": "2024-01-15T10:00:00Z",
        "open": 185.50,
        "high": 186.20,
        "low": 185.10,
        "close": 185.90,
        "volume": 1250000.0,
    }
    response = client.post("/api/v1/market-data", json=payload, headers=auth_headers)
    assert response.status_code == 201, f"Market data creation failed: {response.text}"
    return response.json()


# ══════════════════════════════════════════════════════════════
# PHASE 7: real trained models for anomaly detection tests
# ══════════════════════════════════════════════════════════════
@pytest.fixture(scope="session", autouse=True)
def trained_models_for_tests(tmp_path_factory):
    """
    Trains real mkt_surveillance_ml models ONCE for the whole test
    session (training takes real seconds; doing it per-test would be
    wasteful) and points settings.MODEL_DIR at them.

    autouse=True: every test in the suite gets a working model registry
    without needing to explicitly request this fixture -- anomaly
    detection tests need it to get a real score; other tests are
    unaffected by its presence.
    """
    from mkt_surveillance_ml.data.synthetic import generate_synthetic_market_data
    from mkt_surveillance_ml.detection.multi_pattern import MultiPatternDetector
    from mkt_surveillance_ml.anomaly.isolation_forest import IsolationForestScratch
    from mkt_surveillance_ml.config import BASE_FEATURE_COLUMNS
    import joblib
    import json as json_module
    from datetime import datetime, timezone

    model_dir = tmp_path_factory.mktemp("test_models")
    df = generate_synthetic_market_data(n_days=800, random_state=123)

    detector = MultiPatternDetector(random_state=123)
    detector.fit(df[BASE_FEATURE_COLUMNS], df)
    joblib.dump(detector, model_dir / "multi_pattern_detector.joblib")
    (model_dir / "multi_pattern_detector_metadata.json").write_text(json_module.dumps({
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_columns": BASE_FEATURE_COLUMNS,
    }))

    iso_forest = IsolationForestScratch(n_estimators=100, contamination=0.05, random_state=123)
    iso_forest.fit(df[BASE_FEATURE_COLUMNS].values)
    joblib.dump(iso_forest, model_dir / "isolation_forest_scratch.joblib")
    (model_dir / "isolation_forest_metadata.json").write_text(json_module.dumps({
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_columns": BASE_FEATURE_COLUMNS,
    }))

    settings.MODEL_DIR = str(model_dir)
    anomaly_service._registry = None  # force a fresh load against the new MODEL_DIR
    yield model_dir


@pytest.fixture(scope="function")
def sample_market_data_with_history(client, auth_headers) -> dict:
    """
    Creates 30 sequential daily OHLCV records for one symbol (enough to
    clear MIN_RAW_ROWS_FOR_FEATURES=21 with margin) and returns the
    response body for the LAST one -- same return contract as
    sample_market_data, but with enough trailing history for
    mkt_surveillance_ml's rolling-window features to actually compute.
    """
    import numpy as np
    from datetime import datetime, timedelta
    rng = np.random.RandomState(7)
    n = 30
    close = 150 + np.cumsum(rng.normal(0, 1, n))
    volume = rng.lognormal(13, 0.2, n)
    base_date = datetime(2024, 2, 1, 10, 0, 0)

    last_response = None
    for i in range(n):
        payload = {
            "symbol": "MSFT",
            "timestamp": (base_date + timedelta(days=i)).isoformat() + "Z",
            "open": float(close[i]) - 0.3,
            "high": float(close[i]) + 0.8,
            "low": float(close[i]) - 0.8,
            "close": float(close[i]),
            "volume": float(volume[i]),
        }
        response = client.post("/api/v1/market-data", json=payload, headers=auth_headers)
        assert response.status_code == 201, f"Market data creation failed: {response.text}"
        last_response = response.json()
    return last_response
