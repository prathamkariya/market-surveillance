# Phase 7 Audit — `mkt_surveillance_ml` + FastAPI backend

**Verified by actually running things**, not reading and guessing: both packages were installed, a local Postgres was built, real test suites were run, `alembic upgrade head` was executed against real Postgres for the first time, and the migrated schema was diffed against what the models produce. Every number below is an actual run, not an estimate.

---

## Bottom Line

"Tests pass" is true but narrower than it sounds. The 302 `mkt_surveillance_ml` tests and 156 backend tests genuinely all pass — confirmed independently. But **the one module Phase 7 actually added to the ML package, `serving/model_registry.py`, had zero tests** (`pytest --cov` reports it as "never imported" across all 302), and it has a real bug. The backend rewrite (`anomaly_service.py`) is good work — but Docker, the one deployment piece that's supposed to be Phase 7's other deliverable, doesn't build at all, and there's a leftover dead file the migration doc says was deleted.

14 new tests were written (9 + 5) targeting exactly these gaps. 5 fail out of the box. All 5 were fixed and reverified: **472/472 pass** (311 ML package + 161 backend). Patch package applied, all fixes verified.

---

## Critical

### 1. Docker doesn't build

`Dockerfile` referenced `pyproject.toml` and `poetry.lock` which don't exist at the repo root. `docker build .` fails before installing a single dependency. **Fixed**: swapped the Poetry block for `COPY requirements.txt` + `pip install -r requirements.txt`.

### 2. `docker-compose.yml` sets a secret the app never reads

```yaml
JWT_SECRET_KEY=dev_secret_key_change_me
```

`app/config.py`'s `Settings` class has `SECRET_KEY`, not `JWT_SECRET_KEY`. pydantic-settings silently drops unrecognized names — so `SECRET_KEY` was sitting at its hardcoded default, signing every JWT. **Fixed**: renamed to `SECRET_KEY`. Also fixed `MODEL_DIR=/app/models` → `trained_models/` to match the convention used everywhere else.

Tests written: `TestDockerfileReferencesExistingFiles`, `TestDockerComposeEnvVarsMatchSettings` — parse actual files rather than hardcoding specific mismatches.

---

## Major

### 3. `ModelRegistry.load()` — one bad file blocks a completely unrelated model

Two sequential try-blocks with no isolation between them. An exception from the first block propagates out of `load()`, so the second block never runs even if that file is fine. Confirmed by hand: corrupt `isolation_forest_scratch.joblib` + valid `multi_pattern_detector.joblib` → `registry.has_multi_pattern` returns `False`.

**Fixed**: independent try/except per model type, collecting errors and raising once at the end. 9 new tests in `mkt_surveillance_ml/tests/test_model_registry.py`. Coverage: 0% → 96%.

### 4. `get_model_registry()` lazy singleton has no lock

```python
if _registry is None:
    _registry = ModelRegistry(settings.MODEL_DIR)
    _registry.load()
```

Thread A assigns `_registry` before calling `.load()` — Thread B sees it non-None, skips init, gets a half-loaded object. FastAPI runs sync path operations in a threadpool, so this really does happen concurrently.

**Fixed**: lock around the check-and-load. Build into a local `candidate` first, only publish to `_registry` after `.load()` completes.

### 5. `MarketData` unique constraint doesn't account for users

```python
UniqueConstraint("symbol", "timestamp", name="uq_market_data_symbol_timestamp")
```

Global constraint, but every read path filters by `user_id`. Two users recording the same AAPL candle timestamp collide. `create_market_data()` had no `try/except` around `db.commit()`.

**Shipped fix**: catch `IntegrityError`, return 409. The deeper question (should the constraint include `user_id`?) is a design decision — see `TestCrossUserMarketDataCollision`'s docstring. The test accepts `201` or `409` since both are legitimate; only "unhandled crash" is wrong.

### 6. Leftover dead code the migration doc says was deleted

`app/services/analysis_service.py` existed, importing `app.features.engineer` and `app.models.database` — neither of which exist in Phase 7. Nothing imports it; it can't be imported without an immediate `ModuleNotFoundError`. **Deleted**. Hygiene test added: `TestDeadCodeRemoved`.

---

## What's Actually Solid

- **`anomaly_service.py` rewrite is good work.** Feature computation delegates to `compute_engineered_features` from the ML package — avoids two-implementations-drift. `predict_proba` output columns verified to always be `proba_<pattern>`-prefixed.
- **156 backend tests pass for real**, against real Postgres, via actual `pytest tests/` — first time this specific suite has executed.
- **Migration 004 applies cleanly.** `alembic upgrade head` from scratch — clean. Resulting `anomalies` table matches `app/models.py`'s `Anomaly` class exactly.
- **Schema gap documented**: Alembic migrations add DB-level `server_default`s (for `created_at`, `updated_at`, `is_active`, `status`) that `Base.metadata.create_all()` doesn't. Today latent (ORM writes supply defaults), but important before anyone adds raw-SQL bulk loaders.

---

## Minor / Nitpicks

- `tests/conftest.py`: `.replace("psycopg2", "psycopg2")` — no-op, removed.
- `alembic/versions/001` and `003`: redundant `create_index` calls for columns already covered by `UniqueConstraint` — removed.
- `RefreshToken.is_expired`: used deprecated `datetime.utcnow()` — updated to `datetime.now(timezone.utc)`.
- `check.py` at repo root: 2-line leftover — deleted.
- Real `.env` was included in the patch zip — keep real env files out of anything you hand off.

---

## Final Test Coverage

| Suite | Before | New tests | After | Status |
|---|---|---|---|---|
| `mkt_surveillance_ml/tests/` | 302 | +9 (`test_model_registry.py`) | 311 | **all pass** |
| `tests/` (backend) | 156 | +5 (`test_integration_config.py`) | 161 | **all pass** |
| **Total** | **458** | **+14** | **472** | |

`serving/model_registry.py` coverage: 0% → 96%.
