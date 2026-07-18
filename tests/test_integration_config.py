"""
tests/test_integration_config.py — regression tests for infrastructure and config correctness.

Covers four independent concerns:
  1. Cross-user MarketData collision handling (unique constraint + 409 response)
  2. Dead-code hygiene (analysis_service.py must not exist)
  3. Dockerfile and docker-compose.yml reference real files and real settings fields
  4. ModelRegistry singleton race condition (lock-based lazy init)

TestCrossUserMarketDataCollision and TestDeadCodeRemoved need the normal
`client`/`db_session` fixtures (Postgres, per conftest.py). The other two
classes are pure file/string checks and need neither DB nor app import.
"""
import re
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


# ══════════════════════════════════════════════════════════════
# 1. Cross-user market-data collision
#
# app/models.py: MarketData.__table_args__ has
#   UniqueConstraint("symbol", "timestamp", ...)
# with no user_id in it -- so the DB treats (symbol, timestamp) as
# globally unique across ALL users, even though every service/router
# (market_data.py, anomaly_service.py) queries MarketData scoped by
# user_id, i.e. treats the data as per-user. Two different users
# ingesting the same popular symbol at the same candle timestamp --
# an entirely ordinary thing for two people both watching AAPL --
# collide on a constraint that was never meant to apply across them.
# app/routers/market_data.py's create_market_data() also has no
# try/except around db.commit(), so the IntegrityError this produces
# is unhandled -> the client sees a raw 500, not a clean 409.
# ══════════════════════════════════════════════════════════════
class TestCrossUserMarketDataCollision:
    def _register_and_login(self, client, email, username):
        client.post("/api/v1/auth/register", json={
            "email": email, "username": username, "password": "SecurePass1",
        })
        login = client.post("/api/v1/auth/login", json={"email": email, "password": "SecurePass1"})
        return {"Authorization": f"Bearer {login.json()['access_token']}"}

    def test_second_user_same_symbol_and_timestamp_gets_clean_response_not_500(self, client, auth_headers):
        """
        EXPECTED TO FAIL against the current code -- and it fails in a
        telling way: the second POST doesn't come back as a handled response
        at all. TestClient's default (raise_server_exceptions=True) lets the
        IntegrityError from db.commit() propagate straight out of
        client.post(...) as a raised exception, because
        create_market_data() has no try/except around the commit to catch
        it first. Against a real deployed server (uvicorn, not TestClient),
        Starlette's ServerErrorMiddleware would catch this same unhandled
        exception and turn it into a generic 500 for the actual HTTP
        client -- still wrong, just wrong in a way that doesn't blow up the
        request/response cycle itself.

        This test doesn't prescribe which fix you take -- both are
        legitimate and the right one depends on what MarketData is
        supposed to mean:

          (a) MarketData is genuinely per-user -> scope the constraint:
              UniqueConstraint("user_id", "symbol", "timestamp", ...)
              via a new Alembic migration. Then this becomes a normal
              201 (each user independently owns their own copy of the
              AAPL candle).

          (b) MarketData is meant to be shared, objective market data
              (two users watching AAPL should see the SAME row) -> keep
              the global constraint, but catch IntegrityError in
              create_market_data() and return 409 (or look up and
              return the existing row) instead of letting it propagate.

        Whichever you pick, "second user's request blows up instead of
        getting a handled response" is the one outcome that's wrong under
        both readings.
        """
        payload = {
            "symbol": "AAPL",
            "timestamp": "2024-01-15T10:00:00Z",
            "open": 185.50, "high": 186.20, "low": 185.10, "close": 185.90,
            "volume": 1250000.0,
        }
        first = client.post("/api/v1/market-data", json=payload, headers=auth_headers)
        assert first.status_code == 201

        other_headers = self._register_and_login(client, "collision@example.com", "collideuser")
        second = client.post("/api/v1/market-data", json=payload, headers=other_headers)

        assert second.status_code in (201, 409), (
            f"Expected 201 (per-user scoping) or 409 (clean conflict handling), "
            f"got {second.status_code}: {second.text}"
        )


# ══════════════════════════════════════════════════════════════
# 2. Dead code that PHASE7_INTEGRATION.md says was removed, but wasn't
#
# apply_phase7.py's `to_delete` list removes app/models/, app/api/,
# app/features/, app/db/database.py, and tests/test_analyses.py -- but
# NOT app/services/analysis_service.py. Since shutil.copytree(...,
# dirs_exist_ok=True) merges rather than mirrors, the Phase 7 copy step
# doesn't remove it either (the phase7 staging app/services/ dir never
# had this file to begin with). It survives, unreachable and
# unimportable (imports app.features.engineer and
# app.models.database/app.models.schemas, none of which exist anymore).
# ══════════════════════════════════════════════════════════════
class TestDeadCodeRemoved:
    def test_analysis_service_file_does_not_exist(self):
        """
        EXPECTED TO FAIL against the current code.

        PHASE7_INTEGRATION.md section 2 ("Dead code removed") explicitly
        lists app/services/analysis_service.py as removed. It's still on
        disk. Fix: add it to apply_phase7.py's to_delete list and re-run,
        or just `rm app/services/analysis_service.py` directly -- nothing
        imports it (grep -rn "analysis_service" --include="*.py" . turns
        up only the file importing itself).
        """
        assert not (REPO_ROOT / "app" / "services" / "analysis_service.py").exists(), (
            "app/services/analysis_service.py still exists but PHASE7_INTEGRATION.md "
            "documents it as removed, and it imports app.features.engineer + "
            "app.models.database/schemas, none of which exist anymore -- it cannot "
            "be imported without an immediate ModuleNotFoundError."
        )


# ══════════════════════════════════════════════════════════════
# 3. Docker build is broken: Dockerfile assumes Poetry, repo uses pip
#
# `COPY pyproject.toml poetry.lock ./` -- neither file exists at the
# repo root (the only pyproject.toml in this whole project is
# ml/pyproject.toml, a different subproject). Every
# other setup doc in this repo (requirements.txt itself,
# PHASE7_INTEGRATION.md's "Setup to actually run this",
# apply_phase7.py's printed next-steps) says pip + requirements.txt.
# `docker build .` fails at this COPY step before installing a single
# dependency.
# ══════════════════════════════════════════════════════════════
class TestDockerfileReferencesExistingFiles:
    def test_every_copy_source_exists_in_build_context(self):
        """EXPECTED TO FAIL against the current Dockerfile."""
        dockerfile = (REPO_ROOT / "Dockerfile").read_text()
        missing = []
        for line in dockerfile.splitlines():
            line = line.strip()
            if not line.upper().startswith("COPY ") or "--from=" in line:
                continue  # skip multi-stage COPY --from=<image>, not a build-context path
            parts = line.split()[1:]
            if len(parts) < 2:
                continue
            sources, _dest = parts[:-1], parts[-1]
            for src in sources:
                if src == ".":
                    continue  # "COPY . ." always resolves (the whole context)
                if any(ch in src for ch in "*?["):
                    if not list(REPO_ROOT.glob(src)):
                        missing.append(src)
                elif not (REPO_ROOT / src).exists():
                    missing.append(src)

        assert not missing, (
            f"Dockerfile COPYs {missing} but {['no matching file' if any(ch in m for ch in '*?[') else 'this file' for m in missing]} "
            f"exist(s) at the repo root -- `docker build .` fails immediately on this "
            f"COPY step. The project uses requirements.txt (confirmed present at repo "
            f"root) everywhere else, not Poetry. Fix: replace the poetry install block "
            f"with `COPY requirements.txt . RUN pip install --no-cache-dir -r requirements.txt` "
            f"(plus `RUN pip install -e ml/`, which the Dockerfile "
            f"already does correctly one line down)."
        )


# ══════════════════════════════════════════════════════════════
# 4. docker-compose.yml sets JWT_SECRET_KEY; Settings reads SECRET_KEY
#
# app/config.py's Settings class (pydantic-settings, case_sensitive=True)
# has a field named SECRET_KEY. docker-compose.yml's `api` service sets
# JWT_SECRET_KEY in its environment block. Pydantic only maps env vars
# to fields by exact name match, so JWT_SECRET_KEY is silently ignored
# and SECRET_KEY falls back to its hardcoded default
# ("change-this-in-production-use-openssl-rand-hex-32") -- a value
# visible in this repo's own source. A `docker-compose up` deployment
# signs every JWT with a publicly-readable default secret.
# ══════════════════════════════════════════════════════════════
class TestDockerComposeEnvVarsMatchSettings:
    def test_api_service_env_vars_are_real_settings_fields(self):
        """EXPECTED TO FAIL against the current docker-compose.yml."""
        from app.config import Settings
        valid_fields = set(Settings.model_fields.keys())

        compose_text = (REPO_ROOT / "docker-compose.yml").read_text()
        # crude but sufficient block extraction: from "api:" service's
        # "environment:" list to the next top-level-ish key
        api_block = compose_text.split("api:", 1)[1]
        env_block = api_block.split("environment:", 1)[1].split("depends_on:", 1)[0]

        env_var_names = re.findall(r"^\s*-\s*([A-Z_][A-Z0-9_]*)\s*=", env_block, re.MULTILINE)
        assert env_var_names, "Expected to find at least one KEY=value line under api.environment"

        unrecognized = [name for name in env_var_names if name not in valid_fields]
        assert not unrecognized, (
            f"docker-compose.yml sets {unrecognized} under services.api.environment, but "
            f"{unrecognized} {'is' if len(unrecognized) == 1 else 'are'} not field name(s) on "
            f"Settings (valid: {sorted(valid_fields)}). pydantic-settings matches env vars to "
            f"fields by exact name (case_sensitive=True) -- an unrecognized name is silently "
            f"ignored, not an error, so Settings falls back to that field's hardcoded default "
            f"instead of the value docker-compose provides. Fix: rename JWT_SECRET_KEY to "
            f"SECRET_KEY in docker-compose.yml."
        )


# ══════════════════════════════════════════════════════════════
# 5. get_model_registry()'s lazy singleton has no lock around the
#    check-then-load, so a concurrent caller can observe (and use) a
#    registry object that is still mid-load.
#
# Deterministic reproduction: monkeypatch ModelRegistry.load() to pause
# on an Event so we control the exact interleaving, rather than hoping
# a sleep() wins a timing race.
# ══════════════════════════════════════════════════════════════
class _DummyMultiPatternForRaceTest:
    def predict_proba(self, X):
        return X


class TestModelRegistrySingletonRace:
    def test_concurrent_caller_can_observe_a_still_loading_registry(self, tmp_path, monkeypatch):
        """
        EXPECTED TO FAIL against the current get_model_registry().

        Sequence forced by the two Events below:
          1. Thread A calls get_model_registry(), sees _registry is None,
             constructs it, assigns it to the module global, then calls
             .load() -- which we've made block here.
          2. Main thread waits until we know Thread A is inside .load(),
             then starts Thread B.
          3. Thread B calls get_model_registry(). _registry is already
             non-None (Thread A assigned it in step 1 before calling
             .load()), so Thread B's `if _registry is None` check is
             False -- it returns immediately without waiting for A's
             .load() to finish, holding a reference to an object that
             is still mid-load.
          4. We read has_any_model through Thread B's reference while A
             is still blocked, then release A.

        In production this is a real window, not just a lab construct:
        model files are read from disk (I/O, not instant) on first use
        per process, so the very first burst of concurrent requests
        after a cold start/restart lands exactly here. A request landing
        in the window sees has_any_model still False and gets a spurious
        503 "no trained models available", even though loading succeeds
        moments later for everyone after it.

        Fix -- a lock around the check-and-load (double-checked locking;
        safe in Python because the GIL makes the reference read/write
        atomic on either side of the lock):

            import threading
            _registry: Optional[ModelRegistry] = None
            _registry_lock = threading.Lock()

            def get_model_registry() -> ModelRegistry:
                global _registry
                if _registry is None:
                    with _registry_lock:
                        if _registry is None:
                            candidate = ModelRegistry(settings.MODEL_DIR)
                            try:
                                candidate.load()
                            except ModelLoadError as e:
                                logger.warning(f"Model loading failed: {e}")
                            _registry = candidate
                return _registry

        (Building into a local `candidate` and only publishing to the
        global _registry after .load() finishes/fails is what actually
        closes the window -- a lock alone still leaves the same gap if
        _registry is assigned before .load() is called, as the current
        code does.)
        """
        import app.services.anomaly_service as anomaly_service
        from ml.serving.model_registry import ModelRegistry
        import joblib

        joblib.dump(_DummyMultiPatternForRaceTest(), tmp_path / "multi_pattern_detector.joblib")

        entered_load = threading.Event()
        release_load = threading.Event()

        class SlowLoadRegistry(ModelRegistry):
            def load(self):
                entered_load.set()
                assert release_load.wait(timeout=5), "test setup error: release_load was never set"
                super().load()

        monkeypatch.setattr(anomaly_service, "_registries", {})
        monkeypatch.setattr(anomaly_service, "ModelRegistry", SlowLoadRegistry)
        monkeypatch.setattr(anomaly_service.settings, "MODEL_DIR", str(tmp_path))

        observed = {}

        def thread_a():
            anomaly_service.get_model_registry()

        def thread_b():
            assert entered_load.wait(timeout=5), "Thread A never entered load() -- test setup broken"
            registry = anomaly_service.get_model_registry()
            observed["has_any_model"] = registry.has_any_model

        ta = threading.Thread(target=thread_a)
        tb = threading.Thread(target=thread_b)
        ta.start()
        tb.start()
        # Give thread B a bounded window to attempt get_model_registry() and
        # either (a) return immediately without a lock -- the bug -- or
        # (b) block waiting for a lock A holds -- the fix. Either way,
        # release A now so the run can't hang: (a) already raced past this
        # point regardless of A; (b) is unblocked by it.
        entered_load.wait(timeout=5)
        release_load.set()
        ta.join(timeout=6)
        tb.join(timeout=6)

        assert "has_any_model" in observed, "Thread B did not complete in time"
        assert observed["has_any_model"] is True, (
            "Thread B got a registry reference back from get_model_registry() while "
            "Thread A's .load() call was still in progress, and has_any_model read as "
            "False even though loading the multi-pattern detector succeeds a moment "
            "later. A request landing in this window gets an incorrect 503."
        )
