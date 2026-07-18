"""
tests/test_model_registry.py — ml.serving.model_registry

This module had ZERO test coverage before this file (confirmed via
`pytest --cov=ml.serving`: "Module ... was never
imported" across the entire 302-test suite). It's the one new module
Phase 7 added to this package, and the one thing anomaly_service.py's
whole scoring path depends on to even start.

One test below — TestPartialFailureIsolation::test_corrupted_isolation_forest_does_not_block_valid_multi_pattern_load —
is a REGRESSION test for a real bug and is EXPECTED TO FAIL against the
current model_registry.py. See its docstring for the mechanism and the
one-line fix.
"""
import joblib
import pytest

from ml.serving.model_registry import ModelRegistry, ModelLoadError


# ──────────────────────────────────────────────
# Fakes standing in for real joblib-dumped model objects.
# Nothing here needs to be an actual sklearn estimator — ModelRegistry
# only cares that joblib.load() returns *something*.
# ──────────────────────────────────────────────
class _FakeIsolationForest:
    def score_samples(self, X):
        return [0.42] * len(X)


class _FakeMultiPatternDetector:
    def predict_proba(self, X):
        return X


def _write_valid_iforest(model_dir, metadata=None):
    joblib.dump(_FakeIsolationForest(), model_dir / "isolation_forest_scratch.joblib")
    if metadata is not None:
        (model_dir / "isolation_forest_metadata.json").write_text(metadata)


def _write_valid_multi_pattern(model_dir, metadata=None):
    joblib.dump(_FakeMultiPatternDetector(), model_dir / "multi_pattern_detector.joblib")
    if metadata is not None:
        (model_dir / "multi_pattern_detector_metadata.json").write_text(metadata)


def _write_corrupt_file(path):
    """Not a valid joblib/pickle payload — any load attempt raises."""
    path.write_bytes(b"this is not a pickle, joblib.load() must raise on it")


# ══════════════════════════════════════════════════════════════
# Directory / existence handling
# ══════════════════════════════════════════════════════════════
class TestModelDirHandling:
    def test_raises_when_model_dir_does_not_exist(self, tmp_path):
        registry = ModelRegistry(str(tmp_path / "does_not_exist"))
        with pytest.raises(ModelLoadError, match="does not exist"):
            registry.load()

    def test_empty_dir_loads_without_error_but_has_no_models(self, tmp_path):
        registry = ModelRegistry(str(tmp_path))
        registry.load()  # should NOT raise -- an empty dir is "no models yet", not an error
        assert registry.has_any_model is False
        assert registry.has_isolation_forest is False
        assert registry.has_multi_pattern is False


# ══════════════════════════════════════════════════════════════
# Independent single-model loading
# ══════════════════════════════════════════════════════════════
class TestSingleModelLoad:
    def test_loads_isolation_forest_only(self, tmp_path):
        _write_valid_iforest(tmp_path)
        registry = ModelRegistry(str(tmp_path))
        registry.load()
        assert registry.has_isolation_forest is True
        assert registry.has_multi_pattern is False
        assert registry.has_any_model is True

    def test_loads_multi_pattern_only(self, tmp_path):
        _write_valid_multi_pattern(tmp_path)
        registry = ModelRegistry(str(tmp_path))
        registry.load()
        assert registry.has_multi_pattern is True
        assert registry.has_isolation_forest is False
        assert registry.has_any_model is True

    def test_loads_both_when_both_present(self, tmp_path):
        _write_valid_iforest(tmp_path)
        _write_valid_multi_pattern(tmp_path)
        registry = ModelRegistry(str(tmp_path))
        registry.load()
        assert registry.has_isolation_forest is True
        assert registry.has_multi_pattern is True


# ══════════════════════════════════════════════════════════════
# Metadata handling
# ══════════════════════════════════════════════════════════════
class TestMetadata:
    def test_metadata_loaded_alongside_model(self, tmp_path):
        _write_valid_iforest(tmp_path, metadata='{"trained_at_utc": "2026-07-01T00:00:00Z"}')
        registry = ModelRegistry(str(tmp_path))
        registry.load()
        assert registry.isolation_forest_metadata == {"trained_at_utc": "2026-07-01T00:00:00Z"}

    def test_missing_metadata_file_does_not_block_model_load(self, tmp_path):
        """Metadata is documented as optional provenance, not a requirement."""
        _write_valid_multi_pattern(tmp_path, metadata=None)
        registry = ModelRegistry(str(tmp_path))
        registry.load()  # must not raise just because the .json sidecar is absent
        assert registry.has_multi_pattern is True
        assert registry.multi_pattern_metadata == {}


# ══════════════════════════════════════════════════════════════
# Partial-failure isolation
#
# model_registry.load() attempts isolation-forest first, then
# multi-pattern, as two sequential blocks in one method. There is
# no isolation between them: an exception raised loading the FIRST
# model type propagates straight out of load(), so the SECOND
# model type's `if mp_path.exists(): try: ...` block is never
# reached at all -- even though nothing is wrong with that file.
# ══════════════════════════════════════════════════════════════
class TestPartialFailureIsolation:
    def test_corrupted_multi_pattern_does_not_affect_already_loaded_isolation_forest(self, tmp_path):
        """Control case: multi-pattern is the SECOND block in load(), so when
        IT fails, isolation-forest (already loaded successfully by that point)
        is correctly left intact. This currently passes -- included so the
        contrast with the test below is explicit rather than assumed."""
        _write_valid_iforest(tmp_path)
        _write_corrupt_file(tmp_path / "multi_pattern_detector.joblib")

        registry = ModelRegistry(str(tmp_path))
        with pytest.raises(ModelLoadError, match="Multi Pattern Detector"):
            registry.load()

        assert registry.has_isolation_forest is True, (
            "isolation_forest_scratch.joblib is completely valid and was loaded "
            "before the multi-pattern block ever raised -- it must not be undone."
        )
        assert registry.has_multi_pattern is False

    def test_corrupted_isolation_forest_does_not_block_valid_multi_pattern_load(self, tmp_path):
        """BUG: isolation-forest is the FIRST block in load(). When IT raises,
        the exception exits load() immediately -- the multi-pattern block
        below it never runs, even though multi_pattern_detector.joblib here
        is completely valid and loadable on its own.

        Practical impact: in anomaly_service.py, get_model_registry() catches
        ModelLoadError and just logs a warning, then returns the (silently
        half-populated) registry. If has_any_model ends up False as a result,
        every /anomalies request gets a 503 "no trained models available" --
        even though a perfectly good multi-pattern model is sitting on disk,
        solely because an unrelated file (isolation_forest_scratch.joblib)
        happened to be corrupt.

        EXPECTED TO FAIL against the current model_registry.py.

        Fix -- give each model type its own independent try/except so one
        failing does not prevent the other from being attempted:

            def load(self):
                if not self.model_dir.exists():
                    raise ModelLoadError(f"Model directory {self.model_dir} does not exist.")

                errors = []

                iforest_path = self.model_dir / "isolation_forest_scratch.joblib"
                if iforest_path.exists():
                    try:
                        self.isolation_forest = joblib.load(iforest_path)
                        meta = self.model_dir / "isolation_forest_metadata.json"
                        if meta.exists():
                            self.isolation_forest_metadata = json.loads(meta.read_text())
                    except Exception as e:
                        errors.append(f"Failed to load Isolation Forest: {e}")

                mp_path = self.model_dir / "multi_pattern_detector.joblib"
                if mp_path.exists():
                    try:
                        self.multi_pattern_detector = joblib.load(mp_path)
                        meta = self.model_dir / "multi_pattern_detector_metadata.json"
                        if meta.exists():
                            self.multi_pattern_metadata = json.loads(meta.read_text())
                    except Exception as e:
                        errors.append(f"Failed to load Multi Pattern Detector: {e}")

                if errors:
                    raise ModelLoadError("; ".join(errors))

        (Raising once at the end, after both independent attempts, keeps the
        "surface the error" behavior load() already has -- it just no longer
        lets one error suppress the other attempt.)
        """
        _write_corrupt_file(tmp_path / "isolation_forest_scratch.joblib")
        _write_valid_multi_pattern(tmp_path)

        registry = ModelRegistry(str(tmp_path))
        with pytest.raises(ModelLoadError, match="Isolation Forest"):
            registry.load()

        assert registry.has_multi_pattern is True, (
            "multi_pattern_detector.joblib is completely valid and independent "
            "of the isolation-forest file -- a corrupt isolation-forest artifact "
            "must not prevent it from loading."
        )
