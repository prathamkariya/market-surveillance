"""
tests/test_anomaly.py — Anomaly detection endpoint tests.

Covers: score bounds, threshold logic, feature storage, auth guards.

PHASE 7 UPDATE: anomaly_service now scores with real trained
mkt_surveillance_ml models instead of mock formulas (see
app/services/anomaly_service.py's module docstring). This changed two
things tests must account for, deliberately, not incidentally:

1. Real models need real rolling-window feature history (20+ trailing
   days) to produce a score at all -- sample_market_data (1 record) is
   no longer enough for tests that expect a real score.
   sample_market_data_with_history (conftest.py) provides 30 sequential
   days for exactly this reason. sample_market_data still exists and is
   used deliberately by test_insufficient_history_returns_400.

2. The feature set changed (mock's price_return/price_range/
   volume_zscore/price_volatility/body_ratio -> mkt_surveillance_ml's
   return/volume_ratio_20d/volatility_20d), and xgboost_score was
   renamed to multi_pattern_max_score (see migration 004 and its
   docstring for why: the old name was never accurate even for the mock,
   and MultiPatternDetector's default estimator isn't XGBoost either --
   keeping a misleading name for convenience wasn't worth it).
"""
import json
import pytest


# ══════════════════════════════════════════════════════════════
# DETECT ANOMALY
# ══════════════════════════════════════════════════════════════
class TestDetectAnomaly:
    def test_detect_returns_201(self, client, auth_headers, sample_market_data_with_history):
        response = client.post("/api/v1/anomalies", json={
            "market_data_id": sample_market_data_with_history["id"],
        }, headers=auth_headers)
        assert response.status_code == 201, response.text

    def test_response_has_all_required_fields(self, client, auth_headers, sample_market_data_with_history):
        response = client.post("/api/v1/anomalies", json={
            "market_data_id": sample_market_data_with_history["id"],
        }, headers=auth_headers)
        body = response.json()
        for field in ("id", "market_data_id", "anomaly_score", "is_anomaly",
                      "isolation_forest_score", "multi_pattern_max_score",
                      "pattern_scores", "model_version", "features", "detected_at"):
            assert field in body, f"Missing field: {field}"

    def test_market_data_id_linked_correctly(self, client, auth_headers, sample_market_data_with_history):
        response = client.post("/api/v1/anomalies", json={
            "market_data_id": sample_market_data_with_history["id"],
        }, headers=auth_headers)
        assert response.json()["market_data_id"] == sample_market_data_with_history["id"]

    # ── Score bounds ─────────────────────────────────────────
    def test_anomaly_score_bounded_0_to_1(self, client, auth_headers, sample_market_data_with_history):
        response = client.post("/api/v1/anomalies", json={
            "market_data_id": sample_market_data_with_history["id"],
        }, headers=auth_headers)
        score = response.json()["anomaly_score"]
        assert 0.0 <= score <= 1.0, f"anomaly_score {score} out of [0,1]"

    def test_isolation_forest_score_bounded_0_to_1(self, client, auth_headers, sample_market_data_with_history):
        response = client.post("/api/v1/anomalies", json={
            "market_data_id": sample_market_data_with_history["id"],
        }, headers=auth_headers)
        score = response.json()["isolation_forest_score"]
        assert score is not None
        assert 0.0 <= score <= 1.0, f"isolation_forest_score {score} out of [0,1]"

    def test_multi_pattern_max_score_bounded_0_to_1(self, client, auth_headers, sample_market_data_with_history):
        """Renamed from test_xgboost_score_bounded_0_to_1 -- see migration
        004's docstring for why the field itself was renamed, not just
        this test."""
        response = client.post("/api/v1/anomalies", json={
            "market_data_id": sample_market_data_with_history["id"],
        }, headers=auth_headers)
        score = response.json()["multi_pattern_max_score"]
        assert score is not None
        assert 0.0 <= score <= 1.0, f"multi_pattern_max_score {score} out of [0,1]"

    # ── Threshold logic ──────────────────────────────────────
    def test_threshold_1_means_not_anomaly(self, client, auth_headers, sample_market_data_with_history):
        """With threshold=1.0, is_anomaly is False unless the combined
        score hits EXACTLY 1.0. Both real scores are mathematically
        bounded in [0,1] (not mock-clamped), and reaching exactly 1.0
        would require both models to be maximally confident
        simultaneously -- practically not expected on realistic test
        data, though not a hard impossibility the way the old mock's
        clamped formula guaranteed. See anomaly_service._combine_scores'
        docstring."""
        response = client.post("/api/v1/anomalies", json={
            "market_data_id": sample_market_data_with_history["id"],
            "threshold": 1.0,
        }, headers=auth_headers)
        assert response.json()["is_anomaly"] is False

    def test_threshold_0_means_always_anomaly(self, client, auth_headers, sample_market_data_with_history):
        """With threshold=0.0, any score >= 0 triggers is_anomaly. Both
        real scores are non-negative by construction, so this holds
        genuinely, not just for the old mock."""
        response = client.post("/api/v1/anomalies", json={
            "market_data_id": sample_market_data_with_history["id"],
            "threshold": 0.0,
        }, headers=auth_headers)
        assert response.json()["is_anomaly"] is True

    def test_default_threshold_is_applied(self, client, auth_headers, sample_market_data_with_history):
        """Omitting threshold should still produce a valid boolean result."""
        response = client.post("/api/v1/anomalies", json={
            "market_data_id": sample_market_data_with_history["id"],
        }, headers=auth_headers)
        assert isinstance(response.json()["is_anomaly"], bool)

    # ── Features storage ─────────────────────────────────────
    def test_features_json_is_stored(self, client, auth_headers, sample_market_data_with_history):
        response = client.post("/api/v1/anomalies", json={
            "market_data_id": sample_market_data_with_history["id"],
        }, headers=auth_headers)
        features_str = response.json().get("features")
        assert features_str is not None
        features = json.loads(features_str)
        assert isinstance(features, dict)

    def test_features_contains_expected_keys(self, client, auth_headers, sample_market_data_with_history):
        """Updated for the real feature set -- see this file's module
        docstring for why these are different from the old mock's keys."""
        response = client.post("/api/v1/anomalies", json={
            "market_data_id": sample_market_data_with_history["id"],
        }, headers=auth_headers)
        features = json.loads(response.json()["features"])
        expected_keys = {"return", "volume_ratio_20d", "volatility_20d"}
        assert expected_keys == features.keys(), f"Expected exactly {expected_keys}, got {features.keys()}"

    # ── Per-pattern breakdown (new in Phase 7) ────────────────
    def test_pattern_scores_contains_all_four_patterns(self, client, auth_headers, sample_market_data_with_history):
        response = client.post("/api/v1/anomalies", json={
            "market_data_id": sample_market_data_with_history["id"],
        }, headers=auth_headers)
        pattern_scores = json.loads(response.json()["pattern_scores"])
        assert set(pattern_scores.keys()) == {"pump_and_dump", "wash_trading", "spoofing", "layering"}

    def test_pattern_scores_are_bounded_0_to_1(self, client, auth_headers, sample_market_data_with_history):
        response = client.post("/api/v1/anomalies", json={
            "market_data_id": sample_market_data_with_history["id"],
        }, headers=auth_headers)
        pattern_scores = json.loads(response.json()["pattern_scores"])
        for pattern, score in pattern_scores.items():
            assert 0.0 <= score <= 1.0, f"{pattern} score {score} out of [0,1]"

    def test_multi_pattern_max_score_equals_max_of_pattern_scores(self, client, auth_headers, sample_market_data_with_history):
        response = client.post("/api/v1/anomalies", json={
            "market_data_id": sample_market_data_with_history["id"],
        }, headers=auth_headers)
        body = response.json()
        pattern_scores = json.loads(body["pattern_scores"])
        assert body["multi_pattern_max_score"] == pytest.approx(max(pattern_scores.values()))

    def test_model_version_is_populated(self, client, auth_headers, sample_market_data_with_history):
        """Provenance: which trained model(s) actually produced this
        score. Should mention both model types when both are loaded."""
        response = client.post("/api/v1/anomalies", json={
            "market_data_id": sample_market_data_with_history["id"],
        }, headers=auth_headers)
        model_version = response.json()["model_version"]
        assert model_version
        assert "isolation_forest=" in model_version
        assert "multi_pattern=" in model_version

    # ── Error cases ──────────────────────────────────────────
    def test_nonexistent_market_data_returns_404(self, client, auth_headers):
        response = client.post("/api/v1/anomalies", json={
            "market_data_id": 999999,
        }, headers=auth_headers)
        assert response.status_code == 404

    def test_insufficient_history_returns_400(self, client, auth_headers, sample_market_data):
        """New in Phase 7: real models need 20+ trailing days of history
        to compute rolling-window features. sample_market_data creates
        exactly ONE record -- deliberately not enough -- so this should
        fail clearly with 400, not silently score against a garbage or
        default-filled feature vector. See
        anomaly_service._market_data_to_feature_row's docstring."""
        response = client.post("/api/v1/anomalies", json={
            "market_data_id": sample_market_data["id"],
        }, headers=auth_headers)
        assert response.status_code == 400
        assert "Not enough historical data" in response.json()["detail"]

    def test_threshold_above_1_returns_422(self, client, auth_headers, sample_market_data):
        response = client.post("/api/v1/anomalies", json={
            "market_data_id": sample_market_data["id"],
            "threshold": 1.5,               # max is 1.0
        }, headers=auth_headers)
        assert response.status_code == 422

    def test_threshold_below_0_returns_422(self, client, auth_headers, sample_market_data):
        response = client.post("/api/v1/anomalies", json={
            "market_data_id": sample_market_data["id"],
            "threshold": -0.1,              # min is 0.0
        }, headers=auth_headers)
        assert response.status_code == 422

    def test_missing_market_data_id_returns_422(self, client, auth_headers):
        response = client.post("/api/v1/anomalies", json={}, headers=auth_headers)
        assert response.status_code == 422

    def test_zero_market_data_id_returns_422(self, client, auth_headers):
        response = client.post("/api/v1/anomalies", json={
            "market_data_id": 0,            # must be > 0 per schema
        }, headers=auth_headers)
        assert response.status_code == 422

    def test_requires_auth(self, client, sample_market_data):
        response = client.post("/api/v1/anomalies", json={
            "market_data_id": sample_market_data["id"],
        })
        assert response.status_code in (401, 403)

    # ── Weighted combination ─────────────────────────────────
    def test_composite_score_is_weighted_average(self, client, auth_headers, sample_market_data_with_history):
        """anomaly_score should equal 0.6*IF + 0.4*multi_pattern_max
        (within float tolerance) when both models are loaded."""
        response = client.post("/api/v1/anomalies", json={
            "market_data_id": sample_market_data_with_history["id"],
        }, headers=auth_headers)
        body = response.json()
        if_score = body["isolation_forest_score"]
        mp_score = body["multi_pattern_max_score"]
        expected = round(0.6 * if_score + 0.4 * mp_score, 4)
        assert abs(body["anomaly_score"] - expected) < 1e-4, (
            f"Expected weighted score {expected}, got {body['anomaly_score']}"
        )
