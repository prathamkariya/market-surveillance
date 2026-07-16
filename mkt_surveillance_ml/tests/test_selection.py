import numpy as np
import pandas as pd
import pytest

from mkt_surveillance_ml.features.selection import (
    filter_select,
    wrapper_select,
    embedded_select,
    compare_score_against_combined_vs_per_pattern_label,
)


class TestFilterSelect:
    def test_mutual_info_ranks_strong_signal_above_pure_noise(self):
        rng = np.random.RandomState(0)
        n = 400
        X = pd.DataFrame({
            "strong_signal": rng.normal(0, 1, n),
            "pure_noise": rng.normal(0, 1, n),
        })
        y = (X["strong_signal"] * 3 + rng.normal(0, 1, n) > 0).astype(int)
        ranked = filter_select(X, y, method="mutual_info")
        assert ranked.iloc[0]["feature"] == "strong_signal"

    def test_f_test_misses_nonlinear_relationship(self):
        """File 20 Section 5.2: y depends on |X|, near-zero LINEAR
        correlation. F-test should score this weakly."""
        rng = np.random.RandomState(2)
        n = 400
        X_val = rng.uniform(-3, 3, n)
        y = (np.abs(X_val) > 1.5).astype(int)
        X = pd.DataFrame({"x": X_val})
        f_result = filter_select(X, y, method="f_test")
        assert f_result.iloc[0]["p_value"] > 0.01  # not significant by conventional threshold

    def test_mutual_info_catches_nonlinear_relationship_f_test_misses(self):
        rng = np.random.RandomState(2)
        n = 400
        X_val = rng.uniform(-3, 3, n)
        y = (np.abs(X_val) > 1.5).astype(int)
        X = pd.DataFrame({"x": X_val})
        mi_result = filter_select(X, y, method="mutual_info")
        assert mi_result.iloc[0]["score"] > 0.05

    def test_invalid_method_raises(self):
        X = pd.DataFrame({"x": [1, 2, 3, 4]})
        y = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError, match="Unknown method"):
            filter_select(X, y, method="not_a_real_method")


class TestWrapperSelect:
    def test_recovers_true_signal_columns(self):
        rng = np.random.RandomState(3)
        n, n_features = 300, 15
        X_val = rng.normal(0, 1, (n, n_features))
        true_idx = [2, 5, 9]
        y = (X_val[:, true_idx].sum(axis=1) + rng.normal(0, 0.5, n) > 0).astype(int)
        X = pd.DataFrame(X_val, columns=[f"f{i}" for i in range(n_features)])

        selected = wrapper_select(X, y, n_features_to_select=3, random_state=3)
        selected_idx = sorted(int(c[1:]) for c in selected)
        assert selected_idx == sorted(true_idx)

    def test_respects_n_features_to_select(self):
        rng = np.random.RandomState(4)
        X = pd.DataFrame(rng.normal(0, 1, (200, 10)), columns=[f"f{i}" for i in range(10)])
        y = (X["f0"] > 0).astype(int)
        selected = wrapper_select(X, y, n_features_to_select=4)
        assert len(selected) == 4


class TestEmbeddedSelect:
    def test_ranks_signal_features_above_noise(self):
        rng = np.random.RandomState(5)
        n = 300
        X = pd.DataFrame({
            "signal": rng.normal(0, 1, n),
            "noise_1": rng.normal(0, 1, n),
            "noise_2": rng.normal(0, 1, n),
        })
        y = (X["signal"] * 2 + rng.normal(0, 0.5, n) > 0).astype(int)
        importances = embedded_select(X, y)
        assert importances.iloc[0]["feature"] == "signal"

    def test_top_k_limits_output_rows(self):
        rng = np.random.RandomState(6)
        X = pd.DataFrame(rng.normal(0, 1, (200, 8)), columns=[f"f{i}" for i in range(8)])
        y = (X["f0"] > 0).astype(int)
        result = embedded_select(X, y, top_k=3)
        assert len(result) == 3

    def test_importances_sum_to_one(self):
        rng = np.random.RandomState(7)
        X = pd.DataFrame(rng.normal(0, 1, (200, 5)), columns=[f"f{i}" for i in range(5)])
        y = (X["f0"] > 0).astype(int)
        result = embedded_select(X, y)
        assert abs(result["importance"].sum() - 1.0) < 1e-6


class TestCompareScoreAgainstCombinedVsPerPatternLabel:
    def test_feature_perfect_for_one_pattern_scores_lower_against_combined_label(self):
        """File 20 Section 6's exact experiment: a feature that's a perfect
        proxy for pattern A specifically should score measurably lower
        against a blended label than against pattern A's true label."""
        rng = np.random.RandomState(8)
        n = 400
        pattern_a = np.zeros(n, dtype=int)
        pattern_b = np.zeros(n, dtype=int)
        a_days = rng.choice(n, 40, replace=False)
        remaining = [d for d in range(n) if d not in a_days]
        b_days = rng.choice(remaining, 40, replace=False)
        pattern_a[a_days] = 1
        pattern_b[b_days] = 1

        feature = pd.Series(rng.normal(0, 1, n))
        feature.iloc[a_days] = 5.0  # perfect proxy for pattern A only

        combined = ((pattern_a == 1) | (pattern_b == 1)).astype(int)

        result = compare_score_against_combined_vs_per_pattern_label(
            feature, pattern_a, combined, random_state=8
        )
        assert result["degradation_pct"] > 0
        assert result["mi_against_true_pattern"] > result["mi_against_combined_label"]
