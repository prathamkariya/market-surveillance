import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from mkt_surveillance_ml.features.scaling import (
    StandardScalerScratch,
    MinMaxScalerScratch,
    RobustScalerScratch,
    needs_scaling,
)


@pytest.fixture
def sample_df():
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "a": rng.normal(0, 1, 200),
        "b": rng.lognormal(15, 0.3, 200),
    })


class TestStandardScalerScratch:
    def test_matches_sklearn(self, sample_df):
        scratch = StandardScalerScratch().fit_transform(sample_df)
        library = StandardScaler().fit_transform(sample_df)
        assert np.allclose(scratch, library, rtol=1e-8)

    def test_output_has_zero_mean_unit_std(self, sample_df):
        out = StandardScalerScratch().fit_transform(sample_df)
        assert np.allclose(out.mean(axis=0), 0, atol=1e-8)
        assert np.allclose(out.std(axis=0), 1, atol=1e-8)

    def test_raises_on_constant_feature(self):
        df = pd.DataFrame({"const": [5.0] * 50})
        with pytest.raises(ValueError, match="constant"):
            StandardScalerScratch().fit_transform(df)

    def test_transform_before_fit_raises(self, sample_df):
        with pytest.raises(RuntimeError):
            StandardScalerScratch().transform(sample_df)

    def test_fit_then_transform_separately_matches_fit_transform(self, sample_df):
        s1 = StandardScalerScratch()
        s1.fit(sample_df)
        separate = s1.transform(sample_df)
        combined = StandardScalerScratch().fit_transform(sample_df)
        assert np.allclose(separate, combined)


class TestMinMaxScalerScratch:
    def test_matches_sklearn(self, sample_df):
        scratch = MinMaxScalerScratch().fit_transform(sample_df)
        library = MinMaxScaler().fit_transform(sample_df)
        assert np.allclose(scratch, library, rtol=1e-8)

    def test_output_bounded_zero_one(self, sample_df):
        out = MinMaxScalerScratch().fit_transform(sample_df)
        assert out.min() >= -1e-10
        assert out.max() <= 1 + 1e-10

    def test_single_outlier_compresses_rest_of_distribution(self):
        """File 20 Section 4.2's specific claim: ONE outlier compresses
        the scaled position of every ordinary value toward the same tiny
        sliver of [0,1]."""
        rng = np.random.RandomState(1)
        clean = pd.Series(rng.normal(100, 5, 100))
        contaminated = clean.copy()
        contaminated.iloc[50] = 500

        clean_scaled = MinMaxScalerScratch().fit_transform(clean.to_frame())
        contaminated_scaled = MinMaxScalerScratch().fit_transform(contaminated.to_frame())

        typical_idx = 10
        clean_position = clean_scaled[typical_idx, 0]
        contaminated_position = contaminated_scaled[typical_idx, 0]
        # same raw value should land in a much smaller (compressed) position
        # once the outlier stretches the range it's being scaled against
        assert contaminated_position < clean_position


class TestRobustScalerScratch:
    def test_matches_sklearn(self, sample_df):
        scratch = RobustScalerScratch().fit_transform(sample_df)
        library = RobustScaler().fit_transform(sample_df)
        assert np.allclose(scratch, library, rtol=1e-6)

    def test_resists_outlier_better_than_minmax(self):
        """File 20 Section 4.3's core claim: robust scaling's typical-value
        position should be far less disturbed by one outlier than min-max's."""
        rng = np.random.RandomState(1)
        clean = pd.Series(rng.normal(100, 5, 100))
        contaminated = clean.copy()
        contaminated.iloc[50] = 500

        clean_robust = RobustScalerScratch().fit_transform(clean.to_frame())[10, 0]
        contam_robust = RobustScalerScratch().fit_transform(contaminated.to_frame())[10, 0]

        clean_minmax = MinMaxScalerScratch().fit_transform(clean.to_frame())[10, 0]
        contam_minmax = MinMaxScalerScratch().fit_transform(contaminated.to_frame())[10, 0]

        robust_shift = abs(clean_robust - contam_robust)
        minmax_shift = abs(clean_minmax - contam_minmax)
        assert robust_shift < minmax_shift

    def test_raises_on_degenerate_iqr(self):
        df = pd.DataFrame({"mostly_const": [1.0] * 80 + [1.0] * 20})
        with pytest.raises(ValueError, match="IQR"):
            RobustScalerScratch().fit_transform(df)


class TestNeedsScaling:
    @pytest.mark.parametrize("family", [
        "logistic_regression", "kmeans", "knn", "svm", "isolation_forest", "lof",
    ])
    def test_scale_sensitive_families_need_scaling(self, family):
        assert needs_scaling(family) is True

    @pytest.mark.parametrize("family", [
        "decision_tree", "random_forest", "gradient_boosting", "xgboost",
    ])
    def test_scale_invariant_families_dont_need_scaling(self, family):
        assert needs_scaling(family) is False

    def test_unknown_family_raises_rather_than_guessing(self):
        with pytest.raises(ValueError, match="Unknown model family"):
            needs_scaling("some_new_model_nobody_registered")
