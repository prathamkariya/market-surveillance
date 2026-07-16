import numpy as np
import pandas as pd
import pytest

from mkt_surveillance_ml.config import PatternType, BASE_FEATURE_COLUMNS
from mkt_surveillance_ml.data.synthetic import (
    generate_synthetic_market_data,
    chronological_train_test_split,
    PatternInjectionConfig,
)


class TestGenerateSyntheticMarketData:
    def test_returns_dataframe_with_expected_columns(self):
        df = generate_synthetic_market_data(n_days=300)
        for col in BASE_FEATURE_COLUMNS:
            assert col in df.columns
        for p in PatternType:
            assert f"is_{p.value}" in df.columns
        assert "is_manipulation" in df.columns

    def test_no_nulls_in_output(self):
        df = generate_synthetic_market_data(n_days=300)
        assert not df.isnull().any().any(), "warm-up rows should be dropped, not left as NaN"

    def test_patterns_are_mutually_exclusive(self):
        """A day is at most one pattern -- required for per-pattern labels
        to be meaningfully separable rather than teaching contradictions."""
        df = generate_synthetic_market_data(n_days=500)
        label_cols = [f"is_{p.value}" for p in PatternType]
        assert (df[label_cols].sum(axis=1) <= 1).all()

    def test_is_manipulation_is_logical_or_of_patterns(self):
        df = generate_synthetic_market_data(n_days=500)
        label_cols = [f"is_{p.value}" for p in PatternType]
        expected = (df[label_cols].sum(axis=1) > 0).astype(int)
        assert (df["is_manipulation"] == expected).all()

    def test_injected_day_counts_match_config(self):
        df = generate_synthetic_market_data(n_days=500)
        from mkt_surveillance_ml.data.synthetic import DEFAULT_PATTERN_CONFIGS
        for pattern, cfg in DEFAULT_PATTERN_CONFIGS.items():
            assert df[f"is_{pattern.value}"].sum() == cfg.n_days

    def test_deterministic_given_same_random_state(self):
        df1 = generate_synthetic_market_data(n_days=300, random_state=7)
        df2 = generate_synthetic_market_data(n_days=300, random_state=7)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_random_state_gives_different_data(self):
        df1 = generate_synthetic_market_data(n_days=300, random_state=1)
        df2 = generate_synthetic_market_data(n_days=300, random_state=2)
        assert not df1["close"].equals(df2["close"])

    def test_volume_always_positive(self):
        df = generate_synthetic_market_data(n_days=500)
        assert (df["volume"] > 0).all()

    def test_close_always_positive(self):
        df = generate_synthetic_market_data(n_days=500)
        assert (df["close"] > 0).all()

    def test_index_is_sorted_dates(self):
        df = generate_synthetic_market_data(n_days=300)
        assert df.index.is_monotonic_increasing

    def test_subset_of_patterns_only_injects_those(self):
        df = generate_synthetic_market_data(
            n_days=300, patterns=[PatternType.PUMP_AND_DUMP]
        )
        assert df["is_pump_and_dump"].sum() > 0
        assert df["is_wash_trading"].sum() == 0
        assert df["is_spoofing"].sum() == 0
        assert df["is_layering"].sum() == 0

    def test_pump_and_dump_has_higher_mean_return_than_wash_trading(self):
        """The specific claim this package's whole design rests on: patterns
        need genuinely different signatures, not just 'elevated volume'
        for everything. Regression-guard against that collapsing back."""
        df = generate_synthetic_market_data(n_days=800)
        pump_return = df.loc[df["is_pump_and_dump"] == 1, "return"].mean()
        wash_return = df.loc[df["is_wash_trading"] == 1, "return"].mean()
        assert pump_return > wash_return + 0.03

    def test_manipulation_days_have_higher_volume_ratio_than_normal(self):
        df = generate_synthetic_market_data(n_days=800)
        manip_ratio = df.loc[df["is_manipulation"] == 1, "volume_ratio_20d"].mean()
        normal_ratio = df.loc[df["is_manipulation"] == 0, "volume_ratio_20d"].mean()
        assert manip_ratio > normal_ratio

    def test_raises_when_not_enough_days_for_requested_injections(self):
        with pytest.raises(ValueError):
            generate_synthetic_market_data(n_days=40)  # 30-day warmup leaves ~10 days, not enough for all 4 patterns

    def test_custom_pattern_config_respected(self):
        custom = {
            PatternType.PUMP_AND_DUMP: PatternInjectionConfig(
                n_days=5, return_mean=0.1, return_std=0.01,
                volume_ratio_mean=4.0, volume_ratio_std=0.1,
            )
        }
        df = generate_synthetic_market_data(
            n_days=200, patterns=[PatternType.PUMP_AND_DUMP], pattern_configs=custom
        )
        assert df["is_pump_and_dump"].sum() == 5


class TestChronologicalTrainTestSplit:
    def test_split_preserves_order_no_leakage(self):
        df = generate_synthetic_market_data(n_days=500)
        train, test = chronological_train_test_split(df, test_size=0.2)
        assert train.index.max() < test.index.min()

    def test_split_sizes_approximately_correct(self):
        df = generate_synthetic_market_data(n_days=500)
        train, test = chronological_train_test_split(df, test_size=0.2)
        assert len(train) + len(test) == len(df)
        assert abs(len(test) / len(df) - 0.2) < 0.02

    def test_no_row_duplication_or_loss(self):
        df = generate_synthetic_market_data(n_days=500)
        train, test = chronological_train_test_split(df, test_size=0.25)
        recombined = pd.concat([train, test])
        pd.testing.assert_frame_equal(recombined, df)
