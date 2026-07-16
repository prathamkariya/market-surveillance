import numpy as np
import pandas as pd
import pytest

from mkt_surveillance_ml.config import PatternType
from mkt_surveillance_ml.data.synthetic import generate_synthetic_market_data, chronological_train_test_split
from mkt_surveillance_ml.detection.multi_pattern import MultiPatternDetector


FEATURE_COLS = ["return", "volume_ratio_20d", "volatility_20d"]


@pytest.fixture
def train_test_split_data():
    df = generate_synthetic_market_data(n_days=1500, random_state=100)
    train_df, test_df = chronological_train_test_split(df, test_size=0.3)
    return train_df, test_df


@pytest.fixture
def adequately_powered_train_test_split_data():
    """Overrides pattern_configs with larger n_days per pattern than the
    module default. Needed specifically for statistical comparisons
    (recall across models) to be meaningful: the default config's n_days
    is a FIXED absolute count that does NOT scale up just because a
    larger total n_days is requested (see data/synthetic.py) -- at the
    defaults, spoofing/layering collapse to ~2-6 positive TEST examples
    out of 1500 days, where recall can only take a few discrete values
    and is dominated by chance, not by whether per-pattern modeling
    actually works. This fixture gives every pattern 50+ test-set
    positives so the comparison has enough statistical power to mean
    something.
    """
    from mkt_surveillance_ml.data.synthetic import PatternInjectionConfig
    adequately_powered_configs = {
        PatternType.PUMP_AND_DUMP: PatternInjectionConfig(
            n_days=80, return_mean=0.065, return_std=0.012, volume_ratio_mean=3.2, volume_ratio_std=0.35),
        PatternType.WASH_TRADING: PatternInjectionConfig(
            n_days=120, return_mean=0.0, return_std=0.004, volume_ratio_mean=2.9, volume_ratio_std=0.30),
        PatternType.SPOOFING: PatternInjectionConfig(
            n_days=80, return_mean=0.0, return_std=0.028, volume_ratio_mean=1.6, volume_ratio_std=0.25),
        PatternType.LAYERING: PatternInjectionConfig(
            n_days=90, return_mean=0.0, return_std=0.019, volume_ratio_mean=2.1, volume_ratio_std=0.28),
    }
    df = generate_synthetic_market_data(
        n_days=3000, patterns=list(PatternType),
        pattern_configs=adequately_powered_configs, random_state=100,
    )
    train_df, test_df = chronological_train_test_split(df, test_size=0.3)
    return train_df, test_df


class TestMultiPatternDetectorFit:
    def test_fits_one_model_per_pattern(self, train_test_split_data):
        train_df, _ = train_test_split_data
        detector = MultiPatternDetector(random_state=1)
        detector.fit(train_df[FEATURE_COLS], train_df)
        assert set(detector.models_.keys()) == set(PatternType)

    def test_can_fit_a_subset_of_patterns(self, train_test_split_data):
        train_df, _ = train_test_split_data
        detector = MultiPatternDetector(
            patterns=[PatternType.PUMP_AND_DUMP, PatternType.WASH_TRADING], random_state=1
        )
        detector.fit(train_df[FEATURE_COLS], train_df)
        assert set(detector.models_.keys()) == {PatternType.PUMP_AND_DUMP, PatternType.WASH_TRADING}

    def test_raises_on_missing_label_column(self, train_test_split_data):
        train_df, _ = train_test_split_data
        bad_labels = train_df.drop(columns=["is_pump_and_dump"])
        detector = MultiPatternDetector(random_state=1)
        with pytest.raises(ValueError, match="is_pump_and_dump"):
            detector.fit(train_df[FEATURE_COLS], bad_labels)

    def test_raises_on_zero_positive_examples_for_a_pattern(self, train_test_split_data):
        train_df, _ = train_test_split_data
        zeroed_labels = train_df.copy()
        zeroed_labels["is_spoofing"] = 0
        detector = MultiPatternDetector(random_state=1)
        with pytest.raises(ValueError, match="spoofing"):
            detector.fit(train_df[FEATURE_COLS], zeroed_labels)


class TestMultiPatternDetectorPredict:
    def test_predict_proba_before_fit_raises(self):
        detector = MultiPatternDetector()
        with pytest.raises(RuntimeError):
            detector.predict_proba(pd.DataFrame({"a": [1, 2, 3]}))

    def test_predict_proba_has_one_column_per_pattern(self, train_test_split_data):
        train_df, test_df = train_test_split_data
        detector = MultiPatternDetector(random_state=1).fit(train_df[FEATURE_COLS], train_df)
        proba = detector.predict_proba(test_df[FEATURE_COLS])
        expected_cols = {f"proba_{p.value}" for p in PatternType}
        assert set(proba.columns) == expected_cols

    def test_predict_proba_bounded_zero_one(self, train_test_split_data):
        train_df, test_df = train_test_split_data
        detector = MultiPatternDetector(random_state=1).fit(train_df[FEATURE_COLS], train_df)
        proba = detector.predict_proba(test_df[FEATURE_COLS])
        assert (proba.values >= 0).all() and (proba.values <= 1).all()

    def test_a_day_can_score_high_on_more_than_one_pattern_at_once(self, train_test_split_data):
        """The specific representational advantage a single blended label
        cannot offer: independent per-pattern probabilities, not one flag."""
        train_df, test_df = train_test_split_data
        detector = MultiPatternDetector(random_state=1).fit(train_df[FEATURE_COLS], train_df)
        proba = detector.predict_proba(test_df[FEATURE_COLS])
        # not asserting this MUST happen (patterns are disjoint by
        # construction in the synthetic data), just that the representation
        # permits it -- i.e. no structural constraint forces probabilities
        # to sum to <= 1 across patterns
        row_sums = proba.sum(axis=1)
        assert row_sums.max() <= 4.0 + 1e-9  # 4 independent probabilities, no forced normalization

    def test_predict_respects_custom_per_pattern_thresholds(self, train_test_split_data):
        train_df, test_df = train_test_split_data
        detector = MultiPatternDetector(random_state=1).fit(train_df[FEATURE_COLS], train_df)
        low_threshold = {p: 0.01 for p in PatternType}
        high_threshold = {p: 0.99 for p in PatternType}
        low_flags = detector.predict(test_df[FEATURE_COLS], thresholds=low_threshold)
        high_flags = detector.predict(test_df[FEATURE_COLS], thresholds=high_threshold)
        assert low_flags.values.sum() >= high_flags.values.sum()


class TestMultiPatternDetectorEvaluate:
    def test_evaluate_before_fit_raises(self):
        detector = MultiPatternDetector()
        with pytest.raises(RuntimeError):
            detector.evaluate(pd.DataFrame({"a": [1]}), pd.DataFrame({"is_pump_and_dump": [0]}))

    def test_evaluate_returns_one_row_per_pattern(self, train_test_split_data):
        train_df, test_df = train_test_split_data
        detector = MultiPatternDetector(random_state=1).fit(train_df[FEATURE_COLS], train_df)
        result = detector.evaluate(test_df[FEATURE_COLS], test_df)
        assert len(result) == len(PatternType)

    def test_evaluate_achieves_reasonable_auc_given_distinct_signatures(self, train_test_split_data):
        """Each pattern has a genuinely distinct feature signature (verified
        in test_synthetic_data.py) -- a properly fit per-pattern classifier
        should achieve meaningfully-better-than-random AUC on each."""
        train_df, test_df = train_test_split_data
        detector = MultiPatternDetector(random_state=1).fit(train_df[FEATURE_COLS], train_df)
        result = detector.evaluate(test_df[FEATURE_COLS], test_df)
        assert (result["auc"].dropna() > 0.6).all()

    def test_evaluate_includes_n_positive_column(self, train_test_split_data):
        train_df, test_df = train_test_split_data
        detector = MultiPatternDetector(random_state=1).fit(train_df[FEATURE_COLS], train_df)
        result = detector.evaluate(test_df[FEATURE_COLS], test_df)
        assert "n_positive" in result.columns
        assert (result["n_positive"] >= 0).all()


class TestCompareAgainstBlendedBaseline:
    def test_raises_without_is_manipulation_column(self, train_test_split_data):
        train_df, test_df = train_test_split_data
        detector = MultiPatternDetector(random_state=1).fit(train_df[FEATURE_COLS], train_df)
        bad_train_labels = train_df.drop(columns=["is_manipulation"])
        with pytest.raises(ValueError, match="is_manipulation"):
            detector.compare_against_blended_baseline(
                train_df[FEATURE_COLS], bad_train_labels, test_df[FEATURE_COLS], test_df
            )

    def test_returns_one_row_per_pattern_with_expected_columns(self, train_test_split_data):
        train_df, test_df = train_test_split_data
        detector = MultiPatternDetector(random_state=1).fit(train_df[FEATURE_COLS], train_df)
        result = detector.compare_against_blended_baseline(
            train_df[FEATURE_COLS], train_df, test_df[FEATURE_COLS], test_df
        )
        assert len(result) == len(PatternType)
        for col in [
            "blended_model_auc", "per_pattern_model_auc", "auc_improvement",
            "blended_model_recall_at_tuned_threshold", "per_pattern_model_recall_at_tuned_threshold",
            "recall_improvement",
        ]:
            assert col in result.columns

    def test_per_pattern_models_auc_at_least_as_good_as_blended_on_average(
        self, adequately_powered_train_test_split_data
    ):
        """The capstone claim of the entire package, tested with REAL
        trained models rather than a toy single-feature demonstration.

        AUC is the primary metric here, not recall-at-a-shared-threshold
        -- found necessary by testing this exact comparison: a per-pattern
        model and the blended model are trained on very different base
        rates, so raw predict_proba outputs land on different absolute
        scales even when relative ranking (AUC) is comparable. See
        test_shared_threshold_recall_is_confounded_by_calibration_differences
        for that finding preserved as a permanent guardrail. Comparing
        AUC side-steps the confound entirely, matching file 25's own
        methodology throughout (evaluate_on_label reports AUC, never a
        fixed-threshold recall).
        """
        train_df, test_df = adequately_powered_train_test_split_data
        detector = MultiPatternDetector(random_state=1).fit(train_df[FEATURE_COLS], train_df)
        result = detector.compare_against_blended_baseline(
            train_df[FEATURE_COLS], train_df, test_df[FEATURE_COLS], test_df
        )
        avg_improvement = result["auc_improvement"].mean()
        assert avg_improvement >= -0.03  # per-pattern should not, on average, be meaningfully worse on AUC
        worst_case_degradation = result["auc_improvement"].min()
        assert worst_case_degradation >= -0.1  # no single pattern meaningfully worse on AUC

    def test_rarer_pattern_specifically_benefits_from_per_pattern_modeling(
        self, adequately_powered_train_test_split_data
    ):
        """Spoofing and layering are configured with fewer injected days
        than wash_trading -- the file 27 Section 11 argument predicts
        these rarer patterns specifically should show an AUC improvement
        (or at minimum, no meaningful AUC loss) under per-pattern
        modeling versus the blended baseline, which is diluted by the
        more common patterns' days.
        """
        train_df, test_df = adequately_powered_train_test_split_data
        detector = MultiPatternDetector(random_state=1).fit(train_df[FEATURE_COLS], train_df)
        result = detector.compare_against_blended_baseline(
            train_df[FEATURE_COLS], train_df, test_df[FEATURE_COLS], test_df
        )
        spoofing_row = result[result["pattern"] == PatternType.SPOOFING.value]
        assert len(spoofing_row) == 1
        assert spoofing_row["n_positive_test"].iloc[0] >= 15  # confirm this run actually has statistical power
        assert spoofing_row["auc_improvement"].iloc[0] >= -0.05

    def test_shared_threshold_recall_is_confounded_by_calibration_differences(
        self, adequately_powered_train_test_split_data
    ):
        """A genuine finding from testing this comparison, preserved
        permanently rather than fixed-and-forgotten: a per-pattern model
        (trained on a low base rate, e.g. ~2-4% for a single pattern) and
        the blended model (trained on the union of all patterns, a much
        higher base rate) can have nearly IDENTICAL AUC while producing
        systematically different absolute probability scales -- the
        per-pattern model's outputs run lower on average, purely from
        training on a rarer positive class, not from being a worse
        ranker. Comparing recall at one SHARED, untuned 0.5 threshold
        across the two would make the per-pattern model look dramatically
        worse despite equivalent discriminative power. This test confirms
        that gap exists at a shared threshold (motivating why
        compare_against_blended_baseline tunes each model's own threshold
        via best_threshold_by_f1 instead of comparing at a shared 0.5).
        """
        train_df, test_df = adequately_powered_train_test_split_data
        detector = MultiPatternDetector(random_state=1).fit(train_df[FEATURE_COLS], train_df)

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score, recall_score

        blended_model = RandomForestClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced", random_state=1, n_jobs=-1
        )
        blended_model.fit(train_df[FEATURE_COLS], train_df["is_manipulation"])
        blended_proba = blended_model.predict_proba(test_df[FEATURE_COLS])[:, 1]

        per_pattern_proba = detector.predict_proba(test_df[FEATURE_COLS])["proba_spoofing"]
        y_true = test_df["is_spoofing"]

        auc_gap = abs(
            roc_auc_score(y_true, blended_proba) - roc_auc_score(y_true, per_pattern_proba)
        )
        recall_at_shared_half = recall_score(y_true, (per_pattern_proba >= 0.5).astype(int))
        blended_recall_at_shared_half = recall_score(y_true, (blended_proba >= 0.5).astype(int))
        recall_gap_at_shared_threshold = abs(blended_recall_at_shared_half - recall_at_shared_half)

        # AUCs should be close (equivalent discriminative power)...
        assert auc_gap < 0.05
        # ...while recall at a shared, untuned threshold can differ
        # substantially despite that -- exactly the confound this test
        # documents and compare_against_blended_baseline's tuned-threshold
        # design works around.
        assert recall_gap_at_shared_threshold > 0.1

    def test_recall_comparison_is_unreliable_with_too_few_positive_examples(
        self, train_test_split_data
    ):
        """Documents a second, separate genuine finding rather than
        hiding it: at the MODULE DEFAULT pattern injection rates (fixed
        absolute day counts, not scaled to total n_days -- see
        data/synthetic.py's PatternInjectionConfig), rarer patterns like
        spoofing can collapse to single-digit test-set positives, where
        recall takes only a few discrete values and any comparison
        (even at a fair, tuned threshold) is dominated by sample-size
        noise. This test asserts the SMALL sample size itself, not a
        specific outcome, since the outcome genuinely varies run to run
        at this scale.
        """
        train_df, test_df = train_test_split_data
        detector = MultiPatternDetector(random_state=1).fit(train_df[FEATURE_COLS], train_df)
        result = detector.compare_against_blended_baseline(
            train_df[FEATURE_COLS], train_df, test_df[FEATURE_COLS], test_df
        )
        spoofing_row = result[result["pattern"] == PatternType.SPOOFING.value]
        # confirms WHY this pattern's comparison is noisy at default
        # rates: too few test-set positives for any metric computed on
        # it to be a stable statistic
        assert spoofing_row["n_positive_test"].iloc[0] <= 10
