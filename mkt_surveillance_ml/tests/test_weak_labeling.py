import numpy as np
import pandas as pd
import pytest

from mkt_surveillance_ml.config import PatternType, BASE_FEATURE_COLUMNS
from mkt_surveillance_ml.data.synthetic import generate_synthetic_market_data, chronological_train_test_split
from mkt_surveillance_ml.detection.weak_labeling import (
    build_pattern_prototypes_from_examples,
    build_pattern_prototypes_from_domain_rules,
    attribute_pattern_to_anomalies,
    weak_label_from_isolation_forest,
    train_multi_pattern_detector_with_weak_labels,
    evaluate_weak_labeling_quality,
)


@pytest.fixture(scope="module")
def synthetic_split():
    df = generate_synthetic_market_data(n_days=3000, random_state=7)
    return chronological_train_test_split(df, test_size=0.3)


class TestBuildPatternPrototypesFromExamples:
    def test_returns_one_prototype_per_pattern(self, synthetic_split):
        train_df, _ = synthetic_split
        prototypes = build_pattern_prototypes_from_examples(train_df[BASE_FEATURE_COLUMNS], train_df)
        assert set(prototypes.keys()) == set(PatternType)

    def test_prototype_has_correct_dimensionality(self, synthetic_split):
        train_df, _ = synthetic_split
        prototypes = build_pattern_prototypes_from_examples(train_df[BASE_FEATURE_COLUMNS], train_df)
        for proto in prototypes.values():
            assert len(proto) == len(BASE_FEATURE_COLUMNS)

    def test_pump_and_dump_prototype_has_high_return_and_volume(self, synthetic_split):
        """The true pump_and_dump examples should produce a prototype
        with a clearly positive return z-score and positive volume
        z-score, matching the pattern's actual injected signature."""
        train_df, _ = synthetic_split
        prototypes = build_pattern_prototypes_from_examples(train_df[BASE_FEATURE_COLUMNS], train_df)
        pump_proto = prototypes[PatternType.PUMP_AND_DUMP]
        return_z, volume_z, _ = pump_proto
        assert return_z > 1.0
        assert volume_z > 1.0

    def test_raises_on_missing_label_column(self, synthetic_split):
        train_df, _ = synthetic_split
        bad_labels = train_df.drop(columns=["is_spoofing"])
        with pytest.raises(ValueError, match="is_spoofing"):
            build_pattern_prototypes_from_examples(train_df[BASE_FEATURE_COLUMNS], bad_labels)

    def test_raises_on_zero_examples_for_a_pattern(self, synthetic_split):
        train_df, _ = synthetic_split
        zeroed = train_df.copy()
        zeroed["is_layering"] = 0
        with pytest.raises(ValueError, match="No confirmed examples"):
            build_pattern_prototypes_from_examples(train_df[BASE_FEATURE_COLUMNS], zeroed)


class TestBuildPatternPrototypesFromDomainRules:
    def test_returns_one_prototype_per_pattern(self):
        prototypes = build_pattern_prototypes_from_domain_rules()
        assert set(prototypes.keys()) == set(PatternType)

    def test_pump_and_dump_prototype_has_positive_return_and_volume(self):
        proto = build_pattern_prototypes_from_domain_rules()[PatternType.PUMP_AND_DUMP]
        assert proto[0] > 1.0  # return
        assert proto[1] > 1.0  # volume_ratio

    def test_wash_trading_prototype_has_near_zero_return(self):
        """Wash trading's defining feature: high volume without a
        corresponding price move."""
        proto = build_pattern_prototypes_from_domain_rules()[PatternType.WASH_TRADING]
        assert abs(proto[0]) < 0.5  # return near zero
        assert proto[1] > 1.0  # but volume still elevated

    def test_prototypes_are_pairwise_distinct(self):
        """No two prototypes should be identical -- otherwise nearest-
        centroid attribution could never distinguish them."""
        prototypes = build_pattern_prototypes_from_domain_rules()
        values = list(prototypes.values())
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                assert not np.allclose(values[i], values[j])


class TestAttributePatternToAnomalies:
    def test_unflagged_rows_get_none_attribution(self):
        rng = np.random.RandomState(1)
        X = pd.DataFrame(rng.normal(0, 1, (50, 3)), columns=BASE_FEATURE_COLUMNS)
        flagged_mask = np.zeros(50, dtype=bool)
        prototypes = build_pattern_prototypes_from_domain_rules()
        result = attribute_pattern_to_anomalies(X, flagged_mask, prototypes)
        assert result["attributed_pattern"].isna().all()

    def test_point_matching_a_prototype_exactly_is_attributed_correctly(self):
        """A synthetically constructed point placed EXACTLY at the
        pump_and_dump prototype's location (after standardization)
        should be attributed to pump_and_dump with high confidence."""
        rng = np.random.RandomState(1)
        n = 100
        # background of normal-ish points, mean 0 std 1 per column by construction
        X_vals = rng.normal(0, 1, (n, 3))
        prototypes = build_pattern_prototypes_from_domain_rules()
        pump_proto = prototypes[PatternType.PUMP_AND_DUMP]
        # place one point exactly at the pump prototype's standardized location
        X_vals[0] = pump_proto
        X = pd.DataFrame(X_vals, columns=BASE_FEATURE_COLUMNS)
        flagged_mask = np.zeros(n, dtype=bool)
        flagged_mask[0] = True

        result = attribute_pattern_to_anomalies(X, flagged_mask, prototypes)
        assert result.iloc[0]["attributed_pattern"] == PatternType.PUMP_AND_DUMP.value
        assert result.iloc[0]["confidence"] > 0.3  # noticeably better than random 1/4

    def test_confidence_between_zero_and_one(self, synthetic_split):
        train_df, _ = synthetic_split
        X = train_df[BASE_FEATURE_COLUMNS]
        flagged_mask = np.ones(len(X), dtype=bool)
        prototypes = build_pattern_prototypes_from_domain_rules()
        result = attribute_pattern_to_anomalies(X, flagged_mask, prototypes)
        confidences = result["confidence"].dropna()
        assert (confidences >= 0).all() and (confidences <= 1).all()


class TestWeakLabelFromIsolationForest:
    def test_returns_expected_schema(self, synthetic_split):
        train_df, _ = synthetic_split
        result = weak_label_from_isolation_forest(train_df[BASE_FEATURE_COLUMNS], random_state=1)
        expected_cols = {f"is_{p.value}" for p in PatternType} | {"is_manipulation", "attribution_confidence"}
        assert expected_cols.issubset(set(result.columns))

    def test_is_manipulation_matches_sum_of_pattern_flags_upper_bound(self, synthetic_split):
        """Every flagged day gets attributed to exactly one pattern (or
        none, if unflagged) -- so is_manipulation should equal the
        logical OR of the per-pattern flags."""
        train_df, _ = synthetic_split
        result = weak_label_from_isolation_forest(train_df[BASE_FEATURE_COLUMNS], random_state=1)
        pattern_cols = [f"is_{p.value}" for p in PatternType]
        or_of_patterns = (result[pattern_cols].sum(axis=1) > 0).astype(int)
        assert (result["is_manipulation"] == or_of_patterns).all()

    def test_higher_contamination_flags_more_days(self, synthetic_split):
        train_df, _ = synthetic_split
        low = weak_label_from_isolation_forest(train_df[BASE_FEATURE_COLUMNS], contamination=0.02, random_state=1)
        high = weak_label_from_isolation_forest(train_df[BASE_FEATURE_COLUMNS], contamination=0.15, random_state=1)
        assert high["is_manipulation"].sum() > low["is_manipulation"].sum()

    def test_custom_prototypes_are_respected(self, synthetic_split):
        train_df, _ = synthetic_split
        single_pattern_prototypes = {
            PatternType.PUMP_AND_DUMP: build_pattern_prototypes_from_domain_rules()[PatternType.PUMP_AND_DUMP]
        }
        result = weak_label_from_isolation_forest(
            train_df[BASE_FEATURE_COLUMNS], prototypes=single_pattern_prototypes, random_state=1
        )
        assert "is_pump_and_dump" in result.columns
        assert "is_wash_trading" not in result.columns


class TestTrainMultiPatternDetectorWithWeakLabels:
    def test_returns_fitted_detector_and_weak_labels(self, synthetic_split):
        train_df, _ = synthetic_split
        detector, weak_labels = train_multi_pattern_detector_with_weak_labels(
            train_df[BASE_FEATURE_COLUMNS], random_state=1
        )
        assert len(detector.models_) > 0
        assert len(weak_labels) == len(train_df)

    def test_fitted_detector_can_predict_on_new_data(self, synthetic_split):
        train_df, test_df = synthetic_split
        detector, _ = train_multi_pattern_detector_with_weak_labels(train_df[BASE_FEATURE_COLUMNS], random_state=1)
        proba = detector.predict_proba(test_df[BASE_FEATURE_COLUMNS])
        assert len(proba) == len(test_df)
        assert (proba.values >= 0).all() and (proba.values <= 1).all()

    def test_raises_on_empty_prototypes_dict(self, synthetic_split):
        """Confirms the actual failure mode for an empty prototypes
        dict: attribute_pattern_to_anomalies raises a clear ValueError
        EARLY (there's nothing to attribute to), before
        train_multi_pattern_detector_with_weak_labels's own "zero
        patterns have positives" RuntimeError check is ever reached.
        See that function's docstring for why an infinitesimally small
        contamination does NOT reach either error path: at least one
        row is always flagged and forced to the single closest
        prototype when prototypes is non-empty, confirmed directly
        while writing this test (an earlier version of this test
        expected a RuntimeError via tiny contamination and failed).
        """
        train_df, _ = synthetic_split
        with pytest.raises(ValueError, match="prototypes is empty"):
            train_multi_pattern_detector_with_weak_labels(
                train_df[BASE_FEATURE_COLUMNS], prototypes={}, random_state=1
            )


class TestEvaluateWeakLabelingQuality:
    def test_returns_expected_keys(self, synthetic_split):
        train_df, test_df = synthetic_split
        result = evaluate_weak_labeling_quality(
            train_df[BASE_FEATURE_COLUMNS], train_df, test_df[BASE_FEATURE_COLUMNS], test_df, random_state=7
        )
        for key in [
            "isolation_forest_recall_per_pattern", "attribution_accuracy_given_flagged",
            "attribution_confusion_matrix", "downstream_auc_comparison",
        ]:
            assert key in result

    def test_isolation_forest_recall_is_bounded_zero_one(self, synthetic_split):
        train_df, test_df = synthetic_split
        result = evaluate_weak_labeling_quality(
            train_df[BASE_FEATURE_COLUMNS], train_df, test_df[BASE_FEATURE_COLUMNS], test_df, random_state=7
        )
        recalls = result["isolation_forest_recall_per_pattern"]["isolation_forest_recall"]
        assert (recalls >= 0).all() and (recalls <= 1).all()

    def test_pump_and_dump_has_highest_isolation_forest_recall(self, synthetic_split):
        """Pump-and-dump has the strongest, most separable signal of the
        four patterns (established throughout this project, e.g.
        test_synthetic_data.py's signature checks) -- IsolationForest's
        UNSUPERVISED recall should reflect that, catching pump most
        reliably among the four."""
        train_df, test_df = synthetic_split
        result = evaluate_weak_labeling_quality(
            train_df[BASE_FEATURE_COLUMNS], train_df, test_df[BASE_FEATURE_COLUMNS], test_df, random_state=7
        )
        recall_df = result["isolation_forest_recall_per_pattern"].set_index("pattern")
        pump_recall = recall_df.loc["pump_and_dump", "isolation_forest_recall"]
        assert pump_recall == recall_df["isolation_forest_recall"].max()

    def test_attribution_accuracy_better_than_random_four_way_guess(self, synthetic_split):
        """With 4 patterns, random guessing gets ~25% accuracy --
        domain-rule-based attribution should clearly beat that."""
        train_df, test_df = synthetic_split
        result = evaluate_weak_labeling_quality(
            train_df[BASE_FEATURE_COLUMNS], train_df, test_df[BASE_FEATURE_COLUMNS], test_df, random_state=7
        )
        assert result["attribution_accuracy_given_flagged"] > 0.4

    def test_downstream_auc_comparison_has_expected_columns(self, synthetic_split):
        train_df, test_df = synthetic_split
        result = evaluate_weak_labeling_quality(
            train_df[BASE_FEATURE_COLUMNS], train_df, test_df[BASE_FEATURE_COLUMNS], test_df, random_state=7
        )
        comparison = result["downstream_auc_comparison"]
        for col in ["pattern", "auc_true_labels", "auc_weak_labels", "auc_lost"]:
            assert col in comparison.columns

    def test_pump_and_dump_loses_the_least_auc(self, synthetic_split):
        """Pump-and-dump has 100% IsolationForest recall and the most
        separable prototype -- it should lose noticeably less AUC from
        weak labeling than layering, which has both mediocre IF recall
        and (established in detection/multi_pattern.py) known small-
        sample instability once labels get sparse."""
        train_df, test_df = synthetic_split
        result = evaluate_weak_labeling_quality(
            train_df[BASE_FEATURE_COLUMNS], train_df, test_df[BASE_FEATURE_COLUMNS], test_df, random_state=7
        )
        comparison = result["downstream_auc_comparison"].set_index("pattern")
        pump_loss = comparison.loc["pump_and_dump", "auc_lost"]
        layering_loss = comparison.loc["layering", "auc_lost"]
        assert pump_loss < layering_loss
