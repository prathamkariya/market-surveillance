"""
MultiPatternDetector: the production-facing capstone of this package.

Every file in Phase 5-6 makes some version of the same argument:
  - file 20 (Section 6): a feature's mutual information against its true
    pattern label is measurably higher than against a blended label.
  - file 25 (closing argument): a single "respectable" combined-label AUC
    can hide a wide, undiagnosed gap in how well each pattern is actually
    detected.
  - file 27 (Section 11): one global contamination threshold, even
    calibrated to the exact true combined rate, under-detects rarer
    patterns relative to more common ones.

This class is what those arguments become in production: instead of one
model trained on `is_manipulation` (OR of all patterns), it trains ONE
INDEPENDENT classifier per pattern in config.PatternType. compare_against_blended_baseline
puts a real trained blended-label model side by side with the real
per-pattern models on the SAME data, so the gap is demonstrated with
actual trained models, not just a single feature's mutual information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from mkt_surveillance_ml.config import PatternType, RANDOM_STATE


class MultiPatternDetector:
    """Trains one independent classifier per manipulation pattern rather
    than a single classifier on a blended `is_manipulation` label.

    Each pattern gets its own model, its own decision threshold, and its
    own evaluation -- so a rare pattern being poorly detected is visible
    directly, not averaged away into one respectable-looking combined
    number. class_weight='balanced' by default: every pattern here is a
    small minority of days (see data/synthetic.py's injection rates,
    typically 2-5% of days per pattern), and an unweighted classifier
    would happily predict "normal" for everything and still score high
    on raw accuracy while missing every actual event.

    Honest limitation, found by testing rather than assumed away: the
    per-pattern approach's advantage is CONDITIONAL on having enough
    positive training examples for the pattern in question, AND any
    recall-based comparison against a blended baseline must account for
    the two models being calibrated on different base rates. A
    per-pattern model trained on a rare pattern (say 3% of days) and a
    blended model trained on the union of all patterns (say 12% of
    days) can have nearly IDENTICAL discriminative power (AUC) while
    producing systematically different absolute probability scales --
    the rare-pattern model's outputs simply run lower. Comparing recall
    at one shared, untuned 0.5 threshold across the two would make the
    per-pattern model look dramatically worse despite equivalent
    ranking ability -- see compare_against_blended_baseline's use of
    per-model tuned thresholds instead of a shared 0.5, and
    test_multi_pattern_detector.py's
    test_shared_threshold_recall_is_confounded_by_calibration_differences,
    which confirms this gap exists and is exactly why the fix matters.
    Separately, if a pattern is rare enough that its training set
    collapses to single digits of positive examples, no comparison
    (fair threshold or not) is statistically meaningful at that sample
    size -- see test_recall_comparison_is_unreliable_with_too_few_positive_examples.
    """

    def __init__(
        self,
        patterns: list[PatternType] | None = None,
        estimator_factory=None,
        random_state: int = RANDOM_STATE,
    ):
        self.patterns = patterns or list(PatternType)
        self.random_state = random_state
        self.estimator_factory = estimator_factory or self._default_estimator_factory
        self.models_: dict[PatternType, object] = {}
        self.feature_columns_: list[str] | None = None

    def _default_estimator_factory(self):
        return RandomForestClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced",
            random_state=self.random_state, n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame, label_df: pd.DataFrame) -> "MultiPatternDetector":
        """label_df must contain one column per pattern, named
        `is_{pattern.value}` (matching data/synthetic.py's convention) --
        e.g. is_pump_and_dump, is_wash_trading, is_spoofing, is_layering.
        """
        self.feature_columns_ = list(X.columns)
        self.models_ = {}
        for pattern in self.patterns:
            label_col = f"is_{pattern.value}"
            if label_col not in label_df.columns:
                raise ValueError(
                    f"label_df is missing column '{label_col}' required "
                    f"for pattern {pattern.value}."
                )
            y = label_df[label_col]
            if y.sum() == 0:
                raise ValueError(
                    f"No positive examples for pattern '{pattern.value}' in "
                    f"the training data -- cannot fit a classifier for it. "
                    f"Check the label column or exclude this pattern."
                )
            model = self.estimator_factory()
            model.fit(X, y)
            self.models_[pattern] = model
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """One probability column per pattern -- NOT a single blended
        probability. A day can score high on multiple patterns at once,
        which a single combined label could never represent."""
        if not self.models_:
            raise RuntimeError("Call fit() before predict_proba().")
        result = pd.DataFrame(index=X.index)
        for pattern, model in self.models_.items():
            result[f"proba_{pattern.value}"] = model.predict_proba(X)[:, 1]
        return result

    def predict(self, X: pd.DataFrame, thresholds: dict[PatternType, float] | None = None) -> pd.DataFrame:
        """One binary flag column per pattern. thresholds defaults to 0.5
        per pattern, but callers can pass pattern-specific thresholds
        (e.g. from threshold_sweep in models/logistic_regression.py) --
        different patterns can legitimately warrant different
        precision/recall tradeoffs.
        """
        proba = self.predict_proba(X)
        if thresholds is None:
            thresholds = {p: 0.5 for p in self.models_}
        result = pd.DataFrame(index=X.index)
        for pattern in self.models_:
            t = thresholds.get(pattern, 0.5)
            result[f"flag_{pattern.value}"] = (proba[f"proba_{pattern.value}"] >= t).astype(int)
        return result

    def evaluate(self, X: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
        """AUC, precision, recall, and F1 PER PATTERN at threshold 0.5 --
        this is the table that makes an under-served pattern visible
        directly, rather than averaging it into one number."""
        if not self.models_:
            raise RuntimeError("Call fit() before evaluate().")
        proba = self.predict_proba(X)
        rows = []
        for pattern in self.models_:
            label_col = f"is_{pattern.value}"
            y_true = label_df[label_col]
            y_proba = proba[f"proba_{pattern.value}"]
            y_pred = (y_proba >= 0.5).astype(int)
            rows.append({
                "pattern": pattern.value,
                "n_positive": int(y_true.sum()),
                "auc": roc_auc_score(y_true, y_proba) if y_true.sum() > 0 else float("nan"),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
            })
        return pd.DataFrame(rows)

    def compare_against_blended_baseline(
        self, X_train: pd.DataFrame, label_df_train: pd.DataFrame,
        X_test: pd.DataFrame, label_df_test: pd.DataFrame,
    ) -> pd.DataFrame:
        """The concrete, trained-model version of the argument every file
        in Phase 5-6 makes: train ONE classifier on the blended
        `is_manipulation` label (the naive approach), and compare its
        per-pattern performance against this detector's own per-pattern
        models, trained and evaluated on the exact same data.

        AUC is the PRIMARY comparison metric, matching file 25's own
        methodology throughout (evaluate_on_label returns AUC, never a
        fixed-threshold recall) -- and for a real reason, found by
        testing this exact comparison rather than assumed: a per-pattern
        model and a blended model are trained on very different base
        rates (a single rare pattern vs. the union of all patterns), so
        they produce systematically different ABSOLUTE probability
        scales even when their RELATIVE ranking of positives vs
        negatives (AUC) is nearly identical. Comparing recall at one
        shared, untuned 0.5 threshold across models with different
        probability calibration isn't a fair comparison -- it can make
        a per-pattern model with equivalent AUC look dramatically worse
        purely because its probabilities run lower on an absolute scale.
        recall_at_tuned_threshold uses each model's own F1-maximizing
        threshold (via models.logistic_regression.best_threshold_by_f1)
        instead of a shared 0.5, so the recall comparison is fair too.
        """
        if "is_manipulation" not in label_df_train.columns:
            raise ValueError("label_df_train must include 'is_manipulation' for the baseline comparison.")

        from mkt_surveillance_ml.models.logistic_regression import best_threshold_by_f1

        blended_model = self._default_estimator_factory()
        blended_model.fit(X_train, label_df_train["is_manipulation"])
        blended_proba = blended_model.predict_proba(X_test)[:, 1]

        per_pattern_proba = self.predict_proba(X_test)

        rows = []
        for pattern in self.models_:
            label_col = f"is_{pattern.value}"
            y_true_pattern = label_df_test[label_col]
            per_pattern_proba_for_pattern = per_pattern_proba[f"proba_{pattern.value}"]

            if y_true_pattern.sum() == 0:
                blended_auc = per_pattern_auc = float("nan")
            else:
                blended_auc = roc_auc_score(y_true_pattern, blended_proba)
                per_pattern_auc = roc_auc_score(y_true_pattern, per_pattern_proba_for_pattern)

            # fair recall: each model's OWN F1-optimal threshold, not a
            # shared naive 0.5 that confounds calibration differences
            # driven by different training base rates with actual
            # modeling quality (see docstring above -- found by testing,
            # not assumed).
            blended_threshold = best_threshold_by_f1(y_true_pattern.values, blended_proba)
            per_pattern_threshold = best_threshold_by_f1(y_true_pattern.values, per_pattern_proba_for_pattern.values)

            blended_recall = recall_score(
                y_true_pattern, (blended_proba >= blended_threshold).astype(int), zero_division=0
            )
            per_pattern_recall = recall_score(
                y_true_pattern, (per_pattern_proba_for_pattern >= per_pattern_threshold).astype(int),
                zero_division=0,
            )

            rows.append({
                "pattern": pattern.value,
                "n_positive_test": int(y_true_pattern.sum()),
                "blended_model_auc": blended_auc,
                "per_pattern_model_auc": per_pattern_auc,
                "auc_improvement": per_pattern_auc - blended_auc,
                "blended_model_recall_at_tuned_threshold": blended_recall,
                "per_pattern_model_recall_at_tuned_threshold": per_pattern_recall,
                "recall_improvement": per_pattern_recall - blended_recall,
            })
        return pd.DataFrame(rows)
