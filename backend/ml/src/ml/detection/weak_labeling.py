"""
Weak Labeling: bridging unsupervised anomaly detection and per-pattern
supervised classification, without requiring true labels.

THE TENSION THIS RESOLVES: MultiPatternDetector (detection/multi_pattern.py)
is the more sophisticated architecture -- proven, in compare_against_blended_baseline,
to genuinely out-detect a single blended-label model. But it needs
per-pattern GROUND TRUTH LABELS. Most real deployments don't have those --
that's the whole reason surveillance exists in the first place. Meanwhile
IsolationForestScratch (anomaly/isolation_forest.py) needs no labels at
all, but only ever says "this looks weird," never WHICH KIND of weird.

This module is the bridge: once IsolationForest flags a day as
anomalous, ATTRIBUTE a likely pattern to it via nearest-centroid matching
in feature space against reference prototypes -- built either from a
handful of confirmed true examples (few-shot: "I know of 3 confirmed
pump-and-dump days"), or from pure domain knowledge with ZERO true
examples at all (the harder, more interesting case, and the one this
module treats as the default).

HONESTY, not optimism: weak labels are NOT ground truth. Every function
here that produces them is paired with a validation function
(evaluate_weak_labeling_quality) that measures, against synthetic data
with KNOWN true labels, exactly how much is lost by using this bridge
instead of real labels -- a measured tradeoff, not a hopeful claim.
Real deployments won't have that ground truth to check against; the
measurement here is what tells you how much to trust the bridge when
you can't check it directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix

from ml.config import PatternType, BASE_FEATURE_COLUMNS, RANDOM_STATE
from ml.anomaly.isolation_forest import IsolationForestScratch
from ml.detection.multi_pattern import MultiPatternDetector


def build_pattern_prototypes_from_examples(
    X: pd.DataFrame, label_df: pd.DataFrame, patterns: list[PatternType] | None = None,
) -> dict[PatternType, np.ndarray]:
    """Few-shot case: build each pattern's prototype as the mean
    STANDARDIZED feature vector across whatever confirmed true examples
    you have -- even a handful (3-5 confirmed incidents) is enough to
    define a centroid, though more examples make it more reliable.

    Standardized (z-scored) against X's own mean/std, not raw feature
    values -- this is what lets a prototype built on one stock's price
    scale still make sense when attributing anomalies on a different
    stock with a completely different price range.
    """
    if patterns is None:
        patterns = list(PatternType)
    X_mean, X_std = X.mean(), X.std()
    X_standardized = (X - X_mean) / X_std

    prototypes = {}
    for pattern in patterns:
        label_col = f"is_{pattern.value}"
        if label_col not in label_df.columns:
            raise ValueError(f"label_df is missing column '{label_col}' for pattern {pattern.value}.")
        mask = label_df[label_col] == 1
        if mask.sum() == 0:
            raise ValueError(
                f"No confirmed examples for pattern '{pattern.value}' in label_df -- "
                f"cannot build a prototype from zero examples. Use "
                f"build_pattern_prototypes_from_domain_rules for patterns with no "
                f"confirmed examples at all."
            )
        prototypes[pattern] = X_standardized.loc[mask].mean().values
    return prototypes


def build_pattern_prototypes_from_domain_rules() -> dict[PatternType, np.ndarray]:
    """Zero-example case: prototypes built from pure domain reasoning
    about what each pattern's signature typically looks like in
    STANDARDIZED (z-scored) return / volume_ratio_20d / volatility_20d
    space -- the same directional reasoning that went into designing
    data/synthetic.py's DEFAULT_PATTERN_CONFIGS in the first place, made
    explicit and reusable here rather than left implicit in a data
    generator.

    Directional, not exact-magnitude: these represent RELATIVE
    positioning (e.g. "high positive return, high volume") in z-score
    units, not literal numbers from any one dataset -- because they're
    z-scored against WHATEVER dataset attribute_pattern_to_anomalies is
    later called on, they adapt to that dataset's own scale rather than
    assuming synthetic-data-specific magnitudes.

    Ordered [return_z, volume_ratio_z, volatility_z] to match BASE_FEATURE_COLUMNS.
    """
    return {
        PatternType.PUMP_AND_DUMP: np.array([2.5, 2.5, 1.0]),   # strong positive return + high volume
        PatternType.WASH_TRADING: np.array([0.0, 2.5, 0.5]),    # ~zero return + high volume
        PatternType.SPOOFING: np.array([0.0, 0.8, 1.8]),        # ~zero return + moderate volume + high volatility
        PatternType.LAYERING: np.array([0.0, 1.3, 1.5]),        # ~zero return + moderate-high volume + high volatility
    }


def attribute_pattern_to_anomalies(
    X: pd.DataFrame, flagged_mask: np.ndarray, prototypes: dict[PatternType, np.ndarray],
) -> pd.DataFrame:
    """For each flagged (anomalous) row, find the nearest prototype by
    Euclidean distance in standardized feature space, and report both
    the attributed pattern and a confidence score.

    confidence is a softmax over NEGATIVE distances (closer prototype ->
    higher confidence) -- NOT a calibrated probability, just a relative
    measure of how much closer the winning prototype was than the
    runners-up. A confidence near 1/n_patterns means the attribution is
    barely better than a guess; that's meaningful information for a
    human reviewer, not a footnote to hide.
    """
    if not prototypes:
        raise ValueError(
            "prototypes is empty -- there is nothing to attribute flagged "
            "anomalies to. Pass at least one pattern's prototype, or use "
            "build_pattern_prototypes_from_domain_rules()/"
            "build_pattern_prototypes_from_examples() to build some."
        )

    X_mean, X_std = X.mean(), X.std()
    X_standardized = ((X - X_mean) / X_std).values

    pattern_list = list(prototypes.keys())
    prototype_matrix = np.array([prototypes[p] for p in pattern_list])

    results = []
    for i in range(len(X)):
        if not flagged_mask[i]:
            results.append({"attributed_pattern": None, "confidence": None})
            continue
        point = X_standardized[i]
        distances = np.sqrt(((prototype_matrix - point) ** 2).sum(axis=1))
        # softmax over negative distances -- closer = higher score
        neg_distances = -distances
        exp_scores = np.exp(neg_distances - neg_distances.max())
        softmax_scores = exp_scores / exp_scores.sum()
        best_idx = np.argmax(softmax_scores)
        results.append({
            "attributed_pattern": pattern_list[best_idx].value,
            "confidence": float(softmax_scores[best_idx]),
        })
    return pd.DataFrame(results, index=X.index)


def weak_label_from_isolation_forest(
    X: pd.DataFrame, prototypes: dict[PatternType, np.ndarray] | None = None,
    contamination: float = 0.05, n_estimators: int = 100, random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Full bridge pipeline: fit IsolationForestScratch unsupervised (no
    labels needed), flag anomalies, attribute a likely pattern to each
    flagged day, and return a DataFrame with is_{pattern} columns in
    EXACTLY the schema data/synthetic.py produces and MultiPatternDetector
    expects -- so this is a drop-in replacement for true labels wherever
    those aren't available.

    prototypes defaults to build_pattern_prototypes_from_domain_rules()
    if not given -- the zero-example case.
    """
    if prototypes is None:
        prototypes = build_pattern_prototypes_from_domain_rules()

    iso_forest = IsolationForestScratch(
        n_estimators=n_estimators, contamination=contamination, random_state=random_state,
    )
    iso_forest.fit(X.values)
    flagged_mask = iso_forest.predict(X.values).astype(bool)

    attribution = attribute_pattern_to_anomalies(X, flagged_mask, prototypes)

    weak_labels = pd.DataFrame(index=X.index)
    for pattern in prototypes:
        weak_labels[f"is_{pattern.value}"] = (
            (attribution["attributed_pattern"] == pattern.value)
        ).astype(int)
    weak_labels["is_manipulation"] = flagged_mask.astype(int)
    weak_labels["attribution_confidence"] = attribution["confidence"]
    return weak_labels


def train_multi_pattern_detector_with_weak_labels(
    X: pd.DataFrame, prototypes: dict[PatternType, np.ndarray] | None = None,
    contamination: float = 0.05, random_state: int = RANDOM_STATE,
) -> tuple[MultiPatternDetector, pd.DataFrame]:
    """Bootstraps a MultiPatternDetector using weak labels instead of
    true ones. Returns both the fitted detector AND the weak_labels used
    to fit it, so a caller can inspect exactly what the model was
    actually trained on -- these are proxy labels, not confirmed ground
    truth, and treating the returned detector's predictions as more
    certain than that would misrepresent what was actually done here.

    Patterns with zero weakly-labeled positive examples are silently
    excluded (a real possibility if IsolationForest's flagged days
    happen to all attribute toward other patterns) rather than crashing
    the whole pipeline -- MultiPatternDetector.fit() would otherwise
    raise for that specific pattern.

    Note on the RuntimeError below: given the CURRENT guarantees
    elsewhere in this pipeline (IsolationForestScratch's quantile-based
    threshold always flags at least one row -- see
    test_weak_labeling.py's test_raises_when_prototypes_dict_is_empty
    docstring -- and attribute_pattern_to_anomalies always assigns a
    flagged row to its single closest prototype), this specific check is
    not reachable with a non-empty prototypes dict; at least one pattern
    always ends up with a positive. It's kept anyway as a safeguard
    against a future change (e.g. if attribute_pattern_to_anomalies is
    later extended to abstain below a confidence threshold, which WOULD
    make this reachable again) rather than removed as dead code.
    """
    resolved_prototypes = prototypes if prototypes is not None else build_pattern_prototypes_from_domain_rules()
    weak_labels = weak_label_from_isolation_forest(
        X, prototypes=resolved_prototypes, contamination=contamination, random_state=random_state,
    )
    patterns_with_positives = [
        p for p in resolved_prototypes
        if weak_labels[f"is_{p.value}"].sum() > 0
    ]
    if not patterns_with_positives:
        raise RuntimeError(
            "Zero patterns have any weakly-labeled positive examples -- cannot "
            "train MultiPatternDetector on this. Try a higher contamination "
            "value so more days get flagged, or check that prototypes are "
            "reasonable for this data's scale."
        )
    detector = MultiPatternDetector(patterns=patterns_with_positives, random_state=random_state)
    detector.fit(X, weak_labels)
    return detector, weak_labels


def evaluate_weak_labeling_quality(
    X: pd.DataFrame, true_label_df: pd.DataFrame, X_test: pd.DataFrame, true_label_df_test: pd.DataFrame,
    prototypes: dict[PatternType, np.ndarray] | None = None,
    contamination: float = 0.05, random_state: int = RANDOM_STATE,
) -> dict:
    """THE validation function this whole module rests on: measures,
    against data where true labels ARE available (synthetic, or a
    validation slice of real data with confirmed incidents), exactly how
    much is lost by using the weak-labeling bridge instead of true labels.

    Three separate, honestly-reported numbers, because they measure
    three DIFFERENT bottlenecks that a single "accuracy" figure would
    blur together:
      1. isolation_forest_recall_per_pattern: of the TRUE anomalies of
         each pattern, what fraction does IsolationForest even flag in
         the first place? A pattern IF rarely flags can never be
         correctly weak-labeled no matter how good attribution is --
         this bottleneck is entirely upstream of attribution.
      2. attribution_accuracy_given_flagged: of days IF DID flag, how
         often does nearest-centroid attribution assign the CORRECT
         pattern (vs. a wrong one)? Isolates the attribution step's own
         quality, given a day was already correctly flagged as anomalous.
      3. downstream_auc_true_vs_weak: trains MultiPatternDetector twice
         (once on true labels, once on weak labels from THIS pipeline)
         and compares per-pattern test AUC -- the actual bottom-line
         answer to "how much detection quality do you lose."
    """
    if prototypes is None:
        prototypes = build_pattern_prototypes_from_domain_rules()

    weak_labels_train = weak_label_from_isolation_forest(
        X, prototypes=prototypes, contamination=contamination, random_state=random_state,
    )

    # 1. IsolationForest's own recall per pattern (upstream bottleneck)
    if_recall_rows = []
    for pattern in prototypes:
        true_col = f"is_{pattern.value}"
        if true_col not in true_label_df.columns:
            continue
        true_positive_mask = true_label_df[true_col] == 1
        if true_positive_mask.sum() == 0:
            continue
        flagged_among_true_positives = weak_labels_train.loc[true_positive_mask, "is_manipulation"].mean()
        if_recall_rows.append({"pattern": pattern.value, "isolation_forest_recall": float(flagged_among_true_positives)})
    if_recall_df = pd.DataFrame(if_recall_rows)

    # 2. Attribution accuracy given a day was already correctly flagged
    combined_true_positive = (true_label_df[[f"is_{p.value}" for p in prototypes if f"is_{p.value}" in true_label_df.columns]].sum(axis=1) > 0)
    correctly_flagged_true_positives = combined_true_positive & (weak_labels_train["is_manipulation"] == 1)

    if correctly_flagged_true_positives.sum() > 0:
        true_pattern_for_flagged = true_label_df.loc[correctly_flagged_true_positives, [f"is_{p.value}" for p in prototypes]].idxmax(axis=1).str.replace("is_", "")
        weak_pattern_for_flagged = weak_labels_train.loc[correctly_flagged_true_positives, [f"is_{p.value}" for p in prototypes]].idxmax(axis=1).str.replace("is_", "")
        attribution_accuracy = float((true_pattern_for_flagged.values == weak_pattern_for_flagged.values).mean())
        attribution_confusion = confusion_matrix(true_pattern_for_flagged, weak_pattern_for_flagged, labels=[p.value for p in prototypes])
    else:
        attribution_accuracy = float("nan")
        attribution_confusion = None

    # 3. Downstream detection quality: true labels vs weak labels
    true_detector = MultiPatternDetector(patterns=list(prototypes.keys()), random_state=random_state)
    true_detector.fit(X, true_label_df)
    true_eval = true_detector.evaluate(X_test, true_label_df_test)

    weak_detector, _ = train_multi_pattern_detector_with_weak_labels(
        X, prototypes=prototypes, contamination=contamination, random_state=random_state,
    )
    # evaluate the weak-label-trained detector against TRUE test labels --
    # the whole point is measuring against reality, not against more weak labels
    weak_eval_rows = []
    for pattern in weak_detector.models_:
        true_col = f"is_{pattern.value}"
        if true_col not in true_label_df_test.columns:
            continue
        y_true = true_label_df_test[true_col]
        y_proba = weak_detector.predict_proba(X_test)[f"proba_{pattern.value}"]
        weak_eval_rows.append({
            "pattern": pattern.value,
            "auc": roc_auc_score(y_true, y_proba) if y_true.sum() > 0 else float("nan"),
        })
    weak_eval = pd.DataFrame(weak_eval_rows)

    comparison = true_eval[["pattern", "auc"]].merge(
        weak_eval, on="pattern", suffixes=("_true_labels", "_weak_labels")
    )
    comparison["auc_lost"] = comparison["auc_true_labels"] - comparison["auc_weak_labels"]

    return {
        "isolation_forest_recall_per_pattern": if_recall_df,
        "attribution_accuracy_given_flagged": attribution_accuracy,
        "attribution_confusion_matrix": attribution_confusion,
        "downstream_auc_comparison": comparison,
    }
