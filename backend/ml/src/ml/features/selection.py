"""
Feature selection, matching file 20 Section 5.

Three families, kept as three distinct functions rather than one
do-everything selector, because file 20's own experiments show they
answer different questions:
  - filter (F-test, mutual information): fast, model-agnostic, but the
    F-test specifically misses non-linear relationships (Section 5.2's
    |X| example) -- mutual information does not.
  - wrapper (RFE): uses an actual model's performance to select, more
    expensive, generally more reliable for the model it's built with.
  - embedded (feature_importances_): comes free from fitting a tree
    ensemble, no extra selection step needed.

Also includes the per-pattern-vs-combined-label mutual information
comparison from Section 6 as a reusable function, since that's the
specific numerical argument the per-pattern detection design in this
package rests on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, f_classif, mutual_info_classif


def filter_select(
    X: pd.DataFrame, y: np.ndarray, method: str = "mutual_info", random_state: int = 42
) -> pd.DataFrame:
    """Rank features by a model-agnostic statistic.

    method='f_test': ANOVA F-statistic. Sensitive to LINEAR relationships
        only (file 20 Section 5.2 demonstrates this failing on a clean
        non-linear relationship).
    method='mutual_info': captures non-linear relationships too. Prefer
        this as the default unless there's a specific reason to want the
        linear-only F-test.

    Returns a DataFrame sorted by score, descending.
    """
    if method == "f_test":
        scores, pvalues = f_classif(X, y)
        result = pd.DataFrame(
            {"feature": X.columns, "score": scores, "p_value": pvalues}
        )
    elif method == "mutual_info":
        scores = mutual_info_classif(X, y, random_state=random_state)
        result = pd.DataFrame({"feature": X.columns, "score": scores})
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'f_test' or 'mutual_info'.")
    return result.sort_values("score", ascending=False).reset_index(drop=True)


def wrapper_select(
    X: pd.DataFrame,
    y: np.ndarray,
    n_features_to_select: int,
    estimator=None,
    random_state: int = 42,
) -> list[str]:
    """Recursive Feature Elimination. File 20 Section 5.3.

    More expensive than filter methods (retrains the estimator at each
    elimination step) but selects based on actual downstream model
    performance rather than a proxy statistic.
    """
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=50, random_state=random_state)
    selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
    selector.fit(X, y)
    return list(X.columns[selector.support_])


def embedded_select(
    X: pd.DataFrame, y: np.ndarray, top_k: int | None = None, random_state: int = 42
) -> pd.DataFrame:
    """Feature importance from a fitted Random Forest. File 20 Section 5.4.

    "Free" in the sense that a tree ensemble you were already going to
    fit produces this as a byproduct -- no separate selection pass needed.
    Returns all features ranked unless top_k is given.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X, y)
    importances = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    return importances.head(top_k) if top_k else importances


def compare_score_against_combined_vs_per_pattern_label(
    feature: pd.Series, per_pattern_label: np.ndarray, combined_label: np.ndarray,
    random_state: int = 42,
) -> dict[str, float]:
    """File 20 Section 6's argument, as a reusable function rather than a
    one-off script.

    Scores a single feature's mutual information against its TRUE
    per-pattern target versus a blended/combined label, and reports the
    percentage degradation. This is the concrete, numerical basis for
    this package's per-pattern detection design (see detection/multi_pattern.py)
    -- not an assumption, a measured effect.
    """
    feature_arr = feature.values.reshape(-1, 1)
    mi_per_pattern = mutual_info_classif(
        feature_arr, per_pattern_label, random_state=random_state
    )[0]
    mi_combined = mutual_info_classif(
        feature_arr, combined_label, random_state=random_state
    )[0]
    degradation_pct = (
        (1 - mi_combined / mi_per_pattern) * 100 if mi_per_pattern > 0 else float("nan")
    )
    return {
        "mi_against_true_pattern": float(mi_per_pattern),
        "mi_against_combined_label": float(mi_combined),
        "degradation_pct": float(degradation_pct),
    }
