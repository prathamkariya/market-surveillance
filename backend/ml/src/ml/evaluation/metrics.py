"""
Model Evaluation, matching file 25 Sections 1-4 and its closing argument.

compare_models_time_series_cv turns the notes' one-off loop over four
hardcoded model instances into a function taking any dict of {name: model}.

evaluate_on_label is preserved close to verbatim -- it's the direct
evidence function behind this whole package's per-pattern design: a
single respectable-looking AUC on a COMBINED label can blend two (or
four) genuinely different underlying performances and cannot tell you
which pattern is under-served, or by how much.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, learning_curve
from sklearn.preprocessing import StandardScaler


def compare_models_time_series_cv(
    X: pd.DataFrame,
    y: pd.Series,
    models: dict,
    n_splits: int = 5,
    scale_for: set[str] | None = None,
) -> pd.DataFrame:
    """File 25 Section 2's model-comparison loop, generalized to any
    dict of named models rather than four hardcoded instances.

    TimeSeriesSplit, not KFold or a random split -- each fold trains on
    strictly earlier data and tests on strictly later data, matching the
    chronological_train_test_split convention used everywhere else in
    this package (see data/synthetic.py).

    scale_for: names of models (matching keys in `models`) that need
    scaling (distance/gradient-based -- see features/scaling.py's
    needs_scaling()). Folds where the test set has zero positive
    examples are skipped -- AUC is undefined there, not just noisy.
    """
    if scale_for is None:
        scale_for = set()

    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_results: dict[str, list[float]] = {name: [] for name in models}

    for train_idx, test_idx in tscv.split(X):
        X_train_fold = X.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_train_fold = y.iloc[train_idx]
        y_test_fold = y.iloc[test_idx]

        if y_test_fold.sum() == 0 or y_train_fold.sum() == 0:
            # Both checks matter: a test fold with no positives makes AUC
            # undefined; a TRAINING fold with no positives means the model
            # only ever saw one class and predict_proba won't even have a
            # positive-class column to index -- both are real possibilities
            # in early folds for a genuinely rare pattern, not just edge
            # cases to wave away.
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_test_scaled = scaler.transform(X_test_fold)

        for name, model in models.items():
            X_tr = X_train_scaled if name in scale_for else X_train_fold
            X_te = X_test_scaled if name in scale_for else X_test_fold

            model.fit(X_tr, y_train_fold)
            proba = model.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_test_fold, proba)
            cv_results[name].append(auc)

    rows = []
    for name, aucs in cv_results.items():
        if not aucs:
            rows.append({"model": name, "mean_auc": float("nan"), "std_auc": float("nan"), "n_folds": 0})
        else:
            rows.append({
                "model": name,
                "mean_auc": float(np.mean(aucs)),
                "std_auc": float(np.std(aucs)),
                "n_folds": len(aucs),
            })
    return pd.DataFrame(rows).sort_values("mean_auc", ascending=False).reset_index(drop=True)


def calibration_analysis(
    model, X_test: pd.DataFrame, y_test: pd.Series, n_bins: int = 5
) -> dict:
    """File 25 Section 3. Calibration (do predicted probabilities match
    observed frequencies?) and discrimination (AUC -- does the model rank
    positives above negatives?) are genuinely separate properties. A
    model can have excellent AUC and terrible calibration, or vice versa.
    Post-hoc calibration (CalibratedClassifierCV) should leave AUC nearly
    unchanged, since it only rescales predicted probabilities -- it does
    not change which examples rank above which.
    """
    proba_uncalibrated = model.predict_proba(X_test)[:, 1]
    fraction_pos_before, mean_pred_before = calibration_curve(
        y_test, proba_uncalibrated, n_bins=n_bins
    )
    auc_before = roc_auc_score(y_test, proba_uncalibrated)

    # cv='prefit' (calibrate an already-fitted model without refitting)
    # was removed in current scikit-learn -- FrozenEstimator (1.6+) is
    # the replacement: it wraps an already-fitted estimator so
    # CalibratedClassifierCV treats it as fixed rather than re-fitting it.
    from sklearn.frozen import FrozenEstimator
    calibrated_model = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    calibrated_model.fit(X_test, y_test)
    proba_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
    fraction_pos_after, mean_pred_after = calibration_curve(
        y_test, proba_calibrated, n_bins=n_bins
    )
    auc_after = roc_auc_score(y_test, proba_calibrated)

    return {
        "auc_before": float(auc_before),
        "auc_after": float(auc_after),
        "auc_changed_by": float(abs(auc_after - auc_before)),
        "fraction_of_positives_before": fraction_pos_before,
        "mean_predicted_before": mean_pred_before,
        "fraction_of_positives_after": fraction_pos_after,
        "mean_predicted_after": mean_pred_after,
        "calibrated_model": calibrated_model,
    }


def compute_learning_curve(
    model, X: pd.DataFrame, y: pd.Series, n_splits: int = 4,
    train_sizes: np.ndarray | None = None,
) -> pd.DataFrame:
    """File 25 Section 4. TimeSeriesSplit CV throughout -- a learning
    curve built on a random split would answer "does more DATA help,"
    not "does more data help GIVEN that we can only ever train on the
    past," which is the only question that matters for a system that
    will be deployed to score days it hasn't seen yet.
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.3, 1.0, 6)

    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X, y, cv=TimeSeriesSplit(n_splits=n_splits),
        train_sizes=train_sizes, scoring="roc_auc",
    )
    return pd.DataFrame({
        "train_size": train_sizes_abs,
        "train_score_mean": train_scores.mean(axis=1),
        "train_score_std": train_scores.std(axis=1),
        "test_score_mean": test_scores.mean(axis=1),
        "test_score_std": test_scores.std(axis=1),
    })


def evaluate_on_label(
    X_data: np.ndarray, y_label: np.ndarray, random_state: int = 42
) -> float | None:
    """File 25's closing argument, preserved close to verbatim. A single
    respectable-looking AUC on a COMBINED label can blend genuinely
    different underlying per-pattern performances, and it cannot tell
    you which pattern is under-served or by how much -- this function is
    what's actually called, once per label, to expose that gap (see
    compare_combined_vs_per_pattern_auc below and
    detection/multi_pattern.py, which is built directly on this finding).

    Returns None if the test split ends up with zero positive examples
    for this label (AUC undefined in that case, not just unreliable).
    """
    split = int(len(X_data) * 0.7)
    X_tr, X_te = X_data[:split], X_data[split:]
    y_tr, y_te = y_label[:split], y_label[split:]
    if y_te.sum() == 0:
        return None

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    proba = rf.predict_proba(X_te)[:, 1]
    return float(roc_auc_score(y_te, proba))


def compare_combined_vs_per_pattern_auc(
    X: np.ndarray, pattern_labels: dict[str, np.ndarray], combined_label: np.ndarray,
    random_state: int = 42,
) -> pd.DataFrame:
    """Applies evaluate_on_label once to the combined label and once per
    individual pattern, using the SAME shuffle across all of them (file
    25: "shuffling once, applying the SAME shuffle to all labels so the
    comparison is apples-to-apples") -- returns a table making the
    combined-vs-per-pattern gap directly visible and measurable, instead
    of a single printed number that hides it.
    """
    rng = np.random.RandomState(random_state)
    shuffle_idx = rng.permutation(len(X))
    X_shuffled = X[shuffle_idx]
    combined_shuffled = combined_label[shuffle_idx]

    rows = [{
        "label": "combined (blended)",
        "auc": evaluate_on_label(X_shuffled, combined_shuffled, random_state),
    }]
    for pattern_name, labels in pattern_labels.items():
        labels_shuffled = labels[shuffle_idx]
        rows.append({
            "label": f"{pattern_name} (per-pattern)",
            "auc": evaluate_on_label(X_shuffled, labels_shuffled, random_state),
        })
    return pd.DataFrame(rows)
