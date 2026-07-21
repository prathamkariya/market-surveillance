"""
Feature scaling, matching file 20 Sections 4-4.3.

The three scratch functions from the notes (standardize_from_scratch,
minmax_scale_from_scratch, robust_scale_from_scratch) are preserved exactly
-- same formulas, same behavior -- but wrapped as sklearn-compatible
transformers (fit/transform, not fit-and-immediately-apply) so they can
sit inside a Pipeline instead of being called as one-off functions on a
single global dataframe.

The scratch/library equivalence that file 20 demonstrates block-by-block
is preserved here as an actual assertion in tests, not just a printed
"match: True".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class _ScratchScalerBase(BaseEstimator, TransformerMixin):
    """Shared fit/transform scaffolding. Subclasses only need to define
    _compute_params and _apply -- keeps the three scalers from duplicating
    the same sklearn-compatibility boilerplate three times.
    """

    def fit(self, X: pd.DataFrame | np.ndarray, y=None) -> "_ScratchScalerBase":
        X = self._to_frame(X)
        self.feature_names_in_ = list(X.columns)
        self.params_: dict[str, tuple[float, float]] = {
            col: self._compute_params(X[col]) for col in X.columns
        }
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not hasattr(self, "params_"):
            raise RuntimeError("Call fit() before transform().")
        X = self._to_frame(X)
        out = pd.DataFrame(index=X.index)
        for col in self.feature_names_in_:
            center, spread = self.params_[col]
            out[col] = self._apply(X[col], center, spread)
        return out.values

    @staticmethod
    def _to_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(np.asarray(X))

    def _compute_params(self, series: pd.Series) -> tuple[float, float]:
        raise NotImplementedError

    def _apply(self, series: pd.Series, center: float, spread: float) -> pd.Series:
        raise NotImplementedError


class StandardScalerScratch(_ScratchScalerBase):
    """(x - mean) / std. File 20 Section 4.1.

    ddof=0 (population std, divide by n) deliberately -- this is what
    sklearn.StandardScaler uses internally. pandas' Series.std() defaults
    to ddof=1 (sample std, divide by n-1); using that default here would
    make this "scratch" implementation silently disagree with sklearn by
    a factor of sqrt(n/(n-1)). Caught by test_matches_sklearn, which is
    exactly the point of writing that test.
    """

    def _compute_params(self, series: pd.Series) -> tuple[float, float]:
        return series.mean(), series.std(ddof=0)

    def _apply(self, series: pd.Series, center: float, spread: float) -> pd.Series:
        if spread == 0:
            raise ValueError(
                "Zero standard deviation -- this feature is constant and "
                "cannot be standardized. Drop it before scaling."
            )
        return (series - center) / spread


class MinMaxScalerScratch(_ScratchScalerBase):
    """(x - min) / (max - min). File 20 Section 4.2 -- also demonstrates
    this scaler's outlier sensitivity, reproduced in tests below."""

    def _compute_params(self, series: pd.Series) -> tuple[float, float]:
        return series.min(), series.max() - series.min()

    def _apply(self, series: pd.Series, center: float, spread: float) -> pd.Series:
        if spread == 0:
            raise ValueError(
                "Zero range -- this feature is constant and cannot be "
                "min-max scaled. Drop it before scaling."
            )
        return (series - center) / spread


class RobustScalerScratch(_ScratchScalerBase):
    """(x - median) / IQR. File 20 Section 4.3 -- built specifically to
    resist the outlier sensitivity MinMaxScalerScratch has."""

    def _compute_params(self, series: pd.Series) -> tuple[float, float]:
        median = series.median()
        iqr = series.quantile(0.75) - series.quantile(0.25)
        return median, iqr

    def _apply(self, series: pd.Series, center: float, spread: float) -> pd.Series:
        if spread == 0:
            raise ValueError(
                "Zero IQR -- more than 75% of this feature's values are "
                "identical. Robust scaling is degenerate here; inspect the "
                "feature before proceeding."
            )
        return (series - center) / spread


SCRATCH_TO_LIBRARY = {
    StandardScalerScratch: StandardScaler,
    MinMaxScalerScratch: MinMaxScaler,
    RobustScalerScratch: RobustScaler,
}


def needs_scaling(model_family: str) -> bool:
    """File 20 Section 3: distance/gradient-based models need scaling,
    tree-based models don't (split-finding is scale-invariant). Centralized
    so this judgment call is made once, not re-decided ad hoc per script.
    """
    scale_sensitive = {
        "logistic_regression", "kmeans", "knn", "svm", "neural_network",
        "isolation_forest",  # uses distance/path-length in feature space
        "lof",
    }
    scale_invariant = {
        "decision_tree", "random_forest", "gradient_boosting", "xgboost",
    }
    key = model_family.lower()
    if key in scale_sensitive:
        return True
    if key in scale_invariant:
        return False
    raise ValueError(
        f"Unknown model family '{model_family}'. Add it to needs_scaling() "
        f"explicitly rather than guessing -- getting this wrong silently "
        f"either wastes a scaling step or breaks a distance-based model."
    )
