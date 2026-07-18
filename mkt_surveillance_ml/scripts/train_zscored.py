"""
Phase B: Real-data IsolationForest training with per-symbol rolling z-score
normalization on `return` and `volatility_20d`.

WHY THIS SCRIPT EXISTS (not a standalone curiosity):
  train.py's `real-unsupervised` mode calls `_common.load_real_csv` which
  unconditionally re-runs `compute_engineered_features`, clobbering any
  pre-normalized features in the CSV. Rather than modifying `_common.py`
  (kept untouched per §6 of the implementation plan), this script drives the
  IsolationForest training directly using the same model class, with the same
  hyperparameters, but feeding it the z-score-normalized X instead of raw X.

NORMALIZATION DISCIPLINE (enforced, not just described):
  Z-scores use per-symbol ROLLING mean/std with .shift(1) so that row t is
  scored against statistics computed from rows [t-window, t-1] — the current
  row does NOT contribute to its own normalization baseline. This is the same
  causal discipline as `volatility_20d`'s own 20-day rolling std window.
  Window = 60 trading days (~3 months) with min_periods=20.

FEATURES NORMALIZED: `return` and `volatility_20d` only.
  `volume_ratio_20d` is already dimensionless (volume / rolling mean volume)
  — comparable across symbols by construction. No z-scoring applied there.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

# Resolve project root regardless of cwd
_HERE = Path(__file__).parent
_SRC = _HERE / ".." / "src"
sys.path.insert(0, str(_SRC.resolve()))
sys.path.insert(0, str(_HERE.resolve()))

from mkt_surveillance_ml.config import BASE_FEATURE_COLUMNS
from mkt_surveillance_ml.anomaly.isolation_forest import IsolationForestScratch

ROLLING_WINDOW = 60
MIN_PERIODS = 20


def add_rolling_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-symbol rolling z-score for `return` and `volatility_20d`.

    Critical implementation detail: rolling stats use .shift(1) so row t
    is normalized against [t-window, t-1], NOT including row t itself.
    A genuine outlier at row t does not inflate its own normalization baseline.
    """
    df = df.copy()
    for sym, grp in df.groupby("symbol", sort=False):
        for col in ["return", "volatility_20d"]:
            rolling = grp[col].rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS)
            # .shift(1): each row sees the window ENDING at the previous row
            mu = rolling.mean().shift(1)
            sigma = rolling.std().shift(1).clip(lower=1e-8)
            df.loc[df["symbol"] == sym, col] = ((grp[col] - mu) / sigma).values
    return df


def train_market(market: str, input_csv: str, output_dir: str,
                 contamination: float = 0.05, n_estimators: int = 100,
                 random_state: int = 42) -> None:
    print(f"\n=== {market}: Rolling z-score + IsolationForest training ===")

    df = pd.read_csv(input_csv, index_col=0)
    required = set(BASE_FEATURE_COLUMNS) | {"symbol"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # Drop any NaN rows (warmup rows already dropped by prepare_real_data.py)
    before = len(df)
    df = df.dropna(subset=BASE_FEATURE_COLUMNS)
    if before - len(df) > 0:
        print(f"  Dropped {before - len(df)} NaN rows.")

    print(f"  Applying rolling z-score (window={ROLLING_WINDOW}, shift=1) to "
          f"'return' and 'volatility_20d'...")

    # Extract baselines BEFORE z-scoring, using the trailing window ending at the
    # last row of training data per symbol. This is exactly the baseline the model
    # expects at serving time: the symbol's trailing mean/std as of training end.
    symbol_baselines: dict = {}
    for sym, grp in df.groupby("symbol", sort=False):
        for col in ["return", "volatility_20d"]:
            rolling = grp[col].rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS)
            mu = rolling.mean()
            sigma = rolling.std().clip(lower=1e-8)
            # Use the last non-NaN values — the trailing baseline at training end
            last_mu = float(mu.dropna().iloc[-1]) if not mu.dropna().empty else None
            last_sigma = float(sigma.dropna().iloc[-1]) if not sigma.dropna().empty else None
            if sym not in symbol_baselines:
                symbol_baselines[sym] = {}
            symbol_baselines[sym][col] = {"mean": last_mu, "std": last_sigma}

    df_norm = add_rolling_zscores(df)

    # Drop rows where z-score is NaN (the first MIN_PERIODS rows per symbol
    # have insufficient history for a reliable rolling baseline)
    before = len(df_norm)
    df_norm = df_norm.dropna(subset=BASE_FEATURE_COLUMNS)
    dropped = before - len(df_norm)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with insufficient rolling history "
              f"(first {MIN_PERIODS} rows per symbol - expected).")

    print(f"  Training on {len(df_norm)} rows ({df_norm['symbol'].nunique()} symbols).")
    X = df_norm[BASE_FEATURE_COLUMNS].values

    model = IsolationForestScratch(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X)

    scores = model.score_samples(X)
    flags = model.predict(X)

    df_scored = df_norm.copy()
    df_scored["anomaly_score"] = scores
    df_scored["is_flagged"] = flags

    n_flagged = int(flags.sum())
    print(f"  Flagged {n_flagged} of {len(df_norm)} days "
          f"({n_flagged / len(df_norm) * 100:.1f}%) as anomalous.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / "isolation_forest_scratch.joblib"
    joblib.dump(model, model_path)

    scored_path = out / "scored_days.csv"
    df_scored.to_csv(scored_path)

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "real-unsupervised-zscored",
        "data_source": input_csv,
        "n_rows": len(df_norm),
        "n_flagged": n_flagged,
        "contamination": contamination,
        "n_estimators": n_estimators,
        "random_state": random_state,
        "feature_columns": BASE_FEATURE_COLUMNS,
        "zscore_normalization": {
            "features": ["return", "volatility_20d"],
            "rolling_window": ROLLING_WINDOW,
            "min_periods": MIN_PERIODS,
            "shift": 1,
            "note": "causal: row t uses stats from [t-window, t-1] only",
        },
    }
    meta_path = out / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    # Rename to isolation_forest_metadata.json immediately (§4 naming fix)
    import shutil
    shutil.copy(meta_path, out / "isolation_forest_metadata.json")

    # Save symbol baselines for serving-time z-score transform
    baselines_path = out / "symbol_baselines.json"
    baselines_path.write_text(json.dumps(symbol_baselines, indent=2))

    print(f"  Saved model      -> {model_path}")
    print(f"  Saved scored CSV -> {scored_path}")
    print(f"  Saved metadata   -> {meta_path} + isolation_forest_metadata.json")
    print(f"  Saved baselines  -> {baselines_path} ({len(symbol_baselines)} symbols)")


if __name__ == "__main__":
    train_market(
        "CRYPTO",
        input_csv="trained_models/crypto/real_if_input.csv",
        output_dir="trained_models/crypto",
        random_state=42,
    )
    train_market(
        "US_EQUITY",
        input_csv="trained_models/us_equity/real_if_input.csv",
        output_dir="trained_models/us_equity",
        random_state=43,
    )
