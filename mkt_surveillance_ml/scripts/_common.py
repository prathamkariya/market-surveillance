"""
Shared utilities for scripts/train.py and scripts/forecast.py.

Kept separate specifically so CSV-loading logic isn't duplicated (and
liable to drift) across the two entrypoints.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from mkt_surveillance_ml.config import BASE_FEATURE_COLUMNS
from mkt_surveillance_ml.data.synthetic import compute_engineered_features


def load_real_csv(csv_path: str) -> pd.DataFrame:
    """Loads a real OHLCV CSV, computes the same 3 engineered features
    (return, volume_ratio_20d, volatility_20d) the rest of this package
    uses everywhere else -- so a model trained on real data and one
    trained on synthetic data are directly comparable.

    Requires BOTH 'close' and 'volume' -- this is for train.py's
    classifier/anomaly-detection modes, which need volume_ratio_20d.
    forecast.py's modes only need a price series; use load_price_series
    below for those instead of this function.
    """
    path = Path(csv_path)
    if not path.exists():
        print(f"ERROR: CSV not found at '{csv_path}'.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(path)
    date_col = "date" if "date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    missing = {"close", "volume"} - set(df.columns)
    if missing:
        print(
            f"ERROR: CSV is missing required column(s): {sorted(missing)}. "
            f"Found columns: {list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    df = compute_engineered_features(df)
    n_before = len(df)
    df = df.dropna(subset=BASE_FEATURE_COLUMNS)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(
            f"Dropped {n_dropped} rows at the start (need 20 prior days "
            f"to compute rolling features -- this is expected)."
        )
    if len(df) < 100:
        print(
            f"WARNING: only {len(df)} usable rows after feature computation. "
            f"Models trained on this little data will be unreliable -- "
            f"treat any results as exploratory, not production-ready.",
            file=sys.stderr,
        )
    return df


def load_price_series(csv_path: str) -> pd.Series:
    """Loads just a real price series -- only requires 'close' (and a
    date column), NOT 'volume'. forecast.py's modes (stationarity
    checks, ARIMA/Prophet/LSTM forecasting) operate on a single price
    series and never touch volume_ratio_20d/volatility_20d, so this
    intentionally does NOT route through compute_engineered_features or
    require a 'volume' column the way load_real_csv does for train.py.
    """
    path = Path(csv_path)
    if not path.exists():
        print(f"ERROR: CSV not found at '{csv_path}'.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(path)
    date_col = "date" if "date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    if "close" not in df.columns:
        print(
            f"ERROR: CSV is missing required column 'close'. Found columns: {list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    if len(df) < 100:
        print(
            f"WARNING: only {len(df)} rows. Forecasting on this little data "
            f"will be unreliable -- treat any results as exploratory, not "
            f"production-ready.",
            file=sys.stderr,
        )
    return df["close"]
