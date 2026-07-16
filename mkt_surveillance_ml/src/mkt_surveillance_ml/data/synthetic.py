"""
Canonical synthetic OHLCV data generator, with labeled per-pattern injection.

Every file in the Phase 5-6 notes (20-27) rolled its own synthetic data with
a different seed, different schema, and different injection logic. That's
fine for an isolated demonstration -- it's not fine for a package, where a
model trained against one file's data generator and evaluated against
another's would silently disagree about what "return" or "volume_ratio_20d"
even mean.

This module is the single source of truth for synthetic data going forward.

Honesty note on scope: this generates OHLCV-derived features only (price,
volume, and things computed from them). Spoofing and layering are, in
reality, order-book-level phenomena (orders placed and cancelled before
execution, fake depth at multiple price levels) that OHLCV data cannot
directly observe -- you'd need L2/L3 order book data to detect them
properly. What's injected here for those two patterns is a defensible
OHLCV-visible PROXY (short-lived volatility spikes with reverting price
for spoofing; sustained elevated volatility with a volume signature
distinct from wash trading for layering), not a claim that this is
what real spoofing looks like at the order-book level. Say this plainly
if asked in an interview -- claiming otherwise is the kind of overclaim
the critical-assessment sections in the notes exist to push back against.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mkt_surveillance_ml.config import PatternType, RANDOM_STATE


@dataclass(frozen=True)
class PatternInjectionConfig:
    """How many days of a pattern to inject, and how strongly."""

    n_days: int
    return_mean: float
    return_std: float
    volume_ratio_mean: float
    volume_ratio_std: float


# One config per pattern. Magnitudes are deliberately DIFFERENT per pattern
# (not just "high volume" for everything) so a per-pattern classifier has a
# genuinely distinct signature to learn -- mirroring file 22 Section 7's
# pump-vs-wash contrast, extended to all four patterns.
DEFAULT_PATTERN_CONFIGS: dict[PatternType, PatternInjectionConfig] = {
    PatternType.PUMP_AND_DUMP: PatternInjectionConfig(
        n_days=15, return_mean=0.065, return_std=0.012,
        volume_ratio_mean=3.2, volume_ratio_std=0.35,
    ),
    PatternType.WASH_TRADING: PatternInjectionConfig(
        n_days=25, return_mean=0.0, return_std=0.004,
        volume_ratio_mean=2.9, volume_ratio_std=0.30,
    ),
    PatternType.SPOOFING: PatternInjectionConfig(
        n_days=12, return_mean=0.0, return_std=0.028,
        volume_ratio_mean=1.6, volume_ratio_std=0.25,
    ),
    PatternType.LAYERING: PatternInjectionConfig(
        n_days=18, return_mean=0.0, return_std=0.019,
        volume_ratio_mean=2.1, volume_ratio_std=0.28,
    ),
}


def compute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """return, volume_ratio_20d, volatility_20d -- the same three columns
    used consistently across files 22, 24, 27. Centralized here so every
    caller computes them identically.

    Public (not just for synthetic data): scripts/train.py calls this
    directly on real OHLCV data too, so a model trained on synthetic
    data and one trained on real data compute features identically.
    Requires columns 'close' and 'volume'; index should be date-sorted.
    """
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["volume_ratio_20d"] = df["volume"] / df["volume"].rolling(20, min_periods=1).mean()
    df["volatility_20d"] = df["return"].rolling(20).std() * np.sqrt(252)
    return df


def generate_synthetic_market_data(
    n_days: int = 500,
    patterns: list[PatternType] | None = None,
    pattern_configs: dict[PatternType, PatternInjectionConfig] | None = None,
    start_date: str = "2024-01-01",
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generate n_days of synthetic OHLCV data with disjoint, labeled
    manipulation days injected for each requested pattern.

    Returns a DataFrame indexed by date with columns:
        close, volume, return, volume_ratio_20d, volatility_20d,
        is_pump_and_dump, is_wash_trading, is_spoofing, is_layering,
        is_manipulation (logical OR of the four -- provided ONLY for
        baseline comparison against the per-pattern approach; production
        training code in this package does not use it as a training target)

    Injected days for different patterns never overlap, by construction --
    a day is at most one pattern, which keeps the per-pattern labels
    genuinely separable and avoids silently teaching a model on
    contradictory examples.
    """
    if patterns is None:
        patterns = list(PatternType)
    if pattern_configs is None:
        pattern_configs = DEFAULT_PATTERN_CONFIGS

    rng = np.random.RandomState(random_state)
    dates = pd.date_range(start_date, periods=n_days, freq="D")

    close = 100 + np.cumsum(rng.normal(0.02, 1, n_days))
    volume = rng.lognormal(15, 0.3, n_days)

    label_cols = {p: np.zeros(n_days, dtype=int) for p in PatternType}

    # Reserve the first 30 days untouched -- rolling features need warm-up,
    # matching every source file's `range(30, n)` convention.
    available_days = list(range(30, n_days))
    rng.shuffle(available_days)
    cursor = 0

    for pattern in patterns:
        cfg = pattern_configs[pattern]
        if cursor + cfg.n_days > len(available_days):
            raise ValueError(
                f"Not enough days to inject {pattern.value}: requested "
                f"{cfg.n_days}, only {len(available_days) - cursor} left. "
                f"Increase n_days or reduce pattern day counts."
            )
        chosen_days = available_days[cursor: cursor + cfg.n_days]
        cursor += cfg.n_days

        label_cols[pattern][chosen_days] = 1

        # Inject via volume ratio indirectly: scale raw volume so that,
        # after the rolling-mean ratio is computed downstream, it lands
        # near the pattern's target volume_ratio_mean. Also perturb close
        # price directly for the return signature.
        volume[chosen_days] *= rng.normal(
            cfg.volume_ratio_mean, cfg.volume_ratio_std, cfg.n_days
        ).clip(min=0.3)
        return_shock = rng.normal(cfg.return_mean, cfg.return_std, cfg.n_days)
        close[chosen_days] = close[chosen_days] * (1 + return_shock)

    volume = np.clip(volume, a_min=1.0, a_max=None)

    df = pd.DataFrame({"close": close, "volume": volume}, index=dates)
    df.index.name = "date"
    for pattern in PatternType:
        df[f"is_{pattern.value}"] = label_cols[pattern]

    df = compute_engineered_features(df)
    df = df.dropna()

    df["is_manipulation"] = (
        df[[f"is_{p.value}" for p in PatternType]].sum(axis=1) > 0
    ).astype(int)

    return df


def chronological_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The ONLY split function this package uses for time-series data.

    File 21 Section 2 makes the point directly: a random split leaks
    future information into training for time-ordered data. Every model
    module in this package imports this function rather than calling
    sklearn's train_test_split directly, so that mistake can't silently
    reappear in one module while being correctly avoided in another.
    """
    split_index = int(len(df) * (1 - test_size))
    return df.iloc[:split_index].copy(), df.iloc[split_index:].copy()
