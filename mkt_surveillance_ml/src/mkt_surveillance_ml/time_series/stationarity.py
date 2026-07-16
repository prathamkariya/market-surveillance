"""
Time Series Foundations: Stationarity, matching file 28.

Every function here returns structured data (dicts, dataclasses, or
DataFrames) rather than printing -- the notes demonstrate these as
one-off print statements against a single global series; a production
caller needs the actual numbers back to make a decision (e.g. "should I
difference this series before fitting ARIMA?"), not a printed sentence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose, DecomposeResult
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf


def seasonal_decomposition_comparison(
    series: pd.Series, period: int, model_a: str = "additive", model_b: str = "multiplicative"
) -> dict:
    """File 28 Section 2's argument, as a function: additive vs
    multiplicative isn't a stylistic choice, it's a claim about the
    data's structure that can be measurably right or wrong. The
    correctly-matched model leaves a smaller, more genuinely random
    residual; the mismatched one leaves systematic structure (the
    seasonal swings it failed to account for) baked into the residual.
    """
    decomp_a = seasonal_decompose(series, model=model_a, period=period)
    decomp_b = seasonal_decompose(series, model=model_b, period=period)
    return {
        "residual_std_a": float(decomp_a.resid.std()),
        "residual_std_b": float(decomp_b.resid.std()),
        "better_fit": model_a if decomp_a.resid.std() < decomp_b.resid.std() else model_b,
        "decomposition_a": decomp_a,
        "decomposition_b": decomp_b,
    }


def diagnose_stationarity_violations(series: pd.Series) -> dict:
    """A fast, computational SCREENING check for the three distinct ways
    file 28 Section 3 shows a series can fail to be stationary (non-
    constant mean, non-constant variance, non-constant autocovariance
    structure) -- checked SEPARATELY, since a series can violate one and
    satisfy the other two. This is a quick diagnostic, not a substitute
    for adf_test/kpss_test below, which are the formal statistical tests;
    use this to get oriented before running those.
    """
    series = series.dropna()
    mid = len(series) // 2
    first_half, second_half = series.iloc[:mid], series.iloc[mid:]

    mean_diff = abs(first_half.mean() - second_half.mean())
    pooled_std = series.std()
    mean_violation_score = mean_diff / pooled_std if pooled_std > 0 else 0.0

    std_ratio = max(first_half.std(), second_half.std()) / max(min(first_half.std(), second_half.std()), 1e-10)

    def _lag1_autocorr(x: pd.Series) -> float:
        x = x.values
        if len(x) < 3:
            return float("nan")
        return float(np.corrcoef(x[:-1], x[1:])[0, 1])

    autocorr_first = _lag1_autocorr(first_half)
    autocorr_second = _lag1_autocorr(second_half)
    autocorr_diff = abs(autocorr_first - autocorr_second)

    return {
        "mean_first_half": float(first_half.mean()),
        "mean_second_half": float(second_half.mean()),
        "mean_violation_likely": bool(mean_violation_score > 0.5),
        "std_first_half": float(first_half.std()),
        "std_second_half": float(second_half.std()),
        "variance_violation_likely": bool(std_ratio > 1.5),
        "autocorr_first_half": autocorr_first,
        "autocorr_second_half": autocorr_second,
        "autocovariance_violation_likely": bool(autocorr_diff > 0.3),
    }


@dataclass
class StationarityTestResult:
    test_name: str
    statistic: float
    p_value: float
    is_stationary: bool
    critical_values: dict


def adf_test(series: pd.Series) -> StationarityTestResult:
    """Augmented Dickey-Fuller. File 28 Section 3.1.

    H0: the series has a unit root (is non-stationary, e.g. a random walk).
    H1: the series does not have a unit root (is stationary).
    p < 0.05 -> reject H0 -> conclude STATIONARY.
    """
    result = adfuller(series.dropna())
    return StationarityTestResult(
        test_name="ADF", statistic=float(result[0]), p_value=float(result[1]),
        is_stationary=bool(result[1] < 0.05), critical_values=result[4],
    )


def kpss_test(series: pd.Series) -> StationarityTestResult:
    """KPSS. File 28 Section 3.2 -- hypotheses are the REVERSE of ADF's.

    H0: the series IS stationary.
    H1: the series is NOT stationary (has a unit root).
    p < 0.05 -> reject H0 -> conclude NON-STATIONARY (the opposite
    reading direction from ADF -- easy to get backwards).
    """
    import warnings
    from statsmodels.tools.sm_exceptions import InterpolationWarning

    with warnings.catch_warnings():
        # statsmodels' KPSS p-value comes from a lookup table with a
        # finite range; a statistic extreme enough to fall outside that
        # range returns a boundary p-value with this warning. The
        # DIRECTION of the conclusion (stationary vs not) is unaffected --
        # it just means "at least this extreme," not a precise
        # interpolation -- so this is suppressed as benign rather than
        # left to clutter every caller's output for something not
        # actionable on their end.
        warnings.filterwarnings("ignore", category=InterpolationWarning)
        statistic, p_value, _, critical_values = kpss(series.dropna(), regression="c", nlags="auto")

    return StationarityTestResult(
        test_name="KPSS", statistic=float(statistic), p_value=float(p_value),
        is_stationary=bool(p_value >= 0.05), critical_values=critical_values,
    )


def combined_stationarity_verdict(series: pd.Series) -> dict:
    """File 28 Section 3.3's four-way interpretation table, as a
    structured verdict instead of a printed reference table. Because ADF
    and KPSS have OPPOSITE null hypotheses, agreement between them is
    much stronger evidence than either result alone.
    """
    adf_result = adf_test(series)
    kpss_result = kpss_test(series)

    if adf_result.is_stationary and kpss_result.is_stationary:
        verdict = "confidently_stationary"
    elif not adf_result.is_stationary and not kpss_result.is_stationary:
        verdict = "confidently_non_stationary"
    elif adf_result.is_stationary and not kpss_result.is_stationary:
        verdict = "disagreement_likely_trend_stationary"
    else:
        verdict = "disagreement_borderline_or_low_power"

    return {"adf": adf_result, "kpss": kpss_result, "verdict": verdict}


def difference_scratch(series: pd.Series, order: int = 1) -> pd.Series:
    """First-order differencing, applied `order` times. File 28 Section 4.
    Directly removes a LINEAR trend: a linear trend's period-over-period
    change is constant, so differencing converts "the level is climbing
    steadily" into "the change is roughly constant" -- stationary, if the
    original trend was genuinely linear. Does nothing structural about
    non-constant VARIANCE (a different fix -- log/Box-Cox transform, or
    an explicitly variance-modeling approach like GARCH -- is needed for that).
    """
    result = series.copy()
    for _ in range(order):
        result = result.diff().dropna()
    return result


def compute_acf_pacf(series: pd.Series, nlags: int = 20) -> pd.DataFrame:
    """File 28 Section 5. Structured (lag, acf, pacf) table instead of
    two separately-printed arrays -- makes the AR-vs-MA comparison at
    each lag directly readable in one place.
    """
    acf_values = acf(series.dropna(), nlags=nlags)
    pacf_values = pacf(series.dropna(), nlags=nlags)
    return pd.DataFrame({
        "lag": range(nlags + 1),
        "acf": acf_values,
        "pacf": pacf_values,
    })


def diagnose_ar_or_ma_signature(series: pd.Series, nlags: int = 15) -> dict:
    """File 28 Section 5's AR-vs-MA diagnostic rule, made computational:

    AR-like: ACF decays GRADUALLY across many lags (indirect dependence
        chain propagates weakened correlation forward); PACF cuts off
        SHARPLY (drops inside the ~95% confidence band, roughly
        +/-1.96/sqrt(n), and stays there) after a small number of lags,
        since PACF specifically strips out the indirect chain.
    MA-like: the INVERSE signature -- ACF cuts off sharply, PACF decays
        gradually.

    "Cuts off after lag k" here means: the first lag, scanning forward
    from lag 1, where the value falls inside the confidence band AND
    stays inside it for the rest of the lags checked (a single band
    crossing amid otherwise-significant lags doesn't count as the cutoff).
    """
    table = compute_acf_pacf(series, nlags=nlags)
    n = len(series.dropna())
    conf_band = 1.96 / np.sqrt(n)

    def _cutoff_lag(values: np.ndarray, settle_window: int = 4) -> int:
        """First lag after which the value stays inside the confidence
        band for at least `settle_window` consecutive lags.

        Deliberately NOT "every single remaining lag must be inside the
        band": a 95% band means roughly 1 in 20 lags exceeds it by pure
        chance even for a genuinely clean process, and checking 15 lags
        gives close to even odds that at least one exceeds somewhere in
        the tail. Requiring literal zero-tolerance out to nlags makes
        this fragile to an isolated noise-driven excursion at, say, lag
        11 -- which says nothing about the process's real structure.
        This matches how ACF/PACF plots are actually read in practice:
        look for where it visibly settles and STAYS low for several
        lags, not whether an occasional far-out blip crosses the line.
        """
        # values[0] is lag 0 (always 1.0 for ACF by definition) -- skip it
        for lag in range(1, len(values)):
            window = values[lag: lag + settle_window]
            if len(window) < settle_window:
                break  # not enough remaining lags to confirm settling
            if all(abs(v) < conf_band for v in window):
                return lag
        return len(values)  # never cuts off within the lags checked

    acf_cutoff = _cutoff_lag(table["acf"].values)
    pacf_cutoff = _cutoff_lag(table["pacf"].values)

    if pacf_cutoff < acf_cutoff:
        signature = "AR-like"
    elif acf_cutoff < pacf_cutoff:
        signature = "MA-like"
    else:
        signature = "ambiguous"

    return {
        "acf_cutoff_lag": acf_cutoff,
        "pacf_cutoff_lag": pacf_cutoff,
        "signature": signature,
        "acf_pacf_table": table,
    }
