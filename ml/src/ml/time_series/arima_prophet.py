"""
ARIMA and Prophet, matching file 29.

The capstone of this module (forecast_error_zscore_anomaly_signal) is
the concrete "forecasting as anomaly-detection feeder" mechanism file 29
Sections 6-7 build toward: fit a forecaster on history BEFORE a window,
forecast forward, and score how far actual values deviate from what
this SPECIFIC series' own established pattern would have predicted --
a more targeted, better-conditioned signal than an unconditional
volatility or return threshold, because it's already conditioned on
this stock's own normal dynamics.

Consistent with file 28's closing argument: this module forecasts and
scores deviation from forecast. It does not replace the classification/
anomaly-detection system in models/ and anomaly/ -- it feeds it.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

# Prophet's cmdstanpy backend logs "Chain [1] start/done processing" at
# INFO level on every fit -- accurate, but pure noise for a caller of
# this module. Set at import time (not per-function) since cmdstanpy
# appears to (re)configure its own logging the first time a model is
# actually fit within a process, which can override a level set only
# right before that first fit.
import logging
_cmdstanpy_logger = logging.getLogger("cmdstanpy")
_cmdstanpy_logger.setLevel(logging.WARNING)
_cmdstanpy_logger.propagate = False


def fit_ar_model_scratch(series: pd.Series, p: int) -> np.ndarray:
    """Fits y_t = c + phi_1*y_{t-1} + ... + phi_p*y_{t-p} + error_t via
    OLS. File 29 Section 2.1 -- genuinely just linear regression (file
    21's machinery), with the "features" being the series' own past
    values. Returns [intercept, phi_1, ..., phi_p].
    """
    y = series.values
    n = len(y)

    X_lagged = np.zeros((n - p, p))
    for i in range(p):
        X_lagged[:, i] = y[p - i - 1: n - i - 1]
    y_target = y[p:]

    X_with_intercept = np.column_stack([np.ones(len(X_lagged)), X_lagged])
    coefficients, _, _, _ = np.linalg.lstsq(X_with_intercept, y_target, rcond=None)
    return coefficients


def demonstrate_ma_process(n: int = 300, theta: float = 0.6, random_state: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """A genuine MA(1) process: y_t = error_t + theta * error_{t-1}.
    File 29 Section 2.2. Critically, y_t depends on ERROR TERMS, not
    directly-observable past y values the way AR does -- this is why MA
    fitting needs maximum likelihood (iterative), not AR's closed-form
    OLS: the "error_{t-1}" term is the model's OWN residual from the
    previous step, which depends on the very parameters being estimated.
    """
    rng = np.random.RandomState(random_state)
    errors = rng.normal(0, 1, n + 1)
    y = errors[1:] + theta * errors[:-1]
    return y, errors


@dataclass
class ArimaFitResult:
    fitted_model: object
    rmse: float
    mae: float
    forecast_mean: pd.Series
    forecast_conf_int: pd.DataFrame


def fit_arima_and_forecast(
    train_series: pd.Series, test_series: pd.Series, order: tuple[int, int, int],
) -> ArimaFitResult:
    """File 29 Sections 3.1-3.2. d should come from stationarity testing
    (time_series.stationarity) -- NOT re-derived here; this function
    takes order as given, matching file 29's own point that d=1 was
    chosen directly from file 28's testing, not rediscovered.
    """
    model = ARIMA(train_series, order=order)
    fitted = model.fit()

    forecast_result = fitted.get_forecast(steps=len(test_series))
    forecast_mean = forecast_result.predicted_mean
    forecast_conf_int = forecast_result.conf_int(alpha=0.05)

    rmse = float(np.sqrt(mean_squared_error(test_series, forecast_mean)))
    mae = float(mean_absolute_error(test_series, forecast_mean))

    return ArimaFitResult(
        fitted_model=fitted, rmse=rmse, mae=mae,
        forecast_mean=forecast_mean, forecast_conf_int=forecast_conf_int,
    )


def grid_search_arima_order(
    series: pd.Series, p_range: range, d_range: list[int], q_range: range,
) -> pd.DataFrame:
    """File 29 Section 4. Exhaustively fits ARIMA across (p,d,q)
    combinations, ranking by AIC -- a systematic version of what
    coefficient p-values and file 28's ACF/PACF plots already point
    toward manually.

    Combinations that raise an exception are skipped, not allowed to
    crash the whole search -- AND combinations where statsmodels emits a
    ConvergenceWarning are ALSO skipped, not just ones that raise.
    Confirmed directly (not assumed): statsmodels can return a
    "successful" fit object with a real, plausible-looking AIC even when
    optimization completely failed to converge -- it only warns, it
    doesn't raise. Silently ranking a non-converged fit's meaningless AIC
    alongside genuine fits (where it can even look spuriously BETTER) is
    a real correctness risk for an automated order-selection function,
    not just a cosmetic one.
    """
    import warnings
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    results = []
    for p, d, q in itertools.product(p_range, d_range, q_range):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", category=ConvergenceWarning)
                model = ARIMA(series, order=(p, d, q))
                fitted = model.fit()
                if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    continue  # fit "succeeded" but didn't actually converge -- AIC/BIC meaningless
            results.append({"order": (p, d, q), "aic": fitted.aic, "bic": fitted.bic})
        except Exception:
            continue
    if not results:
        raise RuntimeError(
            "No (p,d,q) combination converged. Widen the search ranges, or "
            "check the series for issues (e.g. insufficient length, extreme "
            "values) that could be preventing convergence across the board."
        )
    return pd.DataFrame(results).sort_values("aic").reset_index(drop=True)


def fit_prophet_and_forecast(
    train_series: pd.Series, test_series: pd.Series, **prophet_kwargs,
):
    """File 29 Section 5. Thin wrapper: builds the ds/y DataFrame Prophet
    requires, fits, forecasts len(test_series) steps ahead, and reports
    RMSE/MAE on the SAME metrics as fit_arima_and_forecast, so the two
    are directly comparable on one series.
    """
    import logging
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    from prophet import Prophet

    prophet_df = pd.DataFrame({"ds": train_series.index, "y": train_series.values})
    defaults = dict(seasonality_mode="additive", daily_seasonality=False)
    model = Prophet(**{**defaults, **prophet_kwargs})
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=len(test_series))
    forecast = model.predict(future)
    forecast_test_period = forecast.iloc[len(train_series):]

    rmse = float(np.sqrt(mean_squared_error(test_series.values, forecast_test_period["yhat"].values)))
    mae = float(mean_absolute_error(test_series.values, forecast_test_period["yhat"].values))

    return {
        "fitted_model": model, "forecast": forecast,
        "forecast_test_period": forecast_test_period, "rmse": rmse, "mae": mae,
    }


def changepoint_sensitivity_sweep(
    train_series: pd.Series, test_series: pd.Series, scales: list[float] | None = None,
) -> pd.DataFrame:
    """File 29 Section 5.1. changepoint_prior_scale controls how flexible
    Prophet's trend is allowed to be -- small values keep the trend
    nearly rigid (few changepoints meaningfully used, even if genuine
    shifts exist); large values let it bend sharply and often, which can
    fit training data closely but risks fitting noise as if it were
    genuine trend shifts. The same overfitting-via-flexibility tradeoff
    as file 22's max_depth, via a completely different mechanism.
    """
    from prophet import Prophet
    import logging
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

    if scales is None:
        scales = [0.001, 0.05, 0.5]

    prophet_df = pd.DataFrame({"ds": train_series.index, "y": train_series.values})
    rows = []
    for scale in scales:
        model = Prophet(changepoint_prior_scale=scale, daily_seasonality=False)
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=len(test_series))
        forecast = model.predict(future)
        forecast_test = forecast.iloc[len(train_series):]
        rmse = float(np.sqrt(mean_squared_error(test_series.values, forecast_test["yhat"].values)))
        n_significant_changepoints = int(np.sum(np.abs(model.params["delta"].mean(axis=0)) > 0.01))
        rows.append({
            "changepoint_prior_scale": scale, "test_rmse": rmse,
            "n_significant_changepoints": n_significant_changepoints,
        })
    return pd.DataFrame(rows)


def forecast_error_zscore_anomaly_signal(
    full_series: pd.Series, train_cutoff: int, event_window: tuple[int, int],
    baseline_windows: list[tuple[int, int]], order: tuple[int, int, int] = (1, 1, 1),
) -> dict:
    """File 29 Sections 6-7's capstone: THE concrete "forecasting as
    anomaly-detection feeder" mechanism.

    Fits ARIMA on data strictly BEFORE train_cutoff, forecasts forward
    through the full remaining series, and expresses forecast error
    DURING event_window as a z-score relative to BASELINE (non-event)
    forecast error measured over baseline_windows. This is file 27's
    anomaly-scoring logic (rolling z-score), applied to FORECAST ERROR
    rather than raw price/volume directly.

    baseline_windows must be given EXPLICITLY and should stay reasonably
    close in horizon-distance to event_window -- e.g. file 29 Section 6's
    own baseline is forecast_error[:10] + forecast_error[17:30], not "the
    rest of the whole horizon." This matters mechanically, not just
    stylistically: forecast error for ANY multi-step forecast grows with
    horizon distance (each step's uncertainty compounds into the next,
    file 29 Section 3.2), which has nothing to do with anomalies. Using
    an unbounded "everything else" baseline pulls in far-future errors
    that are naturally much larger for reasons unrelated to any event,
    inflating the baseline and masking a real, local spike right at the
    event -- confirmed directly while building this function: an
    unbounded 60-step-horizon baseline suppressed an obvious, visually
    clear local spike down to a z-score of ~1.4, while a properly bounded
    near-term baseline recovers a z-score of several sigma for the exact
    same event.

    The advantage over an unconditional threshold: forecast error is
    already conditioned on THIS series' own established drift and
    autocorrelation -- ARIMA's forecast for tomorrow already accounts
    for this stock's typical dynamics, so a large forecast error
    specifically means "this deviated from what THIS STOCK'S OWN
    pattern would have predicted," a more targeted signal than a raw
    volatility or return threshold applied the same way to every stock.

    event_window and baseline_windows are given as (start, end) INDICES
    relative to the forecast horizon (i.e., relative to train_cutoff),
    not the full series.
    """
    train_part = full_series.iloc[:train_cutoff]
    horizon_length = len(full_series) - train_cutoff

    model = ARIMA(train_part, order=order)
    fitted = model.fit()
    forecast_result = fitted.get_forecast(steps=horizon_length)
    forecast_values = forecast_result.predicted_mean.values
    actual_values = full_series.iloc[train_cutoff:].values

    forecast_error = np.abs(actual_values - forecast_values)

    event_start, event_end = event_window
    event_errors = forecast_error[event_start:event_end]
    baseline_errors = np.concatenate([forecast_error[start:end] for start, end in baseline_windows])

    if len(baseline_errors) == 0:
        raise ValueError("baseline_windows produced no baseline observations -- check the window bounds.")
    baseline_mean = baseline_errors.mean()
    baseline_std = baseline_errors.std()
    if baseline_std == 0:
        raise ValueError(
            "Baseline forecast error has zero variance -- cannot compute a "
            "meaningful z-score. This is unusual for real data; check the "
            "series for a data quality issue (e.g. a constant-value stretch)."
        )
    event_zscores = (event_errors - baseline_mean) / baseline_std

    return {
        "forecast_error": forecast_error,
        "event_zscores": event_zscores,
        "baseline_mean_error": float(baseline_mean),
        "baseline_std_error": float(baseline_std),
        "event_mean_zscore": float(event_zscores.mean()),
        "max_event_zscore": float(np.max(np.abs(event_zscores))),
    }
