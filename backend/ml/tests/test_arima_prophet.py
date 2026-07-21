import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.arima.model import ARIMA

from ml.time_series.arima_prophet import (
    fit_ar_model_scratch,
    demonstrate_ma_process,
    fit_arima_and_forecast,
    grid_search_arima_order,
    fit_prophet_and_forecast,
    changepoint_sensitivity_sweep,
    forecast_error_zscore_anomaly_signal,
)


def make_random_walk(random_state=150, n=500, drift=0.03):
    rng = np.random.RandomState(random_state)
    dates = pd.date_range("2024-06-01", periods=n, freq="D")
    close = 100 + np.cumsum(rng.normal(drift, 1, n))
    return pd.Series(close, index=dates)


class TestFitArModelScratch:
    def test_matches_statsmodels_ar_coefficients_approximately(self):
        """File 29 Section 2.1's claim: AR fitting IS ordinary least
        squares -- coefficients should closely match statsmodels' own
        AR-only fit (ARIMA with d=0, q=0)."""
        series = make_random_walk().diff().dropna()
        scratch_coeffs = fit_ar_model_scratch(series, p=2)

        statsmodels_model = ARIMA(series, order=(2, 0, 0), trend="c").fit()
        # statsmodels orders params as [const, ar.L1, ar.L2, sigma2] --
        # compare just the intercept and AR coefficients
        statsmodels_coeffs = statsmodels_model.params.values[:3]

        assert np.allclose(scratch_coeffs, statsmodels_coeffs, atol=0.05)

    def test_returns_p_plus_one_coefficients(self):
        series = make_random_walk().diff().dropna()
        coeffs = fit_ar_model_scratch(series, p=3)
        assert len(coeffs) == 4  # intercept + 3 AR coefficients

    def test_recovers_known_ar1_coefficient(self):
        """A cleanly-constructed AR(1) with phi=0.7 and no other
        structure should recover phi close to 0.7."""
        rng = np.random.RandomState(1)
        n = 2000
        y = np.zeros(n)
        for i in range(1, n):
            y[i] = 0.7 * y[i - 1] + rng.normal(0, 1)
        coeffs = fit_ar_model_scratch(pd.Series(y), p=1)
        assert coeffs[1] == pytest.approx(0.7, abs=0.05)


class TestDemonstrateMaProcess:
    def test_returns_correct_length(self):
        y, errors = demonstrate_ma_process(n=300, theta=0.6, random_state=1)
        assert len(y) == 300
        assert len(errors) == 301

    def test_deterministic_given_random_state(self):
        y1, _ = demonstrate_ma_process(n=200, theta=0.5, random_state=7)
        y2, _ = demonstrate_ma_process(n=200, theta=0.5, random_state=7)
        assert np.array_equal(y1, y2)

    def test_lag1_autocorrelation_matches_theoretical_ma1_formula(self):
        """A genuine MA(1)'s theoretical lag-1 autocorrelation is
        theta / (1 + theta^2) -- checked against the actual generated
        series, not just assumed from the construction."""
        y, _ = demonstrate_ma_process(n=5000, theta=0.6, random_state=1)
        empirical_lag1_autocorr = np.corrcoef(y[:-1], y[1:])[0, 1]
        theoretical = 0.6 / (1 + 0.6 ** 2)
        assert empirical_lag1_autocorr == pytest.approx(theoretical, abs=0.03)

    def test_lag2_autocorrelation_near_zero(self):
        """A genuine MA(1) has NO direct or indirect dependence beyond
        lag 1 -- lag-2 autocorrelation should be close to zero."""
        y, _ = demonstrate_ma_process(n=5000, theta=0.6, random_state=1)
        empirical_lag2_autocorr = np.corrcoef(y[:-2], y[2:])[0, 1]
        assert abs(empirical_lag2_autocorr) < 0.05


class TestFitArimaAndForecast:
    def test_produces_finite_forecast_and_metrics(self):
        series = make_random_walk()
        train, test = series.iloc[:400], series.iloc[400:]
        result = fit_arima_and_forecast(train, test, order=(2, 1, 2))
        assert np.isfinite(result.rmse)
        assert np.isfinite(result.mae)
        assert len(result.forecast_mean) == len(test)

    def test_confidence_interval_widens_with_horizon(self):
        """File 29 Section 3.2's claim: each forecasted step compounds
        uncertainty into every subsequent step, so the confidence
        interval should widen further out."""
        series = make_random_walk()
        train, test = series.iloc[:400], series.iloc[400:]
        result = fit_arima_and_forecast(train, test, order=(2, 1, 2))
        widths = result.forecast_conf_int.iloc[:, 1] - result.forecast_conf_int.iloc[:, 0]
        assert widths.iloc[-1] > widths.iloc[0]

    def test_forecast_index_aligns_with_test_series(self):
        series = make_random_walk()
        train, test = series.iloc[:400], series.iloc[400:]
        result = fit_arima_and_forecast(train, test, order=(1, 1, 1))
        assert list(result.forecast_mean.index) == list(test.index)


class TestGridSearchArimaOrder:
    def test_returns_results_sorted_by_aic(self):
        series = make_random_walk(n=300)
        result = grid_search_arima_order(series, p_range=range(0, 3), d_range=[1], q_range=range(0, 3))
        aics = result["aic"].values
        assert (aics[:-1] <= aics[1:]).all()

    def test_all_returned_orders_have_requested_d(self):
        series = make_random_walk(n=300)
        result = grid_search_arima_order(series, p_range=range(0, 2), d_range=[1], q_range=range(0, 2))
        for order in result["order"]:
            assert order[1] == 1

    def test_raises_when_nothing_converges(self):
        """An empty/degenerate series should leave no successful fits."""
        degenerate_series = pd.Series([1.0] * 5)
        with pytest.raises(RuntimeError, match="No .* converged"):
            grid_search_arima_order(degenerate_series, p_range=range(5, 8), d_range=[2], q_range=range(5, 8))


class TestFitProphetAndForecast:
    def test_produces_finite_forecast_and_metrics(self):
        series = make_random_walk(n=300)
        train, test = series.iloc[:250], series.iloc[250:]
        result = fit_prophet_and_forecast(train, test, weekly_seasonality=True, yearly_seasonality=False)
        assert np.isfinite(result["rmse"])
        assert np.isfinite(result["mae"])
        assert len(result["forecast_test_period"]) == len(test)


class TestChangepointSensitivitySweep:
    def test_returns_one_row_per_scale(self):
        series = make_random_walk(n=300)
        train, test = series.iloc[:250], series.iloc[250:]
        result = changepoint_sensitivity_sweep(train, test, scales=[0.01, 0.5])
        assert len(result) == 2

    def test_includes_expected_columns(self):
        series = make_random_walk(n=300)
        train, test = series.iloc[:250], series.iloc[250:]
        result = changepoint_sensitivity_sweep(train, test, scales=[0.05])
        assert {"changepoint_prior_scale", "test_rmse", "n_significant_changepoints"}.issubset(result.columns)


class TestForecastErrorZscoreAnomalySignal:
    def test_detects_elevated_error_during_injected_pump_and_dump(self):
        """File 29 Section 6's exact construction and claim: forecast
        error during a genuine pump-and-dump event should register as a
        clear outlier (high |z-score|) relative to a NEAR-TERM baseline
        -- matching file 29's own baseline windows (forecast_error[:10]
        and forecast_error[17:30]), not the entire remaining horizon
        (see test_unbounded_baseline_masks_the_same_event below for why
        that specific distinction matters)."""
        rng = np.random.RandomState(152)
        clean_history = 100 + np.cumsum(rng.normal(0.02, 1, 250))
        extension_normal = 100 + np.cumsum(rng.normal(0.02, 1, 50)) + (clean_history[-1] - 100)
        full_series_values = np.concatenate([clean_history, extension_normal])
        full_series_values[200:203] += [4, 7, 10]
        full_series_values[203:207] -= [5, 6, 4, 2]
        full_series = pd.Series(full_series_values)

        result = forecast_error_zscore_anomaly_signal(
            full_series, train_cutoff=190, event_window=(10, 17),
            baseline_windows=[(0, 10), (17, 30)], order=(1, 1, 1),
        )
        assert result["max_event_zscore"] > 2.0  # a clear statistical outlier

    def test_unbounded_baseline_masks_the_same_event(self):
        """Documents the exact failure mode found while building this
        function: forecast error grows naturally with horizon distance
        for ANY multi-step forecast (nothing to do with anomalies). Using
        the ENTIRE remaining horizon as baseline pulls in far-future
        errors that are naturally much larger, inflating the baseline and
        masking a real, local spike. Same event, same data, same
        event_window as the test above -- only the baseline_windows
        differ -- and the z-score for a visually obvious spike collapses
        to a small, unremarkable value.
        """
        rng = np.random.RandomState(152)
        clean_history = 100 + np.cumsum(rng.normal(0.02, 1, 250))
        extension_normal = 100 + np.cumsum(rng.normal(0.02, 1, 50)) + (clean_history[-1] - 100)
        full_series_values = np.concatenate([clean_history, extension_normal])
        full_series_values[200:203] += [4, 7, 10]
        full_series_values[203:207] -= [5, 6, 4, 2]
        full_series = pd.Series(full_series_values)

        near_term_result = forecast_error_zscore_anomaly_signal(
            full_series, train_cutoff=190, event_window=(10, 17),
            baseline_windows=[(0, 10), (17, 30)], order=(1, 1, 1),
        )
        unbounded_result = forecast_error_zscore_anomaly_signal(
            full_series, train_cutoff=190, event_window=(10, 17),
            baseline_windows=[(0, 10), (17, 60)], order=(1, 1, 1),
        )
        assert near_term_result["max_event_zscore"] > unbounded_result["max_event_zscore"] + 1.0

    def test_raises_when_baseline_windows_produce_no_observations(self):
        series = make_random_walk(n=300)
        with pytest.raises(ValueError, match="no baseline observations"):
            forecast_error_zscore_anomaly_signal(
                series, train_cutoff=250, event_window=(0, 50), baseline_windows=[(50, 50)]
            )

    def test_returns_expected_keys(self):
        series = make_random_walk(n=300)
        result = forecast_error_zscore_anomaly_signal(
            series, train_cutoff=250, event_window=(10, 20), baseline_windows=[(0, 10), (20, 40)]
        )
        for key in ["forecast_error", "event_zscores", "baseline_mean_error", "max_event_zscore"]:
            assert key in result

    def test_normal_non_event_window_gives_modest_zscore(self):
        """A window with no injected event should NOT show an extreme
        z-score -- confirms the signal isn't just noisy/always-triggering."""
        series = make_random_walk(n=300, random_state=99)
        result = forecast_error_zscore_anomaly_signal(
            series, train_cutoff=250, event_window=(10, 20), baseline_windows=[(0, 10), (20, 40)]
        )
        assert result["max_event_zscore"] < 3.0
