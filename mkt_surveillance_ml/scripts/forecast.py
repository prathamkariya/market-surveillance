#!/usr/bin/env python3
"""
Forecasting entrypoint for mkt-surveillance-ml (files 28-30: stationarity,
ARIMA/Prophet, LSTM).

This does NOT replace train.py's MultiPatternDetector / IsolationForest --
per file 28's own closing argument, forecasting models and their
stationarity/differencing machinery do not themselves flag manipulation as
anomalous. What they DO offer: a forecast of what a stock's price "should"
look like given its own history, and forecast_error_zscore mode turns
deviation from that forecast into a signal that can feed the anomaly
detection system built in train.py -- a signal already conditioned on this
specific stock's own normal dynamics, which a raw volatility/volume
threshold applied identically to every stock is not.

FOUR MODES:

  check-stationarity     Runs ADF + KPSS + AR/MA signature diagnosis on a
                          series. Use this FIRST, before fitting ARIMA --
                          it tells you what differencing order (d) to use.

  fit-arima               Grid-searches (p,d,q) by AIC, then fits the best
                           order and reports forecast RMSE/MAE on a held-out
                           tail of the series.

  fit-prophet              Fits Prophet, sweeps changepoint_prior_scale,
                            reports forecast RMSE/MAE.

  fit-lstm                 Fits an LSTM forecaster. Slower than the other
                           two; CPU-only training on real data can take
                           several minutes depending on --epochs and series
                           length. No GPU/Colab required for the data sizes
                           this package targets, but it will be the slowest
                           of the three to train.

  forecast-error-zscore     Fits ARIMA on data before --train-cutoff, then
                             scores how far ACTUAL values in --event-window
                             deviate from forecast, relative to a NEAR-TERM
                             baseline window you specify. This is the
                             anomaly-detection-feeder signal -- read the
                             module docstring in time_series/arima_prophet.py
                             for why the baseline window must stay close in
                             horizon-distance to the event window (an
                             unbounded baseline silently masks real events).

USAGE
-----
  python scripts/forecast.py check-stationarity --csv my_data.csv
  python scripts/forecast.py fit-arima --csv my_data.csv --test-size 0.2
  python scripts/forecast.py fit-prophet --csv my_data.csv
  python scripts/forecast.py fit-lstm --csv my_data.csv --epochs 30
  python scripts/forecast.py forecast-error-zscore --csv my_data.csv \\
      --train-cutoff 190 --event-window 200 207 --baseline-windows 180 200 207 230

  Omit --csv to run against generated synthetic data instead (useful to
  verify your environment/install before touching real data).

CSV FORMAT
----------
  Required columns: a date column (first column, or named 'date'), 'close'.
  ('volume' is also read if present but not required for these modes --
  unlike train.py, forecasting here operates on a single price series.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _common import load_price_series

from mkt_surveillance_ml.data.synthetic import generate_synthetic_market_data
from mkt_surveillance_ml.time_series.stationarity import (
    adf_test,
    kpss_test,
    combined_stationarity_verdict,
    diagnose_ar_or_ma_signature,
    difference_scratch,
)
from mkt_surveillance_ml.time_series.arima_prophet import (
    grid_search_arima_order,
    fit_arima_and_forecast,
    fit_prophet_and_forecast,
    changepoint_sensitivity_sweep,
    forecast_error_zscore_anomaly_signal,
)
from mkt_surveillance_ml.time_series.lstm import fit_lstm_forecaster


def _load_series(args: argparse.Namespace) -> pd.Series:
    if args.csv:
        return load_price_series(args.csv)
    print(f"No --csv given -- generating {args.n_days} days of synthetic price data instead.")
    df = generate_synthetic_market_data(n_days=args.n_days, random_state=args.random_state)
    return df["close"]


def run_check_stationarity(args: argparse.Namespace) -> None:
    series = _load_series(args)
    print(f"Series length: {len(series)}")

    verdict = combined_stationarity_verdict(series)
    print(f"\nADF:  statistic={verdict['adf'].statistic:.4f}, p={verdict['adf'].p_value:.4f}, "
          f"is_stationary={verdict['adf'].is_stationary}")
    print(f"KPSS: statistic={verdict['kpss'].statistic:.4f}, p={verdict['kpss'].p_value:.4f}, "
          f"is_stationary={verdict['kpss'].is_stationary}")
    print(f"\nVerdict: {verdict['verdict']}")

    recommended_d = 0
    working_series = series
    if verdict["verdict"] in {"confidently_non_stationary", "disagreement_borderline_or_low_power"}:
        print("\nSeries appears non-stationary -- checking differencing...")
        for d in [1, 2]:
            working_series = difference_scratch(series, order=d)
            check = adf_test(working_series)
            print(f"  after {d} order(s) of differencing: ADF is_stationary={check.is_stationary}")
            if check.is_stationary:
                recommended_d = d
                break
    print(f"\nRecommended ARIMA d: {recommended_d}")

    signature = diagnose_ar_or_ma_signature(working_series if recommended_d > 0 else series, nlags=15)
    print(f"ACF/PACF signature: {signature['signature']} "
          f"(acf_cutoff={signature['acf_cutoff_lag']}, pacf_cutoff={signature['pacf_cutoff_lag']})")
    print(
        f"\n(Signature is a heuristic on ONE finite sample -- 'ambiguous' is a "
        f"legitimate result, not an error. Use it as a starting point for "
        f"fit-arima's grid search, not a final answer.)"
    )


def run_fit_arima(args: argparse.Namespace) -> None:
    series = _load_series(args)
    split = int(len(series) * (1 - args.test_size))
    train, test = series.iloc[:split], series.iloc[split:]
    print(f"train: {len(train)}, test: {len(test)}")

    print(f"\nGrid-searching ARIMA order (p in 0-{args.max_p}, d in {args.d_values}, q in 0-{args.max_q})...")
    search_result = grid_search_arima_order(
        train, p_range=range(0, args.max_p + 1), d_range=args.d_values, q_range=range(0, args.max_q + 1)
    )
    print(search_result.head(5).to_string(index=False))

    best_order = tuple(search_result.iloc[0]["order"])
    print(f"\nFitting best order {best_order} and forecasting {len(test)} steps...")
    result = fit_arima_and_forecast(train, test, order=best_order)
    print(f"Test RMSE: {result.rmse:.4f}")
    print(f"Test MAE:  {result.mae:.4f}")


def run_fit_prophet(args: argparse.Namespace) -> None:
    series = _load_series(args)
    split = int(len(series) * (1 - args.test_size))
    train, test = series.iloc[:split], series.iloc[split:]
    print(f"train: {len(train)}, test: {len(test)}")

    print("\nSweeping changepoint_prior_scale...")
    sweep = changepoint_sensitivity_sweep(train, test, scales=[0.001, 0.05, 0.5])
    print(sweep.to_string(index=False))

    best_scale = float(sweep.sort_values("test_rmse").iloc[0]["changepoint_prior_scale"])
    print(f"\nFitting with best scale ({best_scale})...")
    result = fit_prophet_and_forecast(train, test, changepoint_prior_scale=best_scale)
    print(f"Test RMSE: {result['rmse']:.4f}")
    print(f"Test MAE:  {result['mae']:.4f}")


def run_fit_lstm(args: argparse.Namespace) -> None:
    series = _load_series(args)
    split = int(len(series) * (1 - args.test_size))
    train, test = series.values[:split], series.values[split:]
    print(f"train: {len(train)}, test: {len(test)}")
    print(f"Training LSTM ({args.epochs} epochs, seq_length={args.seq_length}) -- this is the slowest of the three forecasters...")

    result = fit_lstm_forecaster(
        train, test, seq_length=args.seq_length, epochs=args.epochs, verbose=1,
    )
    print(f"\nTest RMSE: {result.rmse:.4f}")
    print(f"Test MAE:  {result.mae:.4f}")
    print(
        "\n(If this barely beats ARIMA/Prophet, that's informative, not a "
        "failure -- file 30's own point: a simple linear structure with "
        "far fewer parameters matching a far more expressive model's "
        "performance is real evidence the underlying pattern doesn't need "
        "LSTM's extra capacity.)"
    )


def run_forecast_error_zscore(args: argparse.Namespace) -> None:
    series = _load_series(args)
    if len(series) <= args.train_cutoff:
        print(f"ERROR: train_cutoff ({args.train_cutoff}) must be less than series length ({len(series)}).",
              file=sys.stderr)
        sys.exit(1)

    baseline_windows = [
        (args.baseline_windows[i], args.baseline_windows[i + 1])
        for i in range(0, len(args.baseline_windows), 2)
    ]
    event_window = tuple(args.event_window)

    print(f"Fitting ARIMA{args.order} on the first {args.train_cutoff} points, "
          f"forecasting the remaining {len(series) - args.train_cutoff}...")
    result = forecast_error_zscore_anomaly_signal(
        series.reset_index(drop=True), train_cutoff=args.train_cutoff,
        event_window=event_window, baseline_windows=baseline_windows, order=tuple(args.order),
    )

    print(f"\nBaseline mean forecast error: {result['baseline_mean_error']:.4f}")
    print(f"Baseline std forecast error:  {result['baseline_std_error']:.4f}")
    print(f"Event window z-scores: {np.round(result['event_zscores'], 3)}")
    print(f"Max |z-score| in event window: {result['max_event_zscore']:.3f}")

    if result["max_event_zscore"] > 3:
        print("\n-> Event window shows a CLEAR statistical outlier relative to baseline forecast error.")
    elif result["max_event_zscore"] > 2:
        print("\n-> Event window shows a MODERATE deviation -- worth a closer look.")
    else:
        print("\n-> Event window forecast error is not notably elevated relative to baseline.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "mode", choices=["check-stationarity", "fit-arima", "fit-prophet", "fit-lstm", "forecast-error-zscore"],
    )
    parser.add_argument("--csv", type=str, default=None, help="Path to real OHLCV CSV. Omit to use synthetic data.")
    parser.add_argument("--n-days", type=int, default=500, dest="n_days", help="(synthetic data only)")
    parser.add_argument("--random-state", type=int, default=42, dest="random_state")
    parser.add_argument("--test-size", type=float, default=0.2, dest="test_size")
    parser.add_argument("--max-p", type=int, default=3, dest="max_p", help="(fit-arima only)")
    parser.add_argument("--max-q", type=int, default=3, dest="max_q", help="(fit-arima only)")
    parser.add_argument("--d-values", nargs="+", type=int, default=[0, 1], dest="d_values", help="(fit-arima only)")
    parser.add_argument("--seq-length", type=int, default=30, dest="seq_length", help="(fit-lstm only)")
    parser.add_argument("--epochs", type=int, default=30, help="(fit-lstm only)")
    parser.add_argument(
        "--train-cutoff", type=int, default=None, dest="train_cutoff",
        help="(forecast-error-zscore only) Index to split history/forecast-horizon at.",
    )
    parser.add_argument(
        "--event-window", nargs=2, type=int, default=None, dest="event_window",
        help="(forecast-error-zscore only) start end indices, RELATIVE TO train_cutoff, e.g. --event-window 10 17",
    )
    parser.add_argument(
        "--baseline-windows", nargs="+", type=int, default=None, dest="baseline_windows",
        help="(forecast-error-zscore only) pairs of start end indices, e.g. --baseline-windows 0 10 17 30",
    )
    parser.add_argument("--order", nargs=3, type=int, default=[1, 1, 1], help="(forecast-error-zscore only) p d q")

    args = parser.parse_args()

    if args.mode == "check-stationarity":
        run_check_stationarity(args)
    elif args.mode == "fit-arima":
        run_fit_arima(args)
    elif args.mode == "fit-prophet":
        run_fit_prophet(args)
    elif args.mode == "fit-lstm":
        run_fit_lstm(args)
    elif args.mode == "forecast-error-zscore":
        if args.train_cutoff is None or args.event_window is None or args.baseline_windows is None:
            print(
                "ERROR: forecast-error-zscore requires --train-cutoff, --event-window, "
                "and --baseline-windows. Example:\n"
                "  python scripts/forecast.py forecast-error-zscore --csv data.csv "
                "--train-cutoff 190 --event-window 10 17 --baseline-windows 0 10 17 30",
                file=sys.stderr,
            )
            sys.exit(1)
        run_forecast_error_zscore(args)


if __name__ == "__main__":
    main()
