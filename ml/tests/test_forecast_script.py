import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "forecast.py"


import tempfile

def run_script(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
    """Run a script and capture output robustly via temp files to avoid Windows pipe limits/debugpy issues."""
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as out, tempfile.TemporaryFile(mode="w+", encoding="utf-8") as err:
        process = subprocess.Popen(
            [sys.executable, str(SCRIPT_PATH)] + args,
            stdout=out, stderr=err, text=True, encoding="utf-8"
        )
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise
        
        out.seek(0)
        err.seek(0)
        return subprocess.CompletedProcess(
            process.args, process.returncode, out.read(), err.read()
        )


@pytest.fixture(scope="module")
def fake_real_csv(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("forecast_fake_data")
    rng = np.random.RandomState(11)
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    close = 80 + np.cumsum(rng.normal(0.05, 1, n))
    df = pd.DataFrame({"date": dates, "close": close})
    path = tmp_dir / "prices.csv"
    df.to_csv(path, index=False)
    return path


class TestCheckStationarity:
    def test_runs_successfully_on_synthetic_data(self):
        result = run_script(["check-stationarity", "--n-days", "400"])
        assert result.returncode == 0, result.stderr
        assert "Recommended ARIMA d:" in result.stdout

    def test_runs_successfully_on_real_csv(self, fake_real_csv):
        result = run_script(["check-stationarity", "--csv", str(fake_real_csv)])
        assert result.returncode == 0, result.stderr
        assert "Verdict:" in result.stdout


class TestFitArima:
    def test_runs_successfully_and_reports_metrics(self):
        result = run_script(["fit-arima", "--n-days", "300", "--max-p", "2", "--max-q", "2"])
        assert result.returncode == 0, result.stderr
        assert "Test RMSE:" in result.stdout
        assert "Test MAE:" in result.stdout

    def test_runs_successfully_on_real_csv(self, fake_real_csv):
        result = run_script(["fit-arima", "--csv", str(fake_real_csv), "--max-p", "2", "--max-q", "2"])
        assert result.returncode == 0, result.stderr
        assert "Test RMSE:" in result.stdout


class TestFitProphet:
    def test_runs_successfully_and_reports_metrics(self):
        result = run_script(["fit-prophet", "--n-days", "300"])
        assert result.returncode == 0, result.stderr
        assert "Test RMSE:" in result.stdout
        assert "changepoint_prior_scale" in result.stdout


class TestFitLstm:
    def test_runs_successfully_with_low_epochs(self):
        """Low epoch count and short sequence length purely to verify
        correctness quickly -- not to verify strong forecasting accuracy."""
        result = run_script(
            ["fit-lstm", "--n-days", "250", "--epochs", "2", "--seq-length", "10"], timeout=180
        )
        assert result.returncode == 0, result.stderr
        assert "Test RMSE:" in result.stdout


class TestForecastErrorZscore:
    def test_runs_successfully_with_valid_args(self):
        result = run_script([
            "forecast-error-zscore", "--n-days", "400",
            "--train-cutoff", "300", "--event-window", "10", "20",
            "--baseline-windows", "0", "10", "20", "40",
        ])
        assert result.returncode == 0, result.stderr
        assert "Max |z-score|" in result.stdout

    def test_fails_clearly_when_required_args_missing(self):
        result = run_script(["forecast-error-zscore", "--n-days", "400"])
        assert result.returncode != 0
        assert "requires --train-cutoff" in result.stderr

    def test_detects_a_real_injected_spike(self):
        """An end-to-end check that a genuine injected deviation is
        actually flagged as elevated by the CLI, not just by the
        underlying library function directly.

        baseline_windows must stay NEAR-TERM relative to event_window --
        this is the exact lesson documented in
        time_series/arima_prophet.py's own module docstring (an earlier
        version of this test used baseline_windows reaching to step 40,
        which is far enough into the horizon that ordinary compounding
        random-walk forecast-error growth -- std grows roughly with
        sqrt(horizon), nothing to do with the injected event -- diluted
        the comparison and masked the real spike; confirmed directly
        while fixing this test).
        """
        rng = np.random.RandomState(20)
        n = 350
        close = 100 + np.cumsum(rng.normal(0.02, 1, n))
        close[300:305] += [4, 8, 12, 9, 5]
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n, freq="D"), "close": close})
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f, index=False)
            csv_path = f.name

        result = run_script([
            "forecast-error-zscore", "--csv", csv_path,
            "--train-cutoff", "290", "--event-window", "10", "17",
            "--baseline-windows", "0", "10", "17", "24",
        ])
        assert result.returncode == 0, result.stderr
        assert "CLEAR statistical outlier" in result.stdout or "MODERATE deviation" in result.stdout
