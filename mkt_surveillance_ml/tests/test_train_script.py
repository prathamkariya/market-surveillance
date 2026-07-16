import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "train.py"


def run_script(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH)] + args,
        capture_output=True, text=True, timeout=120,
    )


@pytest.fixture(scope="module")
def fake_real_csv(tmp_path_factory):
    """An independently-constructed CSV (not using the package's own
    generator) simulating real exported OHLCV data with known incidents --
    genuinely exercises the CSV-loading path rather than round-tripping
    through the same generator the rest of the package already tests.
    """
    tmp_dir = tmp_path_factory.mktemp("fake_real_data")
    rng = np.random.RandomState(7)
    n = 400
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    close = 50 + np.cumsum(rng.normal(0.01, 0.8, n))
    volume = rng.lognormal(14, 0.25, n)

    is_pump = np.zeros(n, dtype=int)
    pump_days = rng.choice(range(30, n), 25, replace=False)
    is_pump[pump_days] = 1
    close[pump_days] *= 1.05

    is_wash = np.zeros(n, dtype=int)
    remaining = [d for d in range(30, n) if d not in pump_days]
    wash_days = rng.choice(remaining, 30, replace=False)
    is_wash[wash_days] = 1
    volume[wash_days] *= 2.5

    df = pd.DataFrame({
        "date": dates, "close": close, "volume": volume,
        "is_pump_and_dump": is_pump, "is_wash_trading": is_wash,
    })
    df["is_manipulation"] = ((df["is_pump_and_dump"] == 1) | (df["is_wash_trading"] == 1)).astype(int)

    labeled_path = tmp_dir / "labeled.csv"
    df.to_csv(labeled_path, index=False)

    unlabeled_path = tmp_dir / "unlabeled.csv"
    df[["date", "close", "volume"]].to_csv(unlabeled_path, index=False)

    return {"labeled": labeled_path, "unlabeled": unlabeled_path}


class TestSyntheticMode:
    def test_runs_successfully_and_produces_artifacts(self, tmp_path):
        output_dir = tmp_path / "synthetic_out"
        result = run_script(["synthetic", "--output-dir", str(output_dir), "--n-days", "1000"])
        assert result.returncode == 0, result.stderr
        assert (output_dir / "multi_pattern_detector.joblib").exists()
        assert (output_dir / "evaluation_results.csv").exists()
        assert (output_dir / "metadata.json").exists()

    def test_metadata_is_valid_json_with_expected_keys(self, tmp_path):
        output_dir = tmp_path / "synthetic_out2"
        run_script(["synthetic", "--output-dir", str(output_dir), "--n-days", "1000"])
        metadata = json.loads((output_dir / "metadata.json").read_text())
        assert metadata["mode"] == "synthetic"
        assert set(metadata["patterns_trained"]) == {
            "pump_and_dump", "wash_trading", "spoofing", "layering"
        }

    def test_saved_model_reloads_and_predicts(self, tmp_path):
        import joblib
        from mkt_surveillance_ml.config import BASE_FEATURE_COLUMNS
        from mkt_surveillance_ml.data.synthetic import generate_synthetic_market_data

        output_dir = tmp_path / "synthetic_out3"
        run_script(["synthetic", "--output-dir", str(output_dir), "--n-days", "1000"])
        detector = joblib.load(output_dir / "multi_pattern_detector.joblib")

        fresh = generate_synthetic_market_data(n_days=100, random_state=999)
        proba = detector.predict_proba(fresh[BASE_FEATURE_COLUMNS])
        assert len(proba) == len(fresh)
        assert (proba.values >= 0).all() and (proba.values <= 1).all()

    def test_larger_n_days_gives_larger_dataset(self, tmp_path):
        output_dir = tmp_path / "synthetic_out4"
        result = run_script(["synthetic", "--output-dir", str(output_dir), "--n-days", "600"])
        assert result.returncode == 0, result.stderr
        assert "train:" in result.stdout


class TestRealSupervisedMode:
    def test_runs_successfully_with_valid_labeled_csv(self, fake_real_csv, tmp_path):
        output_dir = tmp_path / "real_supervised_out"
        result = run_script([
            "real-supervised", "--csv", str(fake_real_csv["labeled"]),
            "--output-dir", str(output_dir), "--patterns", "pump_and_dump", "wash_trading",
        ])
        assert result.returncode == 0, result.stderr
        assert (output_dir / "multi_pattern_detector.joblib").exists()

    def test_trains_only_requested_patterns(self, fake_real_csv, tmp_path):
        output_dir = tmp_path / "real_supervised_out2"
        run_script([
            "real-supervised", "--csv", str(fake_real_csv["labeled"]),
            "--output-dir", str(output_dir), "--patterns", "pump_and_dump",
        ])
        metadata = json.loads((output_dir / "metadata.json").read_text())
        assert metadata["patterns_trained"] == ["pump_and_dump"]

    def test_fails_clearly_when_requested_pattern_has_no_label_column(self, fake_real_csv, tmp_path):
        output_dir = tmp_path / "real_supervised_out3"
        result = run_script([
            "real-supervised", "--csv", str(fake_real_csv["labeled"]),
            "--output-dir", str(output_dir), "--patterns", "pump_and_dump", "spoofing",
        ])
        assert result.returncode != 0
        assert "is_spoofing" in result.stderr

    def test_fails_clearly_on_missing_required_columns(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("date,foo,bar\n2023-01-01,1,2\n")
        result = run_script([
            "real-supervised", "--csv", str(bad_csv), "--output-dir", str(tmp_path / "out"),
        ])
        assert result.returncode != 0
        assert "close" in result.stderr and "volume" in result.stderr

    def test_fails_clearly_on_nonexistent_csv(self, tmp_path):
        result = run_script([
            "real-supervised", "--csv", str(tmp_path / "does_not_exist.csv"),
            "--output-dir", str(tmp_path / "out"),
        ])
        assert result.returncode != 0
        assert "not found" in result.stderr

    def test_requires_csv_argument(self, tmp_path):
        result = run_script(["real-supervised", "--output-dir", str(tmp_path / "out")])
        assert result.returncode != 0
        assert "--csv is required" in result.stderr


class TestRealUnsupervisedMode:
    def test_runs_successfully_on_unlabeled_csv(self, fake_real_csv, tmp_path):
        output_dir = tmp_path / "unsupervised_out"
        result = run_script([
            "real-unsupervised", "--csv", str(fake_real_csv["unlabeled"]),
            "--output-dir", str(output_dir),
        ])
        assert result.returncode == 0, result.stderr
        assert (output_dir / "isolation_forest_scratch.joblib").exists()
        assert (output_dir / "scored_days.csv").exists()

    def test_scored_csv_has_anomaly_score_and_flag_columns(self, fake_real_csv, tmp_path):
        output_dir = tmp_path / "unsupervised_out2"
        run_script([
            "real-unsupervised", "--csv", str(fake_real_csv["unlabeled"]),
            "--output-dir", str(output_dir),
        ])
        scored = pd.read_csv(output_dir / "scored_days.csv")
        assert "anomaly_score" in scored.columns
        assert "is_flagged" in scored.columns

    def test_higher_contamination_flags_more_days(self, fake_real_csv, tmp_path):
        output_dir_low = tmp_path / "low_contam"
        output_dir_high = tmp_path / "high_contam"
        run_script([
            "real-unsupervised", "--csv", str(fake_real_csv["unlabeled"]),
            "--output-dir", str(output_dir_low), "--contamination", "0.02",
        ])
        run_script([
            "real-unsupervised", "--csv", str(fake_real_csv["unlabeled"]),
            "--output-dir", str(output_dir_high), "--contamination", "0.2",
        ])
        low_meta = json.loads((output_dir_low / "metadata.json").read_text())
        high_meta = json.loads((output_dir_high / "metadata.json").read_text())
        assert high_meta["n_flagged"] > low_meta["n_flagged"]

    def test_saved_model_reloads_and_scores(self, fake_real_csv, tmp_path):
        import joblib
        output_dir = tmp_path / "unsupervised_out3"
        run_script([
            "real-unsupervised", "--csv", str(fake_real_csv["unlabeled"]),
            "--output-dir", str(output_dir),
        ])
        model = joblib.load(output_dir / "isolation_forest_scratch.joblib")
        assert model.threshold_ is not None

    def test_does_not_require_label_columns(self, fake_real_csv, tmp_path):
        """The whole point of this mode: it must work on data with
        NO label columns at all."""
        output_dir = tmp_path / "unsupervised_out4"
        result = run_script([
            "real-unsupervised", "--csv", str(fake_real_csv["unlabeled"]),
            "--output-dir", str(output_dir),
        ])
        assert result.returncode == 0, result.stderr
