#!/usr/bin/env python3
"""
Training entrypoint for mkt-surveillance-ml.

THREE MODES -- pick the one that matches what data you actually have:

  synthetic           Generates labeled synthetic data and trains the
                       supervised MultiPatternDetector (one classifier per
                       pattern). Use this to see the whole pipeline work
                       end-to-end, or to sanity-check a change you made to
                       the package, before touching real data.

  real-supervised      Trains MultiPatternDetector on YOUR real OHLCV data.
                        Requires you to supply ground-truth label columns
                        (is_pump_and_dump, is_wash_trading, etc.) -- i.e.
                        you already know, from some other source (SEC
                        enforcement actions, exchange investigations, your
                        own documented incidents), which historical days
                        were which kind of manipulation. Most people doing
                        surveillance from scratch do NOT have this --
                        that's the whole reason surveillance exists.

  real-unsupervised     Trains an Isolation Forest anomaly detector on YOUR
                        real OHLCV data with NO labels required. This is
                        the realistic starting point for most real market
                        data: you don't know in advance which days were
                        manipulated, so there's nothing to supervise on.
                        Output is an anomaly score and flag per day, which
                        a human then reviews -- not a confirmed pattern
                        classification.

Do NOT feed unlabeled real data into --mode real-supervised expecting it
to somehow work "the way pros do it" -- it will either crash (no positive
examples for a pattern) or, worse, silently train on garbage if you patch
in fake labels. If you don't have labels, use real-unsupervised.

USAGE
-----
  python scripts/train.py synthetic
  python scripts/train.py real-supervised --csv my_data.csv --output-dir models/
  python scripts/train.py real-unsupervised --csv my_data.csv --output-dir models/

CSV FORMAT (real-supervised / real-unsupervised)
-------------------------------------------------
  Required columns: a date column (first column, or named 'date'), 'close', 'volume'.
  real-supervised additionally requires: is_pump_and_dump, is_wash_trading,
  is_spoofing, is_layering (0/1 columns) -- provide only the patterns you
  actually have ground truth for via --patterns.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mkt_surveillance_ml.config import PatternType, BASE_FEATURE_COLUMNS
from mkt_surveillance_ml.data.synthetic import (
    generate_synthetic_market_data,
    chronological_train_test_split,
)
from mkt_surveillance_ml.detection.multi_pattern import MultiPatternDetector
from mkt_surveillance_ml.anomaly.isolation_forest import IsolationForestScratch

sys.path.insert(0, str(Path(__file__).parent))
from _common import load_real_csv as _load_real_csv


def train_synthetic(args: argparse.Namespace) -> None:
    from mkt_surveillance_ml.data.synthetic import PatternInjectionConfig

    # The library's built-in DEFAULT_PATTERN_CONFIGS uses small, fixed
    # absolute day-counts (12-25 days) -- deliberately tiny so the test
    # suite runs fast. That's the wrong default for an actual demo: at
    # n_days=2000 it leaves single-digit test-set positives per pattern,
    # where precision/recall is a coin flip (see _warn_if_low_sample_size).
    # This script uses a more generously-sized config instead, matching
    # what test_multi_pattern_detector.py's adequately_powered fixture
    # already validated gives a statistically meaningful comparison.
    generous_configs = {
        PatternType.PUMP_AND_DUMP: PatternInjectionConfig(
            n_days=max(40, args.n_days // 25), return_mean=0.065, return_std=0.012,
            volume_ratio_mean=3.2, volume_ratio_std=0.35),
        PatternType.WASH_TRADING: PatternInjectionConfig(
            n_days=max(60, args.n_days // 17), return_mean=0.0, return_std=0.004,
            volume_ratio_mean=2.9, volume_ratio_std=0.30),
        PatternType.SPOOFING: PatternInjectionConfig(
            n_days=max(40, args.n_days // 25), return_mean=0.0, return_std=0.028,
            volume_ratio_mean=1.6, volume_ratio_std=0.25),
        PatternType.LAYERING: PatternInjectionConfig(
            n_days=max(45, args.n_days // 22), return_mean=0.0, return_std=0.019,
            volume_ratio_mean=2.1, volume_ratio_std=0.28),
    }

    print(f"Generating {args.n_days} days of synthetic market data (random_state={args.random_state})...")
    df = generate_synthetic_market_data(
        n_days=args.n_days, pattern_configs=generous_configs, random_state=args.random_state,
    )
    train_df, test_df = chronological_train_test_split(df, test_size=args.test_size)
    print(f"  train: {len(train_df)} rows, test: {len(test_df)} rows")

    detector = MultiPatternDetector(random_state=args.random_state)
    print("Training MultiPatternDetector (one classifier per pattern)...")
    detector.fit(train_df[BASE_FEATURE_COLUMNS], train_df)

    print("\nPer-pattern evaluation on held-out test data:")
    eval_result = detector.evaluate(test_df[BASE_FEATURE_COLUMNS], test_df)
    print(eval_result.to_string(index=False))
    _warn_if_low_sample_size(eval_result)

    print("\nComparison against a single blended-label baseline model:")
    comparison = detector.compare_against_blended_baseline(
        train_df[BASE_FEATURE_COLUMNS], train_df, test_df[BASE_FEATURE_COLUMNS], test_df
    )
    print(comparison.to_string(index=False))

    _save_supervised_artifacts(detector, args, eval_result, comparison, data_source="synthetic")


def train_real_supervised(args: argparse.Namespace) -> None:
    if not args.csv:
        print("ERROR: --csv is required for real-supervised mode.", file=sys.stderr)
        sys.exit(1)

    df = _load_real_csv(args.csv)

    patterns = _resolve_patterns(args.patterns)
    label_cols = [f"is_{p.value}" for p in patterns]
    missing_labels = [c for c in label_cols if c not in df.columns]
    if missing_labels:
        print(
            f"ERROR: --patterns requested {[p.value for p in patterns]}, but the "
            f"CSV is missing label column(s): {missing_labels}. Either add those "
            f"columns to your CSV, or pass --patterns with only the patterns you "
            f"actually have ground truth for (e.g. --patterns pump_and_dump).",
            file=sys.stderr,
        )
        sys.exit(1)

    train_df, test_df = chronological_train_test_split(df, test_size=args.test_size)
    print(f"train: {len(train_df)} rows, test: {len(test_df)} rows")
    for p in patterns:
        col = f"is_{p.value}"
        print(f"  {p.value}: {int(train_df[col].sum())} positive in train, {int(test_df[col].sum())} in test")

    detector = MultiPatternDetector(patterns=patterns, random_state=args.random_state)
    print("\nTraining MultiPatternDetector on real data...")
    detector.fit(train_df[BASE_FEATURE_COLUMNS], train_df)

    print("\nPer-pattern evaluation on held-out test data:")
    eval_result = detector.evaluate(test_df[BASE_FEATURE_COLUMNS], test_df)
    print(eval_result.to_string(index=False))
    _warn_if_low_sample_size(eval_result)

    comparison = None
    if "is_manipulation" in df.columns:
        print("\nComparison against a single blended-label baseline model:")
        comparison = detector.compare_against_blended_baseline(
            train_df[BASE_FEATURE_COLUMNS], train_df, test_df[BASE_FEATURE_COLUMNS], test_df
        )
        print(comparison.to_string(index=False))
    else:
        print(
            "\n(Skipping blended-baseline comparison: no 'is_manipulation' "
            "column in your CSV. Add one -- OR of your pattern columns -- "
            "to get this comparison.)"
        )

    _save_supervised_artifacts(detector, args, eval_result, comparison, data_source=args.csv)


def train_real_unsupervised(args: argparse.Namespace) -> None:
    if not args.csv:
        print("ERROR: --csv is required for real-unsupervised mode.", file=sys.stderr)
        sys.exit(1)

    df = _load_real_csv(args.csv)
    X = df[BASE_FEATURE_COLUMNS].values

    print(f"Training IsolationForestScratch on {len(df)} real days (contamination={args.contamination})...")
    model = IsolationForestScratch(
        n_estimators=args.n_estimators, contamination=args.contamination, random_state=args.random_state,
    )
    model.fit(X)

    scores = model.score_samples(X)
    flags = model.predict(X)
    df_scored = df.copy()
    df_scored["anomaly_score"] = scores
    df_scored["is_flagged"] = flags

    n_flagged = int(flags.sum())
    print(f"\nFlagged {n_flagged} of {len(df)} days ({n_flagged / len(df) * 100:.1f}%) as anomalous.")
    print(
        "\nReminder: contamination is a THRESHOLD CHOICE, not evidence the model "
        "found exactly this many real manipulation days -- it means these are "
        "whichever days scored worst, whether or not a natural break in the "
        "score distribution falls near this cutoff. Review flagged days as "
        "candidates for human investigation, not confirmed findings."
    )

    print("\nTop 10 highest-scoring (most anomalous) days:")
    top_10 = df_scored.sort_values("anomaly_score", ascending=False).head(10)
    print(top_10[BASE_FEATURE_COLUMNS + ["anomaly_score", "is_flagged"]].to_string())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "isolation_forest_scratch.joblib"
    joblib.dump(model, model_path)
    scored_csv_path = output_dir / "scored_days.csv"
    df_scored.to_csv(scored_csv_path)

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "real-unsupervised",
        "data_source": args.csv,
        "n_rows": len(df),
        "n_flagged": n_flagged,
        "contamination": args.contamination,
        "n_estimators": args.n_estimators,
        "random_state": args.random_state,
        "feature_columns": BASE_FEATURE_COLUMNS,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"\nSaved model to {model_path}")
    print(f"Saved scored days (all rows, with anomaly_score + is_flagged) to {scored_csv_path}")
    print(f"Saved metadata to {metadata_path}")


def _resolve_patterns(pattern_names: list[str] | None) -> list[PatternType]:
    if not pattern_names:
        return list(PatternType)
    resolved = []
    for name in pattern_names:
        try:
            resolved.append(PatternType(name))
        except ValueError:
            valid = [p.value for p in PatternType]
            print(f"ERROR: unknown pattern '{name}'. Valid options: {valid}", file=sys.stderr)
            sys.exit(1)
    return resolved


def _save_supervised_artifacts(
    detector: MultiPatternDetector, args: argparse.Namespace,
    eval_result: pd.DataFrame, comparison: pd.DataFrame | None, data_source: str,
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "multi_pattern_detector.joblib"
    joblib.dump(detector, model_path)

    eval_csv_path = output_dir / "evaluation_results.csv"
    eval_result.to_csv(eval_csv_path, index=False)

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "data_source": data_source,
        "patterns_trained": [p.value for p in detector.models_],
        "feature_columns": BASE_FEATURE_COLUMNS,
        "random_state": args.random_state,
        "test_size": args.test_size,
        "evaluation": eval_result.to_dict(orient="records"),
    }
    if comparison is not None:
        metadata["blended_baseline_comparison"] = comparison.to_dict(orient="records")

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))

    print(f"\nSaved model to {model_path}")
    print(f"Saved evaluation results to {eval_csv_path}")
    print(f"Saved metadata to {metadata_path}")
    print(
        f"\nTo reload later:\n"
        f"  import joblib\n"
        f"  detector = joblib.load('{model_path}')\n"
        f"  detector.predict_proba(new_data[{BASE_FEATURE_COLUMNS!r}])"
    )


def _warn_if_low_sample_size(eval_result: pd.DataFrame, threshold: int = 10) -> None:
    """A pattern with too few positive test examples will show
    precision/recall of 0.0 or 1.0 that looks like a broken model but is
    actually just too small a sample to measure anything reliably --
    documented as a real, tested limitation in
    detection/multi_pattern.py and test_multi_pattern_detector.py's
    test_recall_comparison_is_unreliable_with_too_few_positive_examples.
    Surfacing it here so a first-time run isn't mistaken for a bug.
    """
    low_sample_patterns = eval_result[eval_result["n_positive"] < threshold]
    if len(low_sample_patterns) > 0:
        print(
            f"\nNOTE: pattern(s) {list(low_sample_patterns['pattern'])} have "
            f"fewer than {threshold} positive test examples. Precision/recall "
            f"for these is NOT a reliable measurement at this sample size -- "
            f"it can look like 0.0 (or a suspiciously perfect 1.0) purely from "
            f"having only a handful of test days to score, not because the "
            f"model is broken. Use more historical data, a higher injection "
            f"rate (synthetic mode), or a longer real data window to get a "
            f"meaningful read on these patterns specifically."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode", choices=["synthetic", "real-supervised", "real-unsupervised"],
        help="Which training mode to run -- see the module docstring above for which one matches your data.",
    )
    parser.add_argument("--csv", type=str, default=None, help="Path to real OHLCV CSV (required for real-* modes).")
    parser.add_argument("--output-dir", type=str, default="trained_models", help="Where to save trained artifacts.")
    parser.add_argument("--random-state", type=int, default=42, dest="random_state")
    parser.add_argument("--test-size", type=float, default=0.2, dest="test_size")
    parser.add_argument("--n-days", type=int, default=2000, dest="n_days", help="(synthetic mode only)")
    parser.add_argument(
        "--patterns", nargs="+", default=None,
        help="(real-supervised only) Which patterns you have labels for, e.g. --patterns pump_and_dump wash_trading. Default: all four.",
    )
    parser.add_argument(
        "--contamination", type=float, default=0.05,
        help="(real-unsupervised only) Expected fraction of anomalous days -- a threshold choice, not a known truth.",
    )
    parser.add_argument("--n-estimators", type=int, default=100, dest="n_estimators", help="(real-unsupervised only)")

    args = parser.parse_args()

    if args.mode == "synthetic":
        train_synthetic(args)
    elif args.mode == "real-supervised":
        train_real_supervised(args)
    elif args.mode == "real-unsupervised":
        train_real_unsupervised(args)


if __name__ == "__main__":
    main()
