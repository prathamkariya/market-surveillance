"""
§5.3 Per-symbol metric slice — mandatory post-training evaluation.
Checks whether any symbol's flagged-rate deviates substantially from the market average.
A 2x-or-greater deviation with a reasonably-sized sample is the trigger to flag.
"""
import pandas as pd
import sys, os

THRESHOLD_MULTIPLIER = 2.0
MIN_SAMPLE = 30  # ignore symbols with fewer rows (not enough data to judge)


def evaluate_market(market: str, scored_csv: str, input_csv: str):
    print(f"\n=== {market} Per-Symbol Slice (§5.3) ===")

    scored = pd.read_csv(scored_csv, index_col=0)
    pooled = pd.read_csv(input_csv, index_col=0)

    # scored has no symbol column — join on index (date)
    if 'symbol' not in scored.columns:
        print("  ERROR: No 'symbol' column in scored CSV — skipping (ensure train_zscored.py preserves the symbol column)")
        return

    scored = scored.dropna(subset=['symbol'])
    market_rate = scored['is_flagged'].mean()
    print(f"  Market-wide flagged rate: {market_rate:.3f} ({scored['is_flagged'].sum()}/{len(scored)})")

    per_sym = scored.groupby('symbol').agg(
        n_rows=('is_flagged', 'count'),
        flagged_rate=('is_flagged', 'mean'),
        mean_score=('anomaly_score', 'mean'),
    ).sort_values('flagged_rate', ascending=False)

    print(f"\n  Per-symbol breakdown:")
    print(per_sym.to_string())

    # Check for deviating symbols
    deviating = per_sym[
        (per_sym['n_rows'] >= MIN_SAMPLE) &
        ((per_sym['flagged_rate'] >= market_rate * THRESHOLD_MULTIPLIER) |
         (per_sym['flagged_rate'] <= market_rate / THRESHOLD_MULTIPLIER))
    ]

    if deviating.empty:
        print(f"  OK: No symbol deviates >= {THRESHOLD_MULTIPLIER}x from market average. Pooled model is acceptable.")
    else:
        print(f"  WARNING: DEVIATING SYMBOLS (>={THRESHOLD_MULTIPLIER}x from {market_rate:.3f}):")
        print(deviating.to_string())
        print("  -> Review before shipping. Consider z-score normalization per S0 if disparity is real.")

    out = os.path.join("trained_models", market, "per_symbol_evaluation.csv")
    per_sym.to_csv(out)
    print(f"\n  Saved to {out}")


if __name__ == "__main__":
    for market, scored_csv, input_csv in [
        ("CRYPTO",    "trained_models/crypto/scored_days.csv",    "trained_models/crypto/real_if_input.csv"),
        ("US_EQUITY", "trained_models/us_equity/scored_days.csv", "trained_models/us_equity/real_if_input.csv"),
    ]:
        # Handle the lowercase directory naming when saving output
        mkt_dir = market.lower()
        evaluate_market(mkt_dir, scored_csv, input_csv)
