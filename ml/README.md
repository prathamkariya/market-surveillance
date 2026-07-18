# mkt-surveillance-ml — Complete Phase 5-6 Production Package

Feature engineering through classical ML, anomaly detection, and time
series forecasting (files 20-30 of the Phase 5-6 notes), consolidated
from 11 markdown files into a tested, installable package with two
runnable training/forecasting entrypoints.

## Hardware requirements

Everything here runs on a laptop CPU. No GPU, no Colab, no Kaggle
needed. Files 20-27 (classical ML) train in seconds on a few thousand
rows. LSTM (file 30) is the slowest to train but still CPU-only —
expect minutes, not hours, at the epoch counts and data sizes this
package targets. 8GB RAM is comfortably enough; none of this data
exceeds low megabytes.

## Install

```bash
pip install -e .          # runtime deps (includes statsmodels, prophet, tensorflow-cpu)
pip install -e ".[dev]"   # + pytest for running tests
```

Prophet's install compiles a Stan backend on first use — this can take
a few minutes but should not require any manual intervention.

## Run the tests

```bash
pytest tests/ -v
```

277 tests, all passing, zero warnings (verified with warnings escalated
to errors), as of this build. Runtime ~2-3 minutes, mostly from ARIMA/
Prophet/LSTM fitting in the time-series tests.

## Two ways to actually use this

**`scripts/train.py`** — classification/anomaly detection (files 20-27).
Three modes: `synthetic` (demo), `real-supervised` (needs labeled
incidents), `real-unsupervised` (no labels needed — the realistic
starting point for most real data). Run `python scripts/train.py --help`
for full usage.

**`scripts/forecast.py`** — time series (files 28-30). Five modes:
`check-stationarity`, `fit-arima`, `fit-prophet`, `fit-lstm`,
`forecast-error-zscore` (the anomaly-detection-feeder signal). Run
`python scripts/forecast.py --help` for full usage.

These are NOT redundant with each other — file 28's own closing
argument is that forecasting models don't themselves flag manipulation
as anomalous; `forecast-error-zscore` produces a signal meant to feed
INTO `train.py`'s detection system, not replace it.

## Module map

| Module | File(s) | What's in it |
|---|---|---|
| `config.py` | — | `PatternType` enum, shared defaults, feature column order |
| `data/synthetic.py` | — | Canonical synthetic OHLCV generator with per-pattern injection |
| `features/scaling.py`, `features/selection.py` | 20 | Scratch scalers, filter/wrapper/embedded selection |
| `models/logistic_regression.py` | 21 | `LogisticRegressionScratch`, threshold sweep |
| `models/decision_tree.py` | 22 | `DecisionTreeClassifierScratch` (canonical) |
| `models/random_forest.py` | 23 | `RandomForestScratch` |
| `models/gradient_boosting.py` | 24 | `GradientBoostingScratch`, XGBoost wrapper |
| `evaluation/metrics.py` | 25 | Time-series CV, calibration, combined-vs-per-pattern AUC |
| `models/clustering.py` | 26 | `KMeansScratch` (with `n_init`), silhouette K selection |
| `anomaly/isolation_forest.py` | 27 | `IsolationForestScratch`, IF/LOF comparison |
| `detection/multi_pattern.py` | — | **Capstone.** `MultiPatternDetector` |
| `detection/weak_labeling.py` | — | **Bridge.** Attributes IsolationForest's unsupervised flags to a likely pattern via nearest-centroid matching, so per-pattern models can be bootstrapped without true labels |
| `time_series/stationarity.py` | 28 | ADF/KPSS, differencing, AR/MA signature diagnosis |
| `time_series/arima_prophet.py` | 29 | ARIMA/Prophet wrappers, order search, **forecast-error-zscore anomaly feeder** |
| `time_series/lstm.py` | 30 | LSTM gate mechanics from scratch, Keras pipeline (leak-proof by construction) |

## The labeled-vs-unlabeled tension, and what was built for it

`MultiPatternDetector`'s advantage over a blended model is real and
measured (above) -- but it needs per-pattern ground-truth labels most
real deployments don't have. `detection/weak_labeling.py` bridges this:
`IsolationForestScratch` flags anomalies with zero labels required, then
nearest-centroid matching against pattern prototypes (built from a
handful of confirmed examples, or from pure domain knowledge with none
at all) attributes a likely pattern to each flag -- enough to bootstrap
`MultiPatternDetector` without true labels.

**Measured, not claimed**, via `evaluate_weak_labeling_quality` against
synthetic data with known true labels:

| Pattern | IsolationForest recall | Downstream AUC lost vs. true labels |
|---|---|---|
| pump_and_dump | 100% | ~0.0002 (negligible) |
| wash_trading | 61% | ~0.009 (small) |
| spoofing | 43% | slightly negative (noise around an already-weak, near-random signal — not a real gain) |
| layering | 45% | 0.245 (substantial) |

The pattern here is coherent, not random: loss tracks directly with how
separable each pattern's signal already is. Layering's large loss traces
to the exact same small-sample instability already documented in
`MultiPatternDetector`'s own docstring -- weak-labeling makes an
already-sparse pattern sparser (it has to survive both the IsolationForest
flagging step AND the attribution step). This bridge is a genuine option
for rare, hard-to-detect patterns; it is not a free substitute for real
labels, and the table above is what "not free" costs concretely.

## Design decisions and real bugs found, worth knowing for an interview

**Per-pattern detection is structural.** `MultiPatternDetector` trains
one model per pattern instead of one blended label — every file's
critical-assessment section argues for this.

**The most important finding in the whole build:** comparing per-pattern
vs. blended models at a shared 0.5 threshold looked like per-pattern
modeling LOSING (recall 0.14 vs 0.45) — but AUC was nearly identical
(0.804 vs 0.806). The two models are calibrated on different base rates
and produce different absolute probability scales despite equivalent
ranking ability. Fixed by making AUC the primary metric and tuning
thresholds per model. Left unfixed, this would have been a backwards
conclusion about the package's own thesis.

**Second-most-important finding, this build pass:** an early version of
`forecast_error_zscore_anomaly_signal` used "the rest of the whole
forecast horizon" as its baseline. Forecast error for ANY multi-step
ARIMA forecast grows naturally with horizon distance (compounding
uncertainty) — nothing to do with anomalies. An unbounded baseline
pulled in naturally-large far-future errors, collapsing an obvious,
visually clear injected event's z-score from several sigma down to
~1.4. Fixed by requiring explicit, near-term baseline windows — the
same discipline the notes' own example already used, which I'd
carelessly generalized away from.

**Two removed/changed library APIs, found by testing:**
1. `CalibratedClassifierCV(cv='prefit')` — removed in current sklearn;
   replaced with `FrozenEstimator`.
2. statsmodels' ARIMA can return a "successful" fit with a real,
   plausible AIC even when optimization completely failed to converge —
   it only warns (`ConvergenceWarning`), it doesn't raise. `grid_search_arima_order`
   now treats a `ConvergenceWarning` the same as an exception, or it
   would silently rank meaningless fits alongside genuine ones.

**Statistical heuristics can legitimately disagree with themselves.**
`diagnose_ar_or_ma_signature`'s AR/MA classification, tested on the
EXACT seed the notes themselves use, comes back "ambiguous" — one of
the noisier draws out of several tested. This isn't a bug; it's the
same honest sampling variability that makes ADF and KPSS occasionally
disagree with each other. Tests check majority behavior across several
draws, not one seed's specific outcome.

**`KMeansScratch` gained `n_init` support** after a bad initialization
was directly confirmed (two centroids landed in one blob, a third
merged two others) — matches what sklearn's own `KMeans` does by default.

**`RandomForestScratch`'s `max_features='sqrt'` can hurt, not just
help**, on low-dimensional data where signal is concentrated in a few
non-redundant features — confirmed empirically, not assumed.

## Known limitations, stated plainly

- Spoofing/layering are OHLCV-derived proxies, not literal order-book
  reconstructions (real detection needs L2/L3 data this system lacks).
- Per-pattern modeling needs enough positive examples per pattern to
  pay off — at single digits, any comparison is noise-dominated.
- Prophet's cmdstanpy backend prints "Chain [1] start/done processing"
  to stdout on every fit. This appears to originate from the compiled
  Stan subprocess directly, not Python's `logging` module, so it isn't
  fully suppressible without redirecting file descriptors at the OS
  level — judged not worth the risk of hiding real errors for a purely
  cosmetic issue. Harmless; standard Prophet behavior.

