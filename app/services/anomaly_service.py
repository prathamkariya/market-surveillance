"""
app/services/anomaly_service.py — real ML scoring (Phase 7).

Replaces the Phase 1/2 mock scoring functions (hand-coded formulas,
explicitly commented "In Phase 3: replaced with a real trained model")
with the actual trained mkt_surveillance_ml package: IsolationForestScratch
(unsupervised) and MultiPatternDetector (per-pattern supervised).

Feature computation is NOT reimplemented here. It calls
compute_engineered_features from mkt_surveillance_ml.data.synthetic
directly -- the exact function every model in that package was trained
against. Recomputing the same three features (return, volume_ratio_20d,
volatility_20d) a second time, by hand, in this file would risk two
independent implementations silently drifting apart -- exactly the
kind of duplication mkt_surveillance_ml's own README flags as the
reason data/synthetic.py exists as a single source of truth in the
first place.

Note the feature set changed from the mock version's 5 hand-coded
features (price_return, price_range, volume_zscore, price_volatility,
body_ratio, using full OHLC) to mkt_surveillance_ml's 3 (return,
volume_ratio_20d, volatility_20d, using close+volume only). This is a
real, deliberate change -- the mock's features were never validated
against anything; mkt_surveillance_ml's have 300+ tests behind them --
not an accidental behavior change to paper over.
"""
import json
import logging
import threading
from typing import Optional

import pandas as pd
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Anomaly, MarketData
from mkt_surveillance_ml.config import BASE_FEATURE_COLUMNS, MIN_RAW_ROWS_FOR_FEATURES
from mkt_surveillance_ml.data.synthetic import compute_engineered_features
from mkt_surveillance_ml.serving.model_registry import ModelRegistry, ModelLoadError

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Anomaly threshold
# ──────────────────────────────────────────────
DEFAULT_THRESHOLD = 0.7

# Fetch a bit more than the strict MIN_RAW_ROWS_FOR_FEATURES - 1 (20)
# prior records needed, so a symbol with a couple of same-timestamp or
# near-duplicate rows still clears the minimum after any dedup upstream.
HISTORICAL_FETCH_LIMIT = 30


# ──────────────────────────────────────────────
# Model loading (lazy singleton -- loaded once per process, not once
# per request. A hot-reload endpoint, matching
# mkt_surveillance_ml.serving.app's POST /admin/reload, is a natural
# follow-up if models get retrained while this service is running; not
# added here to keep this change focused on the mock-to-real swap itself.)
# ──────────────────────────────────────────────
_registry: Optional[ModelRegistry] = None
_registry_lock = threading.Lock()


def get_model_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        # Double-checked locking. Safe in Python because the GIL makes the
        # _registry reference read/write atomic on either side of the lock
        # -- the lock is only needed to stop two threads from both doing
        # the load. Building into `candidate` and only publishing to the
        # global once .load() has finished (or failed) is what actually
        # closes the race: a concurrent caller now either gets None-not-yet
        # (retries the lock) or a fully-loaded/fully-failed registry, never
        # one that's still mid-load.
        with _registry_lock:
            if _registry is None:
                candidate = ModelRegistry(settings.MODEL_DIR)
                try:
                    candidate.load()
                except ModelLoadError as e:
                    # Matches mkt_surveillance_ml.serving.app's philosophy: don't
                    # crash the whole service because models aren't trained yet.
                    # detect_anomaly() below checks has_any_model explicitly and
                    # returns a clear 503, not a silent wrong answer.
                    logger.warning(f"Model loading failed: {e}")
                _registry = candidate
    return _registry


# ──────────────────────────────────────────────
# Feature engineering — delegates entirely to mkt_surveillance_ml
# ──────────────────────────────────────────────
def _market_data_to_feature_row(record: MarketData, historical: list[MarketData]) -> dict:
    """Builds the same 3 engineered features (return, volume_ratio_20d,
    volatility_20d) every mkt_surveillance_ml model is trained on, for
    the single most recent row (`record`), using `historical` as the
    trailing context the rolling-window features need.

    Raises HTTPException(400) if there isn't enough history -- silently
    scoring against a partially-computed or default-filled feature
    vector would produce a confident-looking but meaningless number,
    exactly the kind of failure mode mkt_surveillance_ml's own input
    validation (see its serving layer's ScoreRequest schema) is built
    to catch rather than paper over.
    """
    all_records = historical + [record]
    if len(all_records) < MIN_RAW_ROWS_FOR_FEATURES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Not enough historical data for '{record.symbol}' to compute "
                f"features: {len(all_records)} record(s) available, need at "
                f"least {MIN_RAW_ROWS_FOR_FEATURES} (20 trailing days plus the "
                f"current one) for the rolling-window features every model in "
                f"this system was trained on. Ingest more history for this "
                f"symbol via POST /market-data first."
            ),
        )

    df = pd.DataFrame(
        {
            "close": [float(r.close) for r in all_records],
            "volume": [float(r.volume) for r in all_records],
        },
        index=[r.timestamp for r in all_records],
    )
    df = df.sort_index()
    features_df = compute_engineered_features(df).dropna(subset=BASE_FEATURE_COLUMNS)

    if len(features_df) == 0:
        # Should not happen given the length check above, but the
        # rolling-window math is what actually determines this, not the
        # raw count directly -- fail loudly rather than assume.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not compute features for '{record.symbol}' from the available history.",
        )

    last_row = features_df.iloc[-1]
    return {col: float(last_row[col]) for col in BASE_FEATURE_COLUMNS}


# ──────────────────────────────────────────────
# Scoring — real trained models
# ──────────────────────────────────────────────
def _combine_scores(isolation_forest_score: Optional[float], multi_pattern_max_score: Optional[float]) -> float:
    """Weighted average when both models are available (IF 60%, per-
    pattern max 40% -- the same weighting the mock version used, kept
    for continuity now that both sides of it are real). Falls back to
    whichever single score is available if only one model was trained.

    Both isolation_forest_score (IsolationForestScratch's s(x,n) formula)
    and multi_pattern_max_score (a RandomForestClassifier probability)
    are mathematically bounded in [0,1] -- not mock-clamped like the
    Phase 1/2 version, genuinely bounded by what produces them.
    """
    if isolation_forest_score is not None and multi_pattern_max_score is not None:
        return round(0.6 * isolation_forest_score + 0.4 * multi_pattern_max_score, 4)
    if isolation_forest_score is not None:
        return round(isolation_forest_score, 4)
    if multi_pattern_max_score is not None:
        return round(multi_pattern_max_score, 4)
    raise ValueError("At least one of isolation_forest_score/multi_pattern_max_score must be provided.")


# ──────────────────────────────────────────────
# Public service function
# ──────────────────────────────────────────────
def detect_anomaly(
    db: Session,
    market_data_id: int,
    user_id: int,
    threshold: float = DEFAULT_THRESHOLD,
) -> Anomaly:
    """
    Run anomaly detection on a market data record using real trained models.

    Steps:
    1. Fetch the record (404 if not found or wrong user)
    2. Fetch trailing history for the same symbol (for rolling features)
    3. Compute features via mkt_surveillance_ml's compute_engineered_features
    4. Score with whichever of IsolationForestScratch / MultiPatternDetector
       are loaded (503 if neither is available)
    5. Store the Anomaly record, including the full per-pattern breakdown
    6. Return the Anomaly
    """
    # 1. Fetch target record
    record = db.query(MarketData).filter(
        MarketData.id == market_data_id,
        MarketData.user_id == user_id,
    ).first()
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Market data not found")

    # 2. Fetch historical context (trailing candles for this symbol, before this timestamp)
    historical = (
        db.query(MarketData)
        .filter(
            MarketData.symbol == record.symbol,
            MarketData.user_id == user_id,
            MarketData.timestamp < record.timestamp,
        )
        .order_by(MarketData.timestamp.desc())
        .limit(HISTORICAL_FETCH_LIMIT)
        .all()
    )
    historical = list(reversed(historical))  # chronological order

    # 3. Feature engineering (raises 400 if insufficient history)
    features = _market_data_to_feature_row(record, historical)

    # 4. Score with whichever real models are available
    registry = get_model_registry()
    if not registry.has_any_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No trained models available. Train at least one with "
                "mkt_surveillance_ml's scripts/train.py and point MODEL_DIR "
                "at the output directory."
            ),
        )

    isolation_forest_score = None
    multi_pattern_max_score = None
    pattern_scores: Optional[dict] = None
    model_versions = []

    X_row = pd.DataFrame([features], columns=BASE_FEATURE_COLUMNS)

    if registry.has_isolation_forest:
        isolation_forest_score = float(registry.isolation_forest.score_samples(X_row.values)[0])
        model_versions.append(f"isolation_forest={registry.isolation_forest_metadata.get('trained_at_utc', 'unknown')}")

    if registry.has_multi_pattern:
        proba_row = registry.multi_pattern_detector.predict_proba(X_row).iloc[0]
        pattern_scores = {col.replace("proba_", ""): float(val) for col, val in proba_row.items()}
        multi_pattern_max_score = max(pattern_scores.values())
        model_versions.append(f"multi_pattern={registry.multi_pattern_metadata.get('trained_at_utc', 'unknown')}")

    combined = _combine_scores(isolation_forest_score, multi_pattern_max_score)
    is_anomaly = combined >= threshold

    # 5. Store
    anomaly = Anomaly(
        market_data_id=record.id,
        anomaly_score=combined,
        is_anomaly=is_anomaly,
        isolation_forest_score=isolation_forest_score,
        multi_pattern_max_score=multi_pattern_max_score,
        pattern_scores=json.dumps(pattern_scores) if pattern_scores is not None else None,
        model_version="; ".join(model_versions),
        features=json.dumps(features),
    )
    db.add(anomaly)
    db.commit()
    db.refresh(anomaly)
    return anomaly


# ──────────────────────────────────────────────
# Streaming Service Function (Phase 8)
# ──────────────────────────────────────────────
def score_live_trade(
    trade: dict,
    historical_trades: list[dict],
    sentiment_score: float,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """
    Run anomaly detection on a live streaming tick, using in-memory history.
    Does not touch the Postgres database at all.
    Returns a dict containing the alert if is_anomaly=True, else None.
    
    Args:
        trade: The newest tick (dict representation of UnifiedTradeEvent).
        historical_trades: The last 20 ticks for this symbol from Redis.
        sentiment_score: Fused sentiment score from the live_sentiment stream.
    """
    # 1. Feature engineering
    all_records = historical_trades + [trade]
    if len(all_records) < MIN_RAW_ROWS_FOR_FEATURES:
        return None  # Not enough history yet, skip silently in streaming

    df = pd.DataFrame(
        {
            "close": [float(r["price"]) for r in all_records],
            "volume": [float(r["volume"]) for r in all_records],
        },
        index=[pd.to_datetime(r["timestamp_ms"], unit="ms", utc=True) for r in all_records],
    )
    # Deduplicate exact timestamps by taking the last trade
    df = df[~df.index.duplicated(keep="last")].sort_index()
    
    if len(df) < MIN_RAW_ROWS_FOR_FEATURES:
        return None

    features_df = compute_engineered_features(df).dropna(subset=BASE_FEATURE_COLUMNS)
    if len(features_df) == 0:
        return None

    last_row = features_df.iloc[-1]
    features = {col: float(last_row[col]) for col in BASE_FEATURE_COLUMNS}

    # Sentiment fusion is explicitly deferred. 
    # It needs to be a proper input feature to the ML model (retraining required),
    # not a post-hoc rule-based multiplier bolted onto the model's output.
    # See Phase 7 recommendations. We pass it through as metadata for MAR reports.

    # 2. Score with models
    registry = get_model_registry()
    if not registry.has_any_model:
        return None

    isolation_forest_score = None
    multi_pattern_max_score = None
    pattern_scores: Optional[dict] = None

    X_row = pd.DataFrame([features], columns=BASE_FEATURE_COLUMNS)

    if registry.has_isolation_forest:
        isolation_forest_score = float(registry.isolation_forest.score_samples(X_row.values)[0])

    if registry.has_multi_pattern:
        proba_row = registry.multi_pattern_detector.predict_proba(X_row).iloc[0]
        pattern_scores = {col.replace("proba_", ""): float(val) for col, val in proba_row.items()}
        multi_pattern_max_score = max(pattern_scores.values())

    combined = _combine_scores(isolation_forest_score, multi_pattern_max_score)
    is_anomaly = combined >= threshold

    if not is_anomaly:
        return None

    return {
        "event_id": trade["event_id"],
        "symbol": trade["symbol"],
        "timestamp_ms": trade["timestamp_ms"],
        "price": trade["price"],
        "volume": trade["volume"],
        "anomaly_score": combined,
        "sentiment_score": sentiment_score,
        "isolation_forest_score": isolation_forest_score,
        "multi_pattern_max_score": multi_pattern_max_score,
        "pattern_scores": pattern_scores,
        "features": features,
    }
