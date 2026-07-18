"""tests/test_streaming.py — Regression guard for the volume persistence fix.

This test exists specifically to prevent the silent volume=1.0 corruption bug
from re-entering the codebase. It must:
  - Provide >= MIN_RAW_ROWS_FOR_FEATURES (21) records (20 history + 1 current tick)
    so score_live_trade doesn't bail out early.
  - Mock get_model_registry so the test runs without trained .joblib files.
  - Use monkeypatch on the DEFAULT_THRESHOLD so the mock IF score (0.5) clears
    the threshold and we get a returned alert back to assert against.
  - Assert that alert["volume"] equals the volume on the input trade, not 1.0.
"""
from __future__ import annotations

import importlib

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_history(n: int, base_ts: int = 1_718_000_000_000) -> list[dict]:
    """Return n synthetic history ticks, spaced 1 second apart, ending just
    before the current trade tick so timestamps are strictly ordered."""
    return [
        {
            "event_id": f"TEST_BTCUSDT_{base_ts + i * 1_000}",
            "symbol": "BTCUSDT",
            "timestamp_ms": base_ts + i * 1_000,
            "price": 59_000.0 + i * 10,   # gentle upward drift — not flat, avoids NaN volatility
            "volume": 1.0,
        }
        for i in range(n)
    ]


class _MockIsolationForest:
    """Always returns a fixed score just above 0 so _combine_scores produces
    a value above the patched threshold of -1.0."""

    def score_samples(self, X):
        return [0.5]


class _MockRegistryTrue:
    has_any_model = True
    has_isolation_forest = True
    has_multi_pattern = False
    isolation_forest = _MockIsolationForest()


# ──────────────────────────────────────────────────────────────────────────────
# The test
# ──────────────────────────────────────────────────────────────────────────────

def test_score_live_trade_volume_is_preserved(monkeypatch):
    """score_live_trade must return alert["volume"] == trade["volume"].

    This is the regression guard for the silent volume=1.0 corruption fix.
    If this test fails it means volume is being dropped or overwritten
    somewhere in the scoring pipeline.
    """
    import app.services.anomaly_service as svc

    # Patch get_model_registry to avoid requiring trained .joblib files on disk.
    monkeypatch.setattr(svc, "get_model_registry", lambda **_: _MockRegistryTrue())

    # Patch DEFAULT_THRESHOLD so the mock score (0.5) clears it and we get an
    # alert back. The constant used inside score_live_trade is the default arg
    # value — we override the default in the call instead to keep the patch
    # narrow and avoid mutating module state.

    # Build exactly 20 history ticks (MIN_RAW_ROWS_FOR_FEATURES is 21 total
    # including current tick, so 20 history + 1 current = 21).
    history = _make_history(20, base_ts=1_717_999_980_000)

    trade = {
        "event_id": "TEST_BTCUSDT_1718000000000",
        "symbol": "BTCUSDT",
        "timestamp_ms": 1_718_000_000_000,
        "price": 60_000.0,
        "volume": 2.5,          # ← the value we're asserting survives
        "source": "BINANCE",    # Needed for low_confidence=False
    }

    alert = svc.score_live_trade(
        trade,
        history,
        sentiment_score=0.2,
        threshold=-1.0,         # force anomaly — every score clears this
    )

    assert alert is not None, (
        "score_live_trade returned None — history depth or feature engineering "
        "dropped below MIN_RAW_ROWS_FOR_FEATURES. Check _make_history length."
    )
    assert alert["volume"] == 2.5, (
        f"Volume corruption detected: expected 2.5, got {alert['volume']}. "
        "This is the exact bug the fix was meant to prevent."
    )
    assert alert["symbol"] == "BTCUSDT"
    assert alert["event_id"] == trade["event_id"]
    assert alert["price"] == 60_000.0
    assert alert["sentiment_score"] == 0.2
    assert alert["isolation_forest_score"] == pytest.approx(0.5)
    assert alert["market"] == "CRYPTO"
    assert alert["low_confidence"] is False
    assert "features" in alert
