"""
Central configuration for the market surveillance ML package.

Single source of truth for anything that would otherwise be a magic number
copy-pasted across modules: pattern names, default hyperparameters, and the
train/test split convention used everywhere in this package.

Design decision (carried over from every Phase 5-6 file's critical-assessment
section): this package treats "manipulation" as FOUR separate, independently
learnable patterns rather than one blended binary label. PatternType is the
enum that decision threads through the whole package -- data generation,
model training, and evaluation all key off it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PatternType(str, Enum):
    """The four manipulation patterns this system targets, kept distinct
    on purpose. See detection/multi_pattern.py for why collapsing these
    into one label measurably hurts detection of each individual pattern.
    """

    PUMP_AND_DUMP = "pump_and_dump"
    WASH_TRADING = "wash_trading"
    SPOOFING = "spoofing"
    LAYERING = "layering"


@dataclass(frozen=True)
class RandomForestDefaults:
    n_estimators: int = 100
    max_depth: int = 8
    max_features: str = "sqrt"


@dataclass(frozen=True)
class GradientBoostingDefaults:
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 3


@dataclass(frozen=True)
class IsolationForestDefaults:
    n_estimators: int = 100
    # Not a single global constant -- see anomaly/isolation_forest.py.
    # Each pattern gets contamination estimated from ITS OWN base rate,
    # not one dataset-wide guess. This default is only a fallback for
    # ad-hoc / exploratory calls that don't go through the per-pattern path.
    contamination: float = 0.05


@dataclass(frozen=True)
class GlobalDefaults:
    random_state: int = 42
    test_size: float = 0.2
    # Time-series data: never a random split. See data/synthetic.py and
    # evaluation/metrics.py -- this is enforced structurally, not just documented.
    chronological_split: bool = True


MIN_RAW_ROWS_FOR_FEATURES: int = 20


RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2

RF_DEFAULTS = RandomForestDefaults()
GB_DEFAULTS = GradientBoostingDefaults()
IF_DEFAULTS = IsolationForestDefaults()
GLOBAL_DEFAULTS = GlobalDefaults()

# Feature columns every model in this package expects, in a fixed order.
# Centralized here so a model trained in one module and scored in another
# can never silently disagree about column order.
BASE_FEATURE_COLUMNS: list[str] = [
    "return",
    "volume_ratio_20d",
    "volatility_20d",
]
