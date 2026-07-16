import os
import json
import joblib
from pathlib import Path
from typing import Optional

class ModelLoadError(Exception):
    pass

class ModelRegistry:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.isolation_forest = None
        self.isolation_forest_metadata = {}
        self.multi_pattern_detector = None
        self.multi_pattern_metadata = {}

    def load(self):
        if not self.model_dir.exists():
            raise ModelLoadError(f"Model directory {self.model_dir} does not exist.")

        # Each model type gets its own independent try/except -- a
        # corrupted/unloadable artifact for ONE type must not prevent the
        # OTHER type from being attempted. (Previously these were sequential
        # with no isolation: a failure in the isolation-forest block raised
        # immediately, so the multi-pattern block below it never even ran.)
        errors: list[str] = []

        # Try to load Isolation Forest
        iforest_path = self.model_dir / "isolation_forest_scratch.joblib"
        iforest_meta = self.model_dir / "isolation_forest_metadata.json"
        if iforest_path.exists():
            try:
                self.isolation_forest = joblib.load(iforest_path)
                if iforest_meta.exists():
                    with open(iforest_meta, "r") as f:
                        self.isolation_forest_metadata = json.load(f)
            except Exception as e:
                errors.append(f"Failed to load Isolation Forest: {e}")

        # Try to load Multi Pattern Detector
        mp_path = self.model_dir / "multi_pattern_detector.joblib"
        mp_meta = self.model_dir / "multi_pattern_detector_metadata.json"
        if mp_path.exists():
            try:
                self.multi_pattern_detector = joblib.load(mp_path)
                if mp_meta.exists():
                    with open(mp_meta, "r") as f:
                        self.multi_pattern_metadata = json.load(f)
            except Exception as e:
                errors.append(f"Failed to load Multi Pattern Detector: {e}")

        if errors:
            raise ModelLoadError("; ".join(errors))

    @property
    def has_isolation_forest(self) -> bool:
        return self.isolation_forest is not None

    @property
    def has_multi_pattern(self) -> bool:
        return self.multi_pattern_detector is not None

    @property
    def has_any_model(self) -> bool:
        return self.has_isolation_forest or self.has_multi_pattern
