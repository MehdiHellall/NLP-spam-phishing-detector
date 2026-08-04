"""Compatibility imports for the refactored scikit-learn runtime."""

from web.backend.sklearn_runtime import (
    PredictionResult,
    ThreatClassifier,
    load_model,
    save_model,
)

__all__ = ["PredictionResult", "ThreatClassifier", "load_model", "save_model"]
