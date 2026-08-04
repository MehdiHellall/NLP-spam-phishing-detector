"""Model persistence and inference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


@dataclass(frozen=True)
class PredictionResult:
    """Single-message prediction result."""

    label: str
    probabilities: dict[str, float] | None = None


def save_model(model: Any, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
    """Persist a trained sklearn-compatible model."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata or {}}, destination)


def load_model(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load a trusted model saved by :func:`save_model`.

    Joblib uses pickle internally, so artifacts must come from this project or
    another trusted source.
    """
    payload = joblib.load(path)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"], dict(payload.get("metadata", {}))
    return payload, {}


class ThreatClassifier:
    """Small inference wrapper around a sklearn-compatible classifier."""

    def __init__(self, model: Any) -> None:
        self._model = model

    @classmethod
    def from_path(cls, path: str | Path) -> ThreatClassifier:
        """Load a persisted classifier."""
        model, _ = load_model(path)
        return cls(model)

    def predict_one(self, text: str) -> PredictionResult:
        """Predict one message and include probabilities when available."""
        label = str(self._model.predict([text])[0])
        probabilities = None

        if hasattr(self._model, "predict_proba"):
            classes = [str(value) for value in self._model.classes_]
            scores = self._model.predict_proba([text])[0]
            probabilities = {
                label_name: float(score)
                for label_name, score in zip(classes, scores, strict=True)
            }

        return PredictionResult(label=label, probabilities=probabilities)
