"""Validated HTTP request and response schemas for ThreatLens."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from web.backend.settings import MAX_TEXT_CHARS

RiskLevel = Literal["low", "medium", "high"]
ThreatLabel = Literal["ham", "phish", "spam"]


class HealthResponse(BaseModel):
    status: Literal["ok", "error"]
    model_loaded: bool
    model_path: str | None
    detail: str


class LiveResponse(BaseModel):
    status: Literal["ok"]
    app_name: str
    detail: str


class MetadataResponse(BaseModel):
    app_name: str
    labels: list[str]
    max_text_chars: int
    model: dict[str, object]
    metrics: dict[str, object] | None
    privacy: str


class PredictRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_CHARS)

    @field_validator("text", mode="before")
    @classmethod
    def clean_text(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("text must not be empty.")
        if len(trimmed) > MAX_TEXT_CHARS:
            raise ValueError(f"text must be {MAX_TEXT_CHARS} characters or fewer.")
        return trimmed


class PredictResponse(BaseModel):
    """Legacy prediction shape retained for existing clients."""

    label: ThreatLabel
    probabilities: dict[str, float] | None
    risk_level: RiskLevel
    explanation: str
    suggested_action: str


class ModelPredictionOutput(BaseModel):
    label: ThreatLabel
    confidence: float | None
    probabilities: dict[str, float] | None


class ArtifactMetadata(BaseModel):
    """Real deployment metadata; unavailable artifact fields remain explicitly null."""

    artifact: str | None
    model_name: str | None
    metrics_file: str | None


class VersionedPredictResponse(BaseModel):
    """Duel-ready response with the currently deployed model as its sole output."""

    final_label: ThreatLabel
    final_risk_level: RiskLevel
    final_confidence: float | None
    model_outputs: dict[Literal["tfidf_logreg"], ModelPredictionOutput]
    explanation: str
    suggested_action: str
    artifact_metadata: ArtifactMetadata
