"""FastAPI app for artifact-backed message threat classification."""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from hashlib import sha256
from ipaddress import ip_address
from pathlib import Path
from typing import Literal, cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from email_threat_detector.constants import LABEL_NAMES, normalize_label
from email_threat_detector.inference import ThreatClassifier, load_model

MODEL_PATH_ENV = "EMAIL_THREAT_MODEL_PATH"
MODEL_URL_ENV = "EMAIL_THREAT_MODEL_URL"
MODEL_SHA256_ENV = "EMAIL_THREAT_MODEL_SHA256"
MODEL_CACHE_PATH_ENV = "EMAIL_THREAT_MODEL_CACHE_PATH"
BACKGROUND_WARMUP_ENV = "EMAIL_THREAT_BACKGROUND_WARMUP"
ALLOWED_ORIGINS_ENV = "EMAIL_THREAT_ALLOWED_ORIGINS"
DEFAULT_METRICS_PATH = Path("reports/metrics/tfidf_logreg_metrics.json")
DEFAULT_MODEL_CACHE_PATH = Path("/tmp/threatlens/model.joblib")
DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
)
MAX_TEXT_CHARS = 5_000
PUBLIC_METADATA_KEYS = frozenset(
    {
        "model_name",
        "metrics_file",
    }
)
TRANSPARENT_SIGNALS = {
    "urgency": re.compile(r"\b(urgent|immediately|expires?|suspended|locked|final notice)\b", re.I),
    "credential request": re.compile(
        r"\b(password|passcode|login|verify|verification|credentials?|account)\b",
        re.I,
    ),
    "money or prize language": re.compile(
        r"\b(free|prize|winner|cash|coupon|discount|limited offer|buy now)\b",
        re.I,
    ),
    "link or contact prompt": re.compile(r"(https?://|www\.|\bclick\b|\breply\b|\bcall\b)", re.I),
}

RiskLevel = Literal["low", "medium", "high"]
ThreatLabel = Literal["ham", "phish", "spam"]


@dataclass(frozen=True)
class AppSettings:
    """Runtime settings for the API."""

    model_path: Path | None = None
    model_url: str | None = None
    model_sha256: str | None = None
    model_cache_path: Path = DEFAULT_MODEL_CACHE_PATH
    background_warmup: bool = False
    metrics_path: Path = DEFAULT_METRICS_PATH
    allowed_origins: tuple[str, ...] = DEFAULT_ALLOWED_ORIGINS
    max_text_chars: int = MAX_TEXT_CHARS

    @classmethod
    def from_env(cls) -> AppSettings:
        """Create settings from environment variables."""
        raw_model_path = os.getenv(MODEL_PATH_ENV)
        raw_model_url = os.getenv(MODEL_URL_ENV)
        raw_model_cache_path = os.getenv(MODEL_CACHE_PATH_ENV)
        raw_origins = os.getenv(ALLOWED_ORIGINS_ENV)
        origins = (
            tuple(origin.strip() for origin in raw_origins.split(",") if origin.strip())
            if raw_origins
            else DEFAULT_ALLOWED_ORIGINS
        )
        return cls(
            model_path=Path(raw_model_path).expanduser() if raw_model_path else None,
            model_url=raw_model_url.strip() if raw_model_url else None,
            model_sha256=os.getenv(MODEL_SHA256_ENV),
            model_cache_path=(
                Path(raw_model_cache_path).expanduser()
                if raw_model_cache_path
                else DEFAULT_MODEL_CACHE_PATH
            ),
            background_warmup=os.getenv(BACKGROUND_WARMUP_ENV, "").casefold()
            in {"1", "true", "yes"},
            allowed_origins=origins,
        )


@dataclass(frozen=True)
class ModelState:
    """Loaded model state or a clear startup problem."""

    classifier: ThreatClassifier | None
    path: Path | None
    metadata: dict[str, object]
    error: str | None = None

    @property
    def loaded(self) -> bool:
        return self.classifier is not None and self.error is None


class ModelArtifactError(RuntimeError):
    """Raised when an artifact cannot be resolved for model loading."""


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
    label: ThreatLabel
    probabilities: dict[str, float] | None
    risk_level: RiskLevel
    explanation: str
    suggested_action: str


def _resolve_local_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_checksum(path: Path, expected_sha256: str | None) -> None:
    if expected_sha256 is None:
        return

    actual_sha256 = _file_sha256(path)
    if actual_sha256.casefold() != expected_sha256.casefold():
        raise ModelArtifactError(
            "Downloaded model artifact checksum mismatch. "
            f"Expected {expected_sha256}, got {actual_sha256}."
        )


def _is_loopback_host(host: str | None) -> bool:
    if host is None:
        return False
    if host.casefold() == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def _validate_remote_artifact_settings(url: str, expected_sha256: str | None) -> None:
    parsed_url = urllib.parse.urlparse(url)
    is_loopback_http = parsed_url.scheme == "http" and _is_loopback_host(parsed_url.hostname)
    if parsed_url.scheme != "https" and not is_loopback_http:
        raise ModelArtifactError(
            "Remote model artifact URL must use https, except loopback URLs used for local tests."
        )

    if expected_sha256 is None:
        raise ModelArtifactError(f"{MODEL_SHA256_ENV} is required for remote model artifacts.")

    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ModelArtifactError(f"{MODEL_SHA256_ENV} must be a 64-character hex SHA-256 digest.")


def _download_model_artifact(url: str, destination: Path) -> Path:
    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.scheme not in {"http", "https"}:
        raise ModelArtifactError("Model artifact URL must use http or https.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_destination = destination.with_suffix(f"{destination.suffix}.partial")

    try:
        with (
            urllib.request.urlopen(url, timeout=120) as response,
            partial_destination.open("wb") as file,
        ):
            shutil.copyfileobj(response, file)
        partial_destination.replace(destination)
    except (OSError, urllib.error.URLError) as exc:
        partial_destination.unlink(missing_ok=True)
        message = f"Could not download model artifact: {type(exc).__name__}"
        raise ModelArtifactError(message) from exc

    return destination


def _resolve_model_artifact(settings: AppSettings) -> tuple[Path | None, str | None]:
    if settings.model_path is not None:
        model_path = _resolve_local_path(settings.model_path)
        if not model_path.is_file():
            return model_path, f"Configured model artifact does not exist: {model_path.name}"
        return model_path, None

    if settings.model_url is None:
        return None, f"{MODEL_PATH_ENV} or {MODEL_URL_ENV} must be set."

    try:
        _validate_remote_artifact_settings(settings.model_url, settings.model_sha256)
    except ModelArtifactError as exc:
        return _resolve_local_path(settings.model_cache_path), str(exc)

    cache_path = _resolve_local_path(settings.model_cache_path)
    if cache_path.is_file():
        try:
            _validate_checksum(cache_path, settings.model_sha256)
            return cache_path, None
        except ModelArtifactError:
            cache_path.unlink(missing_ok=True)

    try:
        downloaded_path = _download_model_artifact(settings.model_url, cache_path)
        _validate_checksum(downloaded_path, settings.model_sha256)
    except ModelArtifactError as exc:
        cache_path.unlink(missing_ok=True)
        return cache_path, str(exc)

    return cache_path, None


def _public_path(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.name


def _validation_message(exc: RequestValidationError) -> str:
    for error in exc.errors():
        location = tuple(error.get("loc", ()))
        if "text" not in location:
            continue

        if error.get("type") == "missing":
            return "text is required."

        message = str(error.get("msg", "Invalid text."))
        return message.removeprefix("Value error, ")

    return "Invalid request body."


def _load_model_state(settings: AppSettings) -> ModelState:
    model_path, artifact_error = _resolve_model_artifact(settings)
    if artifact_error is not None:
        return ModelState(
            classifier=None,
            path=model_path,
            metadata={},
            error=artifact_error,
        )
    if model_path is None:
        return ModelState(classifier=None, path=None, metadata={}, error="Model artifact missing.")

    try:
        model, metadata = load_model(model_path)
    except Exception as exc:  # pragma: no cover - depends on local artifact corruption.
        return ModelState(
            classifier=None,
            path=model_path,
            metadata={},
            error=f"Could not load model artifact: {type(exc).__name__}",
        )

    return ModelState(
        classifier=ThreatClassifier(model),
        path=model_path,
        metadata=metadata,
    )


class ModelService:
    """Lazily resolve and load a trusted model artifact once per process."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._state: ModelState | None = None
        self._warmup_thread: threading.Thread | None = None

    def get_state(self) -> ModelState:
        """Return the current model state, loading the artifact on first use."""
        if self._state is not None and (self._state.loaded or self._settings.model_url is None):
            return self._state

        with self._lock:
            should_retry_remote_failure = (
                self._state is not None
                and not self._state.loaded
                and self._settings.model_url is not None
            )
            if self._state is None or should_retry_remote_failure:
                self._state = _load_model_state(self._settings)
            return self._state

    def start_background_warmup(self) -> None:
        """Start a non-blocking model load for hosted runtimes."""
        if self._warmup_thread is not None:
            return

        self._warmup_thread = threading.Thread(
            target=self.get_state,
            name="threatlens-model-warmup",
            daemon=True,
        )
        self._warmup_thread.start()


def _load_metrics(path: Path) -> dict[str, object] | None:
    metrics_path = _resolve_local_path(path)
    if not metrics_path.is_file():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _public_model_metadata(metadata: dict[str, object]) -> dict[str, object]:
    """Return artifact metadata fields safe to expose through the browser API."""
    return {key: value for key, value in metadata.items() if key in PUBLIC_METADATA_KEYS}


def _normalize_probabilities(probabilities: dict[str, float] | None) -> dict[str, float] | None:
    if probabilities is None:
        return None

    normalized: dict[str, float] = {}
    for raw_label, score in probabilities.items():
        try:
            label = normalize_label(raw_label)
        except ValueError:
            continue
        normalized[label] = float(score)

    return normalized or None


def _normalize_prediction_label(label: str) -> ThreatLabel:
    normalized = normalize_label(label)
    return cast(ThreatLabel, normalized)


def _confidence(label: str, probabilities: dict[str, float] | None) -> float | None:
    if probabilities is None:
        return None
    return probabilities.get(label)


def _risk_level(label: str, probabilities: dict[str, float] | None) -> RiskLevel:
    confidence = _confidence(label, probabilities)
    if confidence is not None and confidence < 0.55:
        return "medium"

    if label == "phish":
        return "high"
    if label == "spam":
        return "medium"
    return "low"


def _matched_signals(text: str) -> list[str]:
    return [name for name, pattern in TRANSPARENT_SIGNALS.items() if pattern.search(text)]


def _explanation(label: str, probabilities: dict[str, float] | None, text: str) -> str:
    confidence = _confidence(label, probabilities)
    confidence_text = f" with {confidence:.0%} confidence" if confidence is not None else ""
    signals = _matched_signals(text)
    if signals:
        signal_text = ", ".join(signals[:3])
        return (
            f"The trained model classified this message as {label}{confidence_text}. "
            f"Transparent text signals observed: {signal_text}."
        )
    return (
        f"The trained model classified this message as {label}{confidence_text}. "
        "No obvious keyword signal dominated the explanation, so treat the score as model-driven."
    )


def _suggested_action(label: str, risk_level: RiskLevel) -> str:
    if label == "phish":
        return (
            "Do not click links or share credentials. Verify the request through a trusted "
            "channel and report it to your security team."
        )
    if label == "spam":
        return "Avoid engaging with the sender. Mark it as spam or delete it if it is unsolicited."
    if risk_level == "medium":
        return "Review the sender and context before acting; the model confidence is not decisive."
    return "Low apparent risk. Continue normal handling, while still checking sender context."


def _health_response(state: ModelState) -> HealthResponse:
    if state.loaded:
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_path=_public_path(state.path),
            detail="Model artifact loaded.",
        )
    return HealthResponse(
        status="error",
        model_loaded=False,
        model_path=_public_path(state.path),
        detail=state.error or "Model artifact is unavailable.",
    )


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """Create the ThreatLens API app."""
    app_settings = settings or AppSettings.from_env()
    model_service = ModelService(app_settings)
    metrics = _load_metrics(app_settings.metrics_path)

    app = FastAPI(
        title="ThreatLens API",
        summary="Artifact-backed spam and phishing classification.",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(app_settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.state.settings = app_settings
    app.state.model_service = model_service
    app.state.metrics = metrics
    if app_settings.background_warmup:
        model_service.start_background_warmup()

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": _validation_message(exc)})

    @app.get("/health", response_model=HealthResponse)
    def health() -> JSONResponse | HealthResponse:
        response = _health_response(app.state.model_service.get_state())
        if not response.model_loaded:
            return JSONResponse(status_code=503, content=response.model_dump())
        return response

    @app.get("/live", response_model=LiveResponse)
    def live() -> LiveResponse:
        return LiveResponse(
            status="ok",
            app_name="ThreatLens",
            detail="API process is running.",
        )

    @app.get("/metadata", response_model=MetadataResponse)
    def metadata() -> MetadataResponse:
        state: ModelState = app.state.model_service.get_state()
        return MetadataResponse(
            app_name="ThreatLens",
            labels=list(LABEL_NAMES),
            max_text_chars=app.state.settings.max_text_chars,
            model={
                "loaded": state.loaded,
                "artifact": _public_path(state.path),
                "metadata": _public_model_metadata(state.metadata),
                "status": "ready" if state.loaded else state.error,
            },
            metrics=app.state.metrics,
            privacy="Messages are analyzed for the current request only and are not stored.",
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest) -> PredictResponse:
        state: ModelState = app.state.model_service.get_state()
        if not state.loaded or state.classifier is None:
            raise HTTPException(
                status_code=503,
                detail=state.error or "Model artifact is unavailable.",
            )

        result = state.classifier.predict_one(payload.text)
        label = _normalize_prediction_label(result.label)
        probabilities = _normalize_probabilities(result.probabilities)
        risk_level = _risk_level(label, probabilities)
        return PredictResponse(
            label=label,
            probabilities=probabilities,
            risk_level=risk_level,
            explanation=_explanation(label, probabilities, payload.text),
            suggested_action=_suggested_action(label, risk_level),
        )

    return app


app = create_app()
