"""FastAPI application factory and versioned ThreatLens routes."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from web.backend.explanations import (
    confidence_for,
    explanation_for,
    risk_level_for,
    suggested_action_for,
)
from web.backend.labels import LABEL_NAMES
from web.backend.schemas import (
    ArtifactMetadata,
    HealthResponse,
    LiveResponse,
    MetadataResponse,
    ModelPredictionOutput,
    PredictRequest,
    PredictResponse,
    VersionedPredictResponse,
)
from web.backend.security import PredictionSecurityMiddleware
from web.backend.settings import AppSettings
from web.backend.sklearn_runtime import (
    ModelArtifactError,
    ModelService,
    ModelState,
    _download_model_artifact,
    load_metrics,
    normalize_prediction_label,
    normalize_probabilities,
    public_artifact_metadata,
    public_model_metadata,
    public_path,
)

APP_NAME = "ThreatLens"
PRIVACY_NOTICE = "Messages are analyzed for the current request only and are not stored."


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


def _health_response(state: ModelState) -> HealthResponse:
    if state.loaded:
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_path=public_path(state.path),
            detail="Model artifact loaded.",
        )
    return HealthResponse(
        status="error",
        model_loaded=False,
        model_path=public_path(state.path),
        detail=state.error or "Model artifact is unavailable.",
    )


def _classify(payload: PredictRequest, model_service: ModelService) -> PredictResponse:
    state = model_service.get_state()
    if not state.loaded or state.classifier is None:
        raise HTTPException(status_code=503, detail=state.error or "Model artifact is unavailable.")

    result = state.classifier.predict_one(payload.text)
    label = normalize_prediction_label(result.label)
    probabilities = normalize_probabilities(result.probabilities)
    risk_level = risk_level_for(label, probabilities)
    return PredictResponse(
        label=label,
        probabilities=probabilities,
        risk_level=risk_level,
        explanation=explanation_for(label, probabilities, payload.text),
        suggested_action=suggested_action_for(label, risk_level),
    )


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """Create the ThreatLens API while retaining hidden legacy aliases."""
    app_settings = settings or AppSettings.from_env()
    model_service = ModelService(
        app_settings,
        artifact_downloader=lambda url, destination: _download_model_artifact(
            url,
            destination,
            app_settings.max_artifact_bytes,
        ),
    )

    app = FastAPI(
        title="ThreatLens API",
        summary="Artifact-backed spam and phishing classification.",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(app_settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.add_middleware(PredictionSecurityMiddleware, settings=app_settings)

    app.state.settings = app_settings
    app.state.model_service = model_service
    app.state.metrics = load_metrics(app_settings.metrics_path)
    if app_settings.background_warmup:
        model_service.start_background_warmup()

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": _validation_message(exc)})

    @app.get("/health", response_model=HealthResponse, include_in_schema=False)
    @app.get("/v1/ready", response_model=HealthResponse)
    def ready() -> JSONResponse | HealthResponse:
        response = _health_response(model_service.get_state())
        if not response.model_loaded:
            return JSONResponse(status_code=503, content=response.model_dump())
        return response

    @app.get("/live", response_model=LiveResponse, include_in_schema=False)
    @app.get("/v1/live", response_model=LiveResponse)
    def live() -> LiveResponse:
        return LiveResponse(status="ok", app_name=APP_NAME, detail="API process is running.")

    @app.get("/metadata", response_model=MetadataResponse, include_in_schema=False)
    @app.get("/v1/metadata", response_model=MetadataResponse)
    def metadata() -> MetadataResponse:
        state = model_service.get_state()
        return MetadataResponse(
            app_name=APP_NAME,
            labels=list(LABEL_NAMES),
            max_text_chars=app_settings.max_text_chars,
            model={
                "loaded": state.loaded,
                "artifact": public_path(state.path),
                "metadata": public_model_metadata(state.metadata),
                "status": "ready" if state.loaded else state.error,
            },
            metrics=app.state.metrics,
            privacy=PRIVACY_NOTICE,
        )

    @app.post("/predict", response_model=PredictResponse, include_in_schema=False)
    def legacy_predict(payload: PredictRequest) -> PredictResponse:
        return _classify(payload, model_service)

    @app.post("/v1/predict", response_model=VersionedPredictResponse)
    def predict(payload: PredictRequest) -> VersionedPredictResponse:
        legacy = _classify(payload, model_service)
        state = model_service.get_state()
        confidence = confidence_for(legacy.label, legacy.probabilities)
        return VersionedPredictResponse(
            final_label=legacy.label,
            final_risk_level=legacy.risk_level,
            final_confidence=confidence,
            model_outputs={
                "tfidf_logreg": ModelPredictionOutput(
                    label=legacy.label,
                    confidence=confidence,
                    probabilities=legacy.probabilities,
                )
            },
            explanation=legacy.explanation,
            suggested_action=legacy.suggested_action,
            artifact_metadata=ArtifactMetadata(**public_artifact_metadata(state)),
        )

    return app


app = create_app()

__all__ = ["AppSettings", "ModelArtifactError", "app", "create_app"]
