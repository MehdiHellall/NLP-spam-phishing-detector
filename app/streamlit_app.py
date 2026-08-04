"""Small Streamlit app for artifact-backed email threat classification."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import streamlit as st

MODEL_PATH_ENV = "EMAIL_THREAT_MODEL_PATH"
LABELS = ("ham", "phish", "spam")

SAMPLE_MESSAGES = {
    "Ham": "Hi Jordan, can we move our project sync to 2 PM tomorrow? Thanks.",
    "Phish": "Urgent password reset required verify account.",
    "Spam": "Limited offer coupon savings buy now.",
}

@dataclass(frozen=True)
class Prediction:
    """Prediction label and optional class probabilities."""

    label: str
    probabilities: dict[str, float] | None = None


@dataclass(frozen=True)
class ModelState:
    """Loaded model state for the app."""

    mode: str
    model: Any | None = None
    path: Path | None = None
    metadata: dict[str, Any] | None = None
    message: str | None = None


def _add_local_src_to_path() -> None:
    """Allow joblib artifacts to import the local package when run from the repo."""
    src_dir = Path(__file__).resolve().parents[1] / "src"
    src_path = str(src_dir)
    if src_dir.exists() and src_path not in sys.path:
        sys.path.insert(0, src_path)


def _resolve_artifact_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


@st.cache_resource(show_spinner=False)
def load_model_state(raw_path: str | None) -> ModelState:
    """Load a trained artifact if the configured path points to a file."""
    if not raw_path:
        return ModelState(
            mode="missing",
            message=f"{MODEL_PATH_ENV} is not set.",
        )

    artifact_path = _resolve_artifact_path(raw_path)
    if not artifact_path.is_file():
        return ModelState(
            mode="missing",
            path=artifact_path,
            message="Configured artifact path does not exist.",
        )

    try:
        _add_local_src_to_path()
        payload = joblib.load(artifact_path)
    except Exception as exc:  # pragma: no cover - depends on local artifact shape.
        return ModelState(
            mode="error",
            path=artifact_path,
            message=f"Could not load artifact: {type(exc).__name__}",
        )

    if isinstance(payload, dict) and "model" in payload:
        metadata = payload.get("metadata") or {}
        return ModelState(
            mode="artifact",
            model=payload["model"],
            path=artifact_path,
            metadata=dict(metadata),
        )

    return ModelState(mode="artifact", model=payload, path=artifact_path, metadata={})


def normalize_label(value: object) -> str:
    """Map common class ids and aliases to display labels."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int) and 0 <= value < len(LABELS):
        return LABELS[value]

    label = str(value).strip().casefold()
    aliases = {
        "0": "ham",
        "1": "phish",
        "2": "spam",
        "legit": "ham",
        "legitimate": "ham",
        "normal": "ham",
        "phishing": "phish",
        "scam": "phish",
        "junk": "spam",
    }
    return aliases.get(label, label)


def predict_with_model(model: Any, text: str) -> Prediction:
    """Run a scikit-learn compatible model against one message."""
    raw_label = model.predict([text])[0]
    probabilities = None

    if hasattr(model, "predict_proba"):
        scores = [float(score) for score in model.predict_proba([text])[0]]
        raw_classes = getattr(model, "classes_", LABELS)
        classes = [normalize_label(raw_class) for raw_class in raw_classes]
        probabilities = {
            label: score
            for label, score in zip(classes, scores, strict=False)
            if label in LABELS
        }

    return Prediction(label=normalize_label(raw_label), probabilities=probabilities)


def render_sample_buttons() -> None:
    """Render buttons that fill the message box with safe synthetic samples."""
    columns = st.columns(len(SAMPLE_MESSAGES))
    for column, (label, sample) in zip(columns, SAMPLE_MESSAGES.items(), strict=True):
        if column.button(label, use_container_width=True):
            st.session_state["message"] = sample


def render_probabilities(probabilities: dict[str, float]) -> None:
    """Display probabilities only when the active model provides them."""
    st.markdown("**Probabilities**")
    for label in LABELS:
        if label not in probabilities:
            continue
        score = max(0.0, min(1.0, probabilities[label]))
        st.progress(score, text=f"{label}: {score:.1%}")


def render_model_status(state: ModelState) -> None:
    """Show whether predictions are backed by a trained artifact."""
    with st.sidebar:
        st.header("Model")
        st.code(f"{MODEL_PATH_ENV}=<path>", language="text")

        if state.mode == "artifact":
            st.success("Trained artifact loaded.")
            st.caption(state.path.name if state.path is not None else "artifact")
            if state.metadata:
                with st.expander("Artifact metadata"):
                    st.json(state.metadata)
            return

        st.error("No trained artifact loaded.")
        st.caption(state.message or "No trained artifact was loaded.")
        if state.path is not None:
            st.caption(state.path.name)


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title="Email Threat Detector", layout="centered")
    state = load_model_state(os.getenv(MODEL_PATH_ENV))

    st.title("Email Threat Detector")
    st.caption("Paste a message to classify it as ham, phish, or spam.")
    render_model_status(state)

    st.caption("Only load model artifacts that you created or otherwise trust.")

    if state.mode != "artifact" or state.model is None:
        st.stop()

    render_sample_buttons()
    message = st.text_area(
        "Message",
        key="message",
        height=160,
        placeholder="Paste an email, SMS, or short message here.",
    )

    if not message.strip():
        st.info("Enter a message or choose a sample to see a prediction.")
        return

    try:
        prediction = predict_with_model(state.model, message)
    except Exception as exc:  # pragma: no cover - depends on local artifact behavior.
        st.error(f"Could not run prediction with the loaded artifact: {type(exc).__name__}")
        return

    st.subheader("Prediction")
    st.metric("Class", prediction.label)

    if prediction.probabilities:
        render_probabilities(prediction.probabilities)


if __name__ == "__main__":
    main()
