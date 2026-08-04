import importlib
import sys
from types import SimpleNamespace

import joblib
import pytest


class _FakeSidebar:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class _FakeStreamlit:
    sidebar = _FakeSidebar()
    session_state = {}

    def cache_resource(self, **_kwargs):
        def decorator(func):
            return func

        return decorator

    def columns(self, count):
        return [self for _ in range(count)]

    def button(self, *_args, **_kwargs):
        return False

    def expander(self, *_args, **_kwargs):
        return _FakeSidebar()

    def text_area(self, *_args, **_kwargs):
        return ""

    def stop(self):
        raise RuntimeError("streamlit stopped")

    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


@pytest.fixture()
def streamlit_app(monkeypatch):
    monkeypatch.setitem(sys.modules, "streamlit", _FakeStreamlit())
    sys.modules.pop("app.streamlit_app", None)
    return importlib.import_module("app.streamlit_app")


def test_label_normalization(streamlit_app) -> None:
    assert streamlit_app.normalize_label(0) == "ham"
    assert streamlit_app.normalize_label("phishing") == "phish"


def test_model_prediction_includes_probabilities(streamlit_app) -> None:
    model = SimpleNamespace(
        classes_=["ham", "phish", "spam"],
        predict=lambda texts: ["phish"],
        predict_proba=lambda texts: [[0.1, 0.8, 0.1]],
    )

    prediction = streamlit_app.predict_with_model(model, "urgent login")

    assert prediction.label == "phish"
    assert prediction.probabilities == {"ham": 0.1, "phish": 0.8, "spam": 0.1}


def test_load_model_state_requires_artifact(streamlit_app, tmp_path) -> None:
    missing_path = tmp_path / "missing.joblib"

    assert streamlit_app.load_model_state(None).mode == "missing"
    missing = streamlit_app.load_model_state(str(missing_path))
    assert missing.mode == "missing"
    assert missing.path == missing_path

    with pytest.raises(RuntimeError, match="streamlit stopped"):
        streamlit_app.main()


def test_load_model_state_artifact_and_bad_artifact(streamlit_app, tmp_path) -> None:
    artifact_path = tmp_path / "model.joblib"
    bad_artifact_path = tmp_path / "bad.joblib"
    joblib.dump({"model": "model", "metadata": {"kind": "test"}}, artifact_path)
    bad_artifact_path.write_text("not a joblib artifact", encoding="utf-8")

    loaded = streamlit_app.load_model_state(str(artifact_path))
    bad = streamlit_app.load_model_state(str(bad_artifact_path))

    assert loaded.mode == "artifact"
    assert loaded.metadata == {"kind": "test"}
    assert bad.mode == "error"
    assert "Could not load artifact" in bad.message
