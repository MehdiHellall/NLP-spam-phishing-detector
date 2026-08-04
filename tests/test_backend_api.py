from __future__ import annotations

import hashlib
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from web.backend import main as backend_main
from web.backend.main import AppSettings, create_app
from web.backend.model import save_model

CHECKED_IN_ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "tfidf_logreg.joblib"
CHECKED_IN_PHISH_SAMPLE = (
    "replica watches aceyuhardepstatenjus buy replica rolex fraction price safe 15 "
    "buy 2 watches go cocktail party watch sure catch peoples attention youll class "
    "still money come replica watches shop httpgzhifgasolcn thu 07 aug 2008 041343 "
    "0000 tag heuer watches"
)


class PicklableThreatModel:
    classes_ = ["ham", "phish", "spam"]

    def predict(self, texts):
        return ["phish" for _ in texts]

    def predict_proba(self, texts):
        return [[0.03, 0.92, 0.05] for _ in texts]


class NumericClassThreatModel:
    classes_ = [0, 1, 2]

    def predict(self, texts):
        return [1 for _ in texts]

    def predict_proba(self, texts):
        return [[0.04, 0.91, 0.05] for _ in texts]


class QuietStaticFileHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture()
def artifact_path(tmp_path: Path) -> Path:
    path = tmp_path / "threat-model.joblib"
    save_model(
        PicklableThreatModel(),
        path,
        metadata={
            "model_name": "test_model",
            "metrics_file": "test_metrics.json",
            "local_path": str(tmp_path),
        },
    )
    return path


@pytest.fixture()
def artifact_server(tmp_path: Path):
    source_dir = tmp_path / "server"
    source_dir.mkdir()
    source_path = source_dir / "remote-model.joblib"
    save_model(PicklableThreatModel(), source_path)

    handler = partial(QuietStaticFileHandler, directory=str(source_dir))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        yield {
            "url": f"http://127.0.0.1:{server.server_port}/remote-model.joblib",
            "sha256": digest,
        }
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _client(model_path: Path) -> TestClient:
    app = create_app(AppSettings(model_path=model_path))
    return TestClient(app)


def test_health_reports_loaded_model(artifact_path: Path) -> None:
    response = _client(artifact_path).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_path": "threat-model.joblib",
        "detail": "Model artifact loaded.",
    }


def test_live_does_not_require_model_artifact(tmp_path: Path) -> None:
    client = _client(tmp_path / "missing.joblib")

    response = client.get("/live")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app_name": "ThreatLens",
        "detail": "API process is running.",
    }


def test_api_responses_include_security_headers(artifact_path: Path) -> None:
    response = _client(artifact_path).get("/health")

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize("route", ["/predict", "/v1/predict"])
def test_predict_rejects_missing_and_empty_text(artifact_path: Path, route: str) -> None:
    client = _client(artifact_path)

    missing = client.post(route, json={})
    empty = client.post(route, json={"text": "   "})

    assert missing.status_code == 422
    assert empty.status_code == 422
    assert empty.json()["detail"] == "text must not be empty."


@pytest.mark.parametrize("route", ["/predict", "/v1/predict"])
def test_predict_rejects_oversized_request_before_prediction(
    artifact_path: Path,
    route: str,
) -> None:
    app = create_app(AppSettings(model_path=artifact_path, max_body_bytes=16))
    client = TestClient(app)

    response = client.post(route, json={"text": "verify account password"})

    assert response.status_code == 413
    assert "16 bytes or fewer" in response.json()["detail"]


@pytest.mark.parametrize("route", ["/predict", "/v1/predict"])
def test_predict_preflight_is_not_blocked_by_body_guards(
    artifact_path: Path,
    route: str,
) -> None:
    client = TestClient(
        create_app(
            AppSettings(
                model_path=artifact_path,
                allowed_origins=("http://localhost:8080",),
            )
        )
    )

    response = client.options(
        route,
        headers={
            "Origin": "http://localhost:8080",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:8080"


@pytest.mark.parametrize("route", ["/predict", "/v1/predict"])
def test_predict_rate_limits_repeated_requests(artifact_path: Path, route: str) -> None:
    app = create_app(AppSettings(model_path=artifact_path, rate_limit_per_minute=1))
    client = TestClient(app)

    first = client.post(route, json={"text": "verify account"})
    second = client.post(route, json={"text": "verify account"})

    assert first.status_code == 200
    assert second.status_code == 429
    assert "Too many prediction requests" in second.json()["detail"]


def test_predict_returns_real_classifier_result(artifact_path: Path) -> None:
    response = _client(artifact_path).post(
        "/predict",
        json={"text": "  Urgent password reset required verify account.  "},
    )

    body = response.json()

    assert response.status_code == 200
    assert body["label"] == "phish"
    assert body["probabilities"] == {"ham": 0.03, "phish": 0.92, "spam": 0.05}
    assert body["risk_level"] == "high"
    assert "trained model classified" in body["explanation"]
    assert "Do not click" in body["suggested_action"]


def test_checked_in_tfidf_artifact_prediction_is_characterized() -> None:
    client = _client(CHECKED_IN_ARTIFACT)

    phish = client.post("/predict", json={"text": CHECKED_IN_PHISH_SAMPLE})
    ham = client.post("/predict", json={"text": "Hey, are we still meeting for lunch tomorrow?"})
    spam = client.post(
        "/predict",
        json={
            "text": (
                "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. "
                "Text FA to 87121 to receive entry question(std txt rate)"
            )
        },
    )

    assert phish.status_code == 200
    assert phish.json()["label"] == "phish"
    assert phish.json()["probabilities"] == pytest.approx(
        {
            "ham": 0.0003345602296797188,
            "phish": 0.9973158094498464,
            "spam": 0.002349630320473904,
        },
        abs=1e-8,
    )
    assert ham.status_code == 200
    assert ham.json()["label"] == "ham"
    assert spam.status_code == 200
    assert spam.json()["label"] == "spam"


def test_v1_live_preserves_live_contract(artifact_path: Path) -> None:
    client = _client(artifact_path)

    legacy = client.get("/live")
    versioned = client.get("/v1/live")

    assert versioned.status_code == 200
    assert (
        versioned.json()
        == legacy.json()
        == {
            "status": "ok",
            "app_name": "ThreatLens",
            "detail": "API process is running.",
        }
    )


def test_v1_ready_preserves_health_contract(artifact_path: Path) -> None:
    client = _client(artifact_path)

    legacy = client.get("/health")
    versioned = client.get("/v1/ready")

    assert versioned.status_code == 200
    assert (
        versioned.json()
        == legacy.json()
        == {
            "status": "ok",
            "model_loaded": True,
            "model_path": "threat-model.joblib",
            "detail": "Model artifact loaded.",
        }
    )


def test_v1_metadata_preserves_public_metadata_contract(artifact_path: Path) -> None:
    client = _client(artifact_path)

    legacy = client.get("/metadata")
    versioned = client.get("/v1/metadata")

    assert versioned.status_code == 200
    assert versioned.json() == legacy.json()
    assert set(versioned.json()) == {
        "app_name",
        "labels",
        "max_text_chars",
        "model",
        "metrics",
        "privacy",
    }
    assert versioned.json()["model"] == {
        "loaded": True,
        "artifact": "threat-model.joblib",
        "metadata": {
            "metrics_file": "test_metrics.json",
            "model_name": "test_model",
        },
        "status": "ready",
    }


def test_v1_predict_has_duel_ready_shape_and_preserves_legacy_result(
    artifact_path: Path,
) -> None:
    client = _client(artifact_path)
    payload = {"text": "Urgent password reset required verify account."}

    legacy = client.post("/predict", json=payload)
    versioned = client.post("/v1/predict", json=payload)

    expected_legacy = {
        "label": "phish",
        "probabilities": {"ham": 0.03, "phish": 0.92, "spam": 0.05},
        "risk_level": "high",
        "explanation": (
            "The trained model classified this message as phish with 92% confidence. "
            "Transparent text signals observed: urgency, credential request."
        ),
        "suggested_action": (
            "Do not click links or share credentials. Verify the request through a trusted "
            "channel and report it to your security team."
        ),
    }
    assert legacy.status_code == 200
    assert legacy.json() == expected_legacy
    assert versioned.status_code == 200

    body = versioned.json()
    assert set(body) == {
        "final_label",
        "final_risk_level",
        "final_confidence",
        "model_outputs",
        "explanation",
        "suggested_action",
        "artifact_metadata",
    }
    assert set(body["model_outputs"]) == {"tfidf_logreg"}
    assert set(body["model_outputs"]["tfidf_logreg"]) == {
        "label",
        "confidence",
        "probabilities",
    }
    assert body["artifact_metadata"] == {
        "artifact": "threat-model.joblib",
        "model_name": "test_model",
        "metrics_file": "test_metrics.json",
    }
    assert body["final_label"] == expected_legacy["label"]
    assert body["final_risk_level"] == expected_legacy["risk_level"]
    assert body["final_confidence"] == expected_legacy["probabilities"]["phish"]
    assert body["model_outputs"]["tfidf_logreg"] == {
        "label": expected_legacy["label"],
        "confidence": expected_legacy["probabilities"]["phish"],
        "probabilities": expected_legacy["probabilities"],
    }
    assert body["explanation"] == expected_legacy["explanation"]
    assert body["suggested_action"] == expected_legacy["suggested_action"]


def test_predict_normalizes_numeric_class_probabilities(tmp_path: Path) -> None:
    model_path = tmp_path / "numeric.joblib"
    save_model(NumericClassThreatModel(), model_path)

    response = _client(model_path).post(
        "/predict",
        json={"text": "verify account password"},
    )

    assert response.status_code == 200
    assert response.json()["label"] == "phish"
    assert response.json()["probabilities"] == {"ham": 0.04, "phish": 0.91, "spam": 0.05}


def test_metadata_only_returns_public_artifact_metadata(artifact_path: Path) -> None:
    response = _client(artifact_path).get("/metadata")

    model = response.json()["model"]

    assert response.status_code == 200
    assert model["metadata"] == {
        "metrics_file": "test_metrics.json",
        "model_name": "test_model",
    }
    assert "local_path" not in model["metadata"]


def test_remote_model_url_is_downloaded_verified_and_cached(
    artifact_server,
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "cache" / "model.joblib"
    app = create_app(
        AppSettings(
            model_url=artifact_server["url"],
            model_sha256=artifact_server["sha256"],
            model_cache_path=cache_path,
        )
    )
    client = TestClient(app)

    response = client.post("/predict", json={"text": "verify account"})

    assert response.status_code == 200
    assert response.json()["label"] == "phish"
    assert cache_path.is_file()


def test_model_configuration_rejects_path_and_url_together(
    artifact_path: Path,
    artifact_server,
    tmp_path: Path,
) -> None:
    app = create_app(
        AppSettings(
            model_path=artifact_path,
            model_url=artifact_server["url"],
            model_sha256=artifact_server["sha256"],
            model_cache_path=tmp_path / "cache" / "model.joblib",
        )
    )
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 503
    assert "Configure only one" in response.json()["detail"]


def test_remote_model_url_requires_checksum(tmp_path: Path) -> None:
    app = create_app(
        AppSettings(
            model_url="https://example.com/remote-model.joblib",
            model_cache_path=tmp_path / "model.joblib",
        )
    )
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 503
    assert "EMAIL_THREAT_MODEL_SHA256 is required" in response.json()["detail"]


def test_remote_model_url_rejects_malformed_checksum(tmp_path: Path) -> None:
    app = create_app(
        AppSettings(
            model_url="https://example.com/remote-model.joblib",
            model_sha256="not-a-sha",
            model_cache_path=tmp_path / "model.joblib",
        )
    )
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 503
    assert "64-character hex SHA-256" in response.json()["detail"]


def test_remote_model_url_requires_https_for_non_loopback(tmp_path: Path) -> None:
    app = create_app(
        AppSettings(
            model_url="http://example.com/remote-model.joblib",
            model_sha256="0" * 64,
            model_cache_path=tmp_path / "model.joblib",
        )
    )
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 503
    assert "must use https" in response.json()["detail"]


def test_remote_model_url_rejects_checksum_mismatch(artifact_server, tmp_path: Path) -> None:
    cache_path = tmp_path / "cache" / "model.joblib"
    app = create_app(
        AppSettings(
            model_url=artifact_server["url"],
            model_sha256="0" * 64,
            model_cache_path=cache_path,
        )
    )
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 503
    assert "checksum mismatch" in response.json()["detail"]
    assert not cache_path.exists()


def test_remote_model_url_retry_recovers_after_initial_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.joblib"
    cache_path = tmp_path / "cache" / "model.joblib"
    save_model(PicklableThreatModel(), source_path)
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    calls = 0

    def fake_download_model_artifact(
        _url: str,
        destination: Path,
        max_artifact_bytes: int,
    ) -> Path:
        nonlocal calls
        assert max_artifact_bytes == 1_073_741_824
        calls += 1
        if calls == 1:
            raise backend_main.ModelArtifactError("temporary artifact outage")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source_path.read_bytes())
        return destination

    monkeypatch.setattr(
        backend_main,
        "_download_model_artifact",
        fake_download_model_artifact,
    )
    app = create_app(
        AppSettings(
            model_url="https://example.com/remote-model.joblib",
            model_sha256=digest,
            model_cache_path=cache_path,
            model_retry_seconds=0,
        )
    )
    client = TestClient(app)

    first = client.get("/health")
    second = client.get("/health")

    assert first.status_code == 503
    assert "temporary artifact outage" in first.json()["detail"]
    assert second.status_code == 200
    assert second.json()["model_loaded"] is True
    assert calls == 2


def test_missing_model_returns_health_error(tmp_path: Path) -> None:
    client = _client(tmp_path / "missing.joblib")

    health = client.get("/health")
    prediction = client.post("/predict", json={"text": "hello"})

    assert health.status_code == 503
    assert health.json()["model_loaded"] is False
    assert "does not exist" in health.json()["detail"]
    assert prediction.status_code == 503
    assert "does not exist" in prediction.json()["detail"]
