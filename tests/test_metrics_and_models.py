import pandas as pd
import pytest

from email_threat_detector.inference import ThreatClassifier, load_model, save_model
from email_threat_detector.metrics import compute_classification_metrics
from email_threat_detector.models import build_baseline_model


class PicklableProbabilityModel:
    classes_ = ["ham", "phish", "spam"]

    def predict(self, texts):
        return ["phish" for _ in texts]

    def predict_proba(self, texts):
        return [[0.1, 0.8, 0.1] for _ in texts]


def test_compute_classification_metrics_returns_summary_per_label_and_confusion_matrix() -> None:
    metrics = compute_classification_metrics(
        ["ham", "phish", "spam", "spam"],
        ["ham", "phish", "ham", "spam"],
    )

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision_macro"] == pytest.approx((0.5 + 1.0 + 1.0) / 3)
    assert metrics["recall_macro"] == pytest.approx((1.0 + 1.0 + 0.5) / 3)
    assert metrics["f1_macro"] == pytest.approx((2 / 3 + 1.0 + 2 / 3) / 3)
    assert metrics["per_label"]["ham"] == {
        "precision": 0.5,
        "recall": 1.0,
        "f1": pytest.approx(2 / 3),
        "support": 1,
    }
    assert metrics["per_label"]["spam"] == {
        "precision": 1.0,
        "recall": 0.5,
        "f1": pytest.approx(2 / 3),
        "support": 2,
    }
    assert metrics["confusion_matrix"] == [
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 1],
    ]


def test_baseline_pipeline_inference_with_probabilities() -> None:
    training = pd.DataFrame(
        {
            "text": [
                "team lunch schedule",
                "project meeting notes",
                "family dinner plan",
                "verify password immediately",
                "account suspended reset now",
                "confirm bank login",
                "free prize money now",
                "limited offer coupon",
                "cheap meds discount",
            ],
            "label": [
                "ham",
                "ham",
                "ham",
                "phish",
                "phish",
                "phish",
                "spam",
                "spam",
                "spam",
            ],
        }
    )
    model = build_baseline_model("tfidf_complement_nb", random_state=7)

    model.fit(training["text"], training["label"])
    result = ThreatClassifier(model).predict_one("urgent password reset required")

    assert result.label in {"ham", "phish", "spam"}
    assert result.probabilities is not None
    assert set(result.probabilities) == {"ham", "phish", "spam"}
    assert sum(result.probabilities.values()) == pytest.approx(1.0)


def test_model_factories_and_inference_persistence(tmp_path) -> None:
    pipeline = build_baseline_model("tfidf_logreg", random_state=7)
    training = pd.DataFrame(
        {
            "text": [
                "team agenda",
                "office notes",
                "verify password",
                "bank login",
                "free coupon",
                "promo prize",
            ],
            "label": ["ham", "ham", "phish", "phish", "spam", "spam"],
        }
    )
    pipeline.fit(training["text"], training["label"])
    model_path = tmp_path / "model.joblib"

    save_model(PicklableProbabilityModel(), model_path, metadata={"source": "test"})
    loaded, metadata = load_model(model_path)
    result = ThreatClassifier.from_path(model_path).predict_one("verify bank password")

    assert metadata == {"source": "test"}
    assert pipeline.predict(["team agenda"])[0] in {"ham", "phish", "spam"}
    assert loaded.predict(["team agenda"]) == ["phish"]
    assert result.label == "phish"

    with pytest.raises(ValueError, match="Unknown baseline model"):
        build_baseline_model("missing")
