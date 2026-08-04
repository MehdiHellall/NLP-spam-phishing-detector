"""Leakage-safe sklearn baseline model factories."""

from __future__ import annotations

from collections.abc import Callable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

from email_threat_detector.constants import DEFAULT_RANDOM_SEED
from email_threat_detector.preprocessing import basic_text_preprocessor

ModelBuilder = Callable[..., Pipeline]


def build_tfidf_logistic_regression_pipeline(
    *,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> Pipeline:
    """Build a TF-IDF + Logistic Regression baseline pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=basic_text_preprocessor,
                    lowercase=False,
                    ngram_range=(1, 2),
                    min_df=1,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_tfidf_complement_nb_pipeline(
    *,
    alpha: float = 0.5,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> Pipeline:
    """Build a TF-IDF + Complement Naive Bayes baseline pipeline."""
    _ = random_state
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=basic_text_preprocessor,
                    lowercase=False,
                    ngram_range=(1, 2),
                    min_df=1,
                ),
            ),
            ("classifier", ComplementNB(alpha=alpha)),
        ]
    )


BASELINE_MODEL_BUILDERS: dict[str, ModelBuilder] = {
    "tfidf_logreg": build_tfidf_logistic_regression_pipeline,
    "tfidf_complement_nb": build_tfidf_complement_nb_pipeline,
}


def build_baseline_model(
    name: str,
    *,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> Pipeline:
    """Build a named baseline model."""
    try:
        builder = BASELINE_MODEL_BUILDERS[name]
    except KeyError as exc:
        available = ", ".join(sorted(BASELINE_MODEL_BUILDERS))
        raise ValueError(f"Unknown baseline model {name!r}. Available: {available}") from exc
    return builder(random_state=random_state)
