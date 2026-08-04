# NLP Threat Detector

Detect spam and phishing messages with reusable NLP utilities, classical machine-learning baselines, transformer experiments, and security-focused evaluation.

## Project Status

This repository is being refactored from an academic notebook project into a reproducible ML engineering portfolio project. The task is a three-class email/message classification problem:

- `ham`: legitimate message
- `phish`: phishing or attack-oriented message
- `spam`: unwanted promotional or abusive message

The reusable Python package now includes data loading, cleaning, duplicate auditing, deterministic sampling and splitting, leakage checks, sklearn baseline factories, metric helpers, inference utilities, and CLI entry points for baseline and transformer runs. Reproducible baseline and compact-transformer metrics are committed under `reports/metrics/`.

## Result Provenance

Keep these two result types separate when describing the project:

| Result type | Current status |
| --- | --- |
| Reproducible script results | Real Kaggle-backed runs were executed through the leakage-safe CLI. TF-IDF + Logistic Regression reached `95.76%` test accuracy and `95.76%` macro F1. TF-IDF + ComplementNB reached `91.37%` test accuracy and `91.32%` macro F1. BERT Tiny reached `95.09%` test accuracy and `95.08%` macro F1. |
| Historical notebook-only results | The notebooks report a best classical result of `91.44%` accuracy for a Word2Vec + MLP model and a best transformer result near `96.58%` accuracy for RoBERTa. These are retained only as academic history; current claims should use the scripted metrics above. |

The transformer command uses the compact pretrained BERT miniature checkpoint `google/bert_uncased_L-2_H-128_A-2` for local CPU/GPU reproduction. Its metrics are reported separately from the stronger historical RoBERTa notebook result.

The original notebooks removed missing rows and exact duplicate rows, but duplicate or conflicting normalized message text could still leak across train/test boundaries. The refactored package fixes that modeling risk by removing conflicting normalized texts, deduplicating normalized message text before splitting, and checking for split overlap with `assert_no_text_overlap`.

## Repository Map

```text
src/email_threat_detector/   Reusable data, preprocessing, evaluation, and inference utilities
configs/experiments/         Reproducible experiment settings
notebooks/                   Cleaned copies of the academic notebooks
reports/                     Reproducibility notes, model/data cards, metrics, and figures
app/                         Streamlit classifier backed by a trained artifact
web/backend/                 FastAPI API for artifact-backed predictions
web/frontend/                React, TypeScript, Vite, and Tailwind browser app
tests/                       Fast unit and integration tests
```

The original academic notebook content is preserved under `notebooks/`. Duplicate root-level notebook copies were removed so there is one canonical notebook location.

## Dataset

The notebooks use Kaggle's "The Biggest Spam Ham Phish Email Dataset" via:

```python
kagglehub.dataset_download("akshatsharma2/the-biggest-spam-ham-phish-email-dataset-300000")
```

The raw dataset is not committed to this repository. See [data/README.md](data/README.md) for provenance, handling notes, leakage concerns, and privacy guidance.

## Development

Use Python `>=3.10,<3.13`. In this repository, the local virtual environment is at `.venv`. Install the package with development tools, then run linting and tests:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest
```

Install `baseline`, `transformers`, `app`, or `tracking` extras only when working on those surfaces.

Baseline metrics can be generated from a local dataset file with the CLI:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,baseline]"
email-threat-detector train-baseline --data-path path\to\df.csv
```

The command writes leakage-safe metrics under `reports/metrics/`. Do not commit raw data. Large model artifacts should be shared through Git LFS or through a hosted artifact URL with a checksum.

Transformer split files can be prepared with the same cleanup and split logic:

```powershell
email-threat-detector prepare-transformer-splits --data-path path\to\df.csv --output-dir data\processed\splits
```

The optional Streamlit app loads a trained artifact produced with `--model-path`:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,app]"
$env:EMAIL_THREAT_MODEL_PATH = "artifacts\baseline.joblib"
streamlit run app\streamlit_app.py
```

Without `EMAIL_THREAT_MODEL_PATH`, the app stops before prediction instead of using keyword fallback rules.
Model artifacts are loaded with `joblib`, which uses Python pickle internally; only load artifacts that you created or otherwise trust.

## ThreatLens Web App

ThreatLens is a FastAPI + React web app for interactive artifact-backed classification. It does not store submitted messages and it does not use keyword fallback predictions when the model artifact is missing.

Start the backend from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,web]"
$env:EMAIL_THREAT_MODEL_PATH = "artifacts\tfidf_logreg.joblib"
.\.venv\Scripts\python.exe -m uvicorn web.backend.main:app --reload --host 127.0.0.1 --port 8000
```

For hosted backends where the artifact is not present on disk, publish the trusted joblib artifact to a release asset or object storage, then configure an HTTPS URL and required SHA-256 checksum:

```powershell
$env:EMAIL_THREAT_MODEL_URL = "https://your-artifact-host/tfidf_logreg.joblib"
$env:EMAIL_THREAT_MODEL_SHA256 = "d9ed306935e26c9bf7b285a861991bb3539be7089ad9d8bf798d781d01d45981"
$env:EMAIL_THREAT_MODEL_CACHE_PATH = "/tmp/threatlens/model.joblib"
$env:EMAIL_THREAT_BACKGROUND_WARMUP = "true"
```

The backend downloads the artifact once, verifies the required SHA-256 checksum, caches it, and then loads it through the same `ThreatClassifier` path used for local artifacts. Remote joblib artifacts without a checksum are rejected because joblib uses Python pickle internally.
Configure only one artifact source at a time. For Docker or Compose deployments that use `EMAIL_THREAT_MODEL_URL`, set `EMAIL_THREAT_MODEL_PATH` to an empty value so the image's local artifact default does not compete with the remote source.

Useful backend endpoints:

```text
GET  http://127.0.0.1:8000/live
GET  http://127.0.0.1:8000/health
GET  http://127.0.0.1:8000/metadata
POST http://127.0.0.1:8000/predict
```

Use `/live` for container and platform liveness probes. Use `/health` for model readiness; it returns `503` until the real artifact is available and loaded.

Start the frontend in a second terminal:

```powershell
cd web\frontend
pnpm install
$env:VITE_API_BASE_URL = "http://127.0.0.1:8000"
pnpm dev
```

Open [ThreatLens](http://127.0.0.1:5173/), paste a message, and select **Analyze**. The frontend reads real model metadata and committed metrics from the backend. If the artifact is unavailable, `/health` returns a service error and `/predict` returns `503` instead of producing a fake result.

For local checks:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest
cd web\frontend
$env:VITE_API_BASE_URL = "http://127.0.0.1:8000"
pnpm lint
pnpm build
pnpm test:e2e:install
pnpm test:e2e
```

The E2E suite starts a real FastAPI backend and Vite frontend, then checks phish, ham, spam, and responsive overflow states in Chromium desktop and mobile projects.

## Production Deployment

The repository includes Docker packaging for non-local deployment:

```powershell
copy .env.example .env
docker compose up --build
```

The backend image copies `artifacts/tfidf_logreg.joblib` from Git LFS and defaults `EMAIL_THREAT_MODEL_PATH` to that in-image artifact. Compose also mounts `./artifacts` for local iteration.

The compose stack serves:

- [ThreatLens](http://localhost:8080)
- [ThreatLens API](http://localhost:8000/health)

For an open-source repository with the model artifact included, use Git LFS instead of regular Git blobs:

```powershell
git lfs install
git add .gitattributes artifacts\tfidf_logreg.joblib
git commit -m "chore: add trained sklearn artifact via git lfs"
```

`artifacts/tfidf_logreg.joblib` is configured for Git LFS and has this current pointer metadata:

```text
oid sha256:d9ed306935e26c9bf7b285a861991bb3539be7089ad9d8bf798d781d01d45981
size 270487147
```

For cloud platforms, deploy `web/backend/Dockerfile` as the API service and `web/frontend/Dockerfile` as the static frontend service. Point platform health checks at `/live`. Set `VITE_API_BASE_URL` to the public backend origin when building the frontend; production builds intentionally fail when it is missing or when a non-loopback origin uses plain HTTP. Set either `EMAIL_THREAT_MODEL_PATH` for the in-image or mounted LFS artifact, or `EMAIL_THREAT_MODEL_URL` plus `EMAIL_THREAT_MODEL_SHA256` for a remote artifact. Set `EMAIL_THREAT_ALLOWED_ORIGINS` to the deployed frontend origin.

The `Web app` GitHub Actions workflow checks out LFS files, verifies the committed model artifact checksum, builds the frontend with an explicit API origin, runs the artifact-backed Playwright E2E suite against the built static app, and smoke-tests the Docker Compose stack.

## Limitations

- Current committed tests cover package behavior, but they do not retrain full models on every test run.
- Raw data is not committed. The trained sklearn artifact is prepared for Git LFS or remote artifact hosting, not regular Git storage.
- Notebook metrics were produced in exploratory/academic workflows and may not reflect leakage-safe performance.
- The dataset is public, but public message datasets can still contain personal names, email addresses, URLs, and sensitive text.
- The project is intended for research and portfolio presentation, not for unattended production email filtering or security enforcement.

## Privacy And Ethics

Do not commit raw email/message data, credentials, Kaggle tokens, model checkpoints trained on sensitive text, or unredacted examples. Evaluate false positives and false negatives carefully: blocking legitimate messages can harm users, while missed phishing can create security risk. App predictions should be treated as advisory.

## Credits

The original academic notebooks were developed by Group 35: Mehdi Hellal, Antonio Augusto Brito de Sousa, and Adam Oprchal. The repository structure, reproducibility work, tests, documentation, CI, and deployment-oriented additions are part of the later portfolio transformation.
