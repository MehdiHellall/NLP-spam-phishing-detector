# ThreatLens

ThreatLens is a web app that classifies messages as:

- `ham`: legitimate
- `phish`: phishing or attack-oriented
- `spam`: unwanted promotional or abusive

The repo keeps only the deployable app, the trained model artifact, and the source notebooks.

## Results

The packaged model is `artifacts/tfidf_logreg.joblib`.

| Model | Test accuracy | Test macro F1 |
| --- | ---: | ---: |
| TF-IDF + Logistic Regression | `95.76%` | `95.76%` |

The frontend reads these metrics from the backend asset at `web/backend/assets/tfidf_logreg_metrics.json`.

## Repository Map

```text
web/backend/                 FastAPI prediction API
web/frontend/                React frontend
artifacts/tfidf_logreg.joblib Trained sklearn artifact
src/email_threat_detector/   Minimal compatibility module required by the artifact
notebooks/                   Source notebooks
tests/                       Backend API tests
```

## Setup

Use Python `>=3.10,<3.13`.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,web]"
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest
```

## Run Locally

Start the backend:

```powershell
$env:EMAIL_THREAT_MODEL_PATH = "artifacts\tfidf_logreg.joblib"
.\.venv\Scripts\python.exe -m uvicorn web.backend.main:app --reload --host 127.0.0.1 --port 8000
```

Start the frontend:

```powershell
cd web\frontend
pnpm install
$env:VITE_API_BASE_URL = "http://127.0.0.1:8000"
pnpm dev
```

Open [http://127.0.0.1:5173/](http://127.0.0.1:5173/).

## API

```text
GET  /live
GET  /health
GET  /metadata
POST /predict
```

## Docker

```powershell
copy .env.example .env
docker compose up --build
```

Services:

- Web app: [http://localhost:8080](http://localhost:8080)
- API health: [http://localhost:8000/health](http://localhost:8000/health)

## Configuration

Use one model source at a time:

- `EMAIL_THREAT_MODEL_PATH`: local joblib artifact
- `EMAIL_THREAT_MODEL_URL` and `EMAIL_THREAT_MODEL_SHA256`: remote artifact with checksum
- `EMAIL_THREAT_ALLOWED_ORIGINS`: allowed frontend origins
- `VITE_API_BASE_URL`: backend URL used by the frontend

`joblib` uses Python pickle internally. Only load artifacts you trust.

## Checks

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest
cd web\frontend
pnpm lint
pnpm build
pnpm test:e2e:install
pnpm test:e2e
```

## Limits

Predictions are advisory. Do not use this as an unattended mail gateway or load untrusted model files.
