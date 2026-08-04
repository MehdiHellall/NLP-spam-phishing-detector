# ThreatLens

ThreatLens is a web app that classifies messages as:

- `ham`: legitimate
- `phish`: phishing or attack-oriented
- `spam`: unwanted promotional or abusive

The first screen is the classifier itself: model readiness, a message input, the prediction,
confidence and class probabilities, an explanation, a suggested action, and compact artifact
metadata. The UI intentionally has no example-message controls, marketing sections, or simulated
model claims.

## Results

The packaged model is `artifacts/tfidf_logreg.joblib`.

| Model | Test accuracy | Test macro F1 |
| --- | ---: | ---: |
| TF-IDF + Logistic Regression | `95.76%` | `95.76%` |

The frontend reads these metrics from the backend asset at `web/backend/assets/tfidf_logreg_metrics.json`.

## Repository Map

```text
web/backend/
  settings.py                Environment-backed runtime configuration
  schemas.py                 Versioned and compatibility API schemas
  security.py                Request guards, rate limiting, and security headers
  sklearn_runtime.py         Artifact resolution, loading, and sklearn inference
  explanations.py            Risk, explanation, and suggested-action policy
  main.py                    FastAPI composition and route definitions
web/frontend/                React analyst interface and Playwright E2E tests
artifacts/tfidf_logreg.joblib Trained sklearn artifact
src/email_threat_detector/   Compatibility module required by the artifact
notebooks/                   Source notebooks
tests/                       Backend API and real-artifact regression tests
```

The backend modules separate HTTP concerns from artifact-backed inference. Both API generations
use the same TF-IDF + Logistic Regression runtime; the refactor does not replace the trained model
with rules, fixtures, or mock output.

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

The canonical API is versioned under `/v1`:

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/v1/live` | Process liveness; does not require a loaded model |
| `GET` | `/v1/ready` | Inference readiness; returns `503` when the artifact is unavailable |
| `GET` | `/v1/metadata` | Labels, model status, metrics, limits, and privacy information |
| `POST` | `/v1/predict` | Artifact-backed classification using `{"text": "..."}` |

`POST /v1/predict` returns a duel-ready envelope with `final_label`, `final_risk_level`,
`final_confidence`, `model_outputs.tfidf_logreg`, `explanation`, `suggested_action`, and
`artifact_metadata`. Only the current TF-IDF + Logistic Regression output is exposed. The nested
model output includes its label, confidence, and `ham`/`phish`/`spam` probabilities; artifact
metadata identifies the artifact and model without exposing a server filesystem path.

Temporary compatibility routes remain available:

| Compatibility route | Canonical equivalent | Notes |
| --- | --- | --- |
| `GET /live` | `GET /v1/live` | Same liveness response |
| `GET /health` | `GET /v1/ready` | Same readiness response and status code |
| `GET /metadata` | `GET /v1/metadata` | Same metadata response |
| `POST /predict` | `POST /v1/predict` | Preserves the legacy prediction response shape |

## Docker

```powershell
copy .env.example .env
docker compose up --build
```

Services:

- Web app: [http://localhost:8080](http://localhost:8080)
- API readiness: [http://localhost:8000/v1/ready](http://localhost:8000/v1/ready)

The backend image copies the checked-in artifact, and Compose also mounts `./artifacts` read-only.
Container healthchecks call `/v1/live` so liveness polling never triggers an artifact download.
Use `/v1/ready` when checking whether inference is available.

## Configuration

Use one model source at a time:

- `EMAIL_THREAT_MODEL_PATH`: local joblib artifact
- `EMAIL_THREAT_MODEL_URL` and `EMAIL_THREAT_MODEL_SHA256`: remote artifact with checksum
- `EMAIL_THREAT_MODEL_RETRY_SECONDS`: backoff after a failed remote artifact load
- `EMAIL_THREAT_MAX_ARTIFACT_BYTES`: maximum accepted remote artifact size
- `EMAIL_THREAT_ALLOWED_ORIGINS`: allowed frontend origins
- `VITE_API_BASE_URL`: backend URL used by the frontend

`joblib` uses Python pickle internally. Only load artifacts you trust.

## Checks

Run backend tests and lint from the repository root:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest
```

Run frontend lint, production build, and the real prediction E2E flow:

```powershell
cd web\frontend
pnpm install --frozen-lockfile
pnpm lint
$env:VITE_API_BASE_URL = "http://127.0.0.1:8000"
pnpm build
pnpm test:e2e:install
pnpm test:e2e
```

When Docker is available and healthy, build and smoke-test the composed services:

```powershell
docker compose build
docker compose up -d
Invoke-WebRequest http://localhost:8000/v1/ready
Invoke-RestMethod -Method Post -Uri http://localhost:8000/v1/predict `
  -ContentType "application/json" -Body '{"text":"Urgent password reset required verify account."}'
Invoke-WebRequest http://localhost:8080
docker compose down
```

## Limits

Predictions are advisory. Do not use this as an unattended mail gateway or load untrusted model files.
