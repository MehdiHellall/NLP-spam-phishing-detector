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

The command writes leakage-safe metrics under `reports/metrics/`. Do not commit raw data or model artifacts unless they have been reviewed for size, sensitivity, and licensing.

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

## Limitations

- Current committed tests cover package behavior, but they do not retrain full models on every test run.
- No raw data or trained model weights are committed.
- Notebook metrics were produced in exploratory/academic workflows and may not reflect leakage-safe performance.
- The dataset is public, but public message datasets can still contain personal names, email addresses, URLs, and sensitive text.
- The project is intended for research and portfolio presentation, not for unattended production email filtering or security enforcement.

## Privacy And Ethics

Do not commit raw email/message data, credentials, Kaggle tokens, model checkpoints trained on sensitive text, or unredacted examples. Evaluate false positives and false negatives carefully: blocking legitimate messages can harm users, while missed phishing can create security risk. App predictions should be treated as advisory.

## Credits

The original academic notebooks were developed by Group 35: Mehdi Hellal, Antonio Augusto Brito de Sousa, and Adam Oprchal. The repository structure, reproducibility work, tests, documentation, CI, and deployment-oriented additions are part of the later portfolio transformation.
