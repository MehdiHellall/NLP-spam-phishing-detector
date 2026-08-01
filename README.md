# NLP Threat Detector

Detecting spam and phishing messages with production-minded NLP: classical ML baselines, transformer experiments, and security-focused evaluation.

## Project Status

This repository is being transformed from an academic notebook project into a polished ML engineering portfolio project. The original work compares classical NLP approaches with transformer fine-tuning for a three-class email/message classification task:

- `ham`: legitimate message
- `phish`: phishing or attack-oriented message
- `spam`: unwanted promotional or abusive message

The current notebooks report a classical-model peak near `91.44%` accuracy and a transformer peak near `96.58%` accuracy with RoBERTa. These are notebook-reported results and will be treated as provisional until reproduced through the scripted pipeline.

## Current Contents

- [notebooks/01_classical_nlp_baselines.ipynb](notebooks/01_classical_nlp_baselines.ipynb): classical NLP baselines, including bag-of-words, Word2Vec, and scikit-learn classifiers.
- [notebooks/02_transformer_finetuning.ipynb](notebooks/02_transformer_finetuning.ipynb): Hugging Face transformer fine-tuning experiments.

The root-level notebook files are preserved as the original academic submissions while the `notebooks/` copies will become the cleaned portfolio versions.

## Planned Portfolio Architecture

```text
src/email_threat_detector/   Reusable data, preprocessing, training, evaluation, and inference code
configs/experiments/         Reproducible experiment settings
notebooks/                   Clean exploratory and reporting notebooks
reports/                     Metrics, figures, model card, data card, and technical report
app/                         Recruiter-friendly demo app
tests/                       Fast unit and smoke tests
```

## Roadmap

1. Build deterministic data loading, cleanup, sampling, and split logic.
2. Refactor classical baselines into leakage-safe scikit-learn pipelines.
3. Reproduce metrics with scripts and save durable artifacts under `reports/`.
4. Add transformer training scripts with small-model defaults and GPU/runtime notes.
5. Build a Streamlit or Gradio demo for interactive prediction.
6. Add tests, Ruff checks, and GitHub Actions CI.
7. Publish a clear model card, data card, and concise technical report.

## Dataset

The notebooks use the Kaggle dataset "The Biggest Spam Ham Phish Email Dataset" via `kagglehub`. The dataset itself is not committed to this repository. See [data/README.md](data/README.md) for provenance and handling notes.

## Development

This project uses `pyproject.toml` for dependency groups and tooling configuration. The first reproducible commands will be added as the notebook logic is extracted into `src/`.

```powershell
pip install -e ".[dev,baseline]"
pytest
ruff check .
```

## Credits

The original academic notebooks were developed by Group 35: Mehdi Hellal, Antonio Augusto Brito de Sousa, and Adam Oprchal. The repository structure, reproducibility work, tests, documentation, and deployment-oriented additions are part of the portfolio transformation.
