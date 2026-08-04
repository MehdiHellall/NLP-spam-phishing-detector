# Model Card

Last updated: 2026-08-04

## Model Status

No trained model artifact is committed. Reproducible metrics are committed for TF-IDF + Logistic Regression, TF-IDF + ComplementNB, and BERT Tiny runs.
Any sklearn artifact should be treated as trusted local code because `joblib` loading uses Python pickle internally.

## Intended Use

Research, coursework cleanup, and portfolio presentation for message classification into:

- `ham`: legitimate message
- `phish`: phishing or attack-oriented message
- `spam`: unwanted promotional or abusive message

## Not Intended For

This project should not be used as an unattended production email security control. It does not include monitoring, abuse handling, model calibration, adversarial testing, privacy review, or operational rollback procedures.

## Training Data

The original notebooks use Kaggle's "The Biggest Spam Ham Phish Email Dataset". Raw data is not stored in the repository.

## Evaluation

The best committed reproducible result is TF-IDF + Logistic Regression with `95.76%` test accuracy and `95.76%` macro F1 on the leakage-safe balanced split. BERT Tiny reached `95.09%` test accuracy and `95.08%` macro F1 after one epoch on the same prepared split. The stronger RoBERTa notebook number remains historical context unless it is rerun with the package workflow.

## Risks

- False positives can hide legitimate messages.
- False negatives can miss phishing or spam.
- Public datasets can contain sensitive text.
- Models may learn dataset artifacts rather than durable threat patterns.
