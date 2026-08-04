# Technical Report

Last updated: 2026-08-04

## Summary

This project classifies short messages into `ham`, `phish`, and `spam`. It began as two academic notebooks and now has reusable package code for data preparation, leakage-safe classical baselines, metrics, inference, and an optional transformer workflow.

## Reproducible Workflow

The package pipeline:

1. Loads a CSV with `text` and `label` columns.
2. Normalizes labels to `ham`, `phish`, and `spam`.
3. Cleans missing or empty messages.
4. Audits duplicate normalized text and removes conflicts.
5. Deduplicates normalized text before splitting.
6. Creates deterministic stratified train/validation/test splits.
7. Checks that normalized text does not overlap across split boundaries.
8. Fits sklearn vectorizers inside `Pipeline` objects on training data only.
9. Saves metrics and optional model artifacts.

## Current Findings

The most important correction from the notebooks is leakage control. The classical notebook fit a bag-of-words vectorizer before the train/test split, and both notebooks lacked a formal normalized-text overlap check. The scripted baseline path fixes those issues.

The current defensible model-quality results come from leakage-safe CLI runs. TF-IDF + Logistic Regression reached `95.76%` test accuracy and `95.76%` macro F1. TF-IDF + ComplementNB reached `91.37%` test accuracy and `91.32%` macro F1. The compact pretrained BERT Tiny transformer reached `95.09%` test accuracy and `95.08%` macro F1 after one epoch on the prepared balanced real-data split.

## Limitations

- The dataset is not committed and has not been versioned in this repository.
- The stronger RoBERTa metric from the academic notebook remains historical context because it was not rerun through the package split and training path.
- Synthetic tests verify code behavior, not real-world phishing robustness.
- Accuracy is not enough for a security-sensitive workflow; per-class recall, false positives, calibration, and adversarial examples still need review.
