# Data Notes

The source dataset used by the original notebooks is:

- Kaggle: "The Biggest Spam Ham Phish Email Dataset"
- Access pattern in the notebooks: `kagglehub.dataset_download("akshatsharma2/the-biggest-spam-ham-phish-email-dataset-300000")`
- Expected file: `df.csv`
- Expected columns: `label`, `text`

The raw dataset is not committed to this repository. Future pipeline work should:

1. Download or locate the dataset explicitly.
2. Validate required columns and label values.
3. Drop null text rows.
4. Audit duplicate and conflicting text labels before splitting.
5. Create deterministic train/validation/test splits with fixed seeds.

Saved public examples should be redacted or carefully truncated because even public email datasets can contain names, addresses, URLs, or other sensitive content.
