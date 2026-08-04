# Data Notes

## Source

The original notebooks use Kaggle's "The Biggest Spam Ham Phish Email Dataset":

- Kaggle dataset slug: `akshatsharma2/the-biggest-spam-ham-phish-email-dataset-300000`
- Notebook access pattern: `kagglehub.dataset_download("akshatsharma2/the-biggest-spam-ham-phish-email-dataset-300000")`
- Expected downloaded file: `df.csv`
- Expected columns: `label`, `text`
- Label mapping used by the notebooks and package: `0=ham`, `1=phish`, `2=spam`

The raw dataset is not committed to this repository. Keep it outside Git because it is large and may contain sensitive message content even though it is publicly distributed.

## Data Quality And Leakage

The academic notebooks noted more than 360,000 rows but about 280,000 unique message texts, plus a small number of duplicated messages with conflicting labels and some missing values. The original notebook workflow dropped null rows and exact duplicate rows.

The refactored package adds stricter leakage-safe preparation for reproducible scripts:

1. Validate required columns and label values.
2. Normalize message text and labels.
3. Drop null or empty text rows.
4. Remove duplicate normalized texts with conflicting labels.
5. Deduplicate normalized message text before splitting.
6. Create deterministic stratified train/validation/test splits.
7. Assert that normalized text does not overlap across split boundaries.

This matters because duplicate messages split across train and test sets can inflate reported accuracy. Any future reported script metric should state which cleanup and split procedure produced it.

## Privacy And Ethics

Do not commit raw data, unredacted samples, downloaded Kaggle archives, trained model checkpoints, or generated prediction logs. Public spam/phishing corpora can still include personal names, email addresses, phone numbers, URLs, security lures, or other sensitive content.

For reports and demos, prefer aggregate metrics and short redacted examples. Do not present this project as a production security control without additional data governance, monitoring, adversarial evaluation, and human review.
