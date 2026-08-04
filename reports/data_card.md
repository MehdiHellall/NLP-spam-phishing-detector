# Data Card

Last updated: 2026-08-03

## Dataset

The original notebooks use Kaggle's "The Biggest Spam Ham Phish Email Dataset" with expected columns:

- `text`
- `label`

The package maps labels to `ham`, `phish`, and `spam`.

## Repository Handling

Raw data is intentionally not committed. Keep downloaded CSVs under ignored local paths such as `data/raw/`.

## Known Quality Issues

The notebooks observed missing rows, exact duplicates, many repeated message texts, and a small number of same-text examples with conflicting labels. The scripted workflow removes conflicting normalized texts, deduplicates normalized message text, and asserts no text overlap across splits.

## Sensitive Content

Even public spam and phishing datasets can include names, contact details, URLs, security lures, and other sensitive text. Reports should use aggregate metrics and redacted examples.

## Recommended Reporting Fields

Every future metric artifact should record:

- Dataset source and local filename.
- Cleanup counts.
- Removed conflict count.
- Split sizes and class distributions.
- Random seed.
- Model name and package version.
