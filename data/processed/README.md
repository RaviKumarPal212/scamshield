# Processed data

Store reproducible split manifests and derived data metadata here. Avoid committing full datasets or private text. Save the seed, source version, hashes, group strategy, and label mapping with every split.

The current UCI SMS baseline split is recorded in `sms_split_manifest.csv`; `sms_split_summary.json` records its seed, source and manifest hashes, label counts, group counts, and realized ratios. The CSV contains source row IDs and metadata only—never SMS text. Regenerate it with `python scripts/prepare_sms_splits.py`.
