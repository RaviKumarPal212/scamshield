# Evaluation plan (freeze before model training)

**Status:** TF-IDF + Logistic Regression and multilingual DistilBERT were compared on validation, then DistilBERT and its operating threshold were locked before a one-time held-out test evaluation (2026-10-12). Validation reports are `reports/sms_tfidf_logreg_validation.json` and `reports/sms_distilmbert_validation.json`; the lock record is `reports/sms_model_selection.json`; test results are `reports/sms_distilmbert_test.json`. Hinglish and URL data remain outside this SMS model.

## Tasks are evaluated separately

### A. Message classifier

- Initial benchmark: UCI SMS Spam Collection (`ham` vs `spam`).
- Interpretation: generic SMS spam detection, not Indian scam detection.
- UCI exact normalized-text duplicates: 415 rows beyond first. A conservative template-key pass (replacing links and standalone digit sequences, then normalizing punctuation/spacing) found 305 repeated groups / 434 rows beyond first, with no cross-label groups. The 999 pairwise within-group message comparisons had minimum case-folded `difflib.SequenceMatcher` similarity 0.864 (all >=0.8). Treat these as strong near-duplicate/template candidates; keep each group within one partition, and do not automatically delete rows based on this heuristic.
- A deterministic group-aware split manifest is saved at `data/processed/sms_split_manifest.csv`; its metadata and source/manifest hashes are in `data/processed/sms_split_summary.json`. It uses seed 42, maps `ham=0` / `spam=1`, keeps exact normalized texts and template candidates in one group, and contains no message text. The realized row ratios are train 3,902 (70.00%), validation 836 (15.00%), and test 836 (15.00%); each partition preserves the original class proportions exactly at the recorded row counts. Automated checks found 0 groups spanning multiple partitions. The user reviewed the manifest before the first training run; keep the test split locked while selecting the model and threshold.
- First baseline: `TfidfVectorizer` word unigrams+bigrams (`strip_accents="unicode"`, `sublinear_tf=True`) followed by `LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=42)`. The TF-IDF vocabulary and classifier were fit on train only. Validation at the default 0.5 threshold (not tuned): precision 1.000, recall 0.723, spam F1 0.839, macro-F1 0.909, average precision 0.995, accuracy 0.963; confusion counts: TN=724, FP=0, FN=31, TP=81. The 31 false negatives are 27.7% of the 112 validation spam rows. These are validation-only results, not final test performance and not Indian scam-detection performance.
- Validation-only threshold analysis is saved at `reports/sms_threshold_analysis.json`. The TF-IDF validation-F1-maximizing candidate is threshold 0.2791: precision 0.957, recall 0.991, spam F1 0.974, with 5 false positives and 1 false negative. No TF-IDF test evaluation was performed.
- Local CPU single-message latency, including TF-IDF transformation and `predict_proba` but excluding API/network overhead: p50 1.76 ms and p95 2.81 ms over 500 calls after 30 warm-ups (batch size 1; Windows 10 host, 12 logical CPUs). Treat latency as machine-specific.
- Transformer comparison: multilingual DistilBERT (`distilbert/distilbert-base-multilingual-cased`, immutable revision `45c032ab32cc946ad88a166f7cb282f58c753c2e`, Apache-2.0) was fine-tuned for 3 epochs on the same 3,902 train rows, with the best checkpoint selected by validation macro-F1 at threshold 0.5. At threshold 0.5 it scored spam precision 0.949, recall 0.991, spam F1 0.969, macro-F1 0.982, and average precision 0.996; confusion counts: TN=718, FP=6, FN=1, TP=111. It improves validation recall and macro-F1 over TF-IDF at 0.5, with 6 rather than 0 false positives. Its validation-F1-maximizing threshold candidate is 0.9842 (precision=recall=spam F1=0.982, FP=2, FN=2); this is not a locked operating threshold.
- Transformer local CPU single-message latency, including tokenization and model forward pass but excluding API/network overhead: p50 14.78 ms and p95 33.01 ms over 500 calls after 30 warm-ups (batch size 1; same Windows 10 host, 12 logical CPUs). The measurement uses the same CPU host and validation partition as the baseline; the transformer is materially slower. See `scripts/benchmark_sms_transformer.py` and `reports/sms_distilmbert_validation.json` for the protocol and full measurements.
- The selected model is DistilBERT at threshold `0.609175443649292`, the highest-recall validation candidate subject to spam precision >=0.90. The validation point was precision 0.949, recall 0.991, FP=6, FN=1. The selection record includes the model and validation-report hashes and predates test access.
- One-time locked test result at that fixed threshold: precision 0.911, recall 1.000, spam F1 0.953, macro-F1 0.973, average precision 1.000, accuracy 0.987; confusion counts TN=713, FP=11, FN=0, TP=112 on 836 test messages. The report is `reports/sms_distilmbert_test.json`. This result is descriptive only; do not tune on the test partition.
- All model results are on the English UCI SMS dataset, not Hindi/Hinglish or India-specific scam performance.
- Supplementary evaluation: the Hinglish dataset card describes 63 hand-written examples, but an exact normalized overlap audit found 3,700 of 3,787 rows also in UCI; only 87 do not match, and their provenance/composition is unclear. A small cue-word heuristic matched 37 of the unmatched rows but is not language identification. Keep this dataset out of independent validation/test evaluation until the unmatched records and labels are manually reviewed. If retained later, report only counts and qualitative errors, not representative performance.

### B. URL classifier

- Candidate benchmark: PhiUSIIL.
- Use raw URL string and only features available from the submitted URL in the live product for the first benchmark.
- Normalize the target to `is_phishing=1` (UCI label 0=phishing, 1=legitimate).
- Deduplicate URLs and group by registrable domain so the same domain or near-identical URL family cannot appear across train and test. The initial audit found 54 full hostnames represented with both labels. The source `Domain` field matched the parsed hostname in 235,759 of 235,795 rows and is not a registrable-domain grouping key; row-random splitting is unsuitable.

### C. Future job-ad classifier

- Separate dataset, labels, model, and metrics. Do not combine with short SMS metrics.

## Split protocol

1. Inspect and document the data before any split.
2. Standardize labels and do exact-text/URL deduplication. Check near-duplicate campaign/template messages and source overlap.
3. Keep a source column and, where possible, an entity/group key (campaign/template, normalized domain, or dataset source).
4. For the initial SMS baseline, use the saved seed-42, group-aware manifest at `data/processed/sms_split_manifest.csv`. All exact normalized duplicates and template candidate groups stay in one partition. The generated ratios match 70/15/15; the split summary records actual ratios, class counts, group counts, and hashes.
5. For URL data, use a group-aware split by registrable domain, not a random row-level split. Use the raw `URL` field as the initial input and recompute any allowed URL-string features inside the project.
6. Fit vectorizers, feature selection, calibration, and decision thresholds using training/validation only. Never fit a learned text vocabulary or select the threshold from test data.
7. Use identical saved splits and label definitions to compare Logistic Regression and the transformer.
8. Lock the test set. Select models and thresholds on validation; compute final test metrics once for reporting.

## Metrics to report

- Precision and recall for the risky/positive class.
- Macro-F1 and class-wise F1.
- PR-AUC, reported as average precision or area-under-precision-recall with its exact calculation named; do not call ROC-AUC “PR-AUC”.
- Confusion matrix, false-negative count and false-negative rate for the risky class.
- Class prevalence and test sample counts alongside every metric.
- CPU inference latency p50 and p95 with batch size, hardware, warm-up, number of repetitions, and whether preprocessing is included. Optionally report GPU separately.
- Model size and time to load if relevant to deployment.

## Model selection and threshold

- Use validation data to choose the operating threshold according to the intended trade-off. Since missing a scam can be costly, examine recall and false negatives, but do not optimize recall without reporting false-positive burden.
- If probability calibration is used, fit it without test data.
- Report performance at the fixed default threshold and at any chosen operating threshold, clearly distinguished.

## Data leakage checks

- Exact normalized-text/URL duplicates across splits: target zero.
- SMS template candidates: the manifest keeps all 305 candidate groups intact; no group crossed partitions.
- Source overlap: especially UCI rows embedded in the secondary Hinglish dataset.
- Hinglish source overlap observed: 3,700 rows match UCI after NFKC, trim, and case-fold; all matched labels agree. Keep the data sources separate.
- URL hostname overlap observed: 54 full hostnames occur with both labels; calculate registrable-domain groups before splitting.
- Template and campaign overlap: the preliminary candidate groups are strongly similar; manually review them and use group splits. Do not use the similarity score as a semantic duplicate label.
- Label leakage from columns that directly encode the answer.
- Feature availability mismatch: no webpage-derived inputs in a URL-only demo unless the backend truly obtains them.

## Results integrity

Validation results for both models and the selected model's one-time locked test result are saved in `reports/`; artifacts are `models/sms_tfidf_logreg.joblib` and `models/sms_distilmbert/`. README metrics must come from these reports. Synthetic challenge-set results must be labelled as synthetic pattern tests, not as real-world test performance. The test report is not permission to tune further on that split.
