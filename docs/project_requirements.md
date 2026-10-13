# ScamShield India — Step 1 requirements

## Problem statement

Build a maintainable prototype that screens user-provided messages and URLs for scam-like indicators, with an explicit focus on English, Hindi (Devanagari), and Hinglish (including Romanized Hindi). Treat this as **risk screening**, not definitive fraud adjudication.

## Intended users

People who want a second opinion on a suspicious SMS, chat excerpt, job offer, or URL; recruiters and community educators may also use it as a demonstration. It is not an enterprise fraud-prevention control.

## MVP requirements

### Functional

- Accept pasted text and a pasted URL as separate input types.
- Return a calibrated/thresholded risk category (for example: lower concern, review, high concern), a probability/score only when it is appropriately calibrated, and a plain-language caveat.
- Show understandable signals found in the input, e.g. OTP/PIN request, urgent account-blocking threat, upfront payment, prize/loan lure, or link present.
- Keep text and URL model evaluations separate because their datasets and label definitions differ.
- Compare TF-IDF + Logistic Regression and one multilingual transformer using identical message splits.
- Record model version, dataset version, threshold, metrics, inference environment, and latency measurement method.
- Provide FastAPI endpoints, a usable web UI, tests, Docker instructions, and a reproducible README in later steps.

### Non-functional

- Free/open-source tooling where practical.
- No secrets or raw private user input written to logs.
- Deterministic split seeds and reproducible environment instructions.
- Keep raw data and large model weights out of Git by default.
- Graceful handling of empty/very long text and malformed URLs.
- Explanations must not claim that a single keyword proves fraud.

## Explicitly out of scope for the first version

- Automatically opening or crawling user-provided links.
- Blocking payments, sending reports to authorities, or making legal determinations.
- Claiming comprehensive detection of all Indian scams.
- Calling a message “safe” merely because the model scores it low risk.
- Combining SMS, URL, and full job-ad inputs into one classifier before their label semantics and evaluation protocols are established.

## Planned model components

1. Message text: TF-IDF word/character n-grams + Logistic Regression as the explainable baseline; compare with a multilingual transformer.
2. URL: a separate URL-string classifier, plus deterministic URL risk signals. Avoid using webpage-derived features unless the running system computes the same features.
3. Rules: explicit, testable signals. Rules supplement model output and must be evaluated for false positives.
4. Later: long-form fake-job-ad model, after confirming a dataset's usage terms and checking domain/time limitations.

## Success criteria (to be agreed before training)

- All data sources have documented provenance, label meaning, licence/terms, and limitations.
- Exact duplicates and obvious template/campaign leakage are checked before splitting.
- No preprocessing step learns from validation/test data.
- A locked test set is evaluated once for final reporting after model/threshold selection.
- All requested metrics, confusion matrices, class counts, and latency method are reported honestly.
- At least a small unit-test suite passes before deployment.
