# ScamShield India

**A multilingual scam-risk screening project for English, Hindi, and Hinglish.**

ScamShield India is an educational ML engineering project. Its goal is to flag *potentially suspicious* messages and URLs, show understandable risk indicators, and make model quality and limitations inspectable. It is not a guarantee that a message or URL is safe.

## Current status

- [x] Step 1: scope, candidate datasets, environment, and evaluation plan reviewed
- [x] Step 2: audit UCI SMS, supplementary Hinglish, and PhiUSIIL URL data; record limitations
- [x] Step 3: create and review the group-aware UCI SMS train/validation/test manifest
- [x] Step 4: train and compare TF-IDF + Logistic Regression with a transformer on validation data
- [ ] Step 5: add rules and explanations, API, web UI, tests, and deployment (SMS rules, local API, web UI, and local container recipe; container/hosting verification remains)

Both the TF-IDF + Logistic Regression baseline and multilingual DistilBERT have been trained on UCI SMS train data and evaluated on validation only. DistilBERT at threshold `0.609175443649292` was selected before test evaluation, prioritizing recall while retaining at least 0.90 validation precision. The decision and hashes are recorded in `reports/sms_model_selection.json`. The final test report is `reports/sms_distilmbert_test.json`; the test set was evaluated once, with no threshold tuning. These results support only an English SMS spam benchmark, not a Hindi/Hinglish or Indian scam-detection claim.

### SMS validation comparison

| Model / operating point | Spam precision | Spam recall | Spam F1 | Macro-F1 | Average precision | FP | FN | CPU p50 / p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression, threshold 0.5 | 1.000 | 0.723 | 0.839 | 0.909 | 0.995 | 0 | 31 | 1.76 / 2.81 |
| Multilingual DistilBERT, threshold 0.5 | 0.949 | 0.991 | 0.969 | 0.982 | 0.996 | 6 | 1 | 14.78 / 33.01 |

These are validation-only measurements on the same 836-message partition (112 spam messages). The transformer catches more spam at the default threshold but flags 6 ordinary messages and is slower on CPU. Validation-F1-maximizing threshold candidates are 0.2791 for TF-IDF (spam F1 0.974, FP=5, FN=1) and 0.9842 for DistilBERT (spam F1 0.982, FP=2, FN=2). They are candidates, not the chosen threshold. CPU timings include preprocessing/model inference, exclude API overhead, and are specific to this machine; see the saved reports and `scripts/benchmark_sms_transformer.py` for details.

### Locked test result

DistilBERT was evaluated once on the held-out test partition at the preselected threshold `0.609175443649292`: precision **0.911**, recall **1.000**, spam F1 **0.953**, macro-F1 **0.973**, average precision **1.000**, accuracy **0.987**; confusion counts: TN=713, FP=11, FN=0, TP=112 (836 messages total). These are English UCI SMS spam results, not India-specific scam-detection performance. Do not retune the model or threshold based on this test result; use a new external dataset for further model changes.

## Initial scope

1. **Message screening:** classify a message as suspicious/spam-like versus ordinary, with a clear note that spam is not synonymous with fraud.
2. **URL screening:** analyze a pasted URL in a separate component. Do not claim to inspect a live webpage unless the deployed product actually fetches and analyzes it safely.
3. **Explanations:** surface concrete indicators such as requests for OTP/PIN, urgency/threats, advance-fee language, or a suspicious link. Indicators are signals, not proof.
4. **Evaluation:** report precision, recall, macro-F1, PR-AUC/average precision, false-negative counts/rates, and CPU inference latency.
5. **Engineering:** FastAPI, simple user interface, tests, Docker, documented setup, and reproducible results.

Long-form fake-job-ad classification is a separate planned component; it should be added only after its dataset provenance and usage terms are reviewed. UPI-related patterns can be covered by transparent rules and a clearly labelled synthetic challenge set until sufficiently documented real-world evaluation data is available.

## Explainable SMS rule signals

The first rule-only layer is in `src/scamshield/rules/sms.py`, exposed as `scamshield.rules.analyze_message`. It reports credential requests, urgency/threats, up-front fee language, prize/lottery lures, and limited URL characteristics. It does not assign a score, declare a message fraudulent, open links, or echo matched text. Rules are indicators—not proof—and Hindi/Hinglish coverage is partial. See [`docs/rule_indicators.md`](docs/rule_indicators.md).

## Local SMS API

The first API slice is in `src/scamshield/api/app.py`. It exposes `GET /health` and `POST /v1/analyze/sms`. The API loads only the locally saved, hash-verified DistilBERT model and uses the threshold already recorded in `reports/sms_model_selection.json`; it does not download model files or make cloud calls. Start it from the project folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt -r requirements-transformers.txt
.\.venv\Scripts\python.exe -m uvicorn scamshield.api.app:app --app-dir src --host 127.0.0.1 --port 8001
```

Then open `http://127.0.0.1:8001/` for the web interface or `http://127.0.0.1:8001/docs` for the interactive API page. The interface and assets are served from `frontend/` by the same local API; no separate frontend build or development server is required. Port 8000 was already occupied during the local smoke test, so this setup uses 8001. The API-only local run is not a public deployment. Example API request from another PowerShell window:

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8001/v1/analyze/sms' -Method Post -ContentType 'application/json' -Body '{"message":"Share your OTP immediately: http://bit.ly/example"}'
```

The response distinguishes `spam_like` from `not_flagged`; `not_flagged` does not mean safe. This is a localhost development API, not a deployed public service. Do not submit real private messages or credentials. API behavior is covered by `tests/test_sms_api.py` and the full suite runs with `python -m pytest`.

## Local container

`Dockerfile`, `compose.yaml`, and `.dockerignore` provide a CPU-oriented local container setup. Model weights and the lock file stay outside the image and are mounted read-only; the container is bound to `127.0.0.1:8001`. See [`docs/container.md`](docs/container.md) for prerequisites and commands. Docker was not available in the development environment, so the image build remains to be verified on a machine with Docker Desktop. The GitHub Actions workflow runs the unit suite on pushes and pull requests.

## Repository layout

- `data/raw/`: original datasets; keep out of Git unless redistribution rights are confirmed
- `data/interim/`: temporary inspection/cleaning outputs
- `data/processed/`: versioned split manifests and derived data metadata; avoid committing large/raw text by default
- `src/scamshield/`: package code
- `tests/`: automated tests
- `scripts/`: environment/data utilities
- `docs/`: requirements, data sources, and evaluation protocol
- `models/`: local model artifacts; do not commit large weights by default
- `reports/`: small evaluation and audit reports from real runs; model weights remain untracked
- `frontend/`: locally served web UI

## Quick start (Windows PowerShell)

Use the project virtual environment. Re-run the data audit with `python scripts/audit_datasets.py`, reproduce the split manifest with `python scripts/prepare_sms_splits.py`, train the baseline with `python scripts/train_sms_baseline.py`, and train/validate the transformer with `python scripts/train_sms_transformer.py` after installing `requirements-transformers.txt`. The benchmark command is `python scripts/benchmark_sms_transformer.py`. The fixed final evaluator is `python scripts/evaluate_sms_locked_test.py`; it verifies the locked selection and provenance hashes and has no threshold-search logic. It has already been run once for this model selection. Do not repeat the locked test run for tuning.

## Data and evaluation commitments

Read [`docs/data_sources.md`](docs/data_sources.md) and [`docs/evaluation_plan.md`](docs/evaluation_plan.md) before processing data. Keep source attribution and label semantics. Never report synthetic challenge-set performance as real-world accuracy. Never fabricate metrics. The validation result is not a claim about Hindi/Hinglish or India-specific scam detection.

## Automated checks

Run the unit tests from the project virtual environment with `python -m pytest -p no:cacheprovider`. GitHub Actions runs the same tests on pushes and pull requests. The suite uses a test predictor for API behavior and does not load the large local model or access the locked test partition.

## Responsible use

A `low risk` output means only that the current model did not find enough signals. It is not a safety certificate. Users should independently verify payment requests, links, job offers, and requests for credentials. Do not send real OTPs, PINs, account numbers, or private messages to public demos.
