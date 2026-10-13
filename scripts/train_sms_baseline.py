"""Train and validate the first TF-IDF + Logistic Regression SMS baseline.

Run from any directory with:
    python scripts/train_sms_baseline.py

Only train rows fit the model. Validation rows are used for the first report;
test-row text is never loaded into the model workflow or scored here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ZIP = ROOT / "data" / "raw" / "uci_sms_spam_collection.zip"
MANIFEST = ROOT / "data" / "processed" / "sms_split_manifest.csv"
SPLIT_SUMMARY = ROOT / "data" / "processed" / "sms_split_summary.json"
MODEL_PATH = ROOT / "models" / "sms_tfidf_logreg.joblib"
REPORT_PATH = ROOT / "reports" / "sms_tfidf_logreg_validation.json"
LABEL_MAPPING = {"ham": 0, "spam": 1}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_and_validation_rows() -> tuple[list[dict[str, Any]], Counter[str]]:
    if not SOURCE_ZIP.is_file() or not MANIFEST.is_file() or not SPLIT_SUMMARY.is_file():
        raise FileNotFoundError(
            "Required raw SMS archive, split manifest, or split summary is missing."
        )

    split_summary = json.loads(SPLIT_SUMMARY.read_text(encoding="utf-8"))
    if sha256_file(SOURCE_ZIP) != split_summary.get("source_sha256"):
        raise ValueError("The raw archive hash does not match the saved split summary.")
    if sha256_file(MANIFEST) != split_summary.get("manifest_sha256"):
        raise ValueError("The split manifest hash does not match the saved summary.")

    with MANIFEST.open("r", newline="", encoding="utf-8") as source:
        assignments: dict[int, dict[str, str]] = {}
        group_splits: dict[str, str] = {}
        group_labels: dict[str, str] = {}
        for row in csv.DictReader(source):
            row_id = int(row["source_row_id"])
            if row_id in assignments:
                raise ValueError(f"Source row {row_id} occurs more than once in the manifest.")
            assignments[row_id] = row
            previous_split = group_splits.setdefault(row["group_id"], row["split"])
            previous_label = group_labels.setdefault(row["group_id"], row["label"])
            if previous_split != row["split"]:
                raise ValueError(f"Group {row['group_id']} crosses split partitions.")
            if previous_label != row["label"]:
                raise ValueError(f"Group {row['group_id']} contains conflicting labels.")

    selected: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    with zipfile.ZipFile(SOURCE_ZIP) as archive:
        for row_id, line in enumerate(
            archive.read("SMSSpamCollection").decode("utf-8").splitlines(), start=1
        ):
            assignment = assignments.get(row_id)
            if assignment is None:
                raise ValueError(f"Source row {row_id} is missing from the split manifest.")
            split = assignment["split"]
            split_counts[split] += 1
            if split == "test":
                continue
            if split not in {"train", "validation"}:
                raise ValueError(f"Unexpected split name {split!r} in the manifest.")
            if "\t" not in line:
                raise ValueError(f"Malformed source row {row_id}.")
            original_label, text = line.split("\t", 1)
            label = LABEL_MAPPING.get(original_label)
            if label is None or str(label) != assignment["label"]:
                raise ValueError(f"Label mismatch between source and manifest at row {row_id}.")
            selected.append(
                {
                    "source_row_id": row_id,
                    "label": label,
                    "split": split,
                    "text": text,
                }
            )

    if len(assignments) != sum(split_counts.values()):
        raise ValueError("The source row count and split manifest row count differ.")
    if split_counts["train"] == 0 or split_counts["validation"] == 0 or split_counts["test"] == 0:
        raise ValueError("A required train, validation, or test split is empty.")
    return selected, split_counts


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    random_state=42,
                    solver="liblinear",
                ),
            ),
        ]
    )


def validation_metrics(
    y_true: list[int] | np.ndarray, y_probability: np.ndarray
) -> dict[str, Any]:
    y_true_array = np.asarray(y_true, dtype=int)
    y_pred = (y_probability >= 0.5).astype(int)
    matrix = confusion_matrix(y_true_array, y_pred, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    positive_count = int(true_positive + false_negative)
    return {
        "decision_threshold": 0.5,
        "threshold_note": "Default 0.5 probability threshold; no validation-based tuning performed.",
        "accuracy": float(accuracy_score(y_true_array, y_pred)),
        "precision_positive_spam": float(precision_score(y_true_array, y_pred, pos_label=1, zero_division=0)),
        "recall_positive_spam": float(recall_score(y_true_array, y_pred, pos_label=1, zero_division=0)),
        "f1_positive_spam": float(f1_score(y_true_array, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true_array, y_pred, average="macro", zero_division=0)),
        "average_precision_positive_spam": float(average_precision_score(y_true_array, y_probability)),
        "average_precision_note": "Average precision (step-wise precision-recall summary); reported as the PR-AUC-style metric, not ROC-AUC.",
        "confusion_matrix_labels_0_ham_1_spam": {
            "true_negative_ham": int(true_negative),
            "false_positive_ham_flagged_as_spam": int(false_positive),
            "false_negative_spam_missed": int(false_negative),
            "true_positive_spam_detected": int(true_positive),
        },
        "false_negative_count": int(false_negative),
        "false_negative_rate_positive_spam": (
            float(false_negative / positive_count) if positive_count else None
        ),
        "positive_spam_prevalence": float(positive_count / len(y_true_array)) if len(y_true_array) else None,
        "validation_row_count": int(len(y_true_array)),
    }


def measure_single_message_latency(
    model: Pipeline, texts: list[str], warmup: int = 30, repetitions: int = 500
) -> dict[str, Any]:
    if not texts:
        raise ValueError("Cannot measure latency without validation messages.")
    for index in range(warmup):
        model.predict_proba([texts[index % len(texts)]])

    elapsed_ms: list[float] = []
    for index in range(repetitions):
        text = texts[index % len(texts)]
        started = time.perf_counter_ns()
        model.predict_proba([text])
        elapsed_ms.append((time.perf_counter_ns() - started) / 1_000_000)

    return {
        "device": "CPU (scikit-learn estimator)",
        "batch_size": 1,
        "preprocessing_included": True,
        "preprocessing_scope": "TF-IDF vectorization and Logistic Regression predict_proba; excludes HTTP/API overhead.",
        "warmup_requests": warmup,
        "measured_requests": repetitions,
        "p50_ms": float(np.percentile(elapsed_ms, 50)),
        "p95_ms": float(np.percentile(elapsed_ms, 95)),
        "hardware_platform": platform.platform(),
        "logical_cpu_count": __import__("os").cpu_count(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    rows, split_counts = load_training_and_validation_rows()
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    train_texts = [row["text"] for row in train_rows]
    train_labels = [row["label"] for row in train_rows]
    validation_texts = [row["text"] for row in validation_rows]
    validation_labels = [row["label"] for row in validation_rows]

    model = build_pipeline()
    training_started = time.perf_counter()
    model.fit(train_texts, train_labels)
    training_seconds = time.perf_counter() - training_started

    spam_class_index = list(model.named_steps["classifier"].classes_).index(1)
    validation_probabilities = model.predict_proba(validation_texts)[:, spam_class_index]
    metrics = validation_metrics(validation_labels, validation_probabilities)
    latency = measure_single_message_latency(model, validation_texts)

    model_path = args.model if args.model.is_absolute() else ROOT / args.model
    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    report = {
        "run_status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "UCI SMS ham-vs-spam classification; not an India-specific fraud evaluation.",
        "source_file": SOURCE_ZIP.relative_to(ROOT).as_posix(),
        "source_sha256": sha256_file(SOURCE_ZIP),
        "split_manifest": MANIFEST.relative_to(ROOT).as_posix(),
        "split_manifest_sha256": sha256_file(MANIFEST),
        "seed": 42,
        "label_mapping": LABEL_MAPPING,
        "split_row_counts": dict(split_counts),
        "test_split_used_for_fit_or_metrics": False,
        "feature_pipeline": {
            "vectorizer": "TfidfVectorizer",
            "ngram_range": [1, 2],
            "strip_accents": "unicode",
            "sublinear_tf": True,
            "fit_scope": "train split only",
        },
        "classifier": {
            "name": "LogisticRegression",
            "C": 1.0,
            "max_iter": 1000,
            "solver": "liblinear",
            "class_weight": None,
            "random_state": 42,
        },
        "package_versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
        "training_duration_seconds": float(training_seconds),
        "validation_metrics_default_threshold": metrics,
        "single_message_latency": latency,
        "saved_model_path": model_path.relative_to(ROOT).as_posix() if model_path.is_relative_to(ROOT) else str(model_path),
        "saved_model_size_bytes": model_path.stat().st_size,
        "saved_model_sha256": sha256_file(model_path),
        "limitations": [
            "Validation results are for a legacy, non-India-specific SMS spam dataset and must not be presented as Indian scam-detection performance.",
            "The 0.5 threshold is the default and has not been tuned on validation data.",
            "The held-out test set has not been evaluated; use it only after model and threshold selection are complete.",
            "Latency is a local single-message CPU measurement and excludes API/network overhead.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
