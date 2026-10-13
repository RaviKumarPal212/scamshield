"""Evaluate the already-locked SMS model and threshold once on the test split.

This script intentionally contains no threshold search or model-selection logic.
The selection record must be finalized before this script is run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ZIP = ROOT / "data" / "raw" / "uci_sms_spam_collection.zip"
MANIFEST = ROOT / "data" / "processed" / "sms_split_manifest.csv"
SPLIT_SUMMARY = ROOT / "data" / "processed" / "sms_split_summary.json"
SELECTION_PATH = ROOT / "reports" / "sms_model_selection.json"
VALIDATION_REPORT = ROOT / "reports" / "sms_distilmbert_validation.json"
MODEL_DIR = ROOT / "models" / "sms_distilmbert"
REPORT_PATH = ROOT / "reports" / "sms_distilmbert_test.json"
LABEL_MAPPING = {"ham": 0, "spam": 1}
MAX_LENGTH = 256
EVAL_BATCH_SIZE = 32


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_locked_selection() -> tuple[dict[str, Any], dict[str, Any]]:
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    validation_report = json.loads(VALIDATION_REPORT.read_text(encoding="utf-8"))
    if selection.get("decision_status") != "locked_for_single_test_evaluation":
        raise ValueError("Model/threshold selection is not locked for test evaluation.")
    evidence = selection["validation_evidence"]
    if sha256_file(VALIDATION_REPORT) != evidence.get("report_sha256"):
        raise ValueError("Validation report changed after the model/threshold was locked.")
    if validation_report.get("test_split_used_for_fit_or_metrics") is not False:
        raise ValueError("Validation report does not confirm the test split was unused.")
    if selection.get("test_split_used_for_selection") is not False:
        raise ValueError("Selection record must confirm test data was not used.")
    model_info = selection["selected_model"]
    if model_info.get("model_id") != validation_report["base_model"]["model_id"]:
        raise ValueError("Locked model ID does not match the validation report.")
    if model_info.get("revision") != validation_report["base_model"]["revision"]:
        raise ValueError("Locked model revision does not match the validation report.")
    chosen_candidate = validation_report["validation_metrics_and_threshold_candidates"][
        "validation_threshold_candidates"
    ]["highest_recall_under_precision_floors"]["precision_at_least_0.90"]
    if float(selection["locked_threshold"]) != float(chosen_candidate["threshold"]):
        raise ValueError("Locked threshold does not match the recorded validation candidate.")
    for selection_key, candidate_key in (
        ("precision_spam", "precision_spam"),
        ("recall_spam", "recall_spam"),
        ("false_positives", "false_positive_ham_flagged"),
        ("false_negatives", "false_negative_spam_missed"),
    ):
        if selection["validation_evidence"][selection_key] != chosen_candidate[candidate_key]:
            raise ValueError("Selection evidence does not match the frozen validation candidate.")
    weights_path = ROOT / model_info["weights_file"]
    if sha256_file(weights_path) != model_info.get("weights_sha256"):
        raise ValueError("Model weights changed after the model/threshold was locked.")
    return selection, validation_report


def load_test_rows(expected_source_hash: str, expected_manifest_hash: str) -> list[dict[str, Any]]:
    split_summary = json.loads(SPLIT_SUMMARY.read_text(encoding="utf-8"))
    source_hash = sha256_file(SOURCE_ZIP)
    manifest_hash = sha256_file(MANIFEST)
    if source_hash != expected_source_hash or source_hash != split_summary.get("source_sha256"):
        raise ValueError("Raw dataset hash does not match the frozen split provenance.")
    if manifest_hash != expected_manifest_hash or manifest_hash != split_summary.get("manifest_sha256"):
        raise ValueError("Split manifest hash does not match the frozen split provenance.")

    assignments: dict[int, dict[str, str]] = {}
    group_splits: dict[str, str] = {}
    with MANIFEST.open("r", newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            row_id = int(row["source_row_id"])
            if row_id in assignments:
                raise ValueError(f"Duplicate manifest row ID {row_id}.")
            assignments[row_id] = row
            prior_split = group_splits.setdefault(row["group_id"], row["split"])
            if prior_split != row["split"]:
                raise ValueError(f"Group {row['group_id']} crosses split partitions.")

    test_rows: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    with zipfile.ZipFile(SOURCE_ZIP) as archive:
        lines = archive.read("SMSSpamCollection").decode("utf-8").splitlines()
    if len(lines) != len(assignments):
        raise ValueError("Dataset row count and split manifest row count differ.")
    for row_id, line in enumerate(lines, start=1):
        assignment = assignments.get(row_id)
        if assignment is None:
            raise ValueError(f"Source row {row_id} is missing from the split manifest.")
        split_counts[assignment["split"]] += 1
        if assignment["split"] != "test":
            continue
        if "\t" not in line:
            raise ValueError(f"Malformed test source row {row_id}.")
        raw_label, text = line.split("\t", 1)
        label = LABEL_MAPPING.get(raw_label)
        if label is None or str(label) != assignment["label"]:
            raise ValueError(f"Test label mismatch at source row {row_id}.")
        test_rows.append({"source_row_id": row_id, "label": label, "text": text})

    if not test_rows or split_counts["test"] != len(test_rows):
        raise ValueError("Test split is empty or its manifest count is inconsistent.")
    return test_rows


def predict_spam_probabilities(
    model: torch.nn.Module, tokenizer: Any, texts: list[str], device: torch.device
) -> np.ndarray:
    all_probabilities: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), EVAL_BATCH_SIZE):
            batch_texts = texts[start : start + EVAL_BATCH_SIZE]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
            all_probabilities.append(probabilities.cpu().numpy())
    return np.concatenate(all_probabilities)


def fixed_threshold_metrics(
    labels: list[int], probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    y_true = np.asarray(labels, dtype=int)
    y_pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spam_count = int(np.sum(y_true == 1))
    ham_count = int(np.sum(y_true == 0))
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_spam": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_spam": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_spam": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "average_precision_spam": float(average_precision_score(y_true, probabilities)),
        "false_positive_rate_ham": float(fp / ham_count) if ham_count else 0.0,
        "false_negative_rate_spam": float(fn / spam_count) if spam_count else 0.0,
        "true_negative_ham_correct": int(tn),
        "false_positive_ham_flagged": int(fp),
        "false_negative_spam_missed": int(fn),
        "true_positive_spam_detected": int(tp),
        "ham_negative_count": ham_count,
        "spam_positive_count": spam_count,
        "test_row_count": int(len(y_true)),
    }


def main() -> int:
    selection, validation_report = load_locked_selection()
    selected = selection["selected_model"]
    threshold = float(selection["locked_threshold"])
    rows = load_test_rows(
        validation_report["source_sha256"], validation_report["split_manifest_sha256"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, local_files_only=True
    ).to(device)
    probabilities = predict_spam_probabilities(
        model, tokenizer, [row["text"] for row in rows], device
    )
    metrics = fixed_threshold_metrics(
        [row["label"] for row in rows], probabilities, threshold
    )
    report = {
        "run_status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": "One-time evaluation of the locked model and threshold on the UCI SMS test split.",
        "test_split_used_for_model_or_threshold_selection": False,
        "model_selection_record": SELECTION_PATH.relative_to(ROOT).as_posix(),
        "selection_record_sha256": sha256_file(SELECTION_PATH),
        "selected_model": selected,
        "locked_threshold": threshold,
        "source_sha256": validation_report["source_sha256"],
        "split_manifest_sha256": validation_report["split_manifest_sha256"],
        "test_metrics": metrics,
        "inference": {
            "device": str(device),
            "batch_size": EVAL_BATCH_SIZE,
            "max_token_length": MAX_LENGTH,
            "training_report": VALIDATION_REPORT.relative_to(ROOT).as_posix(),
            "validation_best_checkpoint_metric": validation_report["training_configuration"]["best_checkpoint_metric"],
        },
        "limitations": [
            "UCI SMS Spam Collection is an English SMS spam dataset, not India-specific scam data.",
            "The test set was used once with the model and threshold locked in advance; do not tune on these results.",
            "Performance does not establish Hindi or Hinglish effectiveness.",
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
