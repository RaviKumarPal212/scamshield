"""Summarize validation-only threshold trade-offs for the saved SMS model."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

try:
    from .train_sms_baseline import (
        LABEL_MAPPING,
        MANIFEST,
        MODEL_PATH,
        REPORT_PATH,
        ROOT,
        SOURCE_ZIP,
        load_training_and_validation_rows,
        sha256_file,
    )
except ImportError:  # Also support direct execution as a script.
    from train_sms_baseline import (
        LABEL_MAPPING,
        MANIFEST,
        MODEL_PATH,
        REPORT_PATH,
        ROOT,
        SOURCE_ZIP,
        load_training_and_validation_rows,
        sha256_file,
    )


THRESHOLD_REPORT = ROOT / "reports" / "sms_threshold_analysis.json"
DEFAULT_THRESHOLDS = [round(value / 20, 2) for value in range(1, 20)]
PRECISION_FLOORS = (0.80, 0.90, 0.95)


def score_threshold(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (probabilities >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return {
        "threshold": float(threshold),
        "precision_spam": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_spam": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_spam": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "true_positive_spam_detected": tp,
        "false_positive_ham_flagged": fp,
        "false_negative_spam_missed": fn,
        "true_negative_ham_correct": tn,
        "predicted_spam_count": tp + fp,
    }


def choose_candidates(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    observed_thresholds = np.unique(np.concatenate((probabilities, np.array([0.5]))))
    scored = [score_threshold(y_true, probabilities, float(threshold)) for threshold in observed_thresholds]
    best_f1 = max(scored, key=lambda item: (item["f1_spam"], item["precision_spam"], item["threshold"]))
    precision_floor_candidates: dict[str, Any] = {}
    for floor in PRECISION_FLOORS:
        eligible = [item for item in scored if item["precision_spam"] >= floor]
        if not eligible:
            precision_floor_candidates[f"precision_at_least_{floor:.2f}"] = None
            continue
        precision_floor_candidates[f"precision_at_least_{floor:.2f}"] = max(
            eligible,
            key=lambda item: (item["recall_spam"], item["precision_spam"], item["threshold"]),
        )
    return {
        "best_f1_on_validation": best_f1,
        "highest_recall_under_precision_floors": precision_floor_candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--training-report", type=Path, default=REPORT_PATH)
    parser.add_argument("--output", type=Path, default=THRESHOLD_REPORT)
    args = parser.parse_args()

    model_path = args.model if args.model.is_absolute() else ROOT / args.model
    training_report_path = (
        args.training_report
        if args.training_report.is_absolute()
        else ROOT / args.training_report
    )
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
    if sha256_file(model_path) != training_report.get("saved_model_sha256"):
        raise ValueError("The model hash differs from the saved training report.")
    if sha256_file(SOURCE_ZIP) != training_report.get("source_sha256"):
        raise ValueError("The raw archive hash differs from the saved training report.")
    if sha256_file(MANIFEST) != training_report.get("split_manifest_sha256"):
        raise ValueError("The split manifest hash differs from the saved training report.")

    rows, _ = load_training_and_validation_rows()
    validation_rows = [row for row in rows if row["split"] == "validation"]
    texts = [row["text"] for row in validation_rows]
    labels = np.asarray([row["label"] for row in validation_rows], dtype=int)
    model = joblib.load(model_path)
    positive_index = list(model.named_steps["classifier"].classes_).index(
        LABEL_MAPPING["spam"]
    )
    probabilities = model.predict_proba(texts)[:, positive_index]

    report = {
        "run_status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "UCI SMS validation split only; no test rows were loaded or scored.",
        "validation_row_count": len(validation_rows),
        "spam_positive_count": int(np.sum(labels == 1)),
        "ham_negative_count": int(np.sum(labels == 0)),
        "default_threshold": 0.5,
        "default_threshold_metrics": score_threshold(labels, probabilities, 0.5),
        "common_threshold_table": [
            score_threshold(labels, probabilities, threshold)
            for threshold in DEFAULT_THRESHOLDS
        ],
        "validation_candidate_operating_points": choose_candidates(labels, probabilities),
        "selection_note": "These points show validation trade-offs only. They do not select or lock a final threshold; choose according to the intended false-negative/false-positive cost, then evaluate the held-out test once.",
        "source_sha256": sha256_file(SOURCE_ZIP),
        "split_manifest_sha256": sha256_file(MANIFEST),
        "model_sha256": sha256_file(model_path),
        "model_file": model_path.relative_to(ROOT).as_posix() if model_path.is_relative_to(ROOT) else str(model_path),
        "training_report_file": training_report_path.relative_to(ROOT).as_posix() if training_report_path.is_relative_to(ROOT) else str(training_report_path),
        "test_split_used": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
