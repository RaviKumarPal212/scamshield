"""Measure single-message CPU latency for the saved SMS transformer."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from .train_sms_baseline import load_training_and_validation_rows
except ImportError:  # Also support direct execution as a script.
    from train_sms_baseline import load_training_and_validation_rows


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "sms_distilmbert"
DEFAULT_REPORT = ROOT / "reports" / "sms_distilmbert_validation.json"


def measure_latency(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
    max_length: int = 256,
    warmup: int = 30,
    repetitions: int = 500,
) -> dict[str, Any]:
    if not texts:
        raise ValueError("Cannot measure latency without validation messages.")

    def predict(text: str) -> None:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.inference_mode():
            model(**encoded)

    for index in range(warmup):
        predict(texts[index % len(texts)])

    elapsed_ms: list[float] = []
    for index in range(repetitions):
        started = time.perf_counter_ns()
        predict(texts[index % len(texts)])
        elapsed_ms.append((time.perf_counter_ns() - started) / 1_000_000)

    return {
        "device": "CPU (PyTorch transformer)",
        "batch_size": 1,
        "preprocessing_included": True,
        "preprocessing_scope": "Tokenization and model forward pass; excludes HTTP/API overhead.",
        "warmup_requests": warmup,
        "measured_requests": repetitions,
        "p50_ms": float(np.percentile(elapsed_ms, 50)),
        "p95_ms": float(np.percentile(elapsed_ms, 95)),
        "hardware_platform": platform.platform(),
        "logical_cpu_count": __import__("os").cpu_count(),
        "torch_version": torch.__version__,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repetitions", type=int, default=500)
    args = parser.parse_args()

    model_dir = args.model if args.model.is_absolute() else ROOT / args.model
    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    rows, _ = load_training_and_validation_rows()
    validation_texts = [row["text"] for row in rows if row["split"] == "validation"]

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True
    ).to("cpu")
    model.eval()
    latency = measure_latency(
        model,
        tokenizer,
        validation_texts,
        warmup=args.warmup,
        repetitions=args.repetitions,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["cpu_single_message_latency"] = latency
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(latency, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
