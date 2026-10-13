"""Fine-tune multilingual DistilBERT on UCI SMS train and validate only."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    from .analyze_sms_thresholds import choose_candidates, score_threshold
    from .train_sms_baseline import (
        LABEL_MAPPING,
        MANIFEST,
        SOURCE_ZIP,
        load_training_and_validation_rows,
        sha256_file,
    )
except ImportError:  # Also support direct execution as a script.
    from analyze_sms_thresholds import choose_candidates, score_threshold
    from train_sms_baseline import (
        LABEL_MAPPING,
        MANIFEST,
        SOURCE_ZIP,
        load_training_and_validation_rows,
        sha256_file,
    )


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "distilbert/distilbert-base-multilingual-cased"
MODEL_REVISION = "45c032ab32cc946ad88a166f7cb282f58c753c2e"
MAX_LENGTH = 256
MODEL_DIR = ROOT / "models" / "sms_distilmbert"
CHECKPOINT_DIR = ROOT / "models" / "sms_distilmbert_checkpoints"
REPORT_PATH = ROOT / "reports" / "sms_distilmbert_validation.json"


class EncodedMessages(Dataset):
    def __init__(self, encodings: dict[str, list[list[int]]], labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(values[index], dtype=torch.long)
            for key, values in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def make_compute_metrics():
    def compute_metrics(prediction: Any) -> dict[str, float]:
        logits = np.asarray(prediction.predictions)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        spam_probabilities = probabilities[:, 1]
        predicted = (spam_probabilities >= 0.5).astype(int)
        labels = np.asarray(prediction.label_ids, dtype=int)
        return {
            "precision_spam": float(precision_score(labels, predicted, pos_label=1, zero_division=0)),
            "recall_spam": float(recall_score(labels, predicted, pos_label=1, zero_division=0)),
            "f1_spam": float(f1_score(labels, predicted, pos_label=1, zero_division=0)),
            "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
            "average_precision_spam": float(average_precision_score(labels, spam_probabilities)),
        }

    return compute_metrics


def validation_result(labels: list[int], probabilities: np.ndarray) -> dict[str, Any]:
    metrics_at_default = score_threshold(np.asarray(labels, dtype=int), probabilities, 0.5)
    predictions = (probabilities >= 0.5).astype(int)
    metrics_at_default.update(
        {
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
            "average_precision_spam": float(average_precision_score(labels, probabilities)),
            "validation_row_count": len(labels),
            "spam_positive_count": int(sum(label == 1 for label in labels)),
            "ham_negative_count": int(sum(label == 0 for label in labels)),
        }
    )
    return {
        "default_threshold_0_5": metrics_at_default,
        "validation_threshold_candidates": choose_candidates(
            np.asarray(labels, dtype=int), probabilities
        ),
    }


def tokenization_stats(tokenizer: Any, texts: list[str]) -> dict[str, int]:
    all_tokens = tokenizer(texts, truncation=False, padding=False)["input_ids"]
    return {
        "row_count": len(texts),
        "rows_truncated_at_max_length": sum(len(token_ids) > MAX_LENGTH for token_ids in all_tokens),
        "longest_tokenized_length_before_truncation": max(
            (len(token_ids) for token_ids in all_tokens), default=0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available to PyTorch. Resolve the GPU-enabled PyTorch install before training."
        )
    set_seed(42)
    torch.cuda.empty_cache()
    rows, split_counts = load_training_and_validation_rows()
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    train_texts = [row["text"] for row in train_rows]
    train_labels = [row["label"] for row in train_rows]
    validation_texts = [row["text"] for row in validation_rows]
    validation_labels = [row["label"] for row in validation_rows]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    train_token_stats = tokenization_stats(tokenizer, train_texts)
    validation_token_stats = tokenization_stats(tokenizer, validation_texts)
    train_encodings = tokenizer(
        train_texts, truncation=True, max_length=MAX_LENGTH, padding=False
    )
    validation_encodings = tokenizer(
        validation_texts, truncation=True, max_length=MAX_LENGTH, padding=False
    )
    train_dataset = EncodedMessages(train_encodings, train_labels)
    validation_dataset = EncodedMessages(validation_encodings, validation_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        num_labels=2,
        id2label={0: "ham", 1: "spam"},
        label2id={"ham": 0, "spam": 1},
    )

    model_dir = args.model_dir if args.model_dir.is_absolute() else ROOT / args.model_dir
    checkpoint_dir = (
        args.checkpoint_dir
        if args.checkpoint_dir.is_absolute()
        else ROOT / args.checkpoint_dir
    )
    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    training_arguments = TrainingArguments(
        output_dir=str(checkpoint_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=True,
        logging_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=42,
        data_seed=42,
        dataloader_num_workers=0,
        disable_tqdm=False,
    )
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        processing_class=tokenizer,
        compute_metrics=make_compute_metrics(),
    )
    started = time.perf_counter()
    train_result = trainer.train()
    training_seconds = time.perf_counter() - started
    prediction = trainer.predict(validation_dataset)
    logits = np.asarray(prediction.predictions)
    logits -= np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    spam_probabilities = probabilities[:, 1]

    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    best_checkpoint = trainer.state.best_model_checkpoint
    report = {
        "run_status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "UCI SMS ham-vs-spam classification; English-source benchmark only, not an India-specific fraud evaluation.",
        "base_model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": "Apache-2.0",
            "pretraining_languages": "Multilingual model card lists 104 languages; this experiment uses English UCI SMS only.",
        },
        "source_file": SOURCE_ZIP.relative_to(ROOT).as_posix(),
        "source_sha256": sha256_file(SOURCE_ZIP),
        "split_manifest": MANIFEST.relative_to(ROOT).as_posix(),
        "split_manifest_sha256": sha256_file(MANIFEST),
        "split_row_counts": dict(split_counts),
        "label_mapping": LABEL_MAPPING,
        "test_split_used_for_fit_or_metrics": False,
        "training_configuration": {
            "seed": 42,
            "epochs_requested": args.epochs,
            "best_checkpoint": best_checkpoint,
            "best_checkpoint_metric": "validation macro-F1 at 0.5 threshold",
            "train_batch_size_per_device": args.train_batch_size,
            "effective_batch_size": args.train_batch_size * 2,
            "eval_batch_size_per_device": 16,
            "gradient_accumulation_steps": 2,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "max_token_length": MAX_LENGTH,
            "mixed_precision_fp16": True,
            "device": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
        },
        "tokenization": {"train": train_token_stats, "validation": validation_token_stats},
        "training_duration_seconds": float(training_seconds),
        "trainer_final_train_metrics": train_result.metrics,
        "validation_metrics_and_threshold_candidates": validation_result(
            validation_labels, spam_probabilities
        ),
        "saved_model_dir": model_dir.relative_to(ROOT).as_posix() if model_dir.is_relative_to(ROOT) else str(model_dir),
        "limitations": [
            "Validation-only results on UCI SMS spam; do not present them as Hindi/Hinglish or India-specific scam-detection performance.",
            "The best checkpoint is selected by validation macro-F1 at a 0.5 threshold; no test rows are used.",
            "Threshold candidates are diagnostic only and require an operating-cost decision before final test evaluation.",
            "Model and tokenizer are pinned to the reported Hugging Face revision; verify licensing and attribution before redistribution.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
