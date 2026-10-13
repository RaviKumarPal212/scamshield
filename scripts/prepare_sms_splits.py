"""Create a group-aware split manifest for the UCI SMS baseline.

Run from any directory with:
    python scripts/prepare_sms_splits.py

The manifest contains source row IDs, labels, stable group IDs, and split names;
it never copies the SMS text. It does not fit a model or transform the dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .audit_datasets import (
        UCI_SMS_ZIP,
        message_template_signature,
        normalize_text,
        sha256_file,
    )
except ImportError:  # Also support direct execution as a script.
    from audit_datasets import (
        UCI_SMS_ZIP,
        message_template_signature,
        normalize_text,
        sha256_file,
    )


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "processed" / "sms_split_manifest.csv"
SUMMARY_PATH = ROOT / "data" / "processed" / "sms_split_summary.json"
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
LABEL_MAPPING = {"ham": 0, "spam": 1}


def assign_group_splits(
    group_rows: dict[str, tuple[int, int]], seed: int = 42
) -> dict[str, str]:
    """Assign single-label groups to approximate per-class 70/15/15 targets."""
    split_by_group: dict[str, str] = {}
    labels = sorted({label for label, _ in group_rows.values()})

    for label in labels:
        label_groups = [
            (group_id, row_count)
            for group_id, (group_label, row_count) in group_rows.items()
            if group_label == label
        ]
        randomizer = random.Random(seed + label * 104729)
        randomizer.shuffle(label_groups)
        label_groups.sort(key=lambda item: item[1], reverse=True)
        total_rows = sum(row_count for _, row_count in label_groups)
        targets = {
            split: total_rows * SPLIT_RATIOS[split] for split in SPLIT_NAMES
        }
        loads = {split: 0 for split in SPLIT_NAMES}

        for group_id, row_count in label_groups:
            def objective(candidate: str) -> float:
                return sum(
                    (
                        (loads[split] + (row_count if split == candidate else 0) - targets[split])
                        / max(targets[split], 1)
                    )
                    ** 2
                    for split in SPLIT_NAMES
                )

            selected = min(SPLIT_NAMES, key=objective)
            split_by_group[group_id] = selected
            loads[selected] += row_count

    return split_by_group


def stable_group_key(text: str) -> str:
    signature = message_template_signature(text)
    if len(signature) >= 20:
        return f"template:{signature}"
    return f"exact:{normalize_text(text)}"


def load_rows(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        raw = archive.read("SMSSpamCollection").decode("utf-8")

    rows: list[dict[str, Any]] = []
    for source_row_id, line in enumerate(raw.splitlines(), start=1):
        if "\t" not in line:
            raise ValueError(f"Malformed source row at line {source_row_id}.")
        original_label, text = line.split("\t", 1)
        if original_label not in LABEL_MAPPING:
            raise ValueError(f"Unexpected source label at line {source_row_id}.")
        key = stable_group_key(text)
        group_id = hashlib.sha256(key.encode("utf-8")).hexdigest()
        rows.append(
            {
                "source_row_id": source_row_id,
                "original_label": original_label,
                "label": LABEL_MAPPING[original_label],
                "group_id": group_id,
            }
        )
    return rows


def create_manifest(rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    group_labels: dict[str, set[int]] = defaultdict(set)
    group_sizes: Counter[str] = Counter()
    for row in rows:
        group_labels[row["group_id"]].add(row["label"])
        group_sizes[row["group_id"]] += 1

    conflicting_groups = [group for group, labels in group_labels.items() if len(labels) != 1]
    if conflicting_groups:
        raise ValueError(
            f"Cannot split safely: {len(conflicting_groups)} groups contain conflicting labels."
        )

    group_rows = {
        group_id: (next(iter(group_labels[group_id])), group_size)
        for group_id, group_size in group_sizes.items()
    }
    split_by_group = assign_group_splits(group_rows, seed)
    manifest = [
        {
            "source_row_id": row["source_row_id"],
            "original_label": row["original_label"],
            "label": row["label"],
            "group_id": row["group_id"],
            "split": split_by_group[row["group_id"]],
        }
        for row in rows
    ]

    split_rows: Counter[str] = Counter(row["split"] for row in manifest)
    split_labels: dict[str, Counter[int]] = {
        split: Counter() for split in SPLIT_NAMES
    }
    groups_per_split: dict[str, set[str]] = {split: set() for split in SPLIT_NAMES}
    for row in manifest:
        split_labels[row["split"]][row["label"]] += 1
        groups_per_split[row["split"]].add(row["group_id"])

    group_split_counts = Counter(split_by_group.values())
    summary = {
        "run_status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": UCI_SMS_ZIP.relative_to(ROOT).as_posix(),
        "source_sha256": sha256_file(UCI_SMS_ZIP),
        "source_row_count": len(rows),
        "seed": seed,
        "grouping": "Exact normalized messages are grouped; messages with a template signature of at least 20 characters are grouped by NFKC/trim/case-fold plus URL/number/punctuation normalization.",
        "assignment": "Deterministic greedy allocation of whole groups within each label toward 70/15/15 row-count targets; actual ratios may differ because groups cannot be split.",
        "target_split_ratios": SPLIT_RATIOS,
        "label_mapping": LABEL_MAPPING,
        "group_count": len(group_rows),
        "largest_group_rows": max(group_sizes.values(), default=0),
        "group_counts_by_split": {split: group_split_counts[split] for split in SPLIT_NAMES},
        "row_counts_by_split": {split: split_rows[split] for split in SPLIT_NAMES},
        "row_ratios_by_split": {
            split: split_rows[split] / len(rows) if rows else 0 for split in SPLIT_NAMES
        },
        "label_counts_by_split": {
            split: {str(label): split_labels[split][label] for label in sorted(LABEL_MAPPING.values())}
            for split in SPLIT_NAMES
        },
        "groups_appearing_in_multiple_splits": 0,
        "raw_text_in_manifest": False,
    }

    all_group_splits = [groups_per_split[split] for split in SPLIT_NAMES]
    if any(all_group_splits[i] & all_group_splits[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("A group leaked across split partitions.")
    if len(manifest) != len(rows) or len({row["source_row_id"] for row in manifest}) != len(rows):
        raise AssertionError("The manifest does not contain each source row exactly once.")
    if any(split_rows[split] == 0 for split in SPLIT_NAMES):
        raise AssertionError("At least one split is empty.")
    if any(split_labels[split][label] == 0 for split in SPLIT_NAMES for label in LABEL_MAPPING.values()):
        raise AssertionError("At least one split is missing a class label.")

    return manifest, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args()

    rows = load_rows(UCI_SMS_ZIP)
    manifest, summary = create_manifest(rows, args.seed)
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    summary_path = args.summary if args.summary.is_absolute() else ROOT / args.summary
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=("source_row_id", "original_label", "label", "group_id", "split"),
        )
        writer.writeheader()
        writer.writerows(manifest)
    summary["manifest_file"] = manifest_path.relative_to(ROOT).as_posix() if manifest_path.is_relative_to(ROOT) else str(manifest_path)
    summary["manifest_sha256"] = sha256_file(manifest_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
