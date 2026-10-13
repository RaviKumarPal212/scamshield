"""Audit the downloaded ScamShield datasets without printing raw messages or URLs.

Run from any directory with:
    python scripts/audit_datasets.py

The report is JSON on stdout. Pass --output reports/data_audit.json to save a copy.
This script profiles source files only; it does not clean data, create splits, or train.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import io
import json
import re
import statistics
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
UCI_SMS_ZIP = RAW / "uci_sms_spam_collection.zip"
HINGLISH_JSON = RAW / "hinglish_scam_text" / "data.json"
PHIUSIIL_ZIP = RAW / "phiusiil_phishing_url.zip"
ROMAN_HINDI_CUES = {
    "aap", "apka", "bhai", "bhej", "bhejo", "hai", "hain", "jama", "ka",
    "kar", "karo", "ki", "ko", "kripya", "mera", "meri", "nahi", "nahin",
    "paise", "rupaye", "se", "tera", "tum", "upi", "kya", "nikalo",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_info(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": path.relative_to(ROOT).as_posix(), "status": "missing"}
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def normalize_text(text: str) -> str:
    """Normalize for exact duplicate/source-overlap checks, not language detection."""
    return unicodedata.normalize("NFKC", text).strip().casefold()


def message_template_signature(text: str) -> str:
    """Build a conservative review key for messages differing by links/numbers."""
    normalized = normalize_text(text)
    normalized = re.sub(r"(?:https?://|www\.)\S+", " <url> ", normalized)
    normalized = re.sub(r"\b\d+\b", " <num> ", normalized)
    return " ".join(re.sub(r"[^\w<>]+", " ", normalized).split())


def audit_sms(path: Path) -> tuple[dict[str, Any], dict[str, set[str]]]:
    info = asset_info(path)
    if info.get("status") == "missing":
        return {**info, "status": "missing"}, {}

    try:
        with zipfile.ZipFile(path) as archive:
            members = {item.filename: item.file_size for item in archive.infolist()}
            raw = archive.read("SMSSpamCollection").decode("utf-8")
    except (OSError, KeyError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
        return {**info, "status": "error", "error": str(exc)}, {}

    lines = raw.splitlines()
    rows: list[tuple[str, str]] = []
    malformed_lines = 0
    for line in lines:
        if "\t" not in line:
            malformed_lines += 1
            continue
        rows.append(tuple(line.split("\t", 1)))

    label_counts = Counter(label for label, _ in rows)
    exact_counts = Counter(text for _, text in rows)
    normalized_labels: dict[str, set[str]] = defaultdict(set)
    template_labels: dict[str, set[str]] = defaultdict(set)
    template_counts: Counter[str] = Counter()
    template_texts: dict[str, list[str]] = defaultdict(list)
    lengths: list[int] = []
    for label, text in rows:
        normalized_labels[normalize_text(text)].add(label)
        signature = message_template_signature(text)
        if len(signature) >= 20:
            template_counts[signature] += 1
            template_labels[signature].add(label)
            template_texts[signature].append(text)
        lengths.append(len(text))

    template_pairwise_similarities = [
        difflib.SequenceMatcher(None, left.casefold(), right.casefold()).ratio()
        for signature, texts in template_texts.items()
        if template_counts[signature] > 1
        for index, left in enumerate(texts)
        for right in texts[index + 1 :]
    ]

    summary = {
        **info,
        "status": "audited",
        "archive_members": members,
        "row_count": len(rows),
        "malformed_line_count": malformed_lines,
        "label_counts": dict(sorted(label_counts.items())),
        "unexpected_label_count": sum(
            count for label, count in label_counts.items() if label not in {"ham", "spam"}
        ),
        "empty_message_count": sum(not text.strip() for _, text in rows),
        "exact_duplicate_rows_beyond_first": sum(
            count - 1 for count in exact_counts.values() if count > 1
        ),
        "normalized_duplicate_rows_beyond_first": sum(
            count - 1
            for count in Counter(normalize_text(text) for _, text in rows).values()
            if count > 1
        ),
        "normalized_texts_with_conflicting_labels": sum(
            len(labels) > 1 for labels in normalized_labels.values()
        ),
        "template_candidates": {
            "method": "NFKC/strip/case-fold, replace URLs and standalone digit sequences, then normalize punctuation/spacing; signatures shorter than 20 characters excluded",
            "repeated_signature_group_count": sum(count > 1 for count in template_counts.values()),
            "rows_beyond_first_in_repeated_groups": sum(
                count - 1 for count in template_counts.values() if count > 1
            ),
            "repeated_signatures_with_conflicting_labels": sum(
                template_counts[signature] > 1 and len(labels) > 1
                for signature, labels in template_labels.items()
            ),
            "pairwise_raw_message_comparison_count": len(template_pairwise_similarities),
            "pairwise_casefolded_character_similarity": {
                "method": "difflib.SequenceMatcher ratio on raw messages after case-folding; heuristic, not semantic similarity",
                "minimum": min(template_pairwise_similarities) if template_pairwise_similarities else None,
                "median": statistics.median(template_pairwise_similarities) if template_pairwise_similarities else None,
                "pairs_at_or_above_0_8": sum(score >= 0.8 for score in template_pairwise_similarities),
                "pairs_below_0_8": sum(score < 0.8 for score in template_pairwise_similarities),
            },
            "interpretation": "Strong near-duplicate/template candidates, not automatic deduplication decisions; human review and group-aware splitting are still required.",
        },
        "message_length_chars": {
            "min": min(lengths) if lengths else None,
            "median": statistics.median(lengths) if lengths else None,
            "max": max(lengths) if lengths else None,
        },
    }
    return summary, normalized_labels


def audit_hinglish(
    path: Path, uci_labels: dict[str, set[str]]
) -> dict[str, Any]:
    info = asset_info(path)
    if info.get("status") == "missing":
        return {**info, "status": "missing"}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {**info, "status": "error", "error": str(exc)}

    rows = data.get("rows") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return {
            **info,
            "status": "error",
            "error": "Expected a JSON object containing a 'rows' list.",
        }

    malformed_rows = 0
    rows_missing_required_fields = 0
    non_string_texts = 0
    invalid_labels = 0
    empty_texts = 0
    normalized_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    overlap_rows = 0
    overlap_texts: set[str] = set()
    overlap_agreements = 0
    overlap_disagreements = 0
    overlap_ambiguous_uci_labels = 0
    nonoverlap_label_counts: Counter[str] = Counter()
    nonoverlap_devanagari_rows = 0
    nonoverlap_roman_hindi_cue_rows = 0
    lengths: list[int] = []

    for row in rows:
        if not isinstance(row, dict):
            malformed_rows += 1
            continue
        if not {"text", "label"}.issubset(row):
            rows_missing_required_fields += 1
        text = row.get("text")
        label = row.get("label")
        if not isinstance(text, str):
            non_string_texts += 1
            continue
        if type(label) is not int or label not in (0, 1):
            invalid_labels += 1
        label_counts[str(label)] += 1
        if not text.strip():
            empty_texts += 1
        lengths.append(len(text))
        normalized = normalize_text(text)
        normalized_counts[normalized] += 1
        if normalized in uci_labels:
            overlap_rows += 1
            overlap_texts.add(normalized)
            source_labels = uci_labels[normalized]
            if len(source_labels) != 1:
                overlap_ambiguous_uci_labels += 1
            elif label == (1 if "spam" in source_labels else 0):
                overlap_agreements += 1
            else:
                overlap_disagreements += 1
        else:
            nonoverlap_label_counts[str(label)] += 1
            if any("\u0900" <= char <= "\u097f" for char in text):
                nonoverlap_devanagari_rows += 1
            tokens = set(re.findall(r"[a-z]+", normalized))
            if tokens & ROMAN_HINDI_CUES:
                nonoverlap_roman_hindi_cue_rows += 1

    summary = {
        **info,
        "status": "audited",
        "root_type": type(data).__name__,
        "row_count": len(rows),
        "row_fields": sorted(
            {key for row in rows if isinstance(row, dict) for key in row}
        ),
        "malformed_row_count": malformed_rows,
        "rows_missing_required_fields": rows_missing_required_fields,
        "non_string_text_count": non_string_texts,
        "invalid_label_count": invalid_labels,
        "label_counts": dict(sorted(label_counts.items())),
        "empty_text_count": empty_texts,
        "normalized_duplicate_rows_beyond_first": sum(
            count - 1 for count in normalized_counts.values() if count > 1
        ),
        "message_length_chars": {
            "min": min(lengths) if lengths else None,
            "median": statistics.median(lengths) if lengths else None,
            "max": max(lengths) if lengths else None,
        },
        "uci_overlap": {
            "normalization": "Unicode NFKC, strip, case-fold",
            "matching_row_count": overlap_rows,
            "distinct_matching_text_count": len(overlap_texts),
            "label_agreement_count": overlap_agreements,
            "label_disagreement_count": overlap_disagreements,
            "ambiguous_uci_label_count": overlap_ambiguous_uci_labels,
            "nonmatching_row_count": sum(nonoverlap_label_counts.values()),
            "nonmatching_label_counts": dict(sorted(nonoverlap_label_counts.items())),
            "nonmatching_rows_with_devanagari_codepoints": nonoverlap_devanagari_rows,
            "nonmatching_rows_with_romanized_hindi_cues": nonoverlap_roman_hindi_cue_rows,
            "romanized_hindi_cue_method": "At least one token from a small, manually selected cue-word list; heuristic only, not language identification.",
        },
    }
    return summary


def normalized_hostname(url: str) -> str:
    value = url.strip()
    candidate = value if "://" in value else f"//{value}"
    try:
        return (urlsplit(candidate).hostname or "").rstrip(".").casefold()
    except ValueError:
        return ""


def audit_phiusiil(path: Path) -> dict[str, Any]:
    info = asset_info(path)
    if info.get("status") == "missing":
        return {**info, "status": "missing"}

    try:
        with zipfile.ZipFile(path) as archive:
            members = {item.filename: item.file_size for item in archive.infolist()}
            csv_members = [name for name in members if name.lower().endswith(".csv")]
            if len(csv_members) != 1:
                raise ValueError(f"Expected one CSV in archive; found {len(csv_members)}.")
            with archive.open(csv_members[0]) as binary_source:
                source = io.TextIOWrapper(binary_source, encoding="utf-8-sig", newline="")
                reader = csv.reader(source)
                header = next(reader, [])
                if "URL" not in header or "label" not in header:
                    raise ValueError("CSV must contain 'URL' and 'label' columns.")
                url_index = header.index("URL")
                label_index = header.index("label")
                domain_index = header.index("Domain") if "Domain" in header else None

                row_count = 0
                malformed_rows = 0
                empty_cells = 0
                empty_urls = 0
                invalid_labels = 0
                labels: Counter[str] = Counter()
                url_counts: Counter[str] = Counter()
                url_labels: dict[str, set[str]] = defaultdict(set)
                hostname_labels: dict[str, set[str]] = defaultdict(set)
                host_count_by_label: dict[str, set[str]] = defaultdict(set)
                domain_counts: Counter[str] = Counter()
                domain_labels: dict[str, set[str]] = defaultdict(set)
                domain_hostname_match_rows = 0
                blank_hosts = 0

                for row in reader:
                    row_count += 1
                    if len(row) != len(header):
                        malformed_rows += 1
                        continue
                    empty_cells += sum(cell == "" for cell in row)
                    url = row[url_index]
                    label = row[label_index].strip()
                    if not url.strip():
                        empty_urls += 1
                    if label not in {"0", "1"}:
                        invalid_labels += 1
                    labels[label] += 1
                    url_counts[url] += 1
                    url_labels[url].add(label)
                    host = normalized_hostname(url)
                    if domain_index is not None:
                        domain = row[domain_index].strip().casefold()
                        domain_counts[domain] += 1
                        domain_labels[domain].add(label)
                        if domain == host:
                            domain_hostname_match_rows += 1
                    if not host:
                        blank_hosts += 1
                    else:
                        hostname_labels[host].add(label)
                        host_count_by_label[label].add(host)
    except (OSError, KeyError, UnicodeDecodeError, zipfile.BadZipFile, csv.Error, ValueError) as exc:
        return {**info, "status": "error", "error": str(exc)}

    summary = {
        **info,
        "status": "audited",
        "archive_members": members,
        "csv_name": csv_members[0],
        "row_count": row_count,
        "column_count": len(header),
        "columns": header,
        "malformed_row_count": malformed_rows,
        "unique_url_value_count": len(url_counts),
        "empty_cell_count": empty_cells,
        "empty_url_count": empty_urls,
        "invalid_label_count": invalid_labels,
        "label_counts": dict(sorted(labels.items())),
        "exact_duplicate_url_rows_beyond_first": sum(
            count - 1 for count in url_counts.values() if count > 1
        ),
        "exact_duplicate_url_group_count": sum(count > 1 for count in url_counts.values()),
        "duplicate_urls_with_conflicting_labels": sum(
            len(row_labels) > 1 for row_labels in url_labels.values()
        ),
        "hostname_summary": {
            "normalization": "URL parser hostname, trailing dot removed, case-folded",
            "blank_or_unparseable_host_count": blank_hosts,
            "distinct_hostname_count": len(hostname_labels),
            "hostnames_in_both_labels": sum(
                len(host_labels) > 1 for host_labels in hostname_labels.values()
            ),
            "distinct_hostnames_by_label": {
                label: len(hosts) for label, hosts in sorted(host_count_by_label.items())
            },
        },
        "domain_field_diagnostics": {
            "field_present": domain_index is not None,
            "distinct_domain_value_count": len(domain_counts) if domain_index is not None else None,
            "rows_where_domain_equals_parsed_hostname": domain_hostname_match_rows if domain_index is not None else None,
            "domain_hostname_mismatch_count": row_count - domain_hostname_match_rows if domain_index is not None else None,
            "domain_values_with_both_labels": sum(
                len(domain_value_labels) > 1 for domain_value_labels in domain_labels.values()
            ) if domain_index is not None else None,
            "interpretation": "The supplied Domain field is compared with the parsed hostname; it is not assumed to be a registrable domain.",
        },
    }
    return summary


def build_report() -> dict[str, Any]:
    sms, uci_labels = audit_sms(UCI_SMS_ZIP)
    hinglish = audit_hinglish(HINGLISH_JSON, uci_labels)
    phiusiil = audit_phiusiil(PHIUSIIL_ZIP)
    sections = {"uci_sms": sms, "hinglish": hinglish, "phiusiil_url": phiusiil}
    run_status = "complete" if all(item.get("status") == "audited" for item in sections.values()) else "partial"

    return {
        "report_version": 1,
        "run_status": run_status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": "complete means all configured files were audited; it does not mean the data are suitable for training.",
        "normalization": "Text overlap uses Unicode NFKC, strip, and case-fold. This detects exact normalized matches, not semantic near-duplicates.",
        "limitations": [
            "No raw message or URL values are included in this report.",
            "Devanagari detection is a codepoint check, not language identification.",
            "URL hostname counts are not registrable-domain counts; use a Public Suffix List-aware grouping step before URL splits.",
            "No dataset is cleaned, split, or used to train a model.",
        ],
        "datasets": sections,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path; relative paths are resolved from the project root.",
    )
    args = parser.parse_args()

    report = build_report()
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        output_path = args.output if args.output.is_absolute() else ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Report saved to {output_path.relative_to(ROOT).as_posix() if output_path.is_relative_to(ROOT) else output_path}")
    print(rendered)
    return 0 if report["run_status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
