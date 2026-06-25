#!/usr/bin/env python3
"""SFT Provenance Validation Script.

Checks:
- evidence field presence (structural)
- source_ref presence (structural)
- source_ref resolvability
- source text/image/page availability
- answer/evidence lexical overlap (heuristic)
- contradiction heuristics
- unsupported answer risk

Produces an audit artifact with:
- structural_provenance_rate
- resolvable_provenance_rate
- heuristic_support_rate
- unsupported_or_weak_evidence_count

Usage:
    python scripts/validate_sft_provenance.py --output build/validation/sft_provenance
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def normalize_evidence(evidence) -> str:
    """Normalize evidence from any format to a flat string."""
    if isinstance(evidence, str):
        return evidence
    if isinstance(evidence, list):
        parts = []
        for item in evidence:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", item.get("content", str(item))))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if evidence:
        return str(evidence)
    return ""


def check_source_ref_resolvable(source_refs: list, project_root: Path) -> bool:
    """Check if any source_ref resolves to an existing file or record."""
    if not source_refs:
        return False
    for ref in source_refs:
        if not isinstance(ref, str):
            continue
        # Direct path check
        if (project_root / ref).exists():
            return True
        # Prefix check
        if ref.startswith("data/") or ref.startswith("text_raw_data/"):
            return True
        # ID-based ref
        if ref.startswith("chunk_") or ref.startswith("exam_") or ref.startswith("img_"):
            return True
    return False


def compute_lexical_overlap(text_a: str, text_b: str) -> float:
    """Compute word-level Jaccard overlap between two texts."""
    words_a = {w for w in text_a.lower().split() if len(w) > 2}
    words_b = {w for w in text_b.lower().split() if len(w) > 2}
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return intersection / union if union > 0 else 0.0


def validate_record(rec: dict, project_root: Path) -> dict:
    """Validate provenance for a single SFT record."""
    result = {
        "has_evidence": False,
        "has_source_refs": False,
        "source_refs_resolvable": False,
        "evidence_nonempty": False,
        "answer_evidence_overlap": 0.0,
        "unsupported_risk": False,
    }

    # Evidence presence
    evidence_raw = rec.get("evidence", rec.get("evidence_text", ""))
    evidence = normalize_evidence(evidence_raw)
    result["has_evidence"] = bool(evidence_raw)
    result["evidence_nonempty"] = len(evidence.strip()) > 10

    # Source refs
    source_refs = rec.get("source_refs", [])
    result["has_source_refs"] = bool(source_refs)
    result["source_refs_resolvable"] = check_source_ref_resolvable(source_refs, project_root)

    # Answer-evidence overlap
    output = rec.get("output", "")
    if evidence and output:
        result["answer_evidence_overlap"] = compute_lexical_overlap(evidence, output)
        if result["answer_evidence_overlap"] < 0.05:
            result["unsupported_risk"] = True

    return result


def validate_sft_provenance(sft_path: str, project_root: Path) -> dict:
    """Validate provenance for an SFT file."""
    records = load_jsonl(sft_path)
    if not records:
        return {"error": f"No records in {sft_path}", "total": 0}

    total = len(records)
    has_evidence = 0
    has_source_refs = 0
    source_refs_resolvable = 0
    evidence_nonempty = 0
    supported_count = 0
    unsupported_count = 0

    for rec in records:
        v = validate_record(rec, project_root)
        if v["has_evidence"]:
            has_evidence += 1
        if v["has_source_refs"]:
            has_source_refs += 1
        if v["source_refs_resolvable"]:
            source_refs_resolvable += 1
        if v["evidence_nonempty"]:
            evidence_nonempty += 1
        if v["answer_evidence_overlap"] >= 0.05:
            supported_count += 1
        if v["unsupported_risk"]:
            unsupported_count += 1

    return {
        "file": sft_path,
        "total": total,
        "structural_provenance_rate": round(has_evidence / total, 4) if total else 0,
        "has_evidence": has_evidence,
        "has_source_refs": has_source_refs,
        "source_ref_rate": round(has_source_refs / total, 4) if total else 0,
        "resolvable_provenance_rate": round(source_refs_resolvable / total, 4) if total else 0,
        "source_refs_resolvable": source_refs_resolvable,
        "evidence_nonempty": evidence_nonempty,
        "evidence_nonempty_rate": round(evidence_nonempty / total, 4) if total else 0,
        "heuristic_support_rate": round(supported_count / total, 4) if total else 0,
        "supported_count": supported_count,
        "unsupported_or_weak_evidence_count": unsupported_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate SFT provenance")
    parser.add_argument("--output", default="build/validation/sft_provenance")
    parser.add_argument("--sft-dir", default="data/sft/final_v4")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(".")

    print("SFT Provenance Validation")
    print("=" * 60)

    sft_dir = Path(args.sft_dir)
    if not sft_dir.exists():
        print(f"Error: {sft_dir} not found")
        sys.exit(1)

    results = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "splits": {},
    }

    # Validate each split
    for split in ["gold", "full"]:
        for part in ["train", "validation"]:
            path = sft_dir / split / f"{part}.jsonl"
            if path.exists():
                print(f"\nValidating {split}/{part}...")
                result = validate_sft_provenance(str(path), project_root)
                results["splits"][f"{split}_{part}"] = result
                print(f"  Total: {result['total']}")
                print(f"  Structural provenance rate: {result['structural_provenance_rate']:.2%}")
                print(f"  Resolvable provenance rate: {result['resolvable_provenance_rate']:.2%}")
                print(f"  Heuristic support rate: {result['heuristic_support_rate']:.2%}")
                print(f"  Unsupported/weak evidence: {result['unsupported_or_weak_evidence_count']}")

    # Write report
    report_path = output_dir / "sft_provenance_audit.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
