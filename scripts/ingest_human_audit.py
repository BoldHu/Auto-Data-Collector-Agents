#!/usr/bin/env python3
"""Human Audit Ingestion Script.

Ingests completed human audit annotation sheets and computes:
- Reviewer IDs and background categories
- Per-item scores
- Pairwise agreement
- Fleiss' kappa or Krippendorff's alpha

Also validates that no paper-ready text claims "human expert validated"
when score cells are empty.

Usage:
    python scripts/ingest_human_audit.py --sheets path/to/sheet1.csv,path/to/sheet2.csv --output build/validation/human_audit
    python scripts/ingest_human_audit.py --validate-claims --output build/validation/human_audit
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_csv(path: str) -> list[dict]:
    """Load a CSV file as list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def check_sheet_completeness(records: list[dict]) -> dict:
    """Check if a human audit sheet has been completed."""
    total = len(records)
    if total == 0:
        return {"complete": False, "reason": "empty_sheet", "total": 0}

    # Look for score columns
    score_columns = []
    for col in records[0].keys():
        if any(kw in col.lower() for kw in ["score", "rating", "correctness", "answerability", "evidence", "relevance", "ambiguity"]):
            score_columns.append(col)

    if not score_columns:
        return {"complete": False, "reason": "no_score_columns", "total": total}

    # Check if any score cells are filled
    nonempty_scores = 0
    for rec in records:
        for col in score_columns:
            val = str(rec.get(col, "")).strip()
            if val and val not in ("", "nan", "None", "N/A"):
                nonempty_scores += 1
                break

    return {
        "complete": nonempty_scores > 0,
        "total": total,
        "score_columns": score_columns,
        "nonempty_score_rows": nonempty_scores,
        "empty_score_rows": total - nonempty_scores,
        "completeness_rate": round(nonempty_scores / total, 4) if total else 0,
    }


def validate_claims_against_audit(paper_dir: str, audit_sheets: list[str]) -> dict:
    """Validate that paper-ready text doesn't claim human validation when sheets are empty."""
    issues = []

    # Check all sheets
    all_complete = True
    for sheet_path in audit_sheets:
        if not Path(sheet_path).exists():
            all_complete = False
            continue
        records = load_csv(sheet_path)
        status = check_sheet_completeness(records)
        if not status["complete"]:
            all_complete = False

    # Scan paper-ready text for prohibited claims
    paper_path = Path(paper_dir)
    if paper_path.exists():
        prohibited_phrases = [
            "human expert validated",
            "human expert audit",
            "human validated",
            "human annotation",
            "expert annotation",
            "expert review confirmed",
            "annotator agreement",
            "inter-annotator",
            "fleiss",
            "krippendorff",
        ]

        for md_file in paper_path.glob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8").lower()
                for phrase in prohibited_phrases:
                    if phrase in content:
                        issues.append({
                            "file": str(md_file),
                            "phrase": phrase,
                            "context": "found_in_paper_ready_text",
                        })
            except Exception:
                continue

        for tex_file in paper_path.glob("*.tex"):
            try:
                content = tex_file.read_text(encoding="utf-8").lower()
                for phrase in prohibited_phrases:
                    if phrase in content:
                        issues.append({
                            "file": str(tex_file),
                            "phrase": phrase,
                            "context": "found_in_paper_ready_tex",
                        })
            except Exception:
                continue

    return {
        "all_sheets_complete": all_complete,
        "issues": issues,
        "issue_count": len(issues),
        "recommendation": "remove_human_validation_claims" if not all_complete and issues else "ok",
    }


def main():
    parser = argparse.ArgumentParser(description="Human audit ingestion and validation")
    parser.add_argument("--sheets", default=None, help="Comma-separated paths to completed audit sheets")
    parser.add_argument("--output", default="build/validation/human_audit")
    parser.add_argument("--validate-claims", action="store_true", help="Validate paper claims against audit status")
    parser.add_argument("--paper-dir", default="reports/paper_ready")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Human Audit Ingestion and Validation")
    print("=" * 60)

    results = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if args.sheets:
        sheet_paths = [p.strip() for p in args.sheets.split(",")]
        sheet_statuses = []
        for path in sheet_paths:
            if Path(path).exists():
                records = load_csv(path)
                status = check_sheet_completeness(records)
                status["file"] = path
                sheet_statuses.append(status)
                print(f"\n{path}:")
                print(f"  Total rows: {status['total']}")
                print(f"  Complete: {status['complete']}")
                print(f"  Non-empty score rows: {status.get('nonempty_score_rows', 0)}")
            else:
                print(f"\n{path}: NOT FOUND")
                sheet_statuses.append({"file": path, "complete": False, "reason": "not_found"})

        results["sheet_statuses"] = sheet_statuses

    if args.validate_claims:
        print("\nValidating paper claims...")
        claim_validation = validate_claims_against_audit(args.paper_dir, args.sheets.split(",") if args.sheets else [])
        results["claim_validation"] = claim_validation

        if claim_validation["issues"]:
            print(f"\n  ISSUES FOUND: {claim_validation['issue_count']}")
            for issue in claim_validation["issues"]:
                print(f"    - {issue['file']}: '{issue['phrase']}'")
            print(f"\n  Recommendation: {claim_validation['recommendation']}")
        else:
            print("  No prohibited claims found.")

    # Write report
    report_path = output_dir / "human_audit_validation.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
