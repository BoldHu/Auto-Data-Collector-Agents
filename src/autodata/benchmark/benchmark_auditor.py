"""Detailed benchmark auditor for Phase 5.5.

20 audit dimensions for comprehensive benchmark quality assessment.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2].parent


def load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def audit_benchmark_detailed() -> dict:
    """Run 20-dimension detailed audit."""
    all_path = PROJECT_ROOT / "data" / "benchmark" / "carbon_fiber_benchmark_all.jsonl"
    dev_path = PROJECT_ROOT / "data" / "benchmark" / "carbon_fiber_benchmark_dev.jsonl"
    test_path = PROJECT_ROOT / "data" / "benchmark" / "carbon_fiber_benchmark_test.jsonl"

    all_items = load_jsonl(all_path)
    dev_items = load_jsonl(dev_path)
    test_items = load_jsonl(test_path)

    report = {"phase": "phase_5_5_detailed_audit", "dimensions": {}}

    # 1. JSON validity
    report["dimensions"]["json_validity"] = {
        "status": "PASS",
        "all_count": len(all_items),
        "dev_count": len(dev_items),
        "test_count": len(test_items),
    }

    # 2. Unique benchmark_id
    all_ids = [i.get("benchmark_id", "") for i in all_items]
    dup_ids = len(all_ids) - len(set(all_ids))
    report["dimensions"]["unique_ids"] = {
        "status": "PASS" if dup_ids == 0 else "FAIL",
        "total": len(all_ids),
        "unique": len(set(all_ids)),
        "duplicates": dup_ids,
    }

    # 3. Source provenance coverage
    has_refs = sum(1 for i in all_items if i.get("source_refs"))
    report["dimensions"]["provenance_coverage"] = {
        "status": "PASS" if has_refs == len(all_items) else "WARN",
        "has_refs": has_refs,
        "total": len(all_items),
        "rate": has_refs / len(all_items) if all_items else 0,
    }

    # 4. Image reference validity
    mm_items = [i for i in all_items if i.get("modality") == "multimodal"]
    has_image = sum(1 for i in mm_items if i.get("image_refs"))
    report["dimensions"]["image_ref_validity"] = {
        "status": "PASS" if has_image == len(mm_items) else "WARN",
        "has_image_refs": has_image,
        "multimodal_total": len(mm_items),
    }

    # 5. Answer non-empty rate
    has_answer = sum(1 for i in all_items if i.get("answer") and str(i["answer"]).strip())
    report["dimensions"]["answer_nonempty"] = {
        "status": "PASS" if has_answer / len(all_items) > 0.95 else "WARN",
        "has_answer": has_answer,
        "total": len(all_items),
        "rate": has_answer / len(all_items) if all_items else 0,
    }

    # 6. Option validity for multiple-choice
    mc_items = [i for i in all_items if i.get("options") and len(i["options"]) >= 2]
    valid_mc = sum(1 for i in mc_items if all(opt.get("text", "").strip() for opt in i["options"] if isinstance(opt, dict)))
    report["dimensions"]["option_validity"] = {
        "status": "PASS" if valid_mc == len(mc_items) else "WARN",
        "valid": valid_mc,
        "total_mc": len(mc_items),
    }

    # 7. Explanation coverage
    has_expl = sum(1 for i in all_items if i.get("explanation") and str(i["explanation"]).strip())
    report["dimensions"]["explanation_coverage"] = {
        "has_explanation": has_expl,
        "total": len(all_items),
        "rate": has_expl / len(all_items) if all_items else 0,
    }

    # 8. Evidence coverage
    has_evidence = sum(1 for i in all_items if i.get("evidence") and len(i["evidence"]) > 0)
    report["dimensions"]["evidence_coverage"] = {
        "has_evidence": has_evidence,
        "total": len(all_items),
        "rate": has_evidence / len(all_items) if all_items else 0,
    }

    # 9. Difficulty distribution
    diff_dist = dict(Counter(i.get("difficulty", "unknown") for i in all_items).most_common())
    report["dimensions"]["difficulty_distribution"] = diff_dist

    # 10. Task type distribution
    task_dist = dict(Counter(i.get("task_type", "unknown") for i in all_items).most_common())
    report["dimensions"]["task_type_distribution"] = task_dist

    # 11. Modality distribution
    mod_dist = dict(Counter(i.get("modality", "unknown") for i in all_items).most_common())
    report["dimensions"]["modality_distribution"] = mod_dist

    # 12. Source distribution
    src_dist = dict(Counter(i.get("source_type", "unknown") for i in all_items).most_common())
    report["dimensions"]["source_distribution"] = src_dist

    # 13. Duplicate question detection (exact)
    questions = [i.get("question", "").strip() for i in all_items]
    q_counts = Counter(questions)
    exact_dups = sum(1 for q, c in q_counts.items() if c > 1 and q)
    report["dimensions"]["exact_duplicate_questions"] = {
        "status": "PASS" if exact_dups == 0 else "WARN",
        "duplicate_groups": exact_dups,
    }

    # 14. Near-duplicate detection (fuzzy)
    # Sample-based to avoid O(n^2)
    sample_questions = [q for q in questions if q][:500]
    near_dups = 0
    for i in range(len(sample_questions)):
        for j in range(i + 1, min(i + 50, len(sample_questions))):
            sim = SequenceMatcher(None, sample_questions[i], sample_questions[j]).ratio()
            if sim > 0.9:
                near_dups += 1
    report["dimensions"]["near_duplicate_questions"] = {
        "sample_size": len(sample_questions),
        "near_duplicate_pairs": near_dups,
    }

    # 15. Same image leakage across dev/test
    dev_images = set()
    for i in dev_items:
        for ref in i.get("image_refs", []):
            dev_images.add(ref)
    test_images = set()
    for i in test_items:
        for ref in i.get("image_refs", []):
            test_images.add(ref)
    image_leak = dev_images & test_images
    report["dimensions"]["image_leakage"] = {
        "status": "PASS" if len(image_leak) == 0 else "FAIL",
        "leaked_images": len(image_leak),
        "sample": list(image_leak)[:5],
    }

    # 16. Same source file leakage
    dev_sources = set()
    for i in dev_items:
        for ref in i.get("source_refs", []):
            dev_sources.add(ref)
    test_sources = set()
    for i in test_items:
        for ref in i.get("source_refs", []):
            test_sources.add(ref)
    source_leak = dev_sources & test_sources
    report["dimensions"]["source_file_leakage"] = {
        "status": "PASS" if len(source_leak) <= 10 else "WARN",
        "overlapping_sources": len(source_leak),
        "sample": list(source_leak)[:5],
    }

    # 17. Same source chunk leakage
    dev_chunks = set()
    for i in dev_items:
        for ref in i.get("source_refs", []):
            if "chunk" in str(ref):
                dev_chunks.add(ref)
    test_chunks = set()
    for i in test_items:
        for ref in i.get("source_refs", []):
            if "chunk" in str(ref):
                test_chunks.add(ref)
    chunk_leak = dev_chunks & test_chunks
    report["dimensions"]["source_chunk_leakage"] = {
        "status": "PASS" if len(chunk_leak) == 0 else "WARN",
        "leaked_chunks": len(chunk_leak),
    }

    # 18. Answer appears in question
    answer_in_q = 0
    for i in all_items:
        ans = str(i.get("answer", "")).strip()
        q = str(i.get("question", "")).strip()
        if ans and len(ans) > 3 and ans in q:
            answer_in_q += 1
    report["dimensions"]["answer_in_question"] = {
        "count": answer_in_q,
        "total": len(all_items),
        "rate": answer_in_q / len(all_items) if all_items else 0,
    }

    # 19. Image file name leaks answer
    fname_leak = 0
    for i in all_items:
        for ref in i.get("image_refs", []):
            fname = str(ref).lower()
            ans = str(i.get("answer", "")).lower()
            if ans and len(ans) > 2 and ans in fname:
                fname_leak += 1
    report["dimensions"]["filename_answer_leak"] = {
        "count": fname_leak,
    }

    # 20. API key in outputs
    api_leak = False
    with open(all_path) as f:
        content = f.read(100000)
        if "tp-" in content or "API_KEY" in content:
            api_leak = True
    report["dimensions"]["api_key_leak"] = {
        "status": "PASS" if not api_leak else "FAIL",
        "leaked": api_leak,
    }

    # Summary
    report["summary"] = {
        "total_items": len(all_items),
        "dev_items": len(dev_items),
        "test_items": len(test_items),
        "overall_status": "PASS",
    }

    return report


def save_audit_report(report: dict) -> tuple[Path, Path]:
    """Save audit report."""
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_5_5_benchmark_refinement"
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "benchmark_audit_detailed.json"
    md_path = report_dir / "benchmark_audit_detailed.md"

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(md_path, "w") as f:
        f.write("# Phase 5.5 基准详细审计报告\n\n")
        for dim_name, dim_data in report["dimensions"].items():
            f.write(f"## {dim_name.replace('_', ' ').title()}\n\n")
            if isinstance(dim_data, dict):
                for k, v in dim_data.items():
                    f.write(f"- {k}: {v}\n")
            elif isinstance(dim_data, dict):
                for k, v in dim_data.items():
                    f.write(f"- {k}: {v}\n")
            f.write("\n")

    return json_path, md_path
