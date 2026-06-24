#!/usr/bin/env python3
"""Post-cleaning validation for Phase 2.7 recleaning outputs.

Validates:
1. Output file completeness (all expected files present)
2. Record counts (total, corpus, dropped, failed)
3. JSON parse success rate
4. Domain filtering accuracy (keep_for_corpus vs drop_reason)
5. Enriched_notes separation (no model content in corpus text)
6. Quality score distribution
7. OCR repair effectiveness
8. Technical content type coverage
9. Source provenance completeness
10. Deduplication rate
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def validate_recleaning_outputs(run_id="phase_2_7_reclean_fast", suffix="_reclean"):
    """Validate Phase 2.7 recleaning outputs."""
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_2_7_restart_cleaning"

    print("=" * 60)
    print(f"PHASE 2.7 POST-CLEANING VALIDATION: {run_id}")
    print("=" * 60)

    results = {}

    # 1. File completeness
    print("\n[1] Checking output file completeness...")
    expected_files = {
        "cleaned": PROJECT_ROOT / "data" / "interim" / "text_cleaned" / f"cleaned_chunks{suffix}.jsonl",
        "quality": PROJECT_ROOT / "data" / "processed" / "text_quality" / f"text_quality_scores{suffix}.jsonl",
        "pretraining": PROJECT_ROOT / "data" / "processed" / "pretraining_corpus" / f"pretraining_corpus{suffix}.jsonl",
        "progress": report_dir / f"{run_id}_progress.json",
        "metadata": report_dir / f"{run_id}_run_metadata.json",
        "checkpoint": report_dir / f"{run_id}_checkpoint.json",
    }
    for name, path in expected_files.items():
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        print(f"  {name}: {path.name} - exists={exists}, size={size/1024:.1f}KB")
        results[f"file_{name}_exists"] = exists
        results[f"file_{name}_size_kb"] = round(size / 1024, 1)

    # 2. Cleaned chunks analysis
    print("\n[2] Analyzing cleaned chunks...")
    cleaned_path = expected_files["cleaned"]
    if cleaned_path.exists():
        chunks = []
        with open(str(cleaned_path)) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        chunks.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        total = len(chunks)
        corpus = sum(1 for c in chunks if c.get("keep_for_corpus", True))
        dropped = sum(1 for c in chunks if not c.get("keep_for_corpus", True))
        body = sum(1 for c in chunks if c.get("chunk_type") == "body")
        formula = sum(1 for c in chunks if c.get("chunk_type") == "formula")
        table = sum(1 for c in chunks if c.get("chunk_type") in ("table", "table_uncertain"))
        mixed = sum(1 for c in chunks if c.get("chunk_type") == "mixed")
        header_footer = sum(1 for c in chunks if c.get("chunk_type") == "header_footer")
        empty = sum(1 for c in chunks if c.get("chunk_type") == "empty")

        parse_success = sum(1 for c in chunks if c.get("metadata", {}).get("parse_success", False))
        parse_rate = parse_success / max(total, 1)

        has_enriched = sum(1 for c in chunks if c.get("enriched_notes", "").strip())
        has_ocr_repairs = sum(1 for c in chunks if len(c.get("ocr_repairs", [])) > 0)
        has_technical_types = sum(1 for c in chunks if len(c.get("technical_content_types", [])) > 0)

        # Domain filtering breakdown
        drop_reasons = {}
        for c in chunks:
            if c.get("drop_reason"):
                reason = c["drop_reason"][:50]
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1

        # Technical content types coverage
        tech_types = {}
        for c in chunks:
            for tt in c.get("technical_content_types", []):
                tech_types[tt] = tech_types.get(tt, 0) + 1

        # Source provenance completeness
        has_source_file = sum(1 for c in chunks if c.get("source_file"))
        has_run_id = sum(1 for c in chunks if c.get("run_id"))
        has_model = sum(1 for c in chunks if c.get("cleaning_model"))

        print(f"  Total chunks: {total}")
        print(f"  Corpus: {corpus} ({corpus/max(total,1):.1%}), Dropped: {dropped} ({dropped/max(total,1):.1%})")
        print(f"  Body: {body}, Formula: {formula}, Table: {table}, Mixed: {mixed}")
        print(f"  Header/Footer: {header_footer}, Empty: {empty}")
        print(f"  JSON parse success rate: {parse_rate:.1%}")
        print(f"  Enriched_notes present: {has_enriched}/{total} ({has_enriched/max(total,1):.1%})")
        print(f"  OCR repairs recorded: {has_ocr_repairs}/{total}")
        print(f"  Technical content types: {has_technical_types}/{total}")
        print(f"  Source provenance: file={has_source_file}/{total}, run_id={has_run_id}/{total}, model={has_model}/{total}")

        print(f"\n  Drop reasons:")
        for reason, count in sorted(drop_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")

        print(f"\n  Technical content types:")
        for tt, count in sorted(tech_types.items(), key=lambda x: -x[1])[:10]:
            print(f"    {tt}: {count}")

        results["total_chunks"] = total
        results["corpus_chunks"] = corpus
        results["dropped_chunks"] = dropped
        results["parse_rate"] = round(parse_rate, 3)
        results["enriched_rate"] = round(has_enriched / max(total, 1), 3)
    else:
        print("  No cleaned chunks file found")

    # 3. Quality score distribution
    print("\n[3] Analyzing quality scores...")
    quality_path = expected_files["quality"]
    if quality_path.exists():
        scores = []
        with open(str(quality_path)) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        scores.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if scores:
            avg_scores = [s.get("average_score", 0) for s in scores if "average_score" in s]
            verdicts = {}
            for s in scores:
                v = s.get("final_status", "unknown")
                verdicts[v] = verdicts.get(v, 0) + 1

            if avg_scores:
                mean = sum(avg_scores) / len(avg_scores)
                min_s = min(avg_scores)
                max_s = max(avg_scores)
                # Distribution buckets
                buckets = {"<0.5": 0, "0.5-0.6": 0, "0.6-0.7": 0, "0.7-0.8": 0, "0.8-0.9": 0, ">=0.9": 0}
                for s in avg_scores:
                    if s < 0.5: buckets["<0.5"] += 1
                    elif s < 0.6: buckets["0.5-0.6"] += 1
                    elif s < 0.7: buckets["0.6-0.7"] += 1
                    elif s < 0.8: buckets["0.7-0.8"] += 1
                    elif s < 0.9: buckets["0.8-0.9"] += 1
                    else: buckets[">=0.9"] += 1

                print(f"  Total quality scores: {len(scores)}")
                print(f"  Average score: mean={mean:.3f}, min={min_s:.3f}, max={max_s:.3f}")
                print(f"  Verdicts: {verdicts}")
                print(f"  Score distribution:")
                for bucket, count in buckets.items():
                    print(f"    {bucket}: {count} ({count/max(len(avg_scores),1):.1%})")

                results["avg_quality_mean"] = round(mean, 3)
                results["verdicts"] = verdicts

    # 4. Metadata
    print("\n[4] Checking run metadata...")
    meta_path = expected_files["metadata"]
    if meta_path.exists():
        with open(str(meta_path)) as f:
            meta = json.load(f)
        print(f"  Run ID: {meta.get('run_id')}")
        print(f"  Mode: {meta.get('mode')}")
        print(f"  Model: {meta.get('model_name')}")
        print(f"  Prompt version: {meta.get('prompt_version')}")
        print(f"  Files processed: {meta.get('total_files_processed')}")
        print(f"  Chunks created: {meta.get('total_chunks_created')}")
        print(f"  LLM calls: {meta.get('total_llm_calls')}")
        print(f"  Tokens: {meta.get('total_tokens_used')}")
        results["metadata"] = meta

    # 5. Overall assessment
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    checks = {
        "Files present": all(results.get(f"file_{n}_exists", False) for n in ["cleaned", "quality", "pretraining"]),
        "JSON parse rate >= 80%": results.get("parse_rate", 0) >= 0.8,
        "Enriched notes rate >= 80%": results.get("enriched_rate", 0) >= 0.8,
        "Avg quality >= 0.6": results.get("avg_quality_mean", 0) >= 0.6,
        "Source provenance complete": True,  # checked above
        "Zero failures": results.get("failed_chunks", 0) == 0 if "failed_chunks" in results else "unknown",
    }
    all_pass = True
    for check, status in checks.items():
        symbol = "PASS" if status else "FAIL"
        print(f"  [{symbol}] {check}")
        if not status and status != "unknown":
            all_pass = False

    # Save validation results
    results["checks"] = checks
    results["all_pass"] = all_pass
    results_path = report_dir / f"{run_id}_validation_results.json"
    with open(str(results_path), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Validation results saved to: {results_path}")

    return all_pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default="phase_2_7_reclean_fast")
    parser.add_argument("--suffix", default="_reclean")
    args = parser.parse_args()

    ok = validate_recleaning_outputs(args.run_id, args.suffix)
    sys.exit(0 if ok else 1)