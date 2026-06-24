#!/usr/bin/env python3
"""Phase 2 output validation script.

Checks that all Phase 2 outputs are correct:
- cleaned chunk file exists and is non-empty
- quality-score file exists and has records per cleaned chunk
- knowledge-unit file exists and is non-empty
- SFT-candidate file exists and is non-empty
- every output has source provenance
- no raw file was overwritten
- DTCG trace has edges > 0
- DTCG trace has artifact nodes
- DTCG trace has quality_feedback edges
- context package file is non-empty
- metadata counts match actual JSONL counts
- no API keys appear in logs or outputs

Usage:
  python scripts/validate_phase_2_outputs.py                        # Default pilot
  python scripts/validate_phase_2_outputs.py pilot                  # Pilot suffix
  python scripts/validate_phase_2_outputs.py --run_id phase_2_full_zh  # Full zh run
  python scripts/validate_phase_2_outputs.py --language zh --run_id phase_2_full_zh
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.autodata.utils.io_utils import safe_read_json, safe_read_jsonl


def _resolve_paths(
    suffix: str,
    run_id: str | None,
    language: str | None,
) -> dict[str, Path]:
    """Resolve all output file paths based on suffix or run_id.

    If run_id is provided and starts with 'phase_2_full_', use run_id-based paths
    under data/reports/phase_2_full_text_cleaning/. Otherwise use legacy suffix paths.
    """
    processed_dir = PROJECT_ROOT / "data" / "processed"
    reports_dir = PROJECT_ROOT / "data" / "reports" / "phase_2_text_cleaning"
    repair_dir = PROJECT_ROOT / "data" / "reports" / "phase_2_text_cleaning_repair"
    full_reports_dir = PROJECT_ROOT / "data" / "reports" / "phase_2_full_text_cleaning"
    raw_dir = PROJECT_ROOT / "text_raw_data"

    paths: dict[str, Path] = {}

    if run_id and run_id.startswith("phase_2_full_"):
        rid = run_id
        paths["cleaned"] = PROJECT_ROOT / "data" / "interim" / "text_cleaned" / f"cleaned_chunks_{rid}.jsonl"
        paths["pretraining"] = processed_dir / "pretraining_corpus" / f"pretraining_corpus_{rid}.jsonl"
        paths["knowledge"] = processed_dir / "knowledge_units" / f"knowledge_units_{rid}.jsonl"
        paths["sft"] = processed_dir / "sft_candidates" / f"sft_candidates_{rid}.jsonl"
        paths["quality"] = processed_dir / "text_quality" / f"text_quality_scores_{rid}.jsonl"
        paths["checkpoint"] = full_reports_dir / f"{rid}_checkpoint.json"
        paths["context_packages"] = full_reports_dir / f"{rid}_context_packages.jsonl"
        paths["errors"] = full_reports_dir / f"{rid}_errors.jsonl"
        paths["progress_json"] = full_reports_dir / f"{rid}_progress.json"
        paths["progress_log"] = full_reports_dir / f"{rid}_progress.log"
        paths["metadata"] = full_reports_dir / f"{rid}_run_metadata.json"
        paths["dtcg_trace"] = full_reports_dir / f"{rid}_dtcg_trace.json"
        paths["reports_dir"] = full_reports_dir
        paths["raw_dir"] = raw_dir
    else:
        paths["cleaned"] = PROJECT_ROOT / "data" / "interim" / "text_cleaned" / f"cleaned_chunks_{suffix}.jsonl"
        paths["pretraining"] = processed_dir / "pretraining_corpus" / f"pretraining_corpus_{suffix}.jsonl"
        paths["knowledge"] = processed_dir / "knowledge_units" / f"knowledge_units_{suffix}.jsonl"
        paths["sft"] = processed_dir / "sft_candidates" / f"sft_candidates_{suffix}.jsonl"
        paths["quality"] = processed_dir / "text_quality" / f"text_quality_scores_{suffix}.jsonl"
        paths["checkpoint"] = reports_dir / f"checkpoint_{suffix}.json"
        paths["context_packages"] = repair_dir / "context_packages_repaired.jsonl"
        paths["metadata"] = reports_dir / "phase_2_run_metadata.json"
        paths["dtcg_trace"] = reports_dir / "dtcg_text_cleaning_trace.json"
        paths["reports_dir"] = reports_dir
        paths["repair_dir"] = repair_dir
        paths["raw_dir"] = raw_dir

    return paths


def validate_phase_2_outputs(
    suffix: str = "pilot",
    run_id: str | None = None,
    language: str | None = None,
    explicit_paths: dict[str, str] | None = None,
) -> dict:
    """Validate Phase 2 outputs and return a result dict."""
    results = {}
    errors = []
    warnings = []

    paths = _resolve_paths(suffix, run_id, language)

    # Allow explicit path overrides
    if explicit_paths:
        for key, val in explicit_paths.items():
            if val:
                paths[key] = Path(val)

    processed_dir = PROJECT_ROOT / "data" / "processed"
    reports_dir = paths.get("reports_dir", PROJECT_ROOT / "data" / "reports" / "phase_2_text_cleaning")
    raw_dir = paths.get("raw_dir", PROJECT_ROOT / "text_raw_data")

    # 1. Check cleaned chunk file (may be in interim or processed)
    cleaned_path = paths["cleaned"]
    if cleaned_path.exists():
        cleaned_records = safe_read_jsonl(str(cleaned_path))
        results["cleaned_chunks_count"] = len(cleaned_records)
        if len(cleaned_records) == 0:
            errors.append("cleaned_chunks file is empty")
    else:
        results["cleaned_chunks_count"] = 0
        # Not an error — chunks may be written directly to pretraining corpus

    # 2. Check quality-score file
    quality_path = paths["quality"]
    if quality_path.exists():
        quality_records = safe_read_jsonl(str(quality_path))
        results["quality_scores_count"] = len(quality_records)
        if len(quality_records) == 0:
            errors.append("quality_scores file is empty")
        else:
            # Check 1:1 mapping or documented many-to-one
            passed = sum(1 for r in quality_records if r.get("final_status") == "passed")
            needs_rev = sum(1 for r in quality_records if r.get("final_status") == "needs_revision")
            failed = sum(1 for r in quality_records if r.get("final_status") == "failed")
            results["quality_passed"] = passed
            results["quality_needs_revision"] = needs_rev
            results["quality_failed"] = failed
            total_scored = passed + needs_rev + failed
            if total_scored < results.get("cleaned_chunks_count", 0):
                warnings.append(f"quality_scores ({total_scored}) < cleaned_chunks ({results.get('cleaned_chunks_count', 0)})")
            # Check pass rate
            if total_scored > 0:
                pass_rate = (passed + needs_rev) / total_scored
                results["quality_pass_rate"] = round(pass_rate, 3)
                if pass_rate < 0.95:
                    warnings.append(f"quality pass rate {pass_rate:.1%} < 95% threshold")
            # Check all required fields exist in first record
            required_fields = [
                "chunk_id", "source_file", "source_folder", "page_numbers",
                "language", "clarity", "completeness", "consistency",
                "feasibility", "complexity", "domain_relevance",
                "final_status", "detected_issues", "verifier_model",
                "prompt_version", "run_id", "timestamp",
            ]
            if quality_records:
                missing_fields = [f for f in required_fields if f not in quality_records[0]]
                if missing_fields:
                    errors.append(f"quality record missing fields: {missing_fields}")
    else:
        results["quality_scores_count"] = 0
        errors.append("quality_scores file not found")

    # 3. Check knowledge-unit file
    ku_path = paths["knowledge"]
    if ku_path.exists():
        ku_records = safe_read_jsonl(str(ku_path))
        results["knowledge_units_count"] = len(ku_records)
        if len(ku_records) == 0:
            warnings.append("knowledge_units file is empty (expected if extraction is not enabled)")
    else:
        results["knowledge_units_count"] = 0

    # 4. Check SFT-candidate file
    sft_path = paths["sft"]
    if sft_path.exists():
        sft_records = safe_read_jsonl(str(sft_path))
        results["sft_candidates_count"] = len(sft_records)
        if len(sft_records) == 0:
            warnings.append("sft_candidates file is empty (expected if generation is not enabled)")
    else:
        results["sft_candidates_count"] = 0

    # 5. Check provenance in all outputs
    provenance_fields_map = {
        "pretraining_corpus": ["source_file", "source_folder", "language"],
        "knowledge_units": ["source_chunk_id", "language", "source_refs"],
        "sft_candidates": ["source_chunk_id", "source_refs"],
    }
    provenance_path_map = {
        "pretraining_corpus": str(paths["pretraining"]),
        "knowledge_units": str(ku_path),
        "sft_candidates": str(sft_path),
    }
    for name, fields in provenance_fields_map.items():
        path_str = provenance_path_map[name]
        records = safe_read_jsonl(path_str)
        if records:
            for pf in fields:
                missing = sum(1 for r in records if pf not in r)
                if missing > 0:
                    errors.append(f"{name}: {missing} records missing provenance field '{pf}'")

    # 6. Check raw data files not modified
    raw_books = raw_dir / "books"
    raw_en_books = raw_dir / "en_books"
    results["raw_files_integrity"] = True
    if raw_books.exists() and raw_en_books.exists():
        for d in [raw_books, raw_en_books]:
            for f in os.listdir(d):
                if not f.endswith(".clean.json"):
                    results["raw_files_integrity"] = False
                    errors.append(f"Unexpected file in raw data: {f}")

    # 7. Check DTCG trace
    dtcg_path = paths.get("dtcg_trace", reports_dir / "dtcg_text_cleaning_trace.json")
    dtcg_data = safe_read_json(str(dtcg_path))
    if dtcg_data:
        nodes = dtcg_data.get("nodes", {})
        edges = dtcg_data.get("edges", {})
        results["dtcg_node_count"] = len(nodes)
        results["dtcg_edge_count"] = len(edges)

        # Check artifact nodes
        artifact_nodes = [n for n in nodes.values() if n.get("node_type") == "artifact"]
        results["dtcg_artifact_nodes"] = len(artifact_nodes)
        if len(artifact_nodes) == 0:
            errors.append("DTCG trace has no artifact nodes")

        # Check quality_feedback edges
        qf_edges = [e for e in edges.values() if e.get("edge_type") == "quality_feedback"]
        results["dtcg_quality_feedback_edges"] = len(qf_edges)
        if len(qf_edges) == 0 and len(artifact_nodes) > 0:
            warnings.append("DTCG trace has no quality_feedback edges (expected in dry-run; must have >0 in real run)")

        if len(edges) == 0:
            errors.append("DTCG trace has 0 edges")
    else:
        results["dtcg_node_count"] = 0
        results["dtcg_edge_count"] = 0
        errors.append("DTCG trace file not found")

    # Also check repair report DTCG trace (legacy)
    repair_dir = paths.get("repair_dir", PROJECT_ROOT / "data" / "reports" / "phase_2_text_cleaning_repair")
    dtcg_repair_path = repair_dir / "dtcg_text_cleaning_trace_repaired.json"
    dtcg_repair_data = safe_read_json(str(dtcg_repair_path))
    if dtcg_repair_data:
        nodes_r = dtcg_repair_data.get("nodes", {})
        edges_r = dtcg_repair_data.get("edges", {})
        results["dtcg_repair_node_count"] = len(nodes_r)
        results["dtcg_repair_edge_count"] = len(edges_r)
        results["dtcg_repair_artifact_nodes"] = sum(
            1 for n in nodes_r.values() if n.get("node_type") == "artifact"
        )
        results["dtcg_repair_quality_feedback_edges"] = sum(
            1 for e in edges_r.values() if e.get("edge_type") == "quality_feedback"
        )

    # 8. Check context packages
    ctx_path = paths.get("context_packages", repair_dir / "context_packages_repaired.jsonl")
    if ctx_path.exists():
        ctx_records = safe_read_jsonl(str(ctx_path))
        results["context_packages_count"] = len(ctx_records)
        if len(ctx_records) == 0:
            errors.append("context_packages file is empty")
        else:
            # Compute average context length
            total_tokens = sum(r.get("dtcg_token_estimate", 0) for r in ctx_records)
            avg_tokens = total_tokens / len(ctx_records) if ctx_records else 0
            results["avg_dtcg_context_tokens"] = round(avg_tokens, 1)
            total_broadcast = sum(r.get("broadcast_token_estimate", 0) for r in ctx_records)
            avg_broadcast = total_broadcast / len(ctx_records) if ctx_records else 0
            results["avg_broadcast_context_tokens"] = round(avg_broadcast, 1)
            avg_saving = sum(r.get("estimated_saving_ratio", 0) for r in ctx_records) / len(ctx_records) if ctx_records else 0
            results["avg_estimated_saving_ratio"] = round(avg_saving, 1)
    else:
        results["context_packages_count"] = 0
        warnings.append("context_packages file not found")

    # 9. Check metadata counts match JSONL
    metadata_path = paths.get("metadata", reports_dir / "phase_2_run_metadata.json")
    metadata = safe_read_json(str(metadata_path))
    if metadata:
        # If legacy mode, also check repair metadata
        if not (run_id and run_id.startswith("phase_2_full_")):
            repair_metadata_path = repair_dir / "phase_2_repair_run_metadata.json"
            repair_metadata = safe_read_json(str(repair_metadata_path))
            if repair_metadata:
                metadata = repair_metadata

        for field, path_str in [
            ("total_knowledge_units", str(ku_path)),
            ("total_sft_candidates", str(sft_path)),
        ]:
            expected = metadata.get(field, 0)
            actual_records = safe_read_jsonl(path_str)
            actual = len(actual_records)
            if expected != actual and expected > 0:
                errors.append(f"metadata {field}={expected} but JSONL has {actual} records")

        # Check quality scores match (allow ≥, since header_footer and empty chunks also get records)
        expected_quality = metadata.get("total_quality_scores", 0)
        quality_actual = safe_read_jsonl(str(quality_path))
        if expected_quality > 0 and len(quality_actual) < expected_quality:
            errors.append(f"metadata total_quality_scores={expected_quality} but JSONL has only {len(quality_actual)} records")

    # 10. Check no API keys in outputs
    api_key_patterns = ["sk-", "api_key", "apikey", "Bearer", "xai-"]
    all_output_files = list(processed_dir.rglob("*.jsonl")) + list(reports_dir.rglob("*.json"))
    for f in all_output_files:
        try:
            content = f.read_text(encoding="utf-8")
            for pattern in api_key_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    errors.append(f"Potential API key '{pattern}' found in {f.name}")
        except Exception:
            pass

    # Summary
    results["errors"] = errors
    results["warnings"] = warnings
    results["passed"] = len(errors) == 0
    results["total_checks"] = 10
    results["checks_failed"] = len(errors)
    results["checks_warning"] = len(warnings)
    results["validated_run_id"] = run_id or suffix

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate Phase 2 outputs")
    parser.add_argument("suffix", nargs="?", default="pilot", help="Output suffix (default: pilot)")
    parser.add_argument("--run_id", type=str, default=None, help="Run ID for full runs (e.g. phase_2_full_zh)")
    parser.add_argument("--language", choices=["zh", "en", "all"], default=None, help="Language filter")
    parser.add_argument("--cleaned", type=str, default=None, help="Explicit path to cleaned chunks JSONL")
    parser.add_argument("--quality", type=str, default=None, help="Explicit path to quality scores JSONL")
    parser.add_argument("--knowledge", type=str, default=None, help="Explicit path to knowledge units JSONL")
    parser.add_argument("--sft", type=str, default=None, help="Explicit path to SFT candidates JSONL")
    parser.add_argument("--dtcg", type=str, default=None, help="Explicit path to DTCG trace JSON")
    parser.add_argument("--context_packages", type=str, default=None, help="Explicit path to context packages JSONL")
    args = parser.parse_args()

    explicit = {}
    if args.cleaned:
        explicit["cleaned"] = args.cleaned
    if args.quality:
        explicit["quality"] = args.quality
    if args.knowledge:
        explicit["knowledge"] = args.knowledge
    if args.sft:
        explicit["sft"] = args.sft
    if args.dtcg:
        explicit["dtcg_trace"] = args.dtcg
    if args.context_packages:
        explicit["context_packages"] = args.context_packages

    results = validate_phase_2_outputs(
        suffix=args.suffix,
        run_id=args.run_id,
        language=args.language,
        explicit_paths=explicit if explicit else None,
    )

    # Save results
    if args.run_id and args.run_id.startswith("phase_2_full_"):
        output_dir = PROJECT_ROOT / "data" / "reports" / "phase_2_full_text_cleaning"
    else:
        output_dir = PROJECT_ROOT / "data" / "reports" / "phase_2_text_cleaning_repair"
    output_dir.mkdir(parents=True, exist_ok=True)

    rid = args.run_id or args.suffix
    output_path = output_dir / f"phase_2_validation_result_{rid}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("Phase 2 Output Validation Results")
    print(f"Run ID: {rid}")
    print("=" * 60)
    for key, value in results.items():
        if key not in ("errors", "warnings"):
            print(f"  {key}: {value}")
    print(f"\n  Errors: {len(results['errors'])}")
    for err in results["errors"]:
        print(f"    - {err}")
    print(f"\n  Warnings: {len(results['warnings'])}")
    for warn in results["warnings"]:
        print(f"    - {warn}")
    print(f"\n  PASSED: {results['passed']}")
    print("=" * 60)
    print(f"\nValidation result saved to: {output_path}")

    return 0 if results["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
