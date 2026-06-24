"""Run Phase 5.5 text task enhancement.

Generates additional text-only benchmark items from pretraining corpus.
Uses API_KEY1 only.

Usage:
    python scripts/run_phase_5_5_text_task_enhancement.py \
        --target_candidates 1500 \
        --target_passed 800 \
        --max_workers 32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.autodata.benchmark.text_task_enhancer import generate_text_items, validate_text_item
from src.autodata.utils.logging_utils import get_logger

logger = get_logger("text_enhance")


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_candidates", type=int, default=1500)
    parser.add_argument("--target_passed", type=int, default=800)
    parser.add_argument("--max_workers", type=int, default=32)
    args = parser.parse_args()

    from src.autodata.utils.model_pool import get_model_pool
    pool = get_model_pool(use_key2=False)

    # Load source texts
    corpus_path = PROJECT_ROOT / "data" / "processed" / "pretraining_corpus" / "pretraining_corpus_reclean.jsonl"
    ku_path = PROJECT_ROOT / "data" / "processed" / "knowledge_units" / "knowledge_units_pilot.jsonl"

    corpus = load_jsonl(corpus_path)
    ku = load_jsonl(ku_path)

    logger.info(f"Loaded {len(corpus)} corpus chunks, {len(ku)} knowledge units")

    # Filter high-quality corpus chunks
    # Use chunks with domain_relevance > 0.7 if available
    source_chunks = []
    for chunk in corpus[:500]:  # Limit to first 500 for efficiency
        text = chunk.get("text", "")
        if len(text) > 100:  # Skip very short chunks
            source_chunks.append({
                "text": text,
                "source_file": chunk.get("source_file", "unknown"),
            })

    # Add knowledge units
    for k in ku:
        claim = k.get("claim", "")
        evidence = k.get("evidence_text", "")
        if claim and evidence:
            source_chunks.append({
                "text": f"{claim}\n\n证据：{evidence}",
                "source_file": f"knowledge_unit_{k.get('unit_id', '')}",
            })

    logger.info(f"Using {len(source_chunks)} source chunks for generation")

    # Generate candidates
    output_dir = PROJECT_ROOT / "data" / "benchmark_candidates" / "text_enhanced"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / "text_enhanced_candidates.jsonl"
    validated_path = output_dir / "text_enhanced_candidates_validated.jsonl"

    all_candidates = []
    all_validated = []
    start_time = time.time()

    # Generation phase
    logger.info("Starting text benchmark generation...")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        for chunk in source_chunks:
            future = executor.submit(
                generate_text_items,
                pool,
                chunk["text"],
                chunk["source_file"],
                3,  # 3 items per chunk
            )
            futures[future] = chunk

        for future in as_completed(futures):
            try:
                items = future.result()
                for item in items:
                    item["source_file"] = futures[future]["source_file"]
                    item["source_text"] = futures[future]["text"][:500]
                    item["benchmark_id"] = f"text_{hashlib.md5(json.dumps(item, sort_keys=True).encode()).hexdigest()[:16]}"
                    item["source_type"] = "text"
                    item["modality"] = "text"
                    all_candidates.append(item)
            except Exception:
                continue

            if len(all_candidates) >= args.target_candidates:
                break

    logger.info(f"Generated {len(all_candidates)} candidates")

    # Save candidates
    with open(candidates_path, "w") as f:
        for c in all_candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Validation phase
    logger.info("Starting validation...")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        for item in all_candidates:
            future = executor.submit(validate_text_item, pool, item)
            futures[future] = item

        for future in as_completed(futures):
            item = futures[future]
            try:
                score = future.result()
                if score.get("quality_status") == "keep":
                    item["quality_scores"] = score
                    item["validation_status"] = "passed"
                    all_validated.append(item)
            except Exception:
                continue

            if len(all_validated) >= args.target_passed:
                break

    logger.info(f"Validated {len(all_validated)} items")

    # Save validated
    with open(validated_path, "w") as f:
        for v in all_validated:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    report = {
        "total_candidates": len(all_candidates),
        "total_validated": len(all_validated),
        "elapsed_seconds": elapsed,
        "source_chunks_used": len(source_chunks),
    }

    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_5_5_benchmark_refinement"
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "text_task_enhancement_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== Text Task Enhancement Complete ===")
    print(f"Candidates: {len(all_candidates)}")
    print(f"Validated: {len(all_validated)}")
    print(f"Elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
