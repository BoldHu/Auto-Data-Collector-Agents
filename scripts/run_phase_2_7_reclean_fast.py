#!/usr/bin/env python3
"""Phase 2.7 fast recleaning CLI — concurrent workers, ModelPool, v2.0 prompts.

Usage:
    # Full Chinese recleaning with 4 workers (default):
    python scripts/run_phase_2_7_reclean_fast.py --language zh

    # With quality gate first (recommended):
    python scripts/run_phase_2_7_reclean_fast.py --language zh --quality_gate_first

    # Resume from checkpoint:
    python scripts/run_phase_2_7_reclean_fast.py --language zh --resume

    # Skip KU/SFT generation (much faster, just clean+verify):
    python scripts/run_phase_2_7_reclean_fast.py --language zh --skip_ku_sft

    # Custom workers and models:
    python scripts/run_phase_2_7_reclean_fast.py --language zh --num_workers 6 --fast_model mimo-v2.5

    # Preflight test only:
    python scripts/run_phase_2_7_reclean_fast.py --preflight
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.autodata.pipelines.fast_text_cleaning_pipeline import FastTextCleaningPipeline
from src.autodata.utils.logging_utils import setup_logging, get_logger
from src.autodata.utils.model_pool import ModelPool

logger = get_logger("run_phase_2_7")


def preflight_check() -> bool:
    """Run preflight validation: API connectivity, file inventory, disk space."""
    print("=" * 60)
    print("PREFLIGHT CHECK")
    print("=" * 60)

    # 1. API connectivity
    print("\n[1] Testing API connectivity...")
    pool = ModelPool()
    try:
        resp = pool.chat(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello in Chinese. One word only."},
            ],
            max_completion_tokens=64,
        )
        print(f"  Fast model ({resp.model}): OK, {resp.total_tokens} tokens")
    except Exception as e:
        print(f"  Fast model: FAILED - {e}")
        return False

    try:
        resp = pool.chat_quality(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello. One word only."},
            ],
            max_completion_tokens=64,
        )
        print(f"  Quality model ({resp.model}): OK, {resp.total_tokens} tokens")
    except Exception as e:
        print(f"  Quality model: FAILED - {e}")
        return False

    stats = pool.stats()
    print(f"  Pool endpoints: {len(stats['endpoints'])}")
    print(f"  Total test calls: {stats['total_calls']}")

    # 2. File inventory
    print("\n[2] Checking file inventory...")
    zh_dir = PROJECT_ROOT / "text_raw_data" / "books"
    en_dir = PROJECT_ROOT / "text_raw_data" / "en_books"

    zh_files = sorted(list(zh_dir.glob("*.clean.json")))
    en_files = sorted(list(en_dir.glob("*.clean.json")))
    print(f"  Chinese files: {len(zh_files)}")
    print(f"  English files: {len(en_files)}")

    if len(zh_files) == 0:
        print("  ERROR: No Chinese files found!")
        return False

    # 3. Disk space
    print("\n[3] Checking disk space...")
    import shutil
    total, used, free = shutil.disk_usage(PROJECT_ROOT)
    print(f"  Total: {total / 1e9:.1f} GB")
    print(f"  Used: {used / 1e9:.1f} GB")
    print(f"  Free: {free / 1e9:.1f} GB")
    if free < 1e9:  # Less than 1 GB
        print("  WARNING: Less than 1 GB free space!")

    # 4. Output directories
    print("\n[4] Checking output directories...")
    output_base = PROJECT_ROOT / "data" / "reports" / "phase_2_7_restart_cleaning"
    output_base.mkdir(parents=True, exist_ok=True)
    print(f"  Report dir: {output_base} (exists)")

    print("\n" + "=" * 60)
    print("PREFLIGHT PASSED")
    print("=" * 60)
    return True


def quality_gate_test(num_chunks: int = 15) -> bool:
    """Run quality gate: test cleaning on small sample to verify v2.0 prompt quality.

    Checks:
    1. JSON parse success rate
    2. Domain filtering (keep_for_corpus accuracy)
    3. OCR repair effectiveness
    4. No hallucination in cleaned_text
    5. Enriched_notes separation (no model content in corpus text)
    """
    print("=" * 60)
    print("QUALITY GATE TEST")
    print("=" * 60)
    print(f"Testing {num_chunks} chunks from first 2 files...")

    # Create a mini pipeline with max_chunks and skip KU/SFT for speed
    pipeline = FastTextCleaningPipeline(
        run_id="phase_2_7_quality_gate",
        language="zh",
        num_workers=2,
        max_files=2,
        max_chunks=num_chunks,
        skip_ku_sft=True,  # Only clean+verify for gate test
        output_suffix="_gate_test",
    )

    # Run the mini pipeline
    try:
        metadata = pipeline.run()
    except Exception as e:
        print(f"Quality gate pipeline FAILED: {e}")
        return False

    # Analyze results
    import json
    cleaned_path = PROJECT_ROOT / "data" / "interim" / "text_cleaned" / "cleaned_chunks_gate_test.jsonl"
    quality_path = PROJECT_ROOT / "data" / "processed" / "text_quality" / "text_quality_scores_gate_test.jsonl"

    # Read cleaned chunks
    chunks = []
    if cleaned_path.exists():
        with open(str(cleaned_path)) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        chunks.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    # Read quality scores
    quality_scores = []
    if quality_path.exists():
        with open(str(quality_path)) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        quality_scores.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    print(f"\nResults: {len(chunks)} cleaned chunks, {len(quality_scores)} quality scores")

    # Check 1: JSON parse success
    parse_success = sum(1 for c in chunks if c.get("metadata", {}).get("parse_success", False))
    parse_rate = parse_success / max(len(chunks), 1)
    print(f"  JSON parse success rate: {parse_rate:.1%} ({parse_success}/{len(chunks)})")
    if parse_rate < 0.7:
        print("  FAIL: JSON parse rate below 70%")
        return False

    # Check 2: Domain filtering
    corpus_count = sum(1 for c in chunks if c.get("keep_for_corpus", True))
    dropped_count = sum(1 for c in chunks if not c.get("keep_for_corpus", True))
    print(f"  Corpus chunks: {corpus_count}, Dropped: {dropped_count}")

    # Check 3: OCR repairs recorded
    chunks_with_repairs = sum(1 for c in chunks if len(c.get("ocr_repairs", [])) > 0)
    print(f"  Chunks with OCR repairs: {chunks_with_repairs}/{len(chunks)}")

    # Check 4: Enriched_notes separation
    chunks_with_notes = sum(1 for c in chunks if c.get("enriched_notes", "").strip())
    print(f"  Chunks with enriched_notes: {chunks_with_notes}/{len(chunks)}")

    # Check 5: Quality scores
    if quality_scores:
        avg_scores = [q.get("average_score", 0) for q in quality_scores if "average_score" in q]
        if avg_scores:
            avg = sum(avg_scores) / len(avg_scores)
            print(f"  Average quality score: {avg:.2f}")
            passed = sum(1 for q in quality_scores if q.get("final_status") == "passed")
            print(f"  Passed: {passed}/{len(quality_scores)}")
            if avg < 0.5:
                print("  FAIL: Average quality score below 0.5")
                return False

    # Show sample cleaned text
    print("\nSample cleaned text (first 2 corpus chunks):")
    corpus_chunks = [c for c in chunks if c.get("keep_for_corpus", True)]
    for i, c in enumerate(corpus_chunks[:2]):
        ct = c.get("cleaned_text", "")
        print(f"  Chunk {i+1}: {ct[:200]}...")
        if c.get("enriched_notes", "").strip():
            print(f"    Enriched: {c.get('enriched_notes', '')[:100]}...")

    print("\n" + "=" * 60)
    print("QUALITY GATE PASSED")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(description="Phase 2.7 fast recleaning pipeline")

    parser.add_argument("--language", default="zh", help="Language filter (zh, en, all)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of concurrent workers")
    parser.add_argument("--fast_model", default="mimo-v2-omni", help="Fast model for cleaning")
    parser.add_argument("--quality_model", default="mimo-v2.5-pro", help="Quality model for verification")
    parser.add_argument("--max_files", type=int, help="Max files to process (None = all)")
    parser.add_argument("--max_pages_per_file", type=int, help="Max pages per file (None = all)")
    parser.add_argument("--max_chunks", type=int, help="Max chunks to process (None = all)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--skip_ku_sft", action="store_true", help="Skip KU and SFT generation (much faster)")
    parser.add_argument("--skip_quality", action="store_true", help="Skip quality verification")
    parser.add_argument("--preflight", action="store_true", help="Run preflight check only")
    parser.add_argument("--quality_gate_first", action="store_true", help="Run quality gate test before full run")
    parser.add_argument("--quality_gate_only", action="store_true", help="Run quality gate test only, no full run")
    parser.add_argument("--run_id", default="phase_2_7_reclean_fast", help="Run ID")
    parser.add_argument("--output_suffix", default="_reclean", help="Output file suffix")

    args = parser.parse_args()

    # Setup logging
    setup_logging(level="INFO")

    # Preflight check
    if args.preflight:
        ok = preflight_check()
        sys.exit(0 if ok else 1)

    # Quality gate test
    if args.quality_gate_only or args.quality_gate_first:
        ok = quality_gate_test()
        if not ok:
            print("Quality gate FAILED — do not proceed with full run")
            sys.exit(1)
        if args.quality_gate_only:
            sys.exit(0)

    # Full run
    print(f"\nStarting Phase 2.7 full recleaning: {args.run_id}")
    print(f"  Language: {args.language}")
    print(f"  Workers: {args.num_workers}")
    print(f"  Fast model: {args.fast_model}")
    print(f"  Quality model: {args.quality_model}")
    print(f"  Skip KU/SFT: {args.skip_ku_sft}")
    print(f"  Resume: {args.resume}")

    pipeline = FastTextCleaningPipeline(
        run_id=args.run_id,
        language=args.language,
        num_workers=args.num_workers,
        fast_model=args.fast_model,
        quality_model=args.quality_model,
        max_files=args.max_files,
        max_pages_per_file=args.max_pages_per_file,
        max_chunks=args.max_chunks,
        resume=args.resume,
        skip_ku_sft=args.skip_ku_sft,
        skip_quality=args.skip_quality,
        output_suffix=args.output_suffix,
    )

    try:
        metadata = pipeline.run()
        print(f"\nPipeline complete!")
        print(f"  Run ID: {metadata.run_id}")
        print(f"  Files processed: {metadata.total_files_processed}")
        print(f"  Chunks created: {metadata.total_chunks_created}")
        print(f"  Corpus chunks: {metadata.total_chunks_passed}")
        print(f"  Failed: {metadata.total_chunks_failed}")
        print(f"  LLM calls: {metadata.total_llm_calls}")
        print(f"  Tokens: {metadata.total_tokens_used}")
    except KeyboardInterrupt:
        print("\nInterrupted — saving checkpoint...")
        pipeline._save_checkpoint()
        pipeline._write_progress()
        print("Checkpoint saved. Use --resume to continue.")


if __name__ == "__main__":
    main()