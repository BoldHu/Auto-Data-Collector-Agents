"""Phase 7: Prepare fine-tuning data and training infrastructure.

Usage:
    python scripts/run_phase_7_prepare_finetuning.py \
        --build_sft \
        --validate_sft \
        --split_sft \
        --prepare_training_configs \
        --run_training false
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Phase 7 fine-tuning preparation")
    parser.add_argument("--build_sft", action="store_true", help="Build SFT pools")
    parser.add_argument("--validate_sft", action="store_true", help="Validate SFT samples")
    parser.add_argument("--split_sft", action="store_true", help="Split into train/val")
    parser.add_argument("--prepare_training_configs", action="store_true", help="Prepare training configs")
    parser.add_argument("--run_training", type=str, default="false", help="Enable full training")
    parser.add_argument("--max_samples", type=int, default=0, help="Max samples for dry run")
    args = parser.parse_args()

    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_7_finetuning_preparation"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = report_dir / "progress_phase_7.log"

    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_file, "a") as f:
            f.write(line + "\n")

    log("=== Phase 7: Fine-tuning Preparation ===")

    # Step 1: Build SFT pools
    if args.build_sft:
        log("Building SFT pools...")
        from src.autodata.finetuning.sft_data_builder import SFTDataBuilder

        builder = SFTDataBuilder(PROJECT_ROOT)
        pools = builder.build_all_pools()

        sft_dir = PROJECT_ROOT / "data" / "sft" / "pools"
        counts = builder.save_pools(pools, sft_dir)

        total = sum(counts.values())
        log(f"SFT pools built: {counts}")
        log(f"Total SFT samples: {total}")

        # Save pool counts
        with open(report_dir / "sft_pool_counts.json", "w") as f:
            json.dump(counts, f, indent=2)

    # Step 2: Leakage check
    log("Running leakage check...")
    from src.autodata.finetuning.leakage_detector import LeakageDetector

    detector = LeakageDetector()
    dev_path = PROJECT_ROOT / "data" / "benchmark" / "carbon_fiber_benchmark_dev.jsonl"
    test_path = PROJECT_ROOT / "data" / "benchmark" / "carbon_fiber_benchmark_test.jsonl"
    detector.load_benchmark(dev_path, test_path)

    log(f"  Benchmark items loaded: {len(detector.benchmark_ids)} IDs, {len(detector.benchmark_questions)} questions")

    # Check all pools
    all_samples = []
    sft_dir = PROJECT_ROOT / "data" / "sft" / "pools"
    for pool_file in sft_dir.glob("*_sft.jsonl"):
        with open(pool_file) as f:
            for line in f:
                if line.strip():
                    all_samples.append(json.loads(line))

    log(f"  Total samples to check: {len(all_samples)}")

    clean, leaked = detector.filter_samples(all_samples)
    log(f"  Clean: {len(clean)}, Leaked: {len(leaked)}")

    # Save leakage report
    leakage_report = {
        "total_checked": len(all_samples),
        "clean": len(clean),
        "leaked": len(leaked),
        "leakage_rate": len(leaked) / max(len(all_samples), 1),
        "leaked_examples": [
            {"sample_id": s.get("sample_id", ""), "reasons": s.get("_leakage_result", {}).get("reasons", [])}
            for s in leaked[:20]
        ],
    }
    with open(report_dir / "leakage_report.json", "w") as f:
        json.dump(leakage_report, f, indent=2, ensure_ascii=False)

    # Save clean and leaked samples
    from src.autodata.finetuning.sft_data_builder import save_jsonl
    save_jsonl(clean, PROJECT_ROOT / "data" / "sft" / "validated" / "sft_validated_all.jsonl")
    save_jsonl(leaked, PROJECT_ROOT / "data" / "sft" / "validated" / "sft_rejected.jsonl")
    save_jsonl(leaked, PROJECT_ROOT / "data" / "sft" / "leakage_removed_samples.jsonl")

    log(f"  Leakage report saved")

    # Step 3: Quality filtering
    if args.validate_sft:
        log("Running quality filter...")
        from src.autodata.finetuning.sft_quality_filter import filter_samples, deduplicate_samples

        # Deduplicate first
        deduped = deduplicate_samples(clean)
        log(f"  After dedup: {len(deduped)} (removed {len(clean) - len(deduped)})")

        passed, rejected = filter_samples(deduped)
        log(f"  Quality filter: passed={len(passed)}, rejected={len(rejected)}")

        # Save quality scores
        save_jsonl(passed, PROJECT_ROOT / "data" / "sft" / "validated" / "sft_validated_all.jsonl")
        save_jsonl(rejected, PROJECT_ROOT / "data" / "sft" / "validated" / "sft_rejected.jsonl")

        clean = passed

    # Step 4: Split
    if args.split_sft:
        log("Splitting into train/validation...")
        from src.autodata.finetuning.sft_splitter import split_samples, save_splits

        train, val = split_samples(clean, train_ratio=0.9)
        stats = save_splits(train, val, PROJECT_ROOT / "data" / "sft" / "final")

        log(f"  Train: {stats['train']['count']}, Val: {stats['validation']['count']}")
        log(f"  Source types: {stats['train']['source_type_dist']}")

    # Step 5: Generate dataset card
    log("Generating dataset card...")
    _generate_dataset_card(PROJECT_ROOT, report_dir)

    # Step 6: Post-finetune evaluation plan
    log("Creating post-finetune evaluation plan...")
    _generate_eval_plan(PROJECT_ROOT)

    log("=== Phase 7 Preparation Complete ===")

    # Save progress
    with open(report_dir / "progress_phase_7.json", "w") as f:
        json.dump({
            "status": "complete",
            "build_sft": args.build_sft,
            "validate_sft": args.validate_sft,
            "split_sft": args.split_sft,
            "run_training": args.run_training,
        }, f, indent=2)


def _generate_dataset_card(project_root: Path, report_dir: Path):
    """Generate SFT dataset card."""
    stats_path = project_root / "data" / "sft" / "final" / "sft_dataset_statistics.json"
    stats = {}
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)

    card = f"""# SFT Dataset Card - Carbon Fiber Domain

## Overview
This dataset is designed for fine-tuning domain-specific models on carbon fiber knowledge tasks.

## Data Sources
1. **Domain Knowledge**: Cleaned text corpus, knowledge units, text-enhanced candidates
2. **Exam Questions**: Carbon fiber exam questions from professional exams
3. **Agent Tasks**: Data construction and quality verification tasks
4. **DTCG Reasoning**: Evidence-based reasoning from DTCG system traces
5. **Error Correction**: Model error correction and explanation

## Construction Pipeline
1. Source data collected and cleaned (Phases 2-4)
2. Benchmark candidates generated (Phase 5)
3. System ablation traces collected (Phases 6.6-6.9)
4. Leakage detection applied (excluded benchmark dev/test items)
5. Quality filtering applied (rule-based validation)
6. Deduplication applied
7. Train/validation split (90/10)

## Leakage Prevention
- Exact benchmark_id matching
- Exact/fuzzy question matching
- Source reference overlap detection
- Leakage group isolation
- No benchmark dev/test items in training data

## Dataset Statistics
- Total samples: {stats.get('total', 'N/A')}
- Train samples: {stats.get('train', {}).get('count', 'N/A')}
- Validation samples: {stats.get('validation', {}).get('count', 'N/A')}

### Train Distribution
- Source types: {json.dumps(stats.get('train', {}).get('source_type_dist', {}), ensure_ascii=False)}
- Task types: {json.dumps(stats.get('train', {}).get('task_type_dist', {}), ensure_ascii=False)}
- Difficulty: {json.dumps(stats.get('train', {}).get('difficulty_dist', {}), ensure_ascii=False)}
- Avg input length: {stats.get('train', {}).get('avg_input_len', 'N/A')} chars
- Avg output length: {stats.get('train', {}).get('avg_output_len', 'N/A')} chars

## Limitations
- Small dataset size (~500-800 samples)
- Mostly Chinese language
- Text-only (multimodal excluded for text fine-tuning)
- Domain-specific (carbon fiber only)

## Intended Use
- Fine-tuning small models for carbon fiber domain QA
- Improving domain-specific reasoning capabilities
- Training data construction agents

## Not Intended Use
- General-purpose chatbot training
- Production deployment without further validation
- Cross-domain transfer without adaptation
"""

    with open(project_root / "data" / "sft" / "final" / "SFT_DATASET_CARD.md", "w") as f:
        f.write(card)


def _generate_eval_plan(project_root: Path):
    """Generate post-finetune evaluation plan."""
    plan = """# Post-Finetune Evaluation Plan

## Evaluation Benchmarks
1. CFBench-Text (315 items)
2. CFBench-Exam (61 items)
3. CFBench-AgentTask (127 items)
4. CFBench-Core text-only subset
5. DTCG-related AgentTask subset

## Metrics
- Strict accuracy (exact match)
- Judge accuracy (LLM-judged)
- F1 score
- Format validity
- Hallucination rate
- Evidence support score
- Latency
- Token cost

## Comparisons
1. Base model (before fine-tuning)
2. Fine-tuned model
3. Larger baseline model (if available)
4. DTCG system with/without fine-tuned worker

## Expected Claims
- Whether smaller model trained on AutoData SFT improves on domain benchmark
- Whether domain-specific fine-tuning outperforms general models
- Do NOT claim superiority before evaluation

## Dry Run Command
```bash
python scripts/run_phase_7_train_lora.py \\
    --config configs/finetuning/dry_run.yaml \\
    --train_file data/sft/final/train_chatml.jsonl \\
    --validation_file data/sft/final/validation_chatml.jsonl \\
    --run_training false \\
    --max_samples 16
```
"""

    with open(project_root / "data" / "sft" / "final" / "POST_FINETUNE_EVALUATION_PLAN.md", "w") as f:
        f.write(plan)


if __name__ == "__main__":
    main()
