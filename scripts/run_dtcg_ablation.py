#!/usr/bin/env python3
"""DTCG Component Ablation Script.

Runs persistent DTCG component ablation with a shared graph across tasks.
Produces per-task trace JSONL and summary table.

Usage:
    python scripts/run_dtcg_ablation.py --output build/validation/dtcg_ablation
    python scripts/run_dtcg_ablation.py --output build/validation/dtcg_ablation --variants full,no_trust,no_redundancy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_benchmark_items(path: str, max_items: int = 50) -> list:
    """Load benchmark items for ablation."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            if len(records) >= max_items:
                break
    return records


def find_benchmark_file() -> str:
    """Find the benchmark file."""
    candidates = [
        "data/benchmark/versions/cfbench_v1_full.jsonl",
        "data/benchmark/carbon_fiber_benchmark_all.jsonl",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return ""


def main():
    parser = argparse.ArgumentParser(description="DTCG Component Ablation")
    parser.add_argument("--output", default="build/validation/dtcg_ablation")
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--max-items", type=int, default=50)
    parser.add_argument("--variants", default=None, help="Comma-separated variant names")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("DTCG Component Ablation (Persistent Graph)")
    print("=" * 60)

    # Find benchmark
    bench_path = args.benchmark or find_benchmark_file()
    if not bench_path:
        print("Error: No benchmark file found")
        sys.exit(1)

    print(f"Benchmark: {bench_path}")
    print(f"Max items: {args.max_items}")

    # Load items
    items = load_benchmark_items(bench_path, args.max_items)
    print(f"Loaded {len(items)} items")
    if not items:
        print("Error: No items loaded")
        sys.exit(1)

    # Parse variants
    variants = None
    if args.variants:
        variants = [v.strip() for v in args.variants.split(",")]

    # Set up mock client for safe execution
    from unittest.mock import MagicMock
    client = MagicMock()
    client.model_name = "mock_xiaomi"

    # Mock response
    mock_response = MagicMock()
    mock_response.content = '{"answer": "Carbon fiber is a strong lightweight material", "confidence": 0.8}'
    mock_response.usage = {"prompt_tokens": 200, "completion_tokens": 100}
    client.chat.return_value = mock_response

    # Run ablation
    from src.autodata.evaluation.dtcg_persistent_eval import (
        run_persistent_ablation,
        compute_ablation_table,
        write_trace_jsonl,
    )

    print("\nRunning ablation...")
    start = time.time()
    results = run_persistent_ablation(items, client, variants=variants)
    elapsed = time.time() - start
    print(f"Completed in {elapsed:.2f}s")

    # Compute summary table
    table = compute_ablation_table(results)

    # Print table
    print("\nAblation Results:")
    print("-" * 80)
    print(f"{'Variant':<20} {'Total':>6} {'Correct':>8} {'Accuracy':>9} {'AvgCtx':>8} {'AvgLat':>8} {'Fallback':>9} {'CacheHit':>9}")
    print("-" * 80)
    for row in table:
        print(f"{row['variant']:<20} {row['total']:>6} {row.get('correct', 0):>8} {row['accuracy']:>9.4f} "
              f"{row['avg_context_tokens']:>8} {row['avg_latency']:>8.3f} {row['fallback_count']:>9} {row['cache_hit_count']:>9}")

    # Write outputs
    # Summary table
    summary_path = output_dir / "ablation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)
    print(f"\nSummary: {summary_path}")

    # Per-variant trace JSONL
    for variant, (traces, logs) in results.items():
        trace_path = output_dir / f"trace_{variant}.jsonl"
        write_trace_jsonl(logs, str(trace_path))
        print(f"Trace: {trace_path}")

    # Combined manifest
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmark_path": bench_path,
        "max_items": args.max_items,
        "variants": list(results.keys()),
        "elapsed_seconds": elapsed,
        "summary_table": table,
    }
    manifest_path = output_dir / "ablation_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
