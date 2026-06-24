#!/usr/bin/env python3
"""Estimate Phase 2 full-scale text cleaning runtime.

Uses Phase 2.5 pilot throughput data to estimate runtime for
full Chinese and English runs under optimistic, conservative,
and pilot-rate scenarios.

Usage:
  python scripts/estimate_phase_2_full_runtime.py
  python scripts/estimate_phase_2_full_runtime.py --language zh
  python scripts/estimate_phase_2_full_runtime.py --language en
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PilotStats:
    """Phase 2.5 LLM pilot statistics (actual measured)."""
    files: int = 4
    pages: int = 643
    llm_body_chunks: int = 30
    total_chunks: int = 89
    quality_records: int = 138
    knowledge_units: int = 148
    sft_candidates: int = 182
    llm_calls: int = 111
    tokens: int = 345_278
    runtime_seconds: float = 10_440  # ~2.9 hours


@dataclass
class FullTarget:
    """Target corpus statistics."""
    files: int = 0
    pages: int = 0
    estimated_body_chunks: int = 0


# Phase 2.5 inventory data
ZH_TARGET = FullTarget(files=38, pages=10_066, estimated_body_chunks=3_500)
EN_TARGET = FullTarget(files=26, pages=9_123, estimated_body_chunks=3_000)


def _fmt_duration(seconds: float) -> str:
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def estimate_runtime(
    pilot: PilotStats,
    target: FullTarget,
    scenario: str = "pilot",
) -> dict:
    """Estimate runtime for target corpus.

    Scenarios:
    - optimistic: 2x pilot throughput (API warm, no retries)
    - pilot: same as measured pilot throughput
    - conservative: 0.6x pilot throughput (retries, rate limits, larger docs)
    """
    # Pilot throughput metrics
    pilot_chunks_per_sec = pilot.llm_body_chunks / pilot.runtime_seconds
    pilot_llm_per_sec = pilot.llm_calls / pilot.runtime_seconds
    pilot_tokens_per_chunk = pilot.tokens / pilot.llm_body_chunks
    pilot_llm_calls_per_chunk = pilot.llm_calls / pilot.llm_body_chunks

    # Multiplier by scenario
    multipliers = {"optimistic": 2.0, "pilot": 1.0, "conservative": 0.6}
    mult = multipliers.get(scenario, 1.0)

    # Each body chunk triggers: clean + verify + extract_ku + generate_sft
    # = ~4 LLM operations per chunk (pilot showed 111 calls / 30 chunks ≈ 3.7)
    effective_llm_per_chunk = pilot_llm_calls_per_chunk
    total_llm_calls = int(target.estimated_body_chunks * effective_llm_per_chunk)
    total_tokens = int(target.estimated_body_chunks * pilot_tokens_per_chunk)

    # Runtime based on throughput * multiplier
    effective_throughput = pilot_chunks_per_sec * mult
    runtime_seconds = target.estimated_body_chunks / effective_throughput

    # API rate limit analysis
    # Xiaomi RPM=100, TPM=10M
    rpm_limit = 100
    tpm_limit = 10_000_000
    min_runtime_rpm = total_llm_calls / (rpm_limit * 60)  # seconds at RPM limit
    min_runtime_tpm = total_tokens / (tpm_limit * 60)  # seconds at TPM limit
    api_limited_runtime = max(min_runtime_rpm, min_runtime_tpm)

    # Concurrency analysis
    # Current pipeline is strictly sequential
    # Theoretical speedup with N concurrent workers
    conc_2 = runtime_seconds / 1.8  # ~1.8x with 2 workers (overhead)
    conc_4 = runtime_seconds / 3.0  # ~3x with 4 workers (more overhead)

    return {
        "scenario": scenario,
        "target_files": target.files,
        "target_pages": target.pages,
        "target_body_chunks": target.estimated_body_chunks,
        "estimated_llm_calls": total_llm_calls,
        "estimated_tokens": total_tokens,
        "estimated_runtime_seconds": round(runtime_seconds, 1),
        "estimated_runtime_formatted": _fmt_duration(runtime_seconds),
        "throughput_chunks_per_hour": round(effective_throughput * 3600, 1),
        "api_rpm_min_runtime_seconds": round(min_runtime_rpm, 1),
        "api_rpm_min_runtime_formatted": _fmt_duration(min_runtime_rpm),
        "api_tpm_min_runtime_seconds": round(min_runtime_tpm, 1),
        "api_tpm_min_runtime_formatted": _fmt_duration(min_runtime_tpm),
        "api_limited_runtime_seconds": round(api_limited_runtime, 1),
        "api_limited_runtime_formatted": _fmt_duration(api_limited_runtime),
        "concurrency_2x_runtime": _fmt_duration(conc_2),
        "concurrency_4x_runtime": _fmt_duration(conc_4),
    }


def main():
    parser = argparse.ArgumentParser(description="Estimate Phase 2 full runtime")
    parser.add_argument("--language", choices=["zh", "en", "all"], default="all")
    args = parser.parse_args()

    pilot = PilotStats()
    results = {}

    targets = []
    if args.language in ("zh", "all"):
        targets.append(("zh", ZH_TARGET))
    if args.language in ("en", "all"):
        targets.append(("en", EN_TARGET))

    for lang, target in targets:
        scenarios = {}
        for scenario in ("optimistic", "pilot", "conservative"):
            est = estimate_runtime(pilot, target, scenario)
            scenarios[scenario] = est
        results[lang] = scenarios

    # Print report
    print("\n" + "=" * 70)
    print("Phase 2 Full-Scale Runtime Estimation")
    print("=" * 70)
    print(f"\nPilot baseline: {pilot.files} files, {pilot.pages} pages, "
          f"{pilot.llm_body_chunks} LLM body chunks, {pilot.llm_calls} LLM calls")
    print(f"Pilot runtime: {_fmt_duration(pilot.runtime_seconds)}")
    print(f"Pilot throughput: {pilot.llm_body_chunks / pilot.runtime_seconds * 3600:.1f} body chunks/hour")

    for lang, scenarios in results.items():
        target = ZH_TARGET if lang == "zh" else EN_TARGET
        print(f"\n{'─' * 70}")
        print(f"Language: {lang.upper()} ({target.files} files, {target.pages} pages, "
              f"~{target.estimated_body_chunks} body chunks)")
        print(f"{'─' * 70}")

        for scenario, est in scenarios.items():
            print(f"\n  {scenario.upper()}:")
            print(f"    Estimated LLM calls:    {est['estimated_llm_calls']:,}")
            print(f"    Estimated tokens:        {est['estimated_tokens']:,}")
            print(f"    Estimated runtime:       {est['estimated_runtime_formatted']}")
            print(f"    Throughput:              {est['throughput_chunks_per_hour']:.1f} chunks/hour")

        print(f"\n  API rate-limit floor:")
        print(f"    RPM floor:               {scenarios['pilot']['api_rpm_min_runtime_formatted']}")
        print(f"    TPM floor:               {scenarios['pilot']['api_tpm_min_runtime_formatted']}")
        print(f"    Rate-limited minimum:    {scenarios['pilot']['api_limited_runtime_formatted']}")

        print(f"\n  Concurrency speedup (pilot rate):")
        print(f"    2 workers:               {scenarios['pilot']['concurrency_2x_runtime']}")
        print(f"    4 workers:               {scenarios['pilot']['concurrency_4x_runtime']}")

    # Save to JSON
    output_dir = PROJECT_ROOT / "data" / "reports" / "phase_2_full_text_cleaning"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "runtime_estimation.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nEstimation saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
