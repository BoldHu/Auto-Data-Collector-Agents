"""Phase 7 preflight checks.

Verifies all required inputs exist before fine-tuning preparation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def run_preflight(project_root: Path) -> dict:
    """Run preflight checks for Phase 7.

    Returns:
        dict with check results and pass/fail status.
    """
    checks = {}

    # 1. Benchmark files
    benchmark_dir = project_root / "data" / "benchmark"
    checks["benchmark_all"] = (benchmark_dir / "carbon_fiber_benchmark_all.jsonl").exists()
    checks["benchmark_dev"] = (benchmark_dir / "carbon_fiber_benchmark_dev.jsonl").exists()
    checks["benchmark_test"] = (benchmark_dir / "carbon_fiber_benchmark_test.jsonl").exists()

    # 2. SFT candidate sources
    processed_dir = project_root / "data" / "processed"
    checks["pretraining_corpus"] = (processed_dir / "pretraining_corpus").exists()
    checks["knowledge_units"] = (processed_dir / "knowledge_units").exists()
    checks["sft_candidates"] = (processed_dir / "sft_candidates").exists()
    checks["exam_questions"] = (processed_dir / "exam_questions").exists()

    # 3. Benchmark candidates
    candidates_dir = project_root / "data" / "benchmark_candidates"
    checks["text_enhanced"] = (candidates_dir / "text_enhanced").exists()
    checks["agent_task"] = (candidates_dir / "agent_task").exists()

    # 4. Evaluation outputs
    eval_dir = project_root / "data" / "evaluation"
    checks["phase_6_9_traces"] = (eval_dir / "phase_6_9" / "targeted_rerun_traces.jsonl").exists()

    # 5. Existing SFT data
    sft_dir = project_root / "data" / "sft"
    checks["sft_pools"] = (sft_dir / "pools").exists()
    checks["sft_validated"] = (sft_dir / "validated" / "sft_validated_all.jsonl").exists()
    checks["sft_final"] = (sft_dir / "final" / "train.jsonl").exists()
    checks["sft_final_v2"] = (sft_dir / "final_v2" / "train.jsonl").exists()

    # 6. Training configs
    configs_dir = project_root / "configs" / "finetuning"
    checks["lora_config"] = (configs_dir / "lora_default.yaml").exists()
    checks["qlora_config"] = (configs_dir / "qlora_default.yaml").exists()

    # 7. Training scripts
    finetuning_dir = project_root / "src" / "autodata" / "finetuning"
    checks["train_lora_script"] = (finetuning_dir / "train_lora.py").exists()
    checks["dataset_loader"] = (finetuning_dir / "dataset_loader.py").exists()

    # 8. GPU check
    try:
        import torch
        checks["torch_available"] = True
        checks["gpu_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            checks["gpu_name"] = torch.cuda.get_device_name(0)
            checks["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except ImportError:
        checks["torch_available"] = False
        checks["gpu_available"] = False

    # 9. Count items
    def count_lines(path):
        if not path.exists():
            return 0
        with open(path) as f:
            return sum(1 for _ in f)

    checks["benchmark_dev_count"] = count_lines(benchmark_dir / "carbon_fiber_benchmark_dev.jsonl")
    checks["benchmark_test_count"] = count_lines(benchmark_dir / "carbon_fiber_benchmark_test.jsonl")
    checks["sft_train_count"] = count_lines(sft_dir / "final" / "train.jsonl")
    checks["sft_val_count"] = count_lines(sft_dir / "final" / "validation.jsonl")
    checks["sft_v2_train_count"] = count_lines(sft_dir / "final_v2" / "train.jsonl")
    checks["sft_v2_val_count"] = count_lines(sft_dir / "final_v2" / "validation.jsonl")

    # 10. Training disabled by default
    import yaml
    for cfg_name in ["lora_default.yaml", "qlora_default.yaml", "dry_run.yaml"]:
        cfg_path = configs_dir / cfg_name
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            checks[f"{cfg_name}_training_disabled"] = cfg.get("run_training", True) is False

    # Overall status
    critical_checks = ["benchmark_dev", "benchmark_test", "sft_final_v2"]
    checks["all_critical_passed"] = all(checks.get(c, False) for c in critical_checks)
    checks["status"] = "PASS" if checks["all_critical_passed"] else "FAIL"

    return checks


def save_preflight_report(checks: dict, output_dir: Path):
    """Save preflight report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "preflight_phase_7.json", "w") as f:
        json.dump(checks, f, indent=2, default=str)

    with open(output_dir / "preflight_phase_7.md", "w") as f:
        f.write("# Phase 7 Preflight Report\n\n")
        f.write(f"**Status**: {checks['status']}\n\n")
        f.write("## Checks\n\n")
        f.write("| Check | Status |\n|-------|--------|\n")
        for k, v in checks.items():
            if k not in ("status", "all_critical_passed"):
                status = "PASS" if v else "FAIL" if isinstance(v, bool) else str(v)
                f.write(f"| {k} | {status} |\n")
