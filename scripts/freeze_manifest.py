#!/usr/bin/env python3
"""Reproducibility manifest generator.

Records git state, Python version, pip freeze, and SHA-256 hashes of
key artifacts. Produces a JSON manifest that can be regenerated with
a single command without modifying source/data artifacts.

Usage:
    python scripts/freeze_manifest.py --output manifests/freeze_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd: list[str], timeout: int = 30, cwd: str = None) -> str:
    """Run a command and return stdout, or error string."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd
        )
        return result.stdout.strip()
    except Exception as e:
        return f"error: {e}"


def sha256_file(path: str) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "missing"


def collect_hashes(root: Path, patterns: list[str]) -> dict[str, str]:
    """Collect SHA-256 hashes for files matching patterns."""
    hashes = {}
    for pattern in patterns:
        for p in sorted(root.glob(pattern)):
            if p.is_file():
                hashes[str(p.relative_to(root))] = sha256_file(str(p))
    return hashes


def main():
    parser = argparse.ArgumentParser(description="Generate reproducibility manifest")
    parser.add_argument(
        "--output", default="manifests/freeze_manifest.json",
        help="Output path for the manifest JSON",
    )
    parser.add_argument(
        "--root", default=".",
        help="Project root directory",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Git state
    git_commit = run_cmd(["git", "rev-parse", "HEAD"], cwd=str(root))
    git_status = run_cmd(["git", "status", "--short"], cwd=str(root))
    git_branch = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(root))

    # Python environment
    python_version = sys.version
    platform_info = platform.platform()

    # pip freeze
    pip_freeze = run_cmd([sys.executable, "-m", "pip", "freeze"])

    # CUDA / Torch availability
    torch_info = "not installed"
    try:
        import torch
        torch_info = f"{torch.__version__}, cuda={torch.cuda.is_available()}"
    except ImportError:
        pass

    # Source file hashes
    source_patterns = [
        "src/**/*.py",
        "scripts/*.py",
        "tests/*.py",
        "Makefile",
        ".gitignore",
    ]
    source_hashes = collect_hashes(root, source_patterns)

    # Config file hashes
    config_patterns = [
        "configs/**/*",
        "environment.yml",
        "requirements*.txt",
    ]
    config_hashes = collect_hashes(root, config_patterns)

    # SFT file hashes
    sft_patterns = [
        "data/sft/final_v4/**/*.jsonl",
    ]
    sft_hashes = collect_hashes(root, sft_patterns)

    # Benchmark file hashes
    benchmark_patterns = [
        "data/benchmark/**/*.jsonl",
        "data/benchmark/**/*.json",
    ]
    benchmark_hashes = collect_hashes(root, benchmark_patterns)

    # Evaluation manifest hashes
    eval_manifest_patterns = [
        "data/evaluation/**/manifest*.jsonl",
        "data/evaluation/**/manifest*.json",
    ]
    eval_manifest_hashes = collect_hashes(root, eval_manifest_patterns)

    # Raw output JSONL hashes (evaluation outputs)
    raw_output_patterns = [
        "data/evaluation/**/*outputs*.jsonl",
        "data/evaluation/**/*traces*.jsonl",
    ]
    raw_output_hashes = collect_hashes(root, raw_output_patterns)

    # Adapter file hashes
    adapter_patterns = [
        "models/**/*adapter*",
        "models/**/*safetensors*",
        "data/evaluation/**/adapters/**/*",
    ]
    adapter_hashes = collect_hashes(root, adapter_patterns)

    # Paper artifact hashes
    paper_patterns = [
        "reports/paper_ready/*.tex",
        "reports/paper_ready/*.csv",
        "reports/paper_ready/*.md",
        "reports/paper_ready/*.json",
    ]
    paper_hashes = collect_hashes(root, paper_patterns)

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_commit,
        "git_branch": git_branch,
        "git_status": git_status,
        "git_clean": git_status == "",
        "python_version": python_version,
        "platform": platform_info,
        "torch_info": torch_info,
        "pip_freeze": pip_freeze,
        "source_hashes": source_hashes,
        "source_count": len(source_hashes),
        "config_hashes": config_hashes,
        "config_count": len(config_hashes),
        "sft_hashes": sft_hashes,
        "sft_count": len(sft_hashes),
        "benchmark_hashes": benchmark_hashes,
        "benchmark_count": len(benchmark_hashes),
        "eval_manifest_hashes": eval_manifest_hashes,
        "eval_manifest_count": len(eval_manifest_hashes),
        "raw_output_hashes": raw_output_hashes,
        "raw_output_count": len(raw_output_hashes),
        "adapter_hashes": adapter_hashes,
        "adapter_count": len(adapter_hashes),
        "paper_hashes": paper_hashes,
        "paper_count": len(paper_hashes),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)

    print(f"Manifest written to: {output_path}")
    print(f"  Git commit: {git_commit[:12]}")
    print(f"  Git clean: {manifest['git_clean']}")
    print(f"  Source files: {len(source_hashes)}")
    print(f"  Config files: {len(config_hashes)}")
    print(f"  SFT files: {len(sft_hashes)}")
    print(f"  Benchmark files: {len(benchmark_hashes)}")
    print(f"  Eval manifests: {len(eval_manifest_hashes)}")
    print(f"  Raw outputs: {len(raw_output_hashes)}")
    print(f"  Adapter files: {len(adapter_hashes)}")
    print(f"  Paper artifacts: {len(paper_hashes)}")


if __name__ == "__main__":
    main()
