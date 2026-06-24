"""Download Qwen-VL models for Phase 8.0.

Usage:
    python scripts/download_phase_8_0_qwen_vl_models.py \
        --models Qwen/Qwen2.5-VL-3B-Instruct \
        --output_dir models/qwen \
        --resume
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
    parser = argparse.ArgumentParser(description="Download Qwen-VL models")
    parser.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-VL-3B-Instruct"], help="Model IDs")
    parser.add_argument("--output_dir", type=str, default="models/qwen", help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume download")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if model exists")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {"models": {}, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    for model_id in args.models:
        model_name = model_id.split("/")[-1]
        local_path = output_dir / model_name

        print(f"\n=== {model_id} ===")

        if args.skip_existing and local_path.exists():
            files = list(local_path.glob("*"))
            print(f"  Already exists at {local_path} ({len(files)} files), skipping")
            report["models"][model_id] = {"status": "skipped", "path": str(local_path)}
            continue

        try:
            from huggingface_hub import snapshot_download

            print(f"  Downloading to {local_path}...")
            start = time.time()

            downloaded = snapshot_download(
                repo_id=model_id,
                local_dir=str(local_path),
                resume_download=args.resume,
            )

            elapsed = time.time() - start
            files = list(local_path.glob("*"))
            total_size = sum(f.stat().st_size for f in files if f.is_file())

            print(f"  Downloaded: {len(files)} files, {total_size/1e9:.2f}GB, {elapsed:.0f}s")

            report["models"][model_id] = {
                "status": "success",
                "path": str(local_path),
                "files": len(files),
                "size_gb": round(total_size / 1e9, 2),
                "elapsed_seconds": round(elapsed, 1),
            }

        except Exception as e:
            print(f"  Error: {str(e)[:200]}")
            report["models"][model_id] = {"status": "error", "error": str(e)[:200]}

    # Save report
    report_dir = PROJECT_ROOT / "data" / "reports" / "phase_8_0_qwen_vl_dryrun"
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "model_download_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nDownload report saved to {report_dir / 'model_download_report.json'}")


if __name__ == "__main__":
    main()
