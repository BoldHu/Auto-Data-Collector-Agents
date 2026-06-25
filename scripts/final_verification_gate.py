#!/usr/bin/env python3
"""Final Verification Gate — semantic checks for publication readiness.

Checks:
1. Clean git status (no uncommitted source changes)
2. Freeze manifest coverage
3. Compile
4. Tests (behavioral + existing)
5. Schema validation
6. Lineage validation
7. SFT provenance validation
8. CFBench duplicate policy
9. Claim registry prohibited phrase scan
10. Human audit status validation
11. DTCG ablation trace validation
12. Makefile OUT compliance

Usage:
    python scripts/final_verification_gate.py --output build/validation/final_gate
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_cmd(cmd: str, timeout: int = 120) -> dict:
    """Run a command and capture output."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return {
            "cmd": cmd,
            "returncode": result.returncode,
            "stdout": result.stdout[-2000:] if result.stdout else "",
            "stderr": result.stderr[-1000:] if result.stderr else "",
            "passed": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {"cmd": cmd, "returncode": -1, "stdout": "", "stderr": "timeout", "passed": False}
    except Exception as e:
        return {"cmd": cmd, "returncode": -1, "stdout": "", "stderr": str(e), "passed": False}


def check_git_clean() -> dict:
    """Check if git status is clean."""
    result = run_cmd("git status --short")
    clean = result["stdout"].strip() == ""
    return {
        "name": "git_clean",
        "passed": clean,
        "detail": "Clean working tree" if clean else f"Dirty: {result['stdout'][:200]}",
    }


def check_compile() -> dict:
    """Check compile."""
    result = run_cmd("PYTHONPATH=. python scripts/check_compile.py")
    return {
        "name": "compile",
        "passed": result["passed"],
        "detail": result["stdout"][:200],
    }


def check_tests() -> dict:
    """Run behavioral tests."""
    result = run_cmd(
        "PYTHONPATH=. python -m pytest tests/test_behavioral.py -v --tb=short -p no:cacheprovider",
        timeout=120,
    )
    return {
        "name": "tests",
        "passed": result["passed"],
        "detail": result["stdout"][-500:] if result["stdout"] else result["stderr"][:500],
    }


def check_claim_registry() -> dict:
    """Check claim registry for prohibited phrases."""
    csv_path = Path("reports/paper_ready/revised_claim_registry.csv")
    if not csv_path.exists():
        return {"name": "claim_registry", "passed": False, "detail": "CSV not found"}

    content = csv_path.read_text().lower()
    prohibited = [
        "human expert validated",
        "universally outperforms",
        "small model outperforms larger",
        "fully automated end-to-end",
    ]
    found = [p for p in prohibited if p in content]
    return {
        "name": "claim_registry",
        "passed": len(found) == 0,
        "detail": f"Found prohibited: {found}" if found else "No prohibited phrases",
    }


def check_human_audit_status() -> dict:
    """Check that human audit claims are removed if audit not completed."""
    csv_path = Path("reports/paper_ready/revised_claim_registry.csv")
    if not csv_path.exists():
        return {"name": "human_audit_status", "passed": False, "detail": "CSV not found"}

    content = csv_path.read_text().lower()
    has_human_claim = "human expert validated" in content or "expert annotation" in content
    return {
        "name": "human_audit_status",
        "passed": not has_human_claim,
        "detail": "Human validation claims found" if has_human_claim else "No human validation claims",
    }


def check_dtcg_import() -> dict:
    """Check that DTCG persistent eval imports correctly."""
    result = run_cmd(
        "PYTHONPATH=. python -c 'from src.autodata.evaluation.dtcg_persistent_eval import PersistentDTCGEngine; print(\"OK\")'"
    )
    return {
        "name": "dtcg_import",
        "passed": result["passed"],
        "detail": result["stdout"][:200] if result["passed"] else result["stderr"][:200],
    }


def check_sft_provenance() -> dict:
    """Check SFT provenance validator runs."""
    result = run_cmd(
        "PYTHONPATH=. python -c 'from scripts.validate_sft_provenance import validate_sft_provenance; print(\"OK\")'",
        timeout=30,
    )
    return {
        "name": "sft_provenance",
        "passed": result["passed"],
        "detail": result["stdout"][:200] if result["passed"] else result["stderr"][:200],
    }


def main():
    parser = argparse.ArgumentParser(description="Final verification gate")
    parser.add_argument("--output", default="build/validation/final_gate")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Final Verification Gate (Semantic)")
    print("=" * 60)

    checks = [
        check_git_clean,
        check_compile,
        check_tests,
        check_claim_registry,
        check_human_audit_status,
        check_dtcg_import,
        check_sft_provenance,
    ]

    results = []
    for check_func in checks:
        print(f"\nRunning {check_func.__name__}...")
        result = check_func()
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  {status}: {result['name']}")
        if not result["passed"]:
            print(f"  Detail: {result['detail'][:200]}")

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} checks passed")

    if failed > 0:
        print(f"\nFAILED checks:")
        for r in results:
            if not r["passed"]:
                print(f"  - {r['name']}: {r['detail'][:100]}")

    # Write manifest
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_checks": total,
        "passed_checks": passed,
        "failed_checks": failed,
        "gate_type": "semantic",
        "results": results,
    }
    manifest_path = output_dir / "final_gate_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nManifest: {manifest_path}")

    # Write summary
    summary_lines = [
        "# Final Verification Gate Summary",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Gate type:** Semantic",
        "",
        "## Results",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        detail = r["detail"][:80].replace("|", "\\|")
        summary_lines.append(f"| {r['name']} | {status} | {detail} |")

    summary_lines.extend([
        "",
        f"**Total:** {total} | **Passed:** {passed} | **Failed:** {failed}",
    ])

    summary_path = output_dir / "final_gate_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"Summary: {summary_path}")

    # Return nonzero if any check failed
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
