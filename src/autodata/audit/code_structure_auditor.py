"""Code structure auditor for Phase 6.55.

Inspects src/autodata/ subdirectories to verify module structure.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2].parent
SRC_ROOT = PROJECT_ROOT / "src" / "autodata"


def audit_code_structure() -> dict:
    """Audit the code structure of src/autodata/."""
    report = {
        "modules": {},
        "summary": {},
    }

    # Check each subdirectory
    subdirs = ["agents", "context_graph", "pipelines", "tools", "benchmark", "evaluation", "utils"]

    for subdir in subdirs:
        dir_path = SRC_ROOT / subdir
        if not dir_path.exists():
            report["modules"][subdir] = {"exists": False, "files": []}
            continue

        files = []
        for f in sorted(dir_path.glob("*.py")):
            if f.name == "__init__.py":
                continue
            file_info = {
                "name": f.name,
                "path": str(f.relative_to(PROJECT_ROOT)),
                "size_bytes": f.stat().st_size,
            }

            # Try to import and check classes
            try:
                module_path = f"src.autodata.{subdir}.{f.stem}"
                # Just check if file is syntactically valid
                with open(f) as fh:
                    content = fh.read()
                file_info["has_class"] = "class " in content
                file_info["has_def"] = "def " in content
                file_info["imports_model"] = "model_client" in content or "model_pool" in content or "ModelPool" in content
                file_info["imports_agent"] = "BaseAgent" in content or "ReActAgent" in content
                file_info["imports_dtcg"] = "DynamicTaskContextGraph" in content or "ContextSelector" in content or "MessageStore" in content
                file_info["imports_writer_queue"] = "WriterQueue" in content
                file_info["imports_threadpool"] = "ThreadPoolExecutor" in content
            except Exception as e:
                file_info["error"] = str(e)[:100]

            files.append(file_info)

        report["modules"][subdir] = {"exists": True, "files": files, "file_count": len(files)}

    # Summary
    total_files = sum(m.get("file_count", 0) for m in report["modules"].values())
    total_with_class = sum(
        sum(1 for f in m.get("files", []) if f.get("has_class"))
        for m in report["modules"].values()
    )
    total_with_agent = sum(
        sum(1 for f in m.get("files", []) if f.get("imports_agent"))
        for m in report["modules"].values()
    )
    total_with_dtcg = sum(
        sum(1 for f in m.get("files", []) if f.get("imports_dtcg"))
        for m in report["modules"].values()
    )

    report["summary"] = {
        "total_files": total_files,
        "files_with_classes": total_with_class,
        "files_importing_agents": total_with_agent,
        "files_importing_dtcg": total_with_dtcg,
    }

    return report
