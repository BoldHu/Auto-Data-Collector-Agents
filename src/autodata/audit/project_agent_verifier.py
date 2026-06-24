"""Project agent verifier for Phase 6.55.

Verifies that the project uses actual Python agent classes,
not just Claude Code pseudo-agents.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2].parent


def verify_project_agents() -> dict:
    """Verify project agent implementation."""
    report = {
        "agent_inheritance": {},
        "dtcg_usage": {},
        "pipeline_execution_patterns": {},
        "claude_code_risk": {},
        "conclusions": {},
    }

    src_dir = PROJECT_ROOT / "src" / "autodata"
    scripts_dir = PROJECT_ROOT / "scripts"

    # 1. Search for agent class definitions
    agent_classes = []
    for py_file in (src_dir / "agents").glob("*.py"):
        if py_file.name == "__init__.py":
            continue
        with open(py_file) as f:
            content = f.read()
        # Find class definitions
        import re
        for match in re.finditer(r'class\s+(\w+)\s*(?:\(([^)]*)\))?', content):
            class_name = match.group(1)
            base_class = match.group(2) or "None"
            agent_classes.append({
                "class": class_name,
                "base": base_class.strip(),
                "file": py_file.name,
            })

    report["agent_inheritance"]["classes"] = agent_classes

    # 2. Search for BaseAgent/ReActAgent inheritance
    base_agent_usage = []
    for py_file in src_dir.rglob("*.py"):
        with open(py_file) as f:
            content = f.read()
        if "BaseAgent" in content or "ReActAgent" in content:
            base_agent_usage.append(str(py_file.relative_to(PROJECT_ROOT)))
    report["agent_inheritance"]["files_using_base_agent"] = base_agent_usage

    # 3. Search for CentralPlanningAgent usage
    planner_usage = []
    for py_file in src_dir.rglob("*.py"):
        with open(py_file) as f:
            content = f.read()
        if "CentralPlanningAgent" in content:
            planner_usage.append(str(py_file.relative_to(PROJECT_ROOT)))
    for py_file in scripts_dir.glob("*.py"):
        with open(py_file) as f:
            content = f.read()
        if "CentralPlanningAgent" in content:
            planner_usage.append(str(py_file.relative_to(PROJECT_ROOT)))
    report["dtcg_usage"]["central_planner_usage"] = planner_usage

    # 4. Search for ContextSelector usage
    context_selector_usage = []
    for py_file in src_dir.rglob("*.py"):
        with open(py_file) as f:
            content = f.read()
        if "ContextSelector" in content:
            context_selector_usage.append(str(py_file.relative_to(PROJECT_ROOT)))
    report["dtcg_usage"]["context_selector_usage"] = context_selector_usage

    # 5. Search for MessageStore usage
    message_store_usage = []
    for py_file in src_dir.rglob("*.py"):
        with open(py_file) as f:
            content = f.read()
        if "MessageStore" in content:
            message_store_usage.append(str(py_file.relative_to(PROJECT_ROOT)))
    report["dtcg_usage"]["message_store_usage"] = message_store_usage

    # 6. Search for LocalCache usage
    local_cache_usage = []
    for py_file in src_dir.rglob("*.py"):
        with open(py_file) as f:
            content = f.read()
        if "LocalCache" in content:
            local_cache_usage.append(str(py_file.relative_to(PROJECT_ROOT)))
    report["dtcg_usage"]["local_cache_usage"] = local_cache_usage

    # 7. Identify execution patterns in pipelines
    pipeline_patterns = {}
    for py_file in (src_dir / "pipelines").glob("*.py"):
        if py_file.name == "__init__.py":
            continue
        with open(py_file) as f:
            content = f.read()

        uses_agents = "BaseAgent" in content or "ReActAgent" in content or "Agent" in content
        uses_direct_llm = "pool.chat" in content or "model_client.chat" in content
        uses_threadpool = "ThreadPoolExecutor" in content
        uses_dtcg_runtime = "DynamicTaskContextGraph" in content and "add_node" in content

        if uses_agents and uses_dtcg_runtime:
            pattern = "agent_with_dtcg"
        elif uses_agents:
            pattern = "agent_without_dtcg"
        elif uses_direct_llm and uses_threadpool:
            pattern = "pipeline_direct_llm"
        elif uses_direct_llm:
            pattern = "direct_llm_calls"
        else:
            pattern = "utility_no_llm"

        pipeline_patterns[py_file.name] = {
            "pattern": pattern,
            "uses_agents": uses_agents,
            "uses_direct_llm": uses_direct_llm,
            "uses_threadpool": uses_threadpool,
            "uses_dtcg_runtime": uses_dtcg_runtime,
        }

    report["pipeline_execution_patterns"] = pipeline_patterns

    # 8. Check for Claude-Code-only patterns
    # Look for scripts that directly call LLM without going through agents
    direct_llm_scripts = []
    for py_file in scripts_dir.glob("*.py"):
        with open(py_file) as f:
            content = f.read()
        if "pool.chat" in content or "model_client" in content:
            if "Agent" not in content:
                direct_llm_scripts.append(py_file.name)

    report["claude_code_risk"]["direct_llm_scripts"] = direct_llm_scripts
    report["claude_code_risk"]["risk_level"] = "low" if len(direct_llm_scripts) < 5 else "medium"

    # 9. Conclusions
    agent_with_dtcg = sum(1 for v in pipeline_patterns.values() if v["pattern"] == "agent_with_dtcg")
    agent_without_dtcg = sum(1 for v in pipeline_patterns.values() if v["pattern"] == "agent_without_dtcg")
    pipeline_direct = sum(1 for v in pipeline_patterns.values() if v["pattern"] == "pipeline_direct_llm")

    report["conclusions"] = {
        "total_pipeline_files": len(pipeline_patterns),
        "agent_with_dtcg": agent_with_dtcg,
        "agent_without_dtcg": agent_without_dtcg,
        "pipeline_direct_llm": pipeline_direct,
        "direct_llm_scripts": len(direct_llm_scripts),
        "central_planner_used": len(planner_usage) > 0,
        "context_selector_used": len(context_selector_usage) > 0,
        "message_store_used": len(message_store_usage) > 0,
        "local_cache_used": len(local_cache_usage) > 0,
    }

    return report
