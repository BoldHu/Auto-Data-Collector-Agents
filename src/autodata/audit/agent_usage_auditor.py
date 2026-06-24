"""Agent usage auditor for Phase 6.55.

Audits each agent's implementation status and usage.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2].parent


def audit_agent_usage() -> dict:
    """Audit agent implementation and usage."""
    report = {"agents": {}, "summary": {}}

    agents_dir = PROJECT_ROOT / "src" / "autodata" / "agents"
    pipelines_dir = PROJECT_ROOT / "src" / "autodata" / "pipelines"
    scripts_dir = PROJECT_ROOT / "scripts"

    # Define expected agents and their requirements
    expected_agents = {
        "CentralPlanningAgent": {
            "file": "planning_agent.py",
            "base_class": "BaseAgent",
            "required_features": ["plan_and_execute", "context_selector", "message_store", "task_decomposition"],
        },
        "DataCleaningAgent": {
            "file": "data_cleaning_agent.py",
            "base_class": "ReActAgent",
            "required_features": ["llm_calls", "jsonl_output", "quality_prompt"],
        },
        "QualityVerificationAgent": {
            "file": "quality_verification_agent.py",
            "base_class": "ReActAgent",
            "required_features": ["independent_prompt", "multi_dimension_scoring"],
        },
        "DataCollectionAgent": {
            "file": "data_collection_agent.py",
            "base_class": "ReActAgent",
            "required_features": ["crawler_tools", "metadata_collection"],
        },
        "ExamExtractionAgent": {
            "file": "exam_extraction_agent.py",
            "base_class": None,
            "required_features": ["llm_calls", "question_extraction"],
        },
        "ExamQualityAgent": {
            "file": "exam_quality_agent.py",
            "base_class": None,
            "required_features": ["llm_calls", "quality_scoring"],
        },
        "BenchmarkGenerationAgent": {
            "file": "benchmark_generation_agent.py",
            "base_class": "ReActAgent",
            "required_features": ["benchmark_construction", "validation"],
        },
        "ModelEvaluationAgent": {
            "file": "model_evaluation_agent.py",
            "base_class": "ReActAgent",
            "required_features": ["baseline_evaluation", "metric_computation"],
        },
    }

    # Check each agent
    for agent_name, requirements in expected_agents.items():
        agent_info = {
            "name": agent_name,
            "expected_base_class": requirements["base_class"],
            "required_features": requirements["required_features"],
        }

        # Check if file exists
        if requirements["file"]:
            file_path = agents_dir / requirements["file"]
            if file_path.exists():
                with open(file_path) as f:
                    content = f.read()

                agent_info["file_exists"] = True
                agent_info["file_path"] = str(file_path.relative_to(PROJECT_ROOT))
                agent_info["file_size"] = file_path.stat().st_size

                # Check base class
                if requirements["base_class"]:
                    agent_info["inherits_base"] = f"class {agent_name}({requirements['base_class']})" in content or f"class {agent_name}(" in content
                else:
                    agent_info["inherits_base"] = None  # standalone class

                # Check features
                feature_checks = {}
                for feature in requirements["required_features"]:
                    if feature == "plan_and_execute":
                        feature_checks[feature] = "_create_plan" in content or "PlanStep" in content
                    elif feature == "context_selector":
                        feature_checks[feature] = "ContextSelector" in content
                    elif feature == "message_store":
                        feature_checks[feature] = "MessageStore" in content or "send_message" in content
                    elif feature == "task_decomposition":
                        feature_checks[feature] = "PlanStep" in content or "_create_plan" in content
                    elif feature == "llm_calls":
                        feature_checks[feature] = "model_client" in content or "pool.chat" in content or "self.pool" in content
                    elif feature == "jsonl_output":
                        feature_checks[feature] = "jsonl" in content.lower() or "append_jsonl" in content
                    elif feature == "quality_prompt":
                        feature_checks[feature] = "get_cleaning_prompt" in content or "cleaning" in content.lower()
                    elif feature == "independent_prompt":
                        feature_checks[feature] = "verification" in content.lower() or "verify" in content.lower()
                    elif feature == "multi_dimension_scoring":
                        feature_checks[feature] = "clarity" in content and "completeness" in content
                    elif feature == "question_extraction":
                        feature_checks[feature] = "question" in content.lower() and "extract" in content.lower()
                    elif feature == "quality_scoring":
                        feature_checks[feature] = "quality" in content.lower() and "score" in content.lower()
                    elif feature == "crawler_tools":
                        feature_checks[feature] = "crawler" in content.lower() or "bing" in content.lower()
                    elif feature == "metadata_collection":
                        feature_checks[feature] = "metadata" in content.lower()
                    elif feature == "benchmark_construction":
                        feature_checks[feature] = "benchmark" in content.lower()
                    elif feature == "validation":
                        feature_checks[feature] = "validat" in content.lower()
                    elif feature == "baseline_evaluation":
                        feature_checks[feature] = "baseline" in content.lower() or "evaluat" in content.lower()
                    elif feature == "metric_computation":
                        feature_checks[feature] = "metric" in content.lower() or "accuracy" in content.lower()
                    else:
                        feature_checks[feature] = feature in content.lower()
                agent_info["feature_checks"] = feature_checks
                agent_info["features_present"] = sum(1 for v in feature_checks.values() if v)
                agent_info["features_total"] = len(feature_checks)

                # Check usage in pipelines
                used_in_pipelines = []
                for pf in pipelines_dir.glob("*.py"):
                    if pf.name == "__init__.py":
                        continue
                    with open(pf) as f:
                        pf_content = f.read()
                    if agent_name in pf_content:
                        used_in_pipelines.append(pf.name)
                agent_info["used_in_pipelines"] = used_in_pipelines

                # Check usage in scripts
                used_in_scripts = []
                for sf in scripts_dir.glob("*.py"):
                    with open(sf) as f:
                        sf_content = f.read()
                    if agent_name in sf_content:
                        used_in_scripts.append(sf.name)
                agent_info["used_in_scripts"] = used_in_scripts

                # Classify
                if used_in_pipelines or used_in_scripts:
                    agent_info["classification"] = "implemented_and_used"
                elif agent_info.get("features_present", 0) > 0:
                    agent_info["classification"] = "implemented_but_not_used"
                else:
                    agent_info["classification"] = "partially_implemented"
            else:
                agent_info["file_exists"] = False
                agent_info["classification"] = "missing"
        else:
            agent_info["file_exists"] = False
            agent_info["classification"] = "missing"

        report["agents"][agent_name] = agent_info

    # Summary
    classifications = [a["classification"] for a in report["agents"].values()]
    report["summary"] = {
        "total_agents": len(expected_agents),
        "implemented_and_used": classifications.count("implemented_and_used"),
        "implemented_but_not_used": classifications.count("implemented_but_not_used"),
        "partially_implemented": classifications.count("partially_implemented"),
        "missing": classifications.count("missing"),
    }

    return report
