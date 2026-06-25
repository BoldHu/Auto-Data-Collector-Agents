#!/usr/bin/env python3
"""End-to-end orchestration demo script.

Runs the full multi-agent pipeline in smoke mode using existing artifacts
and mock/dry-run clients. Produces execution traces, artifact lineage,
DTCG state snapshots, and context package logs.

Planning modes:
    static          - hardcoded default DAG, no CentralPlanningAgent invocation
    central         - instantiate CentralPlanningAgent and call _create_plan()
    cached-central  - load previously persisted plan artifact, fallback to central

Usage:
    python scripts/run_end_to_end_demo.py --mode smoke --planning-mode static
    python scripts/run_end_to_end_demo.py --mode smoke --planning-mode cached-central
    python scripts/run_end_to_end_demo.py --mode real --planning-mode central

Output:
    <output_dir>/
        execution_trace.jsonl
        artifact_lineage.json
        dtcg_trace.json
        context_packages.jsonl
        orchestration_summary.json
        planner_artifact.json  (if central/cached-central mode)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end orchestration demo")
    parser.add_argument(
        "--mode",
        choices=["smoke", "real"],
        default="smoke",
        help="Run mode: smoke (mock/dry-run) or real (requires API credentials)",
    )
    parser.add_argument(
        "--planning-mode",
        choices=["static", "central", "cached-central"],
        default="static",
        help="Planning mode: static (default DAG), central (CentralPlanningAgent), cached-central (cached plan)",
    )
    parser.add_argument(
        "--max_text_files",
        type=int,
        default=1,
        help="Maximum text files to process in smoke mode",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=5,
        help="Maximum images to process in smoke mode",
    )
    parser.add_argument(
        "--max_exam_files",
        type=int,
        default=1,
        help="Maximum exam files to process in smoke mode",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        default=True,
        help="Skip actual model training (dry-run mode)",
    )
    parser.add_argument(
        "--skip_external_api",
        action="store_true",
        default=True,
        help="Skip external API calls (use mock clients)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/reports/end_to_end_orchestration",
        help="Output directory for trace artifacts",
    )
    parser.add_argument(
        "--domain_goal",
        type=str,
        default="Build carbon-fiber domain data pipeline: collect, clean, annotate, benchmark, evaluate",
        help="Domain goal for the pipeline",
    )
    parser.add_argument(
        "--cached_plan_path",
        type=str,
        default=None,
        help="Path to cached plan artifact (for cached-central mode)",
    )
    return parser.parse_args()


# ReAct-compatible mock responses for each agent type
MOCK_REACT_RESPONSES = {
    "DataCollectionAgent": (
        "Thought: I need to register existing raw data manifests. Let me list the available sources.\n"
        "Action: [register_manifest] [text_raw_data/]\n"
    ),
    "DataCleaningAgent": (
        "Thought: I should clean the OCR text chunks from raw book sources.\n"
        "Action: [text_cleaner] [sample_chunk]\n"
    ),
    "QualityVerificationAgent": (
        "Thought: I need to verify the quality of cleaned text chunks.\n"
        "Action: [quality_verifier] [cleaned_chunk]\n"
    ),
    "DataAnnotationAgent": (
        "Thought: I should generate SFT training samples from cleaned text.\n"
        "Action: [generate_sft_samples] [cleaned_text]\n"
    ),
    "BenchmarkGenerationAgent": (
        "Thought: I need to build benchmark items from all sources.\n"
        "Action: [validate_items] [benchmark_candidates]\n"
    ),
    "ModelEvaluationAgent": (
        "Thought: I should load existing model predictions and compute metrics.\n"
        "Action: [load_predictions] [evaluation/outputs]\n"
    ),
    "FineTuningAgent": (
        "Thought: I need to configure fine-tuning runs. Since skip_training is True, I will only prepare data.\n"
        "Action: [prepare_training_data] [sft/final_v4]\n"
    ),
}

# Fallback response for unknown agents
MOCK_FALLBACK_RESPONSE = (
    "Thought: I need to process this task.\n"
    "Action: [finish] [Task completed]\n"
)

# Mock plan JSON for central planning
MOCK_PLAN_JSON = json.dumps([
    {"step_id": "step_0", "description": "Register existing raw data manifests (text, images, exams)", "assigned_agent": "DataCollectionAgent", "dependencies": []},
    {"step_id": "step_1", "description": "Clean OCR text from raw book sources", "assigned_agent": "DataCleaningAgent", "dependencies": ["step_0"]},
    {"step_id": "step_2", "description": "Verify quality of cleaned text chunks", "assigned_agent": "QualityVerificationAgent", "dependencies": ["step_1"]},
    {"step_id": "step_3", "description": "Generate SFT annotations from cleaned text", "assigned_agent": "DataAnnotationAgent", "dependencies": ["step_2"]},
    {"step_id": "step_4", "description": "Build benchmark items from all sources", "assigned_agent": "BenchmarkGenerationAgent", "dependencies": ["step_2"]},
    {"step_id": "step_5", "description": "Evaluate baseline models on benchmark", "assigned_agent": "ModelEvaluationAgent", "dependencies": ["step_4"]},
    {"step_id": "step_6", "description": "Configure fine-tuning runs (dry-run mode)", "assigned_agent": "FineTuningAgent", "dependencies": ["step_3"]},
])


class SmokeMockClient:
    """Mock client that returns ReAct-compatible responses for smoke mode.

    Returns different responses based on the message content, so each
    agent type gets appropriate tool-call responses.
    """

    def __init__(self):
        self.model_name = "mock_smoke"
        self._call_count = 0

    def chat(self, messages, **kwargs):
        from unittest.mock import MagicMock

        self._call_count += 1
        response = MagicMock()

        # Determine which agent is calling based on system prompt
        system_msg = ""
        user_msg = ""
        for msg in messages:
            if msg.get("role") == "system":
                system_msg = msg.get("content", "")
            elif msg.get("role") == "user":
                user_msg = msg.get("content", "")

        # Check if this is a planning call
        if "Central Planning Agent" in system_msg or "Decompose" in system_msg:
            response.content = MOCK_PLAN_JSON
            response.total_tokens = 150
            response.usage = {"prompt_tokens": 100, "completion_tokens": 50}
            return response

        # Check which agent is calling
        for agent_name, react_response in MOCK_REACT_RESPONSES.items():
            if agent_name in system_msg:
                response.content = react_response
                response.total_tokens = 100
                response.usage = {"prompt_tokens": 80, "completion_tokens": 20}
                return response

        # Fallback
        response.content = MOCK_FALLBACK_RESPONSE
        response.total_tokens = 80
        response.usage = {"prompt_tokens": 60, "completion_tokens": 20}
        return response


def main() -> None:
    args = parse_args()

    print(f"=" * 60)
    print(f"CFDataConstruct End-to-End Orchestration Demo")
    print(f"Mode: {args.mode}")
    print(f"Planning mode: {args.planning_mode}")
    print(f"=" * 60)
    print()

    # Import after path setup
    from src.autodata.orchestration.end_to_end_orchestrator import EndToEndOrchestrator

    # Set up mock client for smoke mode
    model_client = None
    if args.mode == "smoke" or args.skip_external_api:
        model_client = SmokeMockClient()
        print("[smoke] Using ReAct-compatible mock model client")

    orchestrator = EndToEndOrchestrator(
        model_client=model_client,
        output_dir=args.output_dir,
        mode=args.mode,
        planning_mode=args.planning_mode,
        max_text_files=args.max_text_files,
        max_images=args.max_images,
        max_exam_files=args.max_exam_files,
        skip_training=args.skip_training,
        skip_external_api=args.skip_external_api,
        cached_plan_path=args.cached_plan_path,
    )

    print(f"Domain goal: {args.domain_goal}")
    print(f"Output dir:  {args.output_dir}")
    print()

    # Run orchestration
    start = time.time()
    result = orchestrator.run(args.domain_goal)
    elapsed = time.time() - start

    # Print summary
    print(f"=" * 60)
    print(f"Orchestration Complete")
    print(f"=" * 60)
    print(f"Run ID:       {result.run_id}")
    print(f"Duration:     {elapsed:.2f}s")
    print(f"Planning:     {result.planning_mode}")
    print(f"Plan steps:   {len(result.plan_steps)}")
    print(f"Executed:     {len(result.steps)}")
    print(f"Completed:    {sum(1 for s in result.steps if s.status == 'completed')}")
    print(f"Failed:       {sum(1 for s in result.steps if s.status == 'failed')}")
    print(f"DTCG nodes:   {result.dtcg_node_count}")
    print(f"DTCG edges:   {result.dtcg_edge_count}")
    print(f"Messages:     {result.total_messages}")
    print(f"Observations: {result.total_observations}")
    print()
    print(f"Central planner invoked: {result.central_planner_invoked}")
    print(f"Planner artifact ID:     {result.planner_artifact_id}")
    print(f"Worker agents:           {result.worker_agent_count}")
    print(f"Executed agents:         {result.executed_agent_names}")
    print(f"Skipped agents:          {result.skipped_agent_names}")
    print(f"Dry-run stages:          {result.dry_run_stages}")
    print(f"Total parse errors:      {result.total_parse_errors}")
    print(f"Total tool calls:        {result.total_tool_calls}")
    print(f"Successful tool calls:   {result.total_successful_tool_calls}")
    print(f"Total artifacts:         {result.total_artifacts}")
    print()

    # Print step details
    print("Execution Steps:")
    print("-" * 60)
    for step in result.steps:
        status_icon = "[OK]" if step.status == "completed" else "[FAIL]"
        duration = ""
        if step.start_time and step.end_time:
            duration = f" ({step.end_time - step.start_time:.2f}s)"
        print(f"  {status_icon} {step.step_id}: {step.agent_name} - {step.task_description[:50]}{duration}")
        print(f"       Tools: {step.tool_call_count} calls, {step.successful_tool_call_count} successful, "
              f"Artifacts: {step.artifact_count}, Parse errors: {step.parse_error_count}")
        if step.error:
            print(f"       Error: {step.error[:100]}")
    print()

    # Output file listing
    output_dir = Path(args.output_dir)
    print("Output files:")
    print("-" * 60)
    for f in sorted(output_dir.glob("*")):
        size = f.stat().st_size
        print(f"  {f.name} ({size:,} bytes)")
    print()

    # Dry-run notice
    if args.mode == "smoke":
        print("[NOTE] This is a SMOKE DEMO using mock clients and existing artifacts.")
        print("[NOTE] All outputs are marked as dry-run. Real execution requires API credentials.")
        print()

    if "FineTuningAgent" in result.dry_run_stages:
        print("[NOTE] FineTuningAgent executed in dry-run mode (skip_training=True).")
        print()

    print(f"Summary saved to: {output_dir / 'orchestration_summary.json'}")


if __name__ == "__main__":
    main()
