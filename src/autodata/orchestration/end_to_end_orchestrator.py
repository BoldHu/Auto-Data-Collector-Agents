"""End-to-End Orchestrator — coordinates the full multi-agent pipeline.

This module implements the real orchestration layer where:
1. CentralPlanningAgent creates a task DAG from a domain goal
2. Worker agents are instantiated and registered
3. DTCG selects context for each worker
4. Workers execute real tools and produce artifacts
5. Artifacts are registered in the DTCG
6. Quality feedback is recorded
7. Task status is updated
8. Final manifest is produced

Planning modes:
- static: uses a hardcoded default DAG (no CentralPlanningAgent)
- central: instantiates CentralPlanningAgent and calls _create_plan()
- cached-central: loads a previously persisted plan artifact, or falls back to central
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.autodata.agents.base_agent import AgentObservation
from src.autodata.agents.planning_agent import CentralPlanningAgent, PlanStep
from src.autodata.context_graph.context_selector import ContextPackage, ContextSelector
from src.autodata.context_graph.graph_schema import (
    DynamicTaskContextGraph,
    Edge,
    EdgeType,
    Node,
    NodeType,
    TaskStatus,
)
from src.autodata.context_graph.message_store import MessageStore, MessageType, Visibility
from src.autodata.utils.logging_utils import get_logger, safe_serialize
from src.autodata.utils.model_client import XiaomiModelClient, get_default_client

logger = get_logger("orchestrator")

VALID_PLANNING_MODES = ("static", "central", "cached-central")


@dataclass
class ExecutionStep:
    """Record of a single orchestration step."""
    step_id: str
    agent_name: str
    task_description: str
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    observations: list[dict] = field(default_factory=list)
    artifact_refs: list[str] = field(default_factory=list)
    context_package: Optional[dict] = None
    error: Optional[str] = None
    # Structured failure fields
    parse_error_count: int = 0
    tool_call_count: int = 0
    successful_tool_call_count: int = 0
    artifact_count: int = 0
    completion_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "agent_name": self.agent_name,
            "task_description": self.task_description,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "observations": self.observations,
            "artifact_refs": self.artifact_refs,
            "context_package": self.context_package,
            "error": self.error,
            "duration_s": (self.end_time - self.start_time) if self.start_time and self.end_time else None,
            "parse_error_count": self.parse_error_count,
            "tool_call_count": self.tool_call_count,
            "successful_tool_call_count": self.successful_tool_call_count,
            "artifact_count": self.artifact_count,
            "completion_reason": self.completion_reason,
        }


@dataclass
class PlannerArtifact:
    """Persisted planner output for reproducibility."""
    plan_id: str
    domain_request: str
    steps: list[dict]
    dependencies: dict[str, list[str]]
    assigned_agents: dict[str, str]
    constraints: list[str]
    seed: Optional[int]
    planner_mode: str
    model_cache_provenance: str
    raw_planner_response: str = ""
    validation_result: dict = field(default_factory=dict)
    assignment_map: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "domain_request": self.domain_request,
            "steps": self.steps,
            "dependencies": self.dependencies,
            "assigned_agents": self.assigned_agents,
            "constraints": self.constraints,
            "seed": self.seed,
            "planner_mode": self.planner_mode,
            "model_cache_provenance": self.model_cache_provenance,
            "raw_planner_response": self.raw_planner_response,
            "validation_result": self.validation_result,
            "assignment_map": self.assignment_map,
        }


@dataclass
class OrchestrationResult:
    """Final result of an orchestration run."""
    run_id: str
    domain_goal: str
    mode: str
    planning_mode: str
    start_time: float
    end_time: float
    steps: list[ExecutionStep]
    plan_steps: list[dict]
    artifact_manifest: list[dict]
    dtcg_node_count: int
    dtcg_edge_count: int
    total_messages: int
    total_observations: int
    # Execution trace fields
    central_planner_invoked: bool
    planner_artifact_id: Optional[str]
    worker_agent_count: int
    executed_agent_names: list[str]
    skipped_agent_names: list[str]
    dry_run_stages: list[str]
    total_parse_errors: int = 0
    total_tool_calls: int = 0
    total_successful_tool_calls: int = 0
    total_artifacts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "domain_goal": self.domain_goal,
            "mode": self.mode,
            "planning_mode": self.planning_mode,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_s": self.end_time - self.start_time,
            "steps": [s.to_dict() for s in self.steps],
            "plan_steps": self.plan_steps,
            "artifact_manifest": self.artifact_manifest,
            "dtcg_node_count": self.dtcg_node_count,
            "dtcg_edge_count": self.dtcg_edge_count,
            "total_messages": self.total_messages,
            "total_observations": self.total_observations,
            "central_planner_invoked": self.central_planner_invoked,
            "planner_artifact_id": self.planner_artifact_id,
            "worker_agent_count": self.worker_agent_count,
            "executed_agent_names": self.executed_agent_names,
            "skipped_agent_names": self.skipped_agent_names,
            "dry_run_stages": self.dry_run_stages,
            "total_parse_errors": self.total_parse_errors,
            "total_tool_calls": self.total_tool_calls,
            "total_successful_tool_calls": self.total_successful_tool_calls,
            "total_artifacts": self.total_artifacts,
        }


def _validate_plan(plan_steps: list[PlanStep], known_agents: set[str]) -> dict:
    """Validate a planner output for correctness.

    Checks:
    - No unknown agents
    - No cyclic dependencies
    - All dependencies exist
    """
    issues = []
    step_ids = {s.step_id for s in plan_steps}

    for step in plan_steps:
        # Check agent exists
        if step.assigned_agent and step.assigned_agent not in known_agents:
            issues.append(f"Unknown agent: {step.assigned_agent} in step {step.step_id}")

        # Check dependencies exist
        for dep in step.dependencies:
            if dep not in step_ids:
                issues.append(f"Missing dependency: {dep} in step {step.step_id}")

    # Simple cycle detection via topological sort attempt
    visited = set()
    in_stack = set()

    def has_cycle(step_id: str) -> bool:
        if step_id in in_stack:
            return True
        if step_id in visited:
            return False
        visited.add(step_id)
        in_stack.add(step_id)
        step = next((s for s in plan_steps if s.step_id == step_id), None)
        if step:
            for dep in step.dependencies:
                if has_cycle(dep):
                    return True
        in_stack.discard(step_id)
        return False

    for step in plan_steps:
        if has_cycle(step.step_id):
            issues.append(f"Cyclic dependency detected involving {step.step_id}")
            break

    return {"valid": len(issues) == 0, "issues": issues}


class EndToEndOrchestrator:
    """Orchestrates the full multi-agent pipeline from domain goal to final artifacts.

    Planning modes:
    - static: uses a hardcoded default DAG (no CentralPlanningAgent invocation)
    - central: instantiates CentralPlanningAgent and calls _create_plan()
    - cached-central: loads a previously persisted plan artifact, or falls back to central
    """

    # Agent name -> (module_path, class_name, role)
    WORKER_AGENTS = {
        "DataCollectionAgent": ("src.autodata.agents.data_collection_agent", "DataCollectionAgent", "data_collection"),
        "DataCleaningAgent": ("src.autodata.agents.data_cleaning_agent", "DataCleaningAgent", "text_cleaning"),
        "DataAnnotationAgent": ("src.autodata.agents.data_annotation_agent", "DataAnnotationAgent", "data_annotation"),
        "QualityVerificationAgent": ("src.autodata.agents.quality_verification_agent", "QualityVerificationAgent", "quality_verification"),
        "BenchmarkGenerationAgent": ("src.autodata.agents.benchmark_generation_agent", "BenchmarkGenerationAgent", "benchmark_generation"),
        "ModelEvaluationAgent": ("src.autodata.agents.model_evaluation_agent", "ModelEvaluationAgent", "model_evaluation"),
        "FineTuningAgent": ("src.autodata.agents.finetuning_agent", "FineTuningAgent", "finetuning"),
    }

    def __init__(
        self,
        model_client: Optional[XiaomiModelClient] = None,
        graph: Optional[DynamicTaskContextGraph] = None,
        message_store: Optional[MessageStore] = None,
        context_selector: Optional[ContextSelector] = None,
        output_dir: Optional[str] = None,
        mode: str = "smoke",
        planning_mode: str = "static",
        max_text_files: int = 1,
        max_images: int = 5,
        max_exam_files: int = 1,
        skip_training: bool = True,
        skip_external_api: bool = True,
        cached_plan_path: Optional[str] = None,
    ) -> None:
        if planning_mode not in VALID_PLANNING_MODES:
            raise ValueError(f"Invalid planning_mode={planning_mode!r}. Must be one of {VALID_PLANNING_MODES}")

        self.model_client = model_client or get_default_client()
        self.graph = graph or DynamicTaskContextGraph()
        self.message_store = message_store or MessageStore()
        self.context_selector = context_selector or ContextSelector()
        self.output_dir = output_dir or "data/reports/end_to_end_orchestration"
        self.mode = mode
        self.planning_mode = planning_mode
        self.max_text_files = max_text_files
        self.max_images = max_images
        self.max_exam_files = max_exam_files
        self.skip_training = skip_training
        self.skip_external_api = skip_external_api
        self.cached_plan_path = cached_plan_path

        self.run_id = f"orch_{uuid.uuid4().hex[:8]}_{int(time.time())}"

        # Worker agent instances
        self._workers: dict[str, Any] = {}

        # Execution trace
        self._execution_steps: list[ExecutionStep] = []

        # Artifact registry
        self._artifacts: list[dict] = []

        # Planner artifact
        self._planner_artifact: Optional[PlannerArtifact] = None

    def _instantiate_workers(self) -> None:
        """Instantiate all worker agents."""
        import importlib

        for agent_name, (module_path, class_name, role) in self.WORKER_AGENTS.items():
            try:
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)

                kwargs = {
                    "model_client": self.model_client,
                    "graph": self.graph,
                    "message_store": self.message_store,
                }

                if agent_name == "FineTuningAgent":
                    kwargs["skip_training"] = self.skip_training

                agent = cls(**kwargs)
                self._workers[agent_name] = agent
                logger.info(f"Instantiated worker: {agent_name}")

            except Exception as e:
                logger.error(f"Failed to instantiate {agent_name}: {str(e)[:100]}")
                self._workers[agent_name] = None

    def _create_default_plan(self, domain_goal: str) -> list[PlanStep]:
        """Create a default plan when LLM planning is unavailable or for static mode."""
        steps = [
            PlanStep(
                step_id="step_0",
                description="Register existing raw data manifests (text, images, exams)",
                assigned_agent="DataCollectionAgent",
                dependencies=[],
            ),
            PlanStep(
                step_id="step_1",
                description="Clean OCR text from raw book sources",
                assigned_agent="DataCleaningAgent",
                dependencies=["step_0"],
            ),
            PlanStep(
                step_id="step_2",
                description="Verify quality of cleaned text chunks",
                assigned_agent="QualityVerificationAgent",
                dependencies=["step_1"],
            ),
            PlanStep(
                step_id="step_3",
                description="Generate SFT annotations from cleaned text",
                assigned_agent="DataAnnotationAgent",
                dependencies=["step_2"],
            ),
            PlanStep(
                step_id="step_4",
                description="Build benchmark items from all sources",
                assigned_agent="BenchmarkGenerationAgent",
                dependencies=["step_2"],
            ),
            PlanStep(
                step_id="step_5",
                description="Evaluate baseline models on benchmark",
                assigned_agent="ModelEvaluationAgent",
                dependencies=["step_4"],
            ),
            PlanStep(
                step_id="step_6",
                description="Configure fine-tuning runs (dry-run mode)",
                assigned_agent="FineTuningAgent",
                dependencies=["step_3"],
            ),
        ]
        return steps

    def _create_plan_central(self, domain_goal: str) -> list[PlanStep]:
        """Create a plan using CentralPlanningAgent."""
        planner = CentralPlanningAgent(
            model_client=self.model_client,
            graph=self.graph,
            message_store=self.message_store,
            context_selector=self.context_selector,
        )

        # Capture raw planner response
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Central Planning Agent for an automated data construction system. "
                    "Decompose the given task into concrete subtasks. "
                    "Output a JSON list of steps, each with: "
                    "step_id, description, assigned_agent, dependencies."
                ),
            },
            {
                "role": "user",
                "content": f"Task: {domain_goal}",
            },
        ]
        raw_response = self.model_client.chat(messages=messages, max_completion_tokens=4096)
        raw_planner_text = raw_response.content

        plan_obs = planner._create_plan(domain_goal, {})
        plan_steps = planner.current_plan

        # Validate plan
        known_agents = set(self.WORKER_AGENTS.keys()) | {"CentralPlanningAgent"}
        validation = _validate_plan(plan_steps, known_agents)

        # Build assignment map
        assignment_map = {s.step_id: s.assigned_agent for s in plan_steps if s.assigned_agent}

        # Build and persist planner artifact
        self._planner_artifact = PlannerArtifact(
            plan_id=f"plan_{uuid.uuid4().hex[:8]}",
            domain_request=domain_goal,
            steps=[s.to_dict() for s in plan_steps],
            dependencies={s.step_id: s.dependencies for s in plan_steps},
            assigned_agents=assignment_map,
            constraints=[],
            seed=None,
            planner_mode=self.planning_mode,
            model_cache_provenance=f"model={getattr(self.model_client, 'model_name', 'mock')}",
            raw_planner_response=raw_planner_text,
            validation_result=validation,
            assignment_map=assignment_map,
        )

        logger.info(f"CentralPlanningAgent created plan with {len(plan_steps)} steps")
        return plan_steps

    def _load_cached_plan(self, domain_goal: str) -> Optional[list[PlanStep]]:
        """Load a previously persisted plan artifact."""
        cache_path = self.cached_plan_path
        if not cache_path:
            default_path = Path(self.output_dir) / "planner_artifact.json"
            if default_path.exists():
                cache_path = str(default_path)

        if not cache_path or not Path(cache_path).exists():
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                artifact_data = json.load(f)

            plan_steps = [
                PlanStep(
                    step_id=s["step_id"],
                    description=s["description"],
                    assigned_agent=s.get("assigned_agent"),
                    dependencies=s.get("dependencies", []),
                )
                for s in artifact_data.get("steps", [])
            ]

            self._planner_artifact = PlannerArtifact(
                plan_id=artifact_data.get("plan_id", "cached"),
                domain_request=artifact_data.get("domain_request", domain_goal),
                steps=artifact_data.get("steps", []),
                dependencies=artifact_data.get("dependencies", {}),
                assigned_agents=artifact_data.get("assigned_agents", {}),
                constraints=artifact_data.get("constraints", []),
                seed=artifact_data.get("seed"),
                planner_mode="cached-central",
                model_cache_provenance=artifact_data.get("model_cache_provenance", "cached"),
                raw_planner_response=artifact_data.get("raw_planner_response", ""),
                validation_result=artifact_data.get("validation_result", {}),
                assignment_map=artifact_data.get("assignment_map", {}),
            )

            logger.info(f"Loaded cached plan from {cache_path}: {len(plan_steps)} steps")
            return plan_steps
        except Exception as e:
            logger.warning(f"Failed to load cached plan: {e}")
            return None

    def _create_plan(self, domain_goal: str) -> list[PlanStep]:
        """Create a plan based on planning_mode."""
        if self.planning_mode == "static":
            return self._create_default_plan(domain_goal)

        elif self.planning_mode == "central":
            return self._create_plan_central(domain_goal)

        elif self.planning_mode == "cached-central":
            cached = self._load_cached_plan(domain_goal)
            if cached:
                return cached
            logger.info("No cached plan found, falling back to central planning")
            return self._create_plan_central(domain_goal)

        else:
            raise ValueError(f"Unknown planning_mode: {self.planning_mode}")

    def _get_task_for_agent(self, agent_name: str, step: PlanStep) -> str:
        """Generate a concrete task description for a worker agent.

        Uses planner-provided description when available, falls back to
        agent-specific templates.
        """
        # Use planner-provided description if it's detailed enough
        if step.description and len(step.description) > 20:
            return step.description

        task_map = {
            "DataCollectionAgent": (
                f"Register existing raw data manifests from the project. "
                f"Max text files: {self.max_text_files}, max images: {self.max_images}, "
                f"max exam files: {self.max_exam_files}."
            ),
            "DataCleaningAgent": (
                "Clean OCR text chunks from raw book sources. "
                "Process available cleaned.json files and produce cleaned chunks."
            ),
            "QualityVerificationAgent": (
                "Verify quality of cleaned text chunks. "
                "Check for completeness, clarity, and domain relevance."
            ),
            "DataAnnotationAgent": (
                "Generate SFT training samples from cleaned text. "
                "Create evidence-grounded instruction-output pairs."
            ),
            "BenchmarkGenerationAgent": (
                "Build benchmark items from cleaned text, image labels, and exam questions. "
                "Validate items and compute distribution statistics."
            ),
            "ModelEvaluationAgent": (
                "Load existing model predictions and compute evaluation metrics. "
                "Generate comparison tables for baseline models."
            ),
            "FineTuningAgent": (
                f"Configure fine-tuning runs for Qwen models. "
                f"Skip training: {self.skip_training}. "
                "Prepare training data manifests and list available adapters."
            ),
        }
        return task_map.get(agent_name, step.description)

    def _execute_worker_step(
        self,
        agent_name: str,
        step: PlanStep,
    ) -> ExecutionStep:
        """Execute a single worker agent step with DTCG context selection."""
        exec_step = ExecutionStep(
            step_id=step.step_id,
            agent_name=agent_name,
            task_description=step.description,
        )

        worker = self._workers.get(agent_name)
        if worker is None:
            exec_step.status = "failed"
            exec_step.error = f"Agent {agent_name} not instantiated"
            exec_step.completion_reason = "agent_not_instantiated"
            return exec_step

        exec_step.status = "running"
        exec_step.start_time = time.time()

        # Select DTCG context for this agent
        try:
            ctx_package = self.context_selector.select_context(
                graph=self.graph,
                agent_node_id=worker.graph_node_id,
                task_id=step.step_id,
                current_goal=step.description,
                token_budget=worker.context_budget,
            )
            exec_step.context_package = {
                "agent_name": ctx_package.agent_name,
                "task_id": ctx_package.task_id,
                "current_goal": ctx_package.current_goal,
                "selected_memory_count": len(ctx_package.selected_memory),
                "selected_artifacts_count": len(ctx_package.selected_artifacts),
                "constraints_count": len(ctx_package.constraints),
            }
        except Exception as e:
            logger.warning(f"Context selection failed for {agent_name}: {str(e)[:80]}")
            ctx_package = ContextPackage(
                agent_name=agent_name,
                task_id=step.step_id,
                current_goal=step.description,
            )
            exec_step.context_package = {"error": str(e)[:100]}

        # Build context dict for the worker
        task_desc = self._get_task_for_agent(agent_name, step)
        context = {
            "current_goal": task_desc,
            "task_id": step.step_id,
            "selected_memory": ctx_package.selected_memory,
            "selected_artifacts": ctx_package.selected_artifacts,
            "constraints": ctx_package.constraints,
        }

        # Execute the worker
        try:
            observations = worker.run(task=task_desc, context=context)

            # Analyze observations for structured failure fields
            for obs in observations:
                action_type = obs.action_type
                success = obs.success
                artifact_refs = obs.artifact_refs

                if action_type == "think" and not success:
                    exec_step.parse_error_count += 1
                elif action_type not in ("think", "unknown"):
                    exec_step.tool_call_count += 1
                    if success:
                        exec_step.successful_tool_call_count += 1

                if artifact_refs:
                    exec_step.artifact_count += len(artifact_refs)

            exec_step.observations = [
                {
                    "agent_name": obs.agent_name,
                    "action_type": obs.action_type,
                    "content": obs.content[:500],
                    "success": obs.success,
                    "token_usage": obs.token_usage,
                    "artifact_refs": obs.artifact_refs,
                }
                for obs in observations
            ]
            exec_step.artifact_refs = [
                ref for obs in observations for ref in obs.artifact_refs
            ]

            # Determine completion status based on observation quality
            if exec_step.parse_error_count > 0 and exec_step.tool_call_count == 0:
                exec_step.status = "failed"
                exec_step.error = f"All {exec_step.parse_error_count} observations were parse errors"
                exec_step.completion_reason = "all_parse_errors"
            elif exec_step.tool_call_count > 0 and exec_step.successful_tool_call_count == 0:
                exec_step.status = "failed"
                exec_step.error = f"All {exec_step.tool_call_count} tool calls failed"
                exec_step.completion_reason = "all_tools_failed"
            else:
                exec_step.status = "completed"
                if exec_step.tool_call_count > 0:
                    exec_step.completion_reason = "successful_tool_calls"
                else:
                    exec_step.completion_reason = "valid_completion"

        except Exception as e:
            exec_step.status = "failed"
            exec_step.error = str(e)[:200]
            exec_step.completion_reason = "exception"
            logger.error(f"Worker {agent_name} failed: {str(e)[:100]}")

        exec_step.end_time = time.time()

        # Update task status in graph
        for node in self.graph.nodes.values():
            if node.node_id == step.step_id:
                node.properties["status"] = (
                    TaskStatus.COMPLETED.value
                    if exec_step.status == "completed"
                    else TaskStatus.FAILED.value
                )
                break

        # Record execution step
        self._execution_steps.append(exec_step)

        # Register artifact in DTCG only if completed with artifacts
        if exec_step.status == "completed" and exec_step.artifact_count > 0:
            artifact_node = Node(
                node_id=f"art_{step.step_id}_output",
                node_type=NodeType.ARTIFACT,
                name=f"Output of {step.step_id}",
                properties={
                    "agent": agent_name,
                    "status": exec_step.status,
                    "observation_count": len(exec_step.observations),
                    "artifact_refs": exec_step.artifact_refs,
                    "tool_call_count": exec_step.tool_call_count,
                    "successful_tool_call_count": exec_step.successful_tool_call_count,
                },
            )
            self.graph.add_node(artifact_node)

            edge = Edge(
                edge_id=f"edge_{step.step_id}_art",
                source_id=step.step_id,
                target_id=f"art_{step.step_id}_output",
                edge_type=EdgeType.ARTIFACT_DERIVED_FROM,
            )
            self.graph.add_edge(edge)

        return exec_step

    def run(self, domain_goal: str) -> OrchestrationResult:
        """Execute the full orchestration pipeline."""
        start_time = time.time()
        logger.info(f"Starting orchestration: {domain_goal} (mode={self.mode}, planning={self.planning_mode})")

        # Step 1: Instantiate workers
        self._instantiate_workers()

        # Step 2: Create plan based on planning_mode
        plan_steps = self._create_plan(domain_goal)

        # Validate that central mode actually invoked CentralPlanningAgent
        central_planner_invoked = self.planning_mode in ("central", "cached-central") and self._planner_artifact is not None

        # Persist planner artifact if available
        planner_artifact_id = None
        if self._planner_artifact:
            planner_artifact_id = self._planner_artifact.plan_id
            self._persist_planner_artifact()

        # Register plan steps in DTCG
        for step in plan_steps:
            task_node = Node(
                node_id=step.step_id,
                node_type=NodeType.TASK,
                name=step.description[:50],
                properties={
                    "description": step.description,
                    "assigned_agent": step.assigned_agent,
                    "status": TaskStatus.PENDING.value,
                },
            )
            self.graph.add_node(task_node)

            for dep_id in step.dependencies:
                edge = Edge(
                    edge_id=f"edge_{dep_id}_{step.step_id}",
                    source_id=dep_id,
                    target_id=step.step_id,
                    edge_type=EdgeType.TASK_DEPENDENCY,
                )
                self.graph.add_edge(edge)

        # Step 3: Execute steps in dependency order
        completed_ids: set[str] = set()
        executed_agent_names: list[str] = []
        skipped_agent_names: list[str] = []
        dry_run_stages: list[str] = []
        max_rounds = len(plan_steps) + 1

        for _ in range(max_rounds):
            progress = False
            for step in plan_steps:
                if step.step_id in completed_ids:
                    continue
                if step.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    continue

                if all(dep in completed_ids for dep in step.dependencies):
                    agent_name = step.assigned_agent
                    if agent_name and agent_name in self._workers:
                        logger.info(f"Executing {step.step_id}: {step.description[:50]}")
                        exec_step = self._execute_worker_step(agent_name, step)

                        if exec_step.status == "completed":
                            step.status = TaskStatus.COMPLETED
                            step.result = exec_step.observations[-1]["content"] if exec_step.observations else ""
                        else:
                            step.status = TaskStatus.FAILED
                            step.result = exec_step.error

                        executed_agent_names.append(agent_name)

                        if agent_name == "FineTuningAgent" and self.skip_training:
                            dry_run_stages.append(agent_name)

                        completed_ids.add(step.step_id)
                        progress = True
                    else:
                        step.status = TaskStatus.FAILED
                        step.result = f"No agent: {agent_name}"
                        skipped_agent_names.append(agent_name or "unknown")
                        completed_ids.add(step.step_id)
                        progress = True

            if not progress:
                break

        # Step 4: Build final manifest
        end_time = time.time()

        artifact_manifest = []
        for exec_step in self._execution_steps:
            artifact_manifest.append({
                "step_id": exec_step.step_id,
                "agent": exec_step.agent_name,
                "status": exec_step.status,
                "observation_count": len(exec_step.observations),
                "artifact_refs": exec_step.artifact_refs,
                "duration_s": (exec_step.end_time - exec_step.start_time) if exec_step.start_time and exec_step.end_time else None,
                "parse_error_count": exec_step.parse_error_count,
                "tool_call_count": exec_step.tool_call_count,
                "successful_tool_call_count": exec_step.successful_tool_call_count,
                "artifact_count": exec_step.artifact_count,
                "completion_reason": exec_step.completion_reason,
            })

        result = OrchestrationResult(
            run_id=self.run_id,
            domain_goal=domain_goal,
            mode=self.mode,
            planning_mode=self.planning_mode,
            start_time=start_time,
            end_time=end_time,
            steps=self._execution_steps,
            plan_steps=[s.to_dict() for s in plan_steps],
            artifact_manifest=artifact_manifest,
            dtcg_node_count=len(self.graph.nodes),
            dtcg_edge_count=len(self.graph.edges),
            total_messages=len(self.message_store._messages) if hasattr(self.message_store, '_messages') else 0,
            total_observations=sum(len(s.observations) for s in self._execution_steps),
            central_planner_invoked=central_planner_invoked,
            planner_artifact_id=planner_artifact_id,
            worker_agent_count=len([w for w in self._workers.values() if w is not None]),
            executed_agent_names=list(set(executed_agent_names)),
            skipped_agent_names=list(set(skipped_agent_names)),
            dry_run_stages=list(set(dry_run_stages)),
            total_parse_errors=sum(s.parse_error_count for s in self._execution_steps),
            total_tool_calls=sum(s.tool_call_count for s in self._execution_steps),
            total_successful_tool_calls=sum(s.successful_tool_call_count for s in self._execution_steps),
            total_artifacts=sum(s.artifact_count for s in self._execution_steps),
        )

        # Step 5: Persist trace artifacts
        self._persist_traces(result)

        return result

    def _persist_planner_artifact(self) -> None:
        """Persist the planner artifact to disk."""
        if not self._planner_artifact:
            return

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = output_dir / "planner_artifact.json"
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(self._planner_artifact.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"Planner artifact persisted to {artifact_path}")

    def _persist_traces(self, result: OrchestrationResult) -> None:
        """Persist execution trace artifacts to disk."""
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Execution trace
        trace_path = output_dir / "execution_trace.jsonl"
        with open(trace_path, "w", encoding="utf-8") as f:
            for step in result.steps:
                f.write(json.dumps(step.to_dict(), ensure_ascii=False) + "\n")

        # Artifact lineage
        lineage_path = output_dir / "artifact_lineage.json"
        with open(lineage_path, "w", encoding="utf-8") as f:
            json.dump(result.artifact_manifest, f, ensure_ascii=False, indent=2)

        # DTCG trace
        dtcg_path = output_dir / "dtcg_trace.json"
        with open(dtcg_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id": result.run_id,
                    "node_count": result.dtcg_node_count,
                    "edge_count": result.dtcg_edge_count,
                    "nodes": [
                        {
                            "node_id": n.node_id,
                            "node_type": n.node_type.value,
                            "name": n.name,
                            "properties": n.properties,
                        }
                        for n in self.graph.nodes.values()
                    ],
                    "edges": [
                        {
                            "edge_id": e.edge_id,
                            "source": e.source_id,
                            "target": e.target_id,
                            "edge_type": e.edge_type.value,
                            "weight": e.weight,
                        }
                        for e in self.graph.edges.values()
                    ],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        # Context packages
        ctx_path = output_dir / "context_packages.jsonl"
        with open(ctx_path, "w", encoding="utf-8") as f:
            for step in result.steps:
                if step.context_package:
                    f.write(json.dumps({
                        "step_id": step.step_id,
                        "agent_name": step.agent_name,
                        "context_package": step.context_package,
                    }, ensure_ascii=False) + "\n")

        # Overall summary
        summary_path = output_dir / "orchestration_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Traces persisted to {output_dir}")
