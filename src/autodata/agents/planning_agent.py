"""Central Planning Agent — Plan-and-Execute framework.

The CentralPlanningAgent is the core orchestrator of the multi-agent system.
It uses a Plan-and-Execute loop:
  1. Plan: decompose a long-horizon task into subtasks
  2. Execute: assign subtasks to worker agents
  3. Evaluate: review results, update plan if needed
  4. Repeat until all subtasks are completed or max iterations reached

It also manages the DTCG (Dynamic Task-Context Graph), routing context
to worker agents based on graph-based context selection.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from src.autodata.agents.base_agent import (
    AgentFramework,
    AgentObservation,
    BaseAgent,
)
from src.autodata.context_graph.context_selector import ContextPackage, ContextSelector
from src.autodata.context_graph.graph_schema import (
    DynamicTaskContextGraph,
    Node,
    NodeType,
    TaskStatus,
)
from src.autodata.context_graph.local_cache import CacheEntryType
from src.autodata.context_graph.message_store import MessageType, MessageStore, Visibility
from src.autodata.utils.logging_utils import get_logger, safe_serialize
from src.autodata.utils.model_client import XiaomiModelClient, get_default_client


logger = get_logger("planning_agent")


# ── Plan step dataclass ────────────────────────────────────────────

class PlanStep:
    """A single step in the execution plan."""

    def __init__(
        self,
        step_id: str,
        description: str,
        assigned_agent: Optional[str] = None,
        status: TaskStatus = TaskStatus.PENDING,
        dependencies: Optional[list[str]] = None,
    ) -> None:
        self.step_id = step_id
        self.description = description
        self.assigned_agent = assigned_agent
        self.status = status
        self.dependencies = dependencies or []
        self.result: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "assigned_agent": self.assigned_agent,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "result": self.result,
        }


# ── CentralPlanningAgent ────────────────────────────────────────────

class CentralPlanningAgent(BaseAgent):
    """Central planner using Plan-and-Execute framework.

    Responsibilities:
    - Task decomposition
    - Subtask assignment to worker agents
    - DTCG management and context routing
    - Plan revision when subtasks fail
    - Phase-level reporting triggers
    """

    def __init__(
        self,
        model_client: Optional[XiaomiModelClient] = None,
        graph: Optional[DynamicTaskContextGraph] = None,
        message_store: Optional[MessageStore] = None,
        context_selector: Optional[ContextSelector] = None,
        max_iterations: int = 20,
        context_budget: int = 8000,
        model: str = "mimo-v2.5-pro",
    ) -> None:
        super().__init__(
            name="CentralPlanningAgent",
            framework=AgentFramework.PLAN_AND_EXECUTE,
            model=model,
            max_iterations=max_iterations,
            context_budget=context_budget,
            message_store=message_store,
        )
        self.model_client = model_client or get_default_client()
        self.graph = graph or DynamicTaskContextGraph()
        self.context_selector = context_selector or ContextSelector()
        self.current_plan: list[PlanStep] = []

        # Register this agent as a graph node
        self._register_in_graph()

    def _register_in_graph(self) -> None:
        """Register this agent as a node in the DTCG."""
        node = Node(
            node_id=self.graph_node_id,
            node_type=NodeType.AGENT,
            name=self.name,
            properties={
                "framework": self.framework.value,
                "model": self.model,
                "max_iterations": self.max_iterations,
                "context_budget": self.context_budget,
            },
        )
        self.graph.add_node(node)

    def step(self, context: dict[str, Any]) -> AgentObservation:
        """Execute one planning step.

        Steps:
        1. Review current plan and results
        2. Identify the next pending subtask
        3. Select context for the assigned agent via DTCG
        4. Send subtask assignment message
        """
        task_desc = context.get("current_goal", "No goal specified")

        # Build prompt for planning
        messages = self._build_planning_prompt(task_desc, context)
        response = self.model_client.chat(
            messages=messages,
            max_completion_tokens=2048,
        )

        self._total_tokens_used += response.total_tokens

        # Parse the planning response
        plan_text = response.content

        # Cache the plan
        self.add_to_cache(
            CacheEntryType.DECISION,
            content=plan_text,
            relevance_tags=["planning", "plan_step"],
            importance=0.8,
        )

        return AgentObservation(
            agent_name=self.name,
            action_type="plan_step",
            content=plan_text,
            success=True,
            token_usage=response.total_tokens,
            metadata={"step": self._step_count + 1},
        )

    def run(self, task: str, context: Optional[dict] = None) -> list[AgentObservation]:
        """Execute the full planning-and-delegation loop.

        1. Decompose task into subtasks (plan)
        2. For each subtask: select context, assign to agent
        3. Review results, revise plan if needed
        4. Continue until all subtasks done or max_iterations
        """
        context = context or {}
        observations = []

        # Step 1: Create initial plan
        plan_obs = self._create_plan(task, context)
        observations.append(plan_obs)
        self._record_observation(plan_obs)

        # Step 2: Execute subtasks iteratively
        for i in range(self.max_iterations):
            if self._is_plan_complete():
                break

            # Find next pending subtask
            next_step = self._get_next_pending_step()
            if next_step is None:
                break

            # Select context for the assigned agent
            ctx_package = self._select_context_for_agent(
                next_step.assigned_agent or "",
                next_step.description,
            )

            # Send assignment message
            assign_obs = self._assign_subtask(next_step, ctx_package)
            observations.append(assign_obs)
            self._record_observation(assign_obs)

        return observations

    def _create_plan(self, task: str, context: dict) -> AgentObservation:
        """Decompose a task into subtasks using the LLM."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Central Planning Agent for an automated data construction system. "
                    "Decompose the given task into concrete subtasks. "
                    "Output a JSON list of steps, each with: "
                    "step_id, description, assigned_agent (one of: "
                    "DataCollectionAgent, DataCleaningAgent, DataAnnotationAgent, "
                    "QualityVerificationAgent, BenchmarkGenerationAgent, ModelEvaluationAgent), "
                    "dependencies (list of step_ids that must complete first)."
                ),
            },
            {
                "role": "user",
                "content": f"Task: {task}\n\nContext: {json.dumps(context, ensure_ascii=False)[:500]}",
            },
        ]

        response = self.model_client.chat(messages=messages, max_completion_tokens=4096)

        # Parse plan from response
        plan_text = response.content
        try:
            # Try to extract JSON from the response
            json_start = plan_text.find("[")
            json_end = plan_text.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                steps_json = json.loads(plan_text[json_start:json_end])
                self.current_plan = [
                    PlanStep(
                        step_id=s.get("step_id", f"step_{i}"),
                        description=s.get("description", ""),
                        assigned_agent=s.get("assigned_agent"),
                        dependencies=s.get("dependencies", []),
                    )
                    for i, s in enumerate(steps_json)
                ]
            else:
                # Fallback: treat entire response as single step
                self.current_plan = [
                    PlanStep(step_id="step_0", description=plan_text)
                ]
        except json.JSONDecodeError:
            self.current_plan = [
                PlanStep(step_id="step_0", description=plan_text)
            ]

        # Register task nodes in the graph
        for step in self.current_plan:
            task_node = Node(
                node_id=step.step_id,
                node_type=NodeType.TASK,
                name=step.description[:50],
                properties={
                    "description": step.description,
                    "assigned_agent": step.assigned_agent,
                    "status": step.status.value,
                },
            )
            self.graph.add_node(task_node)

        return AgentObservation(
            agent_name=self.name,
            action_type="create_plan",
            content=f"Created plan with {len(self.current_plan)} steps",
            success=True,
            token_usage=response.total_tokens,
            metadata={"plan_steps": len(self.current_plan)},
        )

    def _select_context_for_agent(
        self, agent_name: str, task_desc: str
    ) -> ContextPackage:
        """Select context for a worker agent using DTCG."""
        agent_node_id = f"agent_{agent_name}"

        return self.context_selector.select_context(
            graph=self.graph,
            agent_node_id=agent_node_id,
            task_id="current_task",
            current_goal=task_desc,
            token_budget=self.context_budget,
        )

    def _assign_subtask(
        self, step: PlanStep, context_package: ContextPackage
    ) -> AgentObservation:
        """Send subtask assignment to a worker agent."""
        content = f"Subtask: {step.description}"
        if step.assigned_agent:
            self.send_message(
                receiver=step.assigned_agent,
                content=content,
                task_id=step.step_id,
                message_type=MessageType.PLAN,
                visibility=Visibility.LOCAL,
            )

        step.status = TaskStatus.IN_PROGRESS

        return AgentObservation(
            agent_name=self.name,
            action_type="assign_subtask",
            content=content,
            success=True,
            metadata={"step_id": step.step_id, "assigned_to": step.assigned_agent},
        )

    def _is_plan_complete(self) -> bool:
        """Check if all plan steps are completed or failed."""
        return all(
            s.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            for s in self.current_plan
        )

    def _get_next_pending_step(self) -> Optional[PlanStep]:
        """Get the next step whose dependencies are all completed."""
        completed_ids = {
            s.step_id for s in self.current_plan if s.status == TaskStatus.COMPLETED
        }
        for step in self.current_plan:
            if step.status == TaskStatus.PENDING:
                if all(dep in completed_ids for dep in step.dependencies):
                    return step
        return None

    def _build_planning_prompt(
        self, task_desc: str, context: dict
    ) -> list[dict[str, Any]]:
        """Build the prompt for a planning step."""
        cache_context = self.cache.to_context_string(max_tokens=1500)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Central Planning Agent managing a multi-agent data construction system. "
                    "Review the current state and decide the next action."
                ),
            },
            {
                "role": "user",
                "content": f"Task: {task_desc}\n\nCached context:\n{cache_context}\n\n"
                f"Current plan status: {json.dumps([s.to_dict() for s in self.current_plan], ensure_ascii=False)}",
            },
        ]
        return messages