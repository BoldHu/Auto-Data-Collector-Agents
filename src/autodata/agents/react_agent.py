"""ReAct Agent — Reasoning + Acting framework for worker agents.

Worker agents use the ReAct loop:
  1. Thought: reason about the current task and context
  2. Action: execute a tool or LLM call
  3. Observation: record the result
  4. Repeat until task is completed or max_iterations reached

Each ReAct step uses the Xiaomi LLM to decide what action to take,
then executes the action and feeds the observation back into the next step.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Optional

from src.autodata.agents.base_agent import (
    AgentFramework,
    AgentObservation,
    BaseAgent,
)
from src.autodata.context_graph.local_cache import CacheEntryType
from src.autodata.context_graph.message_store import MessageType, Visibility
from src.autodata.utils.logging_utils import get_logger, safe_serialize
from src.autodata.utils.model_client import XiaomiModelClient, get_default_client


logger = get_logger("react_agent")


# ── ReAct action types ──────────────────────────────────────────────

class ReActAction:
    """A parsed ReAct action from the LLM response."""

    def __init__(
        self,
        action_type: str,
        action_input: str,
        reasoning: Optional[str] = None,
    ) -> None:
        self.action_type = action_type
        self.action_input = action_input
        self.reasoning = reasoning

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "action_input": self.action_input,
            "reasoning": self.reasoning,
        }


# ── Tool registry ────────────────────────────────────────────────

class ToolRegistry:
    """Registry of available tools for a ReAct agent."""

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        description: str,
        func: Optional[Any] = None,
        parameters: Optional[dict] = None,
    ) -> None:
        """Register a tool."""
        self._tools[name] = {
            "name": name,
            "description": description,
            "func": func,
            "parameters": parameters or {},
        }

    def get(self, name: str) -> Optional[dict]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[dict]:
        """List all registered tools."""
        return list(self._tools.values())

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools


# ── ReActAgent ──────────────────────────────────────────────────────

class ReActAgent(BaseAgent):
    """Worker agent using the ReAct (Reasoning + Acting) framework.

    Subclasses should register specific tools and override
    _execute_action() for custom action dispatch.
    """

    def __init__(
        self,
        name: str,
        model_client: Optional[XiaomiModelClient] = None,
        message_store: Optional[Any] = None,
        max_iterations: int = 15,
        context_budget: int = 6000,
        model: str = "mimo-v2.5-pro",
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        super().__init__(
            name=name,
            framework=AgentFramework.REACT,
            model=model,
            max_iterations=max_iterations,
            context_budget=context_budget,
            message_store=message_store,
        )
        self.model_client = model_client or get_default_client()
        self.tool_registry = ToolRegistry()

        # Register provided tools
        if tools:
            for tool in tools:
                self.tool_registry.register(
                    name=tool.get("name", ""),
                    description=tool.get("description", ""),
                    func=tool.get("func"),
                    parameters=tool.get("parameters"),
                )

        # ReAct history (thought/action/observation triples)
        self._react_history: list[dict[str, Any]] = []

    def step(self, context: dict[str, Any]) -> AgentObservation:
        """Execute one ReAct step: Thought → Action → Observation.

        Args:
            context: Context package from DTCG (agent_name, task_id,
                     current_goal, selected_memory, etc.)

        Returns:
            AgentObservation with the result of this step.
        """
        current_goal = context.get("current_goal", "")
        selected_memory = context.get("selected_memory", [])
        selected_artifacts = context.get("selected_artifacts", [])

        # Step 1: Thought — decide what to do next
        thought, action = self._think(current_goal, context)

        # Step 2: Action — execute the decided action
        action_result = self._execute_action(action)

        # Step 3: Observation — record the result
        observation_text = self._observe(action_result)

        # Record in ReAct history
        self._react_history.append({
            "thought": thought,
            "action": action.to_dict() if action else None,
            "observation": observation_text,
            "step": self._step_count + 1,
        })

        # Add to local cache
        self.add_to_cache(
            CacheEntryType.OBSERVATION,
            content=observation_text,
            relevance_tags=[action.action_type if action else "unknown"],
            importance=0.6,
        )

        # Determine if task is complete
        is_finished = self._is_task_finished(observation_text)

        return AgentObservation(
            agent_name=self.name,
            action_type=action.action_type if action else "think",
            content=observation_text,
            success=action_result.get("success", True),
            token_usage=action_result.get("token_usage", 0),
            metadata={
                "thought": thought,
                "action": action.to_dict() if action else None,
                "is_finished": is_finished,
            },
        )

    def run(self, task: str, context: Optional[dict] = None) -> list[AgentObservation]:
        """Execute the full ReAct loop for a task.

        Continues iterating until the task is marked as finished
        or max_iterations is reached.
        """
        context = context or {}
        context["current_goal"] = task
        observations = []

        for i in range(self.max_iterations):
            obs = self.step(context)
            observations.append(obs)
            self._record_observation(obs)

            # Update context with latest observation
            context["last_observation"] = obs.content
            context["react_history"] = self._react_history

            # Check if task is done
            if obs.metadata.get("is_finished", False):
                logger.info(f"Agent {self.name} completed task: {task[:50]}")
                break

        # Send completion message
        if self._message_store:
            self.send_message(
                receiver="CentralPlanningAgent",
                content=f"Task completed: {task[:100]}. Steps: {len(observations)}",
                message_type=MessageType.OBSERVATION,
                visibility=Visibility.LOCAL,
            )

        return observations

    def _think(
        self, goal: str, context: dict
    ) -> tuple[str, Optional[ReActAction]]:
        """Generate a thought and decide on the next action."""
        cache_context = self.cache.to_context_string(max_tokens=1000)
        tool_descriptions = self._format_tool_descriptions()
        history_summary = self._format_history(max_steps=5)

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are {self.name}, a worker agent in a data construction system.\n"
                    f"Use the ReAct framework: output Thought and Action.\n"
                    f"Available tools:\n{tool_descriptions}\n\n"
                    f"Action format: Action: [tool_name] [action_input]\n"
                    f"If the task is complete, output: Action: [finish] [summary]\n"
                    f"Always reason step-by-step before acting."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Goal: {goal}\n\n"
                    f"ReAct history:\n{history_summary}\n\n"
                    f"Cached context:\n{cache_context}\n\n"
                    f"What should you do next?"
                ),
            },
        ]

        response = self.model_client.chat(
            messages=messages,
            max_completion_tokens=2048,
        )
        self._total_tokens_used += response.total_tokens

        # Parse thought and action from response
        thought = response.content
        reasoning = response.reasoning

        # Extract action from response
        action = self._parse_action(response.content)
        if action:
            action.reasoning = reasoning

        return thought, action

    def _execute_action(self, action: Optional[ReActAction]) -> dict:
        """Execute the decided action.

        Dispatches to registered tools or LLM calls.
        """
        if action is None:
            return {"success": False, "result": "No action parsed", "token_usage": 0}

        if action.action_type == "finish":
            return {
                "success": True,
                "result": action.action_input,
                "token_usage": 0,
                "is_finished": True,
            }

        # Check if tool is registered
        tool = self.tool_registry.get(action.action_type)
        if tool and tool.get("func"):
            try:
                result = tool["func"](action.action_input)
                return {
                    "success": True,
                    "result": str(result),
                    "token_usage": 0,
                }
            except Exception as e:
                return {
                    "success": False,
                    "result": f"Tool error: {str(e)[:200]}",
                    "token_usage": 0,
                }

        # Fallback: use LLM to process the action
        messages = [
            {
                "role": "system",
                "content": f"You are {self.name}. Execute the following action.",
            },
            {
                "role": "user",
                "content": f"Action: {action.action_type}\nInput: {action.action_input}",
            },
        ]

        response = self.model_client.chat(
            messages=messages,
            max_completion_tokens=2048,
        )
        self._total_tokens_used += response.total_tokens

        return {
            "success": True,
            "result": response.content,
            "token_usage": response.total_tokens,
        }

    def _observe(self, action_result: dict) -> str:
        """Format the action result as an observation string."""
        success = action_result.get("success", True)
        result = action_result.get("result", "")
        prefix = "Observation:" if success else "Observation (ERROR):"
        return f"{prefix} {result[:500]}"

    def _parse_action(self, response_text: str) -> Optional[ReActAction]:
        """Parse an Action line from the LLM response.

        Expected format: "Action: [tool_name] [action_input]"
        """
        # Look for "Action:" pattern
        for line in response_text.split("\n"):
            line = line.strip()
            if line.lower().startswith("action:"):
                action_text = line[len("action:"):].strip()
                # Try to parse [tool_name] [input]
                if action_text.startswith("[") and "]" in action_text:
                    bracket_end = action_text.index("]")
                    tool_name = action_text[1:bracket_end].strip()
                    remaining = action_text[bracket_end + 1:].strip()
                    # Try to extract input from brackets
                    if remaining.startswith("[") and "]" in remaining:
                        input_end = remaining.index("]")
                        action_input = remaining[1:input_end].strip()
                    else:
                        action_input = remaining.strip()
                    return ReActAction(
                        action_type=tool_name,
                        action_input=action_input,
                    )
                # Simple format: "Action: tool_name input_text"
                parts = action_text.split(None, 1)
                if parts:
                    return ReActAction(
                        action_type=parts[0],
                        action_input=parts[1] if len(parts) > 1 else "",
                    )
        return None

    def _is_task_finished(self, observation_text: str) -> bool:
        """Check if the observation indicates task completion."""
        return "finish" in observation_text.lower() or "task complete" in observation_text.lower()

    def _format_tool_descriptions(self) -> str:
        """Format available tools as a description string."""
        tools = self.tool_registry.list_tools()
        if not tools:
            return "No tools registered."
        lines = []
        for t in tools:
            lines.append(f"- {t['name']}: {t['description']}")
        return "\n".join(lines)

    def _format_history(self, max_steps: int = 5) -> str:
        """Format recent ReAct history as a summary string."""
        recent = self._react_history[-max_steps:]
        if not recent:
            return "No history yet."
        lines = []
        for h in recent:
            lines.append(f"Step {h['step']}:")
            lines.append(f"  Thought: {h.get('thought', '')[:200]}")
            if h.get("action"):
                lines.append(f"  Action: {h['action']['action_type']} — {h['action']['action_input'][:100]}")
            lines.append(f"  Observation: {h.get('observation', '')[:200]}")
        return "\n".join(lines)