"""Unit tests for agent modules."""

import pytest

from src.autodata.agents.base_agent import (
    AgentFramework,
    AgentObservation,
    BaseAgent,
)
from src.autodata.agents.planning_agent import (
    CentralPlanningAgent,
    PlanStep,
)
from src.autodata.agents.react_agent import (
    ReActAction,
    ReActAgent,
    ToolRegistry,
)
from src.autodata.context_graph.graph_schema import TaskStatus
from src.autodata.context_graph.message_store import MessageStore


# ── Base agent tests ──────────────────────────────────────────────────

class TestAgentObservation:
    def test_observation_creation(self):
        obs = AgentObservation(
            agent_name="TestAgent",
            action_type="test_action",
            content="Test result",
            success=True,
        )
        assert obs.agent_name == "TestAgent"
        assert obs.action_type == "test_action"
        assert obs.success is True

    def test_observation_defaults(self):
        obs = AgentObservation(
            agent_name="TestAgent",
            action_type="test",
            content="Result",
        )
        assert obs.artifact_refs == []
        assert obs.source_refs == []
        assert obs.token_usage == 0
        assert obs.metadata == {}


# ── Tool registry tests ──────────────────────────────────────────────

class TestToolRegistry:
    def test_register_tool(self):
        registry = ToolRegistry()
        registry.register("pdf_parser", "Parse PDF files")
        assert registry.has_tool("pdf_parser")

    def test_list_tools(self):
        registry = ToolRegistry()
        registry.register("tool1", "Description 1")
        registry.register("tool2", "Description 2")
        tools = registry.list_tools()
        assert len(tools) == 2

    def test_get_tool(self):
        registry = ToolRegistry()
        registry.register("ocr", "OCR processor", func=lambda x: x)
        tool = registry.get("ocr")
        assert tool is not None
        assert tool["name"] == "ocr"

    def test_missing_tool(self):
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None
        assert not registry.has_tool("nonexistent")


# ── ReAct action tests ───────────────────────────────────────────────

class TestReActAction:
    def test_action_creation(self):
        action = ReActAction(
            action_type="text_cleaner",
            action_input="Clean OCR text from page 1",
        )
        assert action.action_type == "text_cleaner"
        assert action.action_input == "Clean OCR text from page 1"

    def test_action_with_reasoning(self):
        action = ReActAction(
            action_type="deduplicator",
            action_input="Remove duplicates",
            reasoning="Found 3 duplicate entries",
        )
        assert action.reasoning == "Found 3 duplicate entries"

    def test_action_serialization(self):
        action = ReActAction(action_type="tool", action_input="input")
        d = action.to_dict()
        assert d["action_type"] == "tool"


# ── PlanStep tests ───────────────────────────────────────────────────

class TestPlanStep:
    def test_step_creation(self):
        step = PlanStep(
            step_id="step_1",
            description="Clean OCR text",
            assigned_agent="DataCleaningAgent",
        )
        assert step.step_id == "step_1"
        assert step.status == TaskStatus.PENDING
        assert step.dependencies == []

    def test_step_with_dependencies(self):
        step = PlanStep(
            step_id="step_2",
            description="Label cleaned text",
            assigned_agent="DataAnnotationAgent",
            dependencies=["step_1"],
        )
        assert step.dependencies == ["step_1"]

    def test_step_serialization(self):
        step = PlanStep(step_id="s1", description="Task", status=TaskStatus.IN_PROGRESS)
        d = step.to_dict()
        assert d["step_id"] == "s1"
        assert d["status"] == "in_progress"


# ── Agent instantiation tests ────────────────────────────────────────

class TestAgentInstantiation:
    def test_planning_agent_creation(self):
        store = MessageStore()
        planner = CentralPlanningAgent(message_store=store)
        assert planner.name == "CentralPlanningAgent"
        assert planner.framework == AgentFramework.PLAN_AND_EXECUTE
        assert planner.model == "mimo-v2.5-pro"

    def test_react_agent_creation(self):
        store = MessageStore()
        worker = ReActAgent(name="DataCleaningAgent", message_store=store)
        assert worker.name == "DataCleaningAgent"
        assert worker.framework == AgentFramework.REACT
        assert worker.model == "mimo-v2.5-pro"

    def test_agent_has_cache(self):
        store = MessageStore()
        worker = ReActAgent(name="TestWorker", message_store=store)
        assert worker.cache is not None
        assert worker.cache.agent_name == "TestWorker"

    def test_agent_messaging(self):
        store = MessageStore()
        planner = CentralPlanningAgent(message_store=store)
        worker = ReActAgent(name="Worker", message_store=store)

        # Planner sends message
        msg = planner.send_message(
            receiver="Worker",
            content="Start task",
            task_id="task_1",
        )
        assert msg.sender_agent == "CentralPlanningAgent"
        assert msg.receiver_agent == "Worker"

        # Worker receives
        received = worker.receive_messages()
        assert len(received) >= 1

    def test_agent_cache_operations(self):
        store = MessageStore()
        worker = ReActAgent(name="TestWorker", message_store=store)
        from src.autodata.context_graph.local_cache import CacheEntryType
        entry = worker.add_to_cache(
            entry_type=CacheEntryType.OBSERVATION,
            content="Test observation",
        )
        assert worker.cache.entry_count == 1