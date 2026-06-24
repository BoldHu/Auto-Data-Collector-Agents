"""Unit tests for context graph modules."""

import time

import pytest

from src.autodata.context_graph.graph_schema import (
    DynamicTaskContextGraph,
    Edge,
    EdgeType,
    Node,
    NodeType,
    TaskStatus,
)
from src.autodata.context_graph.context_selector import (
    ContextPackage,
    ContextSelector,
    ContextSelectorConfig,
)
from src.autodata.context_graph.message_store import (
    Message,
    MessageStore,
    MessageType,
    Visibility,
)
from src.autodata.context_graph.local_cache import (
    CacheEntry,
    CacheEntryType,
    LocalCache,
)


# ── Graph schema tests ────────────────────────────────────────────────

class TestGraphSchema:
    def test_create_graph(self):
        g = DynamicTaskContextGraph()
        assert g.node_count == 0
        assert g.edge_count == 0

    def test_add_node(self):
        g = DynamicTaskContextGraph()
        node = Node(node_id="n1", node_type=NodeType.AGENT, name="Agent1")
        g.add_node(node)
        assert g.node_count == 1
        assert g.get_node("n1") is not None
        assert g.get_node("n1").name == "Agent1"

    def test_add_multiple_nodes(self):
        g = DynamicTaskContextGraph()
        for i in range(5):
            g.add_node(Node(node_id=f"n{i}", node_type=NodeType.TASK, name=f"Task{i}"))
        assert g.node_count == 5

    def test_remove_node(self):
        g = DynamicTaskContextGraph()
        g.add_node(Node(node_id="n1", node_type=NodeType.AGENT, name="Agent1"))
        g.remove_node("n1")
        assert g.node_count == 0
        assert g.get_node("n1") is None

    def test_add_edge(self):
        g = DynamicTaskContextGraph()
        g.add_node(Node(node_id="a", node_type=NodeType.AGENT, name="Agent"))
        g.add_node(Node(node_id="t", node_type=NodeType.TASK, name="Task"))
        edge = Edge.new(source_id="a", target_id="t", edge_type=EdgeType.AGENT_ASSIGNMENT)
        g.add_edge(edge)
        assert g.edge_count == 1

    def test_edge_weight_computation(self):
        edge = Edge.new(
            source_id="a",
            target_id="t",
            edge_type=EdgeType.CONTEXT_RELEVANCE,
        )
        edge.relevance_score = 0.9
        edge.dependency_score = 0.7
        edge.recency_score = 0.8
        edge.trust_score = 0.6
        edge.redundancy_score = 0.2
        edge.cost_score = 0.1
        weight = edge.compute_weight()
        assert 0 < weight < 1  # sigmoid output

    def test_neighbors(self):
        g = DynamicTaskContextGraph()
        g.add_node(Node(node_id="a", node_type=NodeType.AGENT, name="Agent"))
        g.add_node(Node(node_id="t1", node_type=NodeType.TASK, name="Task1"))
        g.add_node(Node(node_id="t2", node_type=NodeType.TASK, name="Task2"))
        g.add_edge(Edge.new(source_id="a", target_id="t1", edge_type=EdgeType.AGENT_ASSIGNMENT))
        g.add_edge(Edge.new(source_id="a", target_id="t2", edge_type=EdgeType.AGENT_ASSIGNMENT))
        neighbors = g.get_neighbors("a")
        assert len(neighbors) == 2

    def test_serialization(self):
        g = DynamicTaskContextGraph()
        g.add_node(Node(node_id="n1", node_type=NodeType.AGENT, name="Agent1"))
        d = g.to_dict()
        assert "nodes" in d
        assert "edges" in d
        assert "t" in d


# ── Context selector tests ────────────────────────────────────────────

class TestContextSelector:
    def test_create_selector(self):
        selector = ContextSelector()
        assert isinstance(selector, ContextSelector)

    def test_config_defaults(self):
        config = ContextSelectorConfig()
        assert config.beta == 0.5
        assert config.gamma == 0.4
        assert config.lam == 0.6
        assert config.default_token_budget == 8000

    def test_select_context_empty_graph(self):
        g = DynamicTaskContextGraph()
        selector = ContextSelector()
        package = selector.select_context(
            graph=g,
            agent_node_id="agent_1",
            task_id="task_1",
            token_budget=2000,
            current_goal="Test goal",
        )
        assert isinstance(package, ContextPackage)

    def test_select_context_with_nodes(self):
        g = DynamicTaskContextGraph()
        g.add_node(Node(node_id="agent_1", node_type=NodeType.AGENT, name="Agent"))
        g.add_node(Node(node_id="task_1", node_type=NodeType.TASK, name="Task"))
        g.add_node(Node(
            node_id="mem_1", node_type=NodeType.MEMORY,
            name="Memory 1", properties={"content": "Important context"},
        ))
        # Connect nodes via edges
        g.add_edge(Edge.new(source_id="agent_1", target_id="mem_1", edge_type=EdgeType.CONTEXT_RELEVANCE))
        selector = ContextSelector()
        package = selector.select_context(
            graph=g,
            agent_node_id="agent_1",
            task_id="task_1",
            token_budget=2000,
            current_goal="Test goal",
        )
        assert isinstance(package, ContextPackage)


# ── Message store tests ──────────────────────────────────────────────

class TestMessageStore:
    def test_create_store(self):
        store = MessageStore()
        assert isinstance(store, MessageStore)

    def test_add_and_retrieve(self):
        store = MessageStore()
        msg = Message(
            message_id="msg_1",
            timestamp=time.time(),
            sender_agent="Agent1",
            receiver_agent="Agent2",
            task_id="task_1",
            message_type=MessageType.OBSERVATION,
            content="Test message",
            relevance_tags=["test"],
            token_estimate=50,
            visibility=Visibility.LOCAL,
        )
        store.add(msg)
        received = store.get_by_receiver("Agent2")
        assert len(received) >= 1
        assert received[0].content == "Test message"

    def test_get_by_sender(self):
        store = MessageStore()
        msg = Message(
            message_id="msg_2",
            timestamp=time.time(),
            sender_agent="Sender",
            receiver_agent="Receiver",
            task_id="task_1",
            message_type=MessageType.PLAN,
            content="Plan message",
        )
        store.add(msg)
        sent = store.get_by_sender("Sender")
        assert len(sent) >= 1

    def test_get_by_task(self):
        store = MessageStore()
        msg = Message(
            message_id="msg_3",
            timestamp=time.time(),
            sender_agent="A1",
            receiver_agent="A2",
            task_id="task_abc",
            message_type=MessageType.OBSERVATION,
            content="Task-specific message",
        )
        store.add(msg)
        task_msgs = store.get_by_task("task_abc")
        assert len(task_msgs) >= 1


# ── Local cache tests ────────────────────────────────────────────────

class TestLocalCache:
    def test_create_cache(self):
        cache = LocalCache(agent_name="TestAgent")
        assert cache.entry_count == 0
        assert cache.total_tokens == 0

    def test_add_entry(self):
        cache = LocalCache(agent_name="TestAgent")
        entry = CacheEntry.new(
            entry_type=CacheEntryType.OBSERVATION,
            content="Test observation",
            relevance_tags=["test"],
        )
        cache.add(entry)
        assert cache.entry_count == 1

    def test_get_by_type(self):
        cache = LocalCache(agent_name="TestAgent")
        for i in range(3):
            cache.add(CacheEntry.new(
                entry_type=CacheEntryType.OBSERVATION,
                content=f"Obs {i}",
            ))
        cache.add(CacheEntry.new(
            entry_type=CacheEntryType.VERIFIED_FACT,
            content="A fact",
        ))
        obs = cache.get_by_type(CacheEntryType.OBSERVATION)
        assert len(obs) == 3

    def test_get_recent(self):
        cache = LocalCache(agent_name="TestAgent")
        for i in range(5):
            cache.add(CacheEntry.new(
                entry_type=CacheEntryType.OBSERVATION,
                content=f"Obs {i}",
            ))
        recent = cache.get_recent(n=3)
        assert len(recent) == 3

    def test_search_by_similarity(self):
        cache = LocalCache(agent_name="TestAgent")
        cache.add(CacheEntry.new(
            entry_type=CacheEntryType.OBSERVATION,
            content="Carbon fiber production process",
            relevance_tags=["carbon_fiber"],
        ))
        cache.add(CacheEntry.new(
            entry_type=CacheEntryType.OBSERVATION,
            content="Image labeling pipeline",
            relevance_tags=["image"],
        ))
        results = cache.search_by_similarity("carbon fiber", top_k=2)
        assert len(results) >= 1

    def test_summarization_trigger(self):
        cache = LocalCache(agent_name="TestAgent", max_entries=10, summary_threshold=0.8)
        # Add entries with low importance — triggers summarization at entry 8
        for i in range(9):  # add enough to trigger and complete summarization
            cache.add(CacheEntry.new(
                entry_type=CacheEntryType.OBSERVATION,
                content=f"Obs {i}",
                importance=0.3,
            ))
        # After summarization, some entries should be compressed into summaries
        summaries = cache.get_by_type(CacheEntryType.SUMMARY)
        # The summarization removes low-importance entries and creates summaries
        assert cache.entry_count < 9  # some entries were compressed

    def test_context_string(self):
        cache = LocalCache(agent_name="TestAgent")
        cache.add(CacheEntry.new(
            entry_type=CacheEntryType.OBSERVATION,
            content="Test content for context string",
        ))
        ctx = cache.to_context_string(max_tokens=500)
        assert len(ctx) > 0