"""Phase 1 smoke test — validate system foundation.

Verifies that all core modules can be instantiated and work together:
1. Environment and dependencies
2. API configuration loading
3. Model client creation
4. DTCG graph operations
5. Context selection
6. Message store
7. Local cache
8. Agent instantiation
9. Agent messaging
10. Baseline model loading

Outputs structured validation results to console and JSON report.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ── Test result tracking ──────────────────────────────────────────

results: dict[str, dict] = {}
total_passed = 0
total_failed = 0


def record_test(group: str, name: str, passed: bool, detail: str = "") -> None:
    """Record a test result."""
    global total_passed, total_failed
    key = f"{group}/{name}"
    results[key] = {"passed": passed, "detail": detail}
    if passed:
        total_passed += 1
        print(f"  [PASS] {name}")
    else:
        total_failed += 1
        print(f"  [FAIL] {name}: {detail}")


# ── 1. Environment and dependencies ───────────────────────────────

def test_environment():
    print("\n[1] Environment and dependencies")
    G = "env"

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    record_test(G, "python_version", sys.version_info.major == 3, py_ver)

    for display_name, import_name in [
        ("openai", "openai"), ("anthropic", "anthropic"),
        ("networkx", "networkx"), ("yaml", "yaml"),
        ("loguru", "loguru"), ("pandas", "pandas"),
        ("tenacity", "tenacity"),
    ]:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "OK")
            record_test(G, f"import_{display_name}", True, f"v{ver}")
        except ImportError as e:
            record_test(G, f"import_{display_name}", False, str(e))


# ── 2. API configuration loading ────────────────────────────────

def test_api_loading():
    print("\n[2] API configuration loading")
    G = "api"

    try:
        from src.autodata.utils.api_loader import load_xiaomi_config, load_baseline_configs
        cfg = load_xiaomi_config()
        record_test(G, "xiaomi_config_loaded", True, f"model={cfg.default_model}, key_loaded={bool(cfg.api_key)}")
        record_test(G, "xiaomi_default_model", cfg.default_model == "mimo-v2.5-pro", f"default_model={cfg.default_model}")
        record_test(G, "xiaomi_openai_url", bool(cfg.openai_base_url), cfg.openai_base_url)
        record_test(G, "xiaomi_anthropic_url", bool(cfg.anthropic_base_url), cfg.anthropic_base_url)
    except Exception as e:
        record_test(G, "xiaomi_config_loaded", False, str(e))

    try:
        configs = load_baseline_configs()
        record_test(G, "baseline_configs_loaded", True, f"{len(configs)} models")
        record_test(G, "baseline_models_count", len(configs) >= 5, f"models: {[c.name for c in configs]}")
    except Exception as e:
        record_test(G, "baseline_configs_loaded", False, str(e))


# ── 3. Model client creation ──────────────────────────────────────

def test_model_client():
    print("\n[3] Model client creation")
    G = "model_client"

    try:
        from src.autodata.utils.model_client import XiaomiModelClient, ChatResponse
        client = XiaomiModelClient()
        record_test(G, "model_client_created", True, f"model={client.default_model}")
        record_test(G, "model_client_default_model", client.default_model == "mimo-v2.5-pro", f"default_model={client.default_model}")
    except Exception as e:
        record_test(G, "model_client_created", False, str(e))
        return

    try:
        response = client.chat(
            messages=[{"role": "user", "content": "Say 'OK' if you can respond."}],
            max_completion_tokens=64,
            model="mimo-v2.5-pro",
        )
        record_test(G, "model_client_api_call", True, f"response_len={len(response.content)}, tokens={response.total_tokens}")
        record_test(G, "model_client_response_type", isinstance(response, ChatResponse), str(type(response)))
    except Exception as e:
        record_test(G, "model_client_api_call", False, str(e)[:200])


# ── 4. DTCG graph operations ────────────────────────────────────

def test_dtcg_graph():
    print("\n[4] DTCG graph operations")
    G = "dtcg"

    try:
        from src.autodata.context_graph.graph_schema import (
            DynamicTaskContextGraph, Node, NodeType, Edge, EdgeType, TaskStatus
        )

        g = DynamicTaskContextGraph()
        record_test(G, "graph_created", True, f"t={g.t}")

        agent_node = Node(node_id="agent_test", node_type=NodeType.AGENT, name="TestAgent", properties={"framework": "react"})
        task_node = Node(node_id="task_test", node_type=NodeType.TASK, name="Test task", properties={"status": "pending"})
        g.add_node(agent_node)
        g.add_node(task_node)
        record_test(G, "add_nodes", g.node_count == 2, f"node_count={g.node_count}")

        edge = Edge.new(
            source_id="agent_test", target_id="task_test", edge_type=EdgeType.AGENT_ASSIGNMENT,
        )
        edge.relevance_score = 0.9
        edge.dependency_score = 0.8
        edge.recency_score = 0.7
        edge.trust_score = 0.6
        edge.redundancy_score = 0.1
        edge.cost_score = 0.05
        weight = edge.compute_weight()
        g.add_edge(edge)
        record_test(G, "add_edge", True, f"weight={weight:.4f}")
        record_test(G, "edge_weight_positive", weight > 0, f"weight={weight:.4f}")

        neighbors = g.get_neighbors("agent_test")
        record_test(G, "neighbors", len(neighbors) >= 1, f"neighbors={len(neighbors)}")

        g_dict = g.to_dict()
        record_test(G, "serialization", "nodes" in g_dict, f"keys={list(g_dict.keys())}")

    except Exception as e:
        record_test(G, "graph_created", False, str(e))


# ── 5. Context selection ──────────────────────────────────────────

def test_context_selector():
    print("\n[5] Context selection")
    G = "context"

    try:
        from src.autodata.context_graph.context_selector import ContextSelector, ContextSelectorConfig
        from src.autodata.context_graph.graph_schema import DynamicTaskContextGraph, Node, NodeType

        g = DynamicTaskContextGraph()
        for i in range(5):
            g.add_node(Node(
                node_id=f"memory_{i}", node_type=NodeType.MEMORY, name=f"Memory {i}",
                properties={"content": f"Test memory content {i}", "importance": 0.5 + i * 0.1},
            ))

        selector = ContextSelector()
        record_test(G, "selector_created", True)

        package = selector.select_context(
            graph=g, agent_node_id="memory_0",
            task_id="task_1",
            token_budget=2000, current_goal="Test context selection",
        )
        record_test(G, "selection_done", True, f"items={len(package.selected_memory)}")
        record_test(G, "package_type", hasattr(package, "agent_name"), str(type(package).__name__))

    except Exception as e:
        record_test(G, "selector_created", False, str(e))


# ── 6. Message store ──────────────────────────────────────────────

def test_message_store():
    print("\n[6] Message store")
    G = "msg_store"

    try:
        from src.autodata.context_graph.message_store import Message, MessageStore, MessageType, Visibility

        store = MessageStore()
        record_test(G, "store_created", True)

        msg = Message(
            message_id="msg_test_001", timestamp=time.time(),
            sender_agent="TestAgent1", receiver_agent="TestAgent2",
            task_id="task_001", message_type=MessageType.OBSERVATION,
            content="Test observation message", relevance_tags=["test"],
            token_estimate=50, visibility=Visibility.LOCAL,
        )
        store.add(msg)
        record_test(G, "add_message", True)

        received = store.get_by_receiver("TestAgent2")
        record_test(G, "get_by_receiver", len(received) >= 1, f"count={len(received)}")

        sent = store.get_by_sender("TestAgent1")
        record_test(G, "get_by_sender", len(sent) >= 1, f"count={len(sent)}")

    except Exception as e:
        record_test(G, "store_created", False, str(e))


# ── 7. Local cache ────────────────────────────────────────────────

def test_local_cache():
    print("\n[7] Local cache")
    G = "cache"

    try:
        from src.autodata.context_graph.local_cache import CacheEntry, CacheEntryType, LocalCache

        cache = LocalCache(agent_name="TestAgent", max_entries=50, max_tokens=2000)
        record_test(G, "cache_created", True)

        for i in range(5):
            cache.add(CacheEntry.new(
                entry_type=CacheEntryType.OBSERVATION,
                content=f"Test observation {i}",
                relevance_tags=["test", f"step_{i}"],
                importance=0.5 + i * 0.1,
            ))
        record_test(G, "add_entries", cache.entry_count == 5, f"count={cache.entry_count}")

        obs_entries = cache.get_by_type(CacheEntryType.OBSERVATION)
        record_test(G, "get_by_type", len(obs_entries) == 5, f"count={len(obs_entries)}")

        recent = cache.get_recent(n=3)
        record_test(G, "get_recent", len(recent) == 3, f"count={len(recent)}")

        similar = cache.search_by_similarity("Test observation", top_k=3)
        record_test(G, "similarity_search", len(similar) >= 1, f"count={len(similar)}")

        ctx_str = cache.to_context_string(max_tokens=500)
        record_test(G, "context_string", len(ctx_str) > 0, f"len={len(ctx_str)}")

    except Exception as e:
        record_test(G, "cache_created", False, str(e))


# ── 8. Agent instantiation ────────────────────────────────────────

def test_agents():
    print("\n[8] Agent instantiation")
    G = "agents"

    try:
        from src.autodata.agents import CentralPlanningAgent, ReActAgent, ToolRegistry
        from src.autodata.context_graph.message_store import MessageStore

        store = MessageStore()
        planner = CentralPlanningAgent(message_store=store)
        record_test(G, "planner_created", True, f"name={planner.name}, model={planner.model}")
        record_test(G, "planner_model", planner.model == "mimo-v2.5-pro", planner.model)

        registry = ToolRegistry()
        registry.register("test_tool", "A test tool for validation")
        worker = ReActAgent(name="DataCleaningAgent", message_store=store, tools=registry.list_tools())
        record_test(G, "worker_created", True, f"name={worker.name}, model={worker.model}")
        record_test(G, "worker_has_cache", worker.cache.entry_count == 0, f"cache_entries={worker.cache.entry_count}")
        record_test(G, "worker_has_tools", worker.tool_registry.has_tool("test_tool"), "test_tool registered")

    except Exception as e:
        record_test(G, "planner_created", False, str(e)[:200])


# ── 9. Agent messaging ────────────────────────────────────────────

def test_agent_messaging():
    print("\n[9] Agent messaging")
    G = "messaging"

    try:
        from src.autodata.agents import CentralPlanningAgent, ReActAgent
        from src.autodata.context_graph.message_store import MessageStore

        store = MessageStore()
        planner = CentralPlanningAgent(message_store=store)
        worker = ReActAgent(name="DataCleaningAgent", message_store=store)

        msg = planner.send_message(
            receiver="DataCleaningAgent",
            content="Clean OCR text from book_001.json",
            task_id="task_clean_001",
        )
        record_test(G, "planner_send", True, f"msg_id={msg.message_id}")

        received = worker.receive_messages()
        record_test(G, "worker_receive", len(received) >= 1, f"count={len(received)}")

        reply = worker.send_message(
            receiver="CentralPlanningAgent",
            content="OCR text cleaning started for book_001.json",
            task_id="task_clean_001",
        )
        record_test(G, "worker_reply", True, f"msg_id={reply.message_id}")

        planner_received = planner.receive_messages()
        record_test(G, "planner_receive", len(planner_received) >= 1, f"count={len(planner_received)}")

    except Exception as e:
        record_test(G, "planner_send", False, str(e)[:200])


# ── 10. Baseline model loading ────────────────────────────────────

def test_baseline_models():
    print("\n[10] Baseline model loading")
    G = "baseline"

    try:
        from src.autodata.utils.baseline_model_loader import load_baseline_models, BaselineModelRunner

        models = load_baseline_models()
        record_test(G, "models_loaded", len(models) >= 5, f"count={len(models)}")

        if models:
            runner = BaselineModelRunner(models[0])
            record_test(G, "runner_created", True, f"model={runner.model_config.name}")
            record_test(G, "runner_display_name", bool(runner.display_name), runner.display_name)

            thinking_models = [m.name for m in models if m.supports_thinking]
            record_test(G, "thinking_models", len(thinking_models) >= 2, f"thinking_models={thinking_models}")

    except Exception as e:
        record_test(G, "models_loaded", False, str(e)[:200])


# ── Main ────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("AutoData Phase 1 — Smoke Test")
    print("=" * 60)

    test_environment()
    test_api_loading()
    test_model_client()
    test_dtcg_graph()
    test_context_selector()
    test_message_store()
    test_local_cache()
    test_agents()
    test_agent_messaging()
    test_baseline_models()

    print("\n" + "=" * 60)
    print(f"Results: {total_passed} passed, {total_failed} failed")
    print("=" * 60)

    report = {
        "phase": "1_smoke_test",
        "timestamp": time.time(),
        "total_passed": total_passed,
        "total_failed": total_failed,
        "results": results,
    }

    report_dir = PROJECT_ROOT / "reports" / "phase_1_system_design"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "SMOKE_TEST_RESULTS.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nReport saved to: {report_path}")

    if total_failed > 0:
        print(f"\nWARNING: {total_failed} tests failed. Review details above.")
    else:
        print("\nAll tests passed! System foundation is validated.")

    return total_failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)