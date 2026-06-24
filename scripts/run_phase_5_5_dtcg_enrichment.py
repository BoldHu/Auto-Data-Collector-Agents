"""Run Phase 5.5 DTCG enrichment.

Usage:
    python scripts/run_phase_5_5_dtcg_enrichment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.autodata.context_graph.phase_5_5_trace_enricher import build_enriched_trace, save_trace


def main():
    print("Building enriched DTCG trace...")
    result = build_enriched_trace()
    trace_path, packages_path, stats_path = save_trace(result)

    stats = result["statistics"]
    print(f"\n=== DTCG Enrichment Complete ===")
    print(f"Nodes: {stats['node_count']}")
    print(f"Edges: {stats['edge_count']}")
    print(f"Context packages: {stats['context_package_count']}")
    print(f"Estimated broadcast tokens: {stats['estimated_broadcast_tokens']}")
    print(f"Estimated DTCG tokens: {stats['estimated_dtcg_tokens']}")
    print(f"Context saving ratio: {stats['context_saving_ratio']:.2%}")
    print(f"\nTrace: {trace_path}")
    print(f"Packages: {packages_path}")
    print(f"Statistics: {stats_path}")


if __name__ == "__main__":
    main()
