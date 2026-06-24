"""Run Phase 3.6: Build DTCG trace for image labeling pipeline.

Usage:
    python scripts/run_phase_3_6_dtcg_trace.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.autodata.context_graph.phase_3_trace import main as build_trace
from src.autodata.utils.logging_utils import get_logger

logger = get_logger("run_phase_3_6")


def main():
    logger.info("Starting Phase 3.6: DTCG Trace")
    result = build_trace()

    print(f"\n=== Phase 3.6 DTCG Trace Complete ===")
    print(f"Nodes: {result['node_count']}")
    print(f"Edges: {result['edge_count']}")
    print(f"Trace file: {result['trace_path']}")
    print(f"Packages file: {result['packages_path']}")


if __name__ == "__main__":
    main()