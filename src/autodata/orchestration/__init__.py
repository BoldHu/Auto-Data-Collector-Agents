"""Orchestration layer for AutoData multi-agent pipeline.

Provides:
  - EndToEndOrchestrator: coordinates central planner and worker agents
"""

from src.autodata.orchestration.end_to_end_orchestrator import EndToEndOrchestrator

__all__ = ["EndToEndOrchestrator"]
