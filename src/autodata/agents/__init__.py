"""Agent abstractions for AutoData multi-agent system.

Provides:
  - BaseAgent: shared interface for all agents
  - CentralPlanningAgent: Plan-and-Execute orchestrator
  - ReActAgent: Reasoning + Acting worker agent
  - ToolRegistry: tool management for ReAct agents
  - DataCollectionAgent: data acquisition agent
  - DataCleaningAgent: text/image cleaning agent
  - QualityVerificationAgent: independent quality critic
  - BenchmarkGenerationAgent: benchmark construction agent
  - ModelEvaluationAgent: model evaluation agent
  - ExamExtractionAgent: exam question extraction
  - ExamQualityAgent: exam quality verification
"""

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
from src.autodata.agents.data_collection_agent import DataCollectionAgent
from src.autodata.agents.data_cleaning_agent import DataCleaningAgent
from src.autodata.agents.quality_verification_agent import QualityVerificationAgent
from src.autodata.agents.benchmark_generation_agent import BenchmarkGenerationAgent
from src.autodata.agents.model_evaluation_agent import ModelEvaluationAgent
from src.autodata.agents.exam_extraction_agent import ExamExtractionAgent
from src.autodata.agents.exam_quality_agent import ExamQualityAgent

__all__ = [
    "AgentFramework",
    "AgentObservation",
    "BaseAgent",
    "CentralPlanningAgent",
    "PlanStep",
    "ReActAction",
    "ReActAgent",
    "ToolRegistry",
    "DataCollectionAgent",
    "DataCleaningAgent",
    "QualityVerificationAgent",
    "BenchmarkGenerationAgent",
    "ModelEvaluationAgent",
    "ExamExtractionAgent",
    "ExamQualityAgent",
]