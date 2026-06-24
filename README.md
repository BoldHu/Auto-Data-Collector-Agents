# AutoData: Dynamic Task-Context Graph for Long-Horizon Multi-Agent Data Construction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

**AutoData** is a multi-agent research system for automated domain data construction, corpus cleaning, supervised fine-tuning (SFT) sample generation, and benchmark generation. The case study domain is **carbon fiber**.

This project is designed as a research system with a clear technical contribution: **Dynamic Task-Context Graph (DTCG)** — a graph-based communication and context-management mechanism that replaces broadcast-style multi-agent communication. Each agent receives only the information relevant to its role, current task, local memory, and dependency neighborhood.

## Core Contribution

Traditional multi-agent systems often use broadcast-style communication where every agent receives all historical messages. This causes:

- Context overload and long prompts
- High token cost and duplicated reasoning
- Poor scalability in long-horizon tasks

**DTCG** solves this by managing long-term multi-agent communication using a **dynamic heterogeneous graph**. At each time step, the system constructs a role-specific context package for each agent by:

1. Retrieving candidate nodes from the agent-task neighborhood in the graph
2. Ranking by relevance, dependency, recency, and trust
3. Penalizing redundancy
4. Selecting under a token budget using a greedy MMR/knapsack approximation

## System Architecture

```
                    ┌──────────────────────┐
                    │  Central Planning     │
                    │  Agent (Plan-Execute) │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
     ┌────────▼──┐    ┌───────▼───┐    ┌───────▼───┐
     │ Data       │    │ Data      │    │ Benchmark │
     │ Collection │    │ Cleaning  │    │ Generation│
     │ Agent      │    │ Agent     │    │ Agent     │
     └────────────┘    └───────────┘    └───────────┘
              │                │                │
     ┌────────▼──┐    ┌───────▼───┐    ┌───────▼───┐
     │ Data       │    │ Quality   │    │ Model     │
     │ Annotation │    │ Verify    │    │ Eval &    │
     │ Agent      │    │ Agent     │    │ Finetune  │
     └────────────┘    └───────────┘    └───────────┘
```

### Agents

| Agent | Framework | Description |
|-------|-----------|-------------|
| Central Planning Agent | Plan-and-Execute | Task decomposition, coordination, graph management |
| Data Collection Agent | ReAct | Paper, patent, image, and material crawling |
| Data Cleaning Agent | ReAct | PDF/TXT/DOCX/CSV cleaning, OCR post-processing |
| Data Annotation Agent | ReAct | Pretraining corpus, SFT sample, benchmark candidate generation |
| Quality Verification Agent | ReAct | Duplication, hallucination, and consistency checks |
| Benchmark Generation Agent | ReAct | Four-stage benchmark construction (plan → generate → validate → evaluate) |
| Model Evaluation Agent | ReAct | Baseline evaluation and fine-tuning preparation |

### Dynamic Task-Context Graph (DTCG)

The DTCG is a heterogeneous graph `G_t = (V_t, E_t)` with:

- **Node types**: agent, task, artifact, memory, tool, constraint
- **Edge types**: task-dependency, agent-assignment, artifact-derived-from, context-relevance, quality-feedback, tool-usage, duplication/conflict, benchmark-source
- **Dynamic edge weights**: `w_ij = sigmoid(α1·Rel + α2·Dep + α3·Rec + α4·Trust - α5·Red - α6·Cost)`

Context selection maximizes relevance, dependency, and trust while minimizing redundancy and token cost under a budget constraint.

## Project Structure

```
.
├── CLAUDE.md                       # Project instructions and research spec
├── README.md                       # This file
├── environment.yml                 # Conda environment specification
├── requirements.txt                # Python dependencies
├── configs/                        # Configuration files
│   ├── agents/default.yaml         # Agent configurations
│   ├── benchmark/default.yaml      # Benchmark task taxonomy
│   ├── evaluation/                 # Evaluation configs
│   ├── finetuning/                 # LoRA/QLoRA training configs
│   └── paths.yaml                  # Data and output paths
├── src/autodata/                   # Main Python package
│   ├── agents/                     # Agent implementations
│   │   ├── base_agent.py           # BaseAgent, AgentObservation
│   │   ├── planning_agent.py       # CentralPlanningAgent
│   │   ├── react_agent.py          # ReActAgent, ToolRegistry
│   │   ├── data_collection_agent.py
│   │   ├── data_cleaning_agent.py
│   │   ├── benchmark_generation_agent.py
│   │   ├── quality_verification_agent.py
│   │   ├── exam_extraction_agent.py
│   │   └── model_evaluation_agent.py
│   ├── context_graph/              # DTCG core implementation
│   │   ├── graph_schema.py         # Node, Edge, DynamicTaskContextGraph
│   │   ├── context_selector.py     # ContextSelector with MMR/knapsack
│   │   ├── message_store.py        # Structured inter-agent messaging
│   │   ├── local_cache.py          # Per-agent local cache
│   │   └── pipeline_dtcg_integration.py
│   ├── tools/                      # Agent tools (OCR, document converter)
│   ├── pipelines/                  # Data processing pipelines
│   │   ├── text_cleaning_pipeline.py
│   │   ├── fast_text_cleaning_pipeline.py
│   │   ├── image_labeling_pipeline.py
│   │   ├── image_deduplicator.py
│   │   ├── exam_question_extraction_pipeline.py
│   │   ├── knowledge_extractor.py
│   │   └── sft_candidate_generator.py
│   ├── benchmark/                  # Benchmark construction
│   │   ├── benchmark_builder.py
│   │   ├── benchmark_schema.py
│   │   ├── benchmark_validator.py
│   │   ├── agent_task_generator.py
│   │   └── text_task_enhancer.py
│   ├── evaluation/                 # Model evaluation framework
│   │   ├── evaluation_runner.py
│   │   ├── system_ablation.py
│   │   ├── system_baselines.py
│   │   ├── llm_judge.py
│   │   ├── metric_calculator.py
│   │   └── unified_model_client.py
│   ├── finetuning/                 # Fine-tuning code (implemented, not executed)
│   │   ├── sft_data_builder.py
│   │   ├── sft_validator.py
│   │   ├── train_lora.py
│   │   └── leakage_detector.py
│   └── utils/                      # Shared utilities
│       ├── api_loader.py           # Safe API key loading
│       ├── model_client.py         # Xiaomi LLM client
│       ├── llm_api_loader.py       # Multi-provider LLM loader
│       ├── logging_utils.py        # Structured logging
│       └── io_utils.py             # Atomic JSON/JSONL I/O
├── scripts/                        # Phase execution scripts (~80 scripts)
│   ├── setup_env.sh                # One-command environment setup
│   ├── run_phase_*.py              # Phase runners
│   └── validate_phase_*.py         # Phase validators
├── tests/                          # Unit tests
└── docs/                           # Documentation
```

## Setup

### Prerequisites

- Python 3.10+
- Conda (recommended) or pip
- Xiaomi MiMo API key (configured in `LLM_API/`)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/BoldHu/Auto-Data-Collector-Agents.git
cd Auto-Data-Collector-Agents

# Option 1: Conda environment
conda env create -f environment.yml
conda activate autodata

# Option 2: Pip only
pip install -r requirements.txt

# Configure API keys (create LLM_API/ directory with your keys)
mkdir -p LLM_API
# Add your API configuration files to LLM_API/

# Verify setup
python -c "from src.autodata.utils.api_loader import load_xiaomi_config; print('Setup OK')"
```

### Configuration

API keys and model endpoints are configured in `LLM_API/` (not committed to Git). You need:

- `xiaomi_mimo.md` — Xiaomi MiMo model documentation and API config
- `llm_api.txt` — API keys for baseline evaluation models

## Execution Pipeline

The system runs in sequential phases:

| Phase | Description | Key Scripts |
|-------|-------------|-------------|
| **0** | Repository audit & architecture design | `run_phase_0_audit.py` |
| **1** | System foundation validation | `run_phase_1_smoke_test.py` |
| **2** | Text cleaning pipeline | `run_phase_2_text_cleaning.py` |
| **3** | Image labeling & deduplication | `run_phase_3_full_image_labeling.py` |
| **4** | Exam question extraction | `run_phase_4_exam_extraction.py` |
| **5** | Benchmark construction | `run_phase_5_benchmark_construction.py` |
| **6** | Baseline evaluation & ablation | `run_phase_6_baseline_evaluation.py` |
| **7** | SFT data preparation | `run_phase_7_build_sft_data.py` |
| **8** | Model evaluation & comparison | `run_phase_8_*.py` |

## Research Baselines

The experimental design compares:

1. **Single LLM** — Direct prompting with one large model
2. **Single ReAct agent** — One agent with tools, no multi-agent decomposition
3. **Plan-and-Execute without graph context** — Workers receive full history
4. **Broadcast multi-agent** — Every agent receives all messages
5. **Static-router multi-agent** — Fixed communication routes
6. **DTCG (proposed)** — Graph-based dynamic context selection

Additional lightweight baselines: AutoGen-style, CrewAI-style, LangGraph-style, CAMEL-style.

## Benchmark

The carbon-fiber benchmark includes:

- Domain knowledge QA
- Exam-style multiple-choice QA
- Short-answer QA
- Knowledge extraction
- Process planning & constraint satisfaction
- Causal reasoning & error diagnosis
- Source-grounded reasoning
- Multimodal image understanding

Each benchmark item preserves full source provenance and passes a four-stage validation pipeline (plan → generate → validate → evaluate).

## Key Constraints

- **Xiaomi MiMo models** are the default backbone for all agent execution, data cleaning, labeling, and generation
- **Other LLMs** (configured in `env_llm.txt`) are used only for baseline evaluation
- Fine-tuning code is implemented but **not executed** at the current stage
- All outputs are reproducible with structured intermediate files
- Reports are written in Chinese; code comments in English

## Citation

If you use this work, please cite:

```bibtex
@software{autodata2026,
  title={AutoData: Dynamic Task-Context Graph for Long-Horizon Multi-Agent Data Construction},
  author={BoldHu},
  year={2026},
  url={https://github.com/BoldHu/Auto-Data-Collector-Agents}
}
```

## License

MIT License
