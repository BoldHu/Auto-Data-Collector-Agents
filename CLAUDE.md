# CLAUDE.md

## 0. Project Identity

You are working on an academic multi-agent research project for automated domain data construction, domain corpus construction, supervised fine-tuning sample generation, and benchmark generation. The target is a high-level AI conference paper. The case study domain is carbon fiber.

The project is not merely an engineering pipeline. It must be designed as a research system with a clear technical contribution:

1. Long-horizon multi-agent coordination.
2. Graph-based context management and communication routing.
3. Automated domain data collection, cleaning, annotation, benchmark generation, and model evaluation.
4. Carbon-fiber case study demonstrating the whole workflow.

The core research problem is:

Traditional multi-agent systems often use broadcast-style communication. Every agent receives all historical messages, causing context overload, long prompts, high token cost, duplicated reasoning, and poor scalability in long-horizon tasks. This project should design a graph-based communication and context-management mechanism so that each agent only receives the information relevant to its role, current task, local memory, and dependency neighborhood.

The system should ultimately show that:
- the proposed multi-agent system performs better on long-horizon data-construction tasks;
- collected data has low duplication and high domain relevance;
- generated benchmark is useful for evaluating domain agents;
- after training on the collected/constructed data, smaller models can outperform larger general models on the domain benchmark;
- the benchmark itself is also generated and validated through the multi-agent system.

At the current stage, do NOT actually run fine-tuning. Implement fine-tuning code only. Focus on data cleaning, annotation, benchmark construction, and baseline evaluation using models in `env_llm.txt`.

---

## 1. Repository Context

Important project files and folders:

### 1.1 LLM_API

`LLM_API/` contains model documentation and API keys.

Files:
- `LLM_API/xiaomi_mimo.md`: Xiaomi model documentation.
- `LLM_API/xiaomi_llm_api.txt`: Xiaomi API key and endpoint. Xiaomi models should be used as the main agent models. The quota can be treated as unlimited.
- `LLM_API/env_llm.txt`: API configuration for other large models. These models are used only for experimental evaluation and baselines, not for routine data cleaning or generation.

Rules:
- Never print, expose, commit, or log API keys.
- Load keys through environment variables or local config loaders.
- Create `.gitignore` rules for API files, cache files, temporary outputs, and generated logs containing model responses.
- Xiaomi models should be used as the default model for agent execution, data cleaning, data labeling, OCR post-processing, benchmark generation, and internal critique.
- Models in `env_llm.txt` should be used only in the evaluation harness as baseline models.

### 1.2 text_raw_data

`text_raw_data/` contains two folders of preliminary text corpus extracted from books. The text is noisy because it was obtained by OCR and page-by-page cleaning.

Tasks:
- Build a text-cleaning pipeline.
- Use Xiaomi LLMs to clean noisy OCR text.
- Convert raw text into:
  - clean pretraining corpus;
  - structured domain knowledge units;
  - candidate supervised fine-tuning samples;
  - source-grounded benchmark material.

### 1.3 imgs_raw_data

`imgs_raw_data/carbon_fiber_mm/` contains many folders of carbon-fiber images. Folder names correspond to keyword combinations used during crawling.

`imgs_raw_data/carbon_fiber_corpus_5911_6000.jsonl` contains crawling metadata.

Tasks:
- Combine image folders and metadata.
- Build a multimodal labeling pipeline.
- Assign image labels useful for benchmark construction and future multimodal fine-tuning.
- Preserve provenance: image path, search keyword, metadata, inferred category, confidence, and potential benchmark usage.

### 1.4 exam_raw_data

`exam_raw_data/` contains carbon-fiber-related Chinese exam papers in mixed formats.

Tasks:
- Build an adaptive document parser.
- Support PDF, image, Word, text, CSV, and scanned documents.
- Use OCR when necessary.
- Extract clean questions, options, answers, explanations, knowledge points, difficulty, and source.
- Use these questions as part of the carbon-fiber benchmark.

### 1.5 Existing Python crawler files

There are existing `.py` files for crawling image data from Bing using combined keywords.

Tasks:
- Do not discard these scripts.
- Refactor them into reusable tools/skills for the Data Collection Agent.
- Wrap them as callable modules under the new project architecture.
- Add clean configuration, logging, retry, deduplication, and metadata persistence.

---

## 2. Target System Architecture

The project should implement a multi-agent system with one central planning agent and multiple task-oriented worker agents.

### 2.1 Central Planning Agent

The Central Planning Agent uses a Plan-and-Execute framework.

Responsibilities:
- Interpret the domain goal.
- Decompose long-horizon data-construction tasks.
- Decide what data to collect.
- Decide what data to clean.
- Decide what samples to annotate.
- Decide what benchmark tasks should be generated.
- Coordinate all worker agents.
- Manage graph-based communication and context routing.
- Maintain and update the dynamic task-context graph.
- Enforce data-generation constraints.
- Trigger independent critique and quality verification.
- Produce phase-level reports.

The Central Planning Agent is the core technical contribution of the project. Its context-management method must be implemented as an algorithm, not merely as prompt engineering.

### 2.2 Worker Agents

Worker agents use the ReAct framework.

Required worker agents:
1. Data Collection Agent
   - Collect papers, patents, books, teaching materials, images, and other carbon-fiber domain data.
   - Current priority: supplement paper data.
   - Refactor existing image crawler scripts as tools.

2. Data Cleaning Agent
   - Clean PDF, TXT, DOC/DOCX, CSV, image, and scanned files.
   - Include OCR capability.
   - Convert noisy text into clean pretraining corpus and structured text units.

3. Data Annotation Agent
   - Build task-specific annotation datasets.
   - Generate pretraining corpus, SFT samples, and benchmark candidates according to the Central Planning Agent’s task design.

4. Data Quality Verification Agent
   - Evaluate clarity, completeness, consistency, feasibility, and complexity.
   - Detect duplication, hallucination, low-quality content, source mismatch, and weak domain relevance.

5. Benchmark Generation Agent
   - Build the carbon-fiber benchmark.
   - Follow a BENCHAGENTS-inspired four-stage process:
     - planning;
     - generation;
     - validation;
     - evaluation.
   - Cover language and visual modalities when data allows.
   - Include tasks related to planning, constraint satisfaction, causal reasoning, domain knowledge QA, multimodal recognition, and exam-style reasoning.

6. Model Fine-tuning and Evaluation Agent
   - Implement fine-tuning code but do not run it yet.
   - Evaluate baseline models from `env_llm.txt`.
   - Compare single LLMs, simple multi-agent baselines, broadcast-based multi-agent baselines, and the proposed graph-context multi-agent system.

---

## 3. Core Research Method: Dynamic Task-Context Graph

Implement a graph-based context-management framework.

Suggested method name:

Dynamic Task-Context Graph for Long-Horizon Multi-Agent Data Construction

Abbreviation:

DTCG

The method should manage long-term multi-agent communication using a dynamic heterogeneous graph.

### 3.1 Graph Definition

At time step `t`, define a dynamic heterogeneous graph:

`G_t = (V_t, E_t)`

Node types:
- agent nodes: worker agents and central planner;
- task nodes: current task, subtask, pending task, completed task;
- artifact nodes: files, cleaned documents, generated samples, benchmark items, evaluation results;
- memory nodes: summarized context chunks, prior decisions, constraints, error reports;
- tool nodes: crawler, OCR, parser, LLM API caller, evaluator, deduplicator;
- constraint nodes: domain constraints, quality rules, benchmark rules, provenance rules.

Edge types:
- task-dependency edge;
- agent-assignment edge;
- artifact-derived-from edge;
- context-relevance edge;
- quality-feedback edge;
- tool-usage edge;
- duplication/conflict edge;
- benchmark-source edge.

Each edge has a dynamic weight:

`w_ij^(t) = sigmoid(α1 Rel(i,j) + α2 Dep(i,j) + α3 Rec(i,j,t) + α4 Trust(i,j) - α5 Red(i,j) - α6 Cost(j))`

Where:
- `Rel(i,j)` is semantic relevance between current agent/task query and node `j`;
- `Dep(i,j)` measures task dependency;
- `Rec(i,j,t)` is recency with time decay;
- `Trust(i,j)` is source quality or verification score;
- `Red(i,j)` is redundancy with already selected context;
- `Cost(j)` is estimated token or computation cost.

### 3.2 Context Selection Objective

For each agent `a` at step `t`, select a context subset `S_a^t` from graph neighborhood `N(a,t)` under a token budget `B_a`:

Maximize:

`Σ relevance(v) + β Σ dependency(v) + γ Σ trust(v) - λ redundancy(S) - μ token_cost(S)`

Subject to:

`Σ token_cost(v) <= B_a`

Use a greedy MMR/knapsack approximation:
1. retrieve candidate nodes from the agent-task neighborhood;
2. rank by relevance, dependency, recency, and trust;
3. penalize redundancy;
4. select until token budget is reached;
5. compress selected nodes into a role-specific context package.

This is the key alternative to broadcast-style multi-agent communication.

### 3.3 Local Cache

Each agent should maintain a local cache:

`C_a^t = {recent observations, tool results, verified facts, unresolved issues, agent-specific summaries}`

The local cache should be:
- compact;
- role-specific;
- updated after each action;
- periodically summarized;
- linked to graph nodes;
- retrievable by semantic similarity and graph dependency.

### 3.4 Message Schema

All inter-agent messages should be stored in structured format.

Use a JSONL-style schema:

```json
{
  "message_id": "...",
  "timestamp": "...",
  "sender_agent": "...",
  "receiver_agent": "...",
  "task_id": "...",
  "message_type": "plan|observation|tool_result|critique|constraint|summary|decision",
  "content": "...",
  "artifact_refs": ["..."],
  "source_refs": ["..."],
  "embedding_id": "...",
  "quality_score": null,
  "relevance_tags": ["..."],
  "token_estimate": 0,
  "visibility": "global|local|restricted"
}
3.5 Context Package Schema
Before invoking an agent, the system should construct a context package:
{
  "agent_name": "...",
  "task_id": "...",
  "current_goal": "...",
  "allowed_tools": ["..."],
  "relevant_plan": "...",
  "selected_memory": [],
  "selected_artifacts": [],
  "constraints": [],
  "quality_requirements": [],
  "output_schema": {},
  "forbidden_actions": []
}
Agents should not receive full history. They should receive only graph-selected context.
________________________________________
4. Research Baselines
The experimental design should compare the proposed graph-context system with:
1.	Single LLM baseline
o	Direct prompting with one large model.
2.	Single ReAct agent baseline
o	One agent with tools but without multi-agent decomposition.
3.	Plan-and-Execute without graph context
o	Central planner plus workers, but each worker receives full history or manually summarized history.
4.	Broadcast multi-agent baseline
o	Every agent receives all messages.
5.	Static-router multi-agent baseline
o	Fixed communication routes based on agent role, without dynamic graph update.
6.	Recent multi-agent framework-style baselines
o	AutoGen-style group chat baseline.
o	CrewAI-style role pipeline baseline.
o	LangGraph-style supervisor baseline.
o	CAMEL/AgentVerse-style role interaction baseline if implementation is feasible.
Do not spend excessive time implementing complete third-party systems. Lightweight faithful baselines are acceptable if clearly documented.
________________________________________
5. Data Quality Metrics
Data collection and cleaning should be evaluated using:
1.	Clarity
o	Text readability.
o	OCR noise reduction.
o	Format correctness.
o	Question/item clarity.
2.	Completeness
o	Whether source content is preserved.
o	Whether important fields are missing.
o	Whether multimodal metadata is complete.
3.	Consistency
o	Internal consistency.
o	Label consistency.
o	Source-label consistency.
o	Duplicate/conflict detection.
4.	Feasibility
o	Whether generated samples can be used for pretraining, SFT, or benchmark evaluation.
o	Whether the pipeline can run end-to-end.
5.	Complexity
o	Task difficulty.
o	Reasoning depth.
o	Multistep dependency.
o	Cross-document or cross-modal requirement.
Additional metrics:
•	domain relevance;
•	duplication rate;
•	source coverage;
•	provenance completeness;
•	human-review pass rate if manual review exists;
•	token cost;
•	execution time;
•	API call count;
•	average context length;
•	quality per unit cost.
________________________________________
6. Benchmark Requirements
The benchmark should be constructed on the carbon-fiber case.
Benchmark sources:
•	cleaned text corpus from text_raw_data;
•	extracted questions from exam_raw_data;
•	image data from imgs_raw_data/carbon_fiber_mm;
•	metadata from carbon_fiber_corpus_5911_6000.jsonl;
•	newly collected paper data;
•	generated and validated task samples.
Benchmark task types should include:
1.	Domain knowledge QA.
2.	Exam-style multiple-choice QA.
3.	Short-answer QA.
4.	Knowledge extraction.
5.	Process planning.
6.	Constraint satisfaction.
7.	Causal reasoning.
8.	Error diagnosis.
9.	Source-grounded reasoning.
10.	Multimodal image understanding when image labels are reliable.
Each benchmark item should include:
{
  "item_id": "...",
  "task_type": "...",
  "modality": "text|image|multimodal",
  "question": "...",
  "options": [],
  "answer": "...",
  "explanation": "...",
  "source_refs": [],
  "difficulty": "easy|medium|hard",
  "required_knowledge": [],
  "reasoning_type": [],
  "quality_scores": {
    "clarity": null,
    "completeness": null,
    "consistency": null,
    "feasibility": null,
    "complexity": null
  },
  "validation_status": "pending|passed|failed",
  "validator_notes": ""
}
The benchmark construction process should follow four stages:
1.	Planning.
2.	Generation.
3.	Validation.
4.	Evaluation.
For each stage, generate a report.
________________________________________
7. Experimental Requirements
The final experimental reports should include:
7.1 Experimental Setting
Describe:
•	Xiaomi MiMo model as the default agent backbone.
•	Other models from env_llm.txt as baseline models.
•	Agent settings:
o	number of agents;
o	tool availability;
o	context budget;
o	graph update frequency;
o	local cache size;
o	maximum iterations;
o	quality thresholds.
•	Baselines:
o	single LLM;
o	ReAct single agent;
o	broadcast multi-agent;
o	static-router multi-agent;
o	graph-context multi-agent;
o	lightweight versions of recent multi-agent systems.
7.2 Benchmark Statistics
Use tables and figures to report:
•	number of documents;
•	number of cleaned text chunks;
•	number of images;
•	number of labeled images;
•	number of extracted exam questions;
•	number of benchmark items;
•	task type distribution;
•	modality distribution;
•	difficulty distribution;
•	source distribution;
•	quality score distribution;
•	duplication rate.
7.3 Model Evaluation
Evaluate all models in env_llm.txt on the benchmark.
Report:
•	accuracy;
•	exact match where applicable;
•	F1 where applicable;
•	reasoning pass rate;
•	format validity;
•	hallucination rate;
•	average token cost;
•	average latency;
•	error types.
7.4 Test-Time Compute
Compare performance under different compute budgets:
•	small context budget;
•	medium context budget;
•	large context budget;
•	small number of agent steps;
•	larger number of agent steps;
•	different retrieval/top-k settings.
Report:
•	performance vs. token cost;
•	performance vs. time;
•	performance vs. number of agent messages;
•	graph-context system vs. broadcast system.
7.5 Carbon-Fiber Case Study
Show the end-to-end workflow:
1.	raw data;
2.	cleaned corpus;
3.	labeled samples;
4.	benchmark items;
5.	baseline evaluation;
6.	generated reports;
7.	future fine-tuning code.
________________________________________
8. Required Project Structure
Create or refactor the project into a clear structure:
.
├── CLAUDE.md
├── README.md
├── environment.yml
├── requirements.txt
├── .gitignore
├── configs/
│   ├── agents/
│   ├── models/
│   ├── benchmark/
│   └── paths.yaml
├── src/
│   └── autodata/
│       ├── agents/
│       ├── context_graph/
│       ├── tools/
│       ├── pipelines/
│       ├── data_quality/
│       ├── benchmark/
│       ├── evaluation/
│       ├── finetuning/
│       └── utils/
├── scripts/
│   ├── setup_env.sh
│   ├── run_phase_0_audit.py
│   ├── run_phase_1_clean_text.py
│   ├── run_phase_2_label_images.py
│   ├── run_phase_3_extract_exam.py
│   ├── run_phase_4_build_benchmark.py
│   ├── run_phase_5_evaluate_baselines.py
│   └── run_phase_6_prepare_finetuning.py
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── benchmark/
│   ├── sft/
│   └── reports/
├── reports/
│   ├── phase_0_audit/
│   ├── phase_1_system_design/
│   ├── phase_2_text_cleaning/
│   ├── phase_3_image_labeling/
│   ├── phase_4_exam_extraction/
│   ├── phase_5_benchmark_construction/
│   ├── phase_6_model_evaluation/
│   └── paper_ready/
├── tests/
└── docs/
Do not delete user data. Do not overwrite raw data.
________________________________________
9. Execution Rules
Follow these rules strictly:
1.	Work phase by phase.
2.	Do not implement everything in one pass.
3.	At the end of every phase, output:
o	completed work;
o	created/modified files;
o	how to run;
o	preliminary results;
o	known problems;
o	next recommended phase.
4.	Stop after each phase and wait for the next user instruction.
5.	Use Xiaomi LLMs heavily for data cleaning, labeling, and generation.
6.	Use env_llm.txt models only for baseline evaluation.
7.	Do not run fine-tuning at the current stage.
8.	Implement fine-tuning code but leave execution disabled by default.
9.	Keep all outputs reproducible.
10.	Save structured intermediate files, not only free-text reports.
11.	Every benchmark item must preserve source provenance.
12.	Every generated item must pass validation before entering the final benchmark.
13.	The graph-context mechanism must be implemented as code and documented as a research method.
14.	Reports should be written in Chinese unless code comments or schemas require English.
15.	Academic framing should emphasize method novelty, experimental controllability, benchmark validity, and ablation studies.
________________________________________
10. First Priority
The first priority is NOT to clean all data immediately.
The first priority is:
1.	audit the repository;
2.	inspect available files and formats;
3.	read Xiaomi model documentation;
4.	design the project architecture;
5.	create the conda environment specification;
6.	create safe API loading utilities;
7.	create the graph-context method skeleton;
8.	output a Phase 0 audit report;
9.	propose the next phase.
Do not start large-scale data processing in Phase 0.
