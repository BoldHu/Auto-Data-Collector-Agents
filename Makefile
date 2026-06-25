.PHONY: help compile test schema lineage sft-validate bench-validate recompute smoke dtcg-ablation freeze-manifest final-gate claim-check human-audit-validate cfbench-v1-validate all clean

# ── Clean-room-safe defaults ──────────────────────────────────────────
# Every target writes only to OUT (default: build/validation/<target>)
# No target writes to data/registry, data/reports, or project root
OUT ?= build/validation
PLANNING_MODE ?= static
PYTHONPATH_CMD = PYTHONPATH=.
PYTHON = python
PYTEST = python -m pytest
PYTEST_FLAGS = -p no:cacheprovider -o cache_dir=/dev/null

help: ## Show this help message
	@echo "CFDataConstruct Reproducibility Commands"
	@echo "========================================="
	@echo ""
	@echo "  compile             Compile-check all Python files"
	@echo "  test                Run safe test suite (no external API)"
	@echo "  schema              Validate artifact schemas"
	@echo "  lineage             Validate artifact lineage"
	@echo "  sft-validate        Validate SFT v4 dataset"
	@echo "  bench-validate      Validate CFBench v1"
	@echo "  recompute           Recompute final tables from raw outputs"
	@echo "  smoke               Run end-to-end smoke demo"
	@echo "  dtcg-ablation       Run DTCG component ablation"
	@echo "  freeze-manifest     Generate reproducibility manifest"
	@echo "  final-gate          Run final verification gate"
	@echo "  claim-check         Check claim registry for prohibited phrases"
	@echo "  human-audit-validate Validate human audit status"
	@echo "  cfbench-v1-validate Validate CFBench v1.0"
	@echo "  all                 Run all validation steps"
	@echo "  clean               Remove build artifacts"
	@echo ""
	@echo "Options:"
	@echo "  OUT=<dir>           Output directory (default: build/validation)"
	@echo "  PLANNING_MODE=<m>   Planning mode for smoke: static|central|cached-central (default: static)"

compile: ## Compile-check all Python files
	@echo "=== Compile Check ==="
	@mkdir -p $(OUT)
	@$(PYTHONPATH_CMD) PYTHONPYCACHEPREFIX=$(OUT)/pycache $(PYTHON) scripts/check_compile.py

test: ## Run safe test suite (no external API calls)
	@echo "=== Running Tests ==="
	@mkdir -p $(OUT)
	@$(PYTHONPATH_CMD) PYTHONDONTWRITEBYTECODE=1 $(PYTEST) \
		tests/test_agents.py \
		tests/test_agent_instantiation_all.py \
		tests/test_agent_tool_dispatch.py \
		tests/test_context_graph.py \
		tests/test_behavioral.py \
		-v --tb=short $(PYTEST_FLAGS) \
		--junitxml=$(OUT)/test_results.xml

schema: ## Validate artifact schemas
	@echo "=== Schema Validation ==="
	@mkdir -p $(OUT)
	@$(PYTHONPATH_CMD) $(PYTHON) scripts/check_schema.py

lineage: ## Validate artifact lineage
	@echo "=== Lineage Validation ==="
	@mkdir -p $(OUT)
	@$(PYTHONPATH_CMD) $(PYTHON) scripts/validate_artifact_lineage.py --output $(OUT)/lineage_report.json --data-dir data

sft-validate: ## Validate SFT v4 dataset
	@echo "=== SFT v4 Validation ==="
	@mkdir -p $(OUT)
	@$(PYTHONPATH_CMD) $(PYTHON) scripts/validate_final_sft_v4.py --output $(OUT)/sft_v4_validation.json

bench-validate: ## Validate CFBench v1
	@echo "=== CFBench v1 Validation ==="
	@mkdir -p $(OUT)
	@$(PYTHONPATH_CMD) $(PYTHON) scripts/validate_cfbench_v1.py --output $(OUT)/cfbench_v1_validation.json

recompute: ## Recompute final tables from raw outputs
	@echo "=== Recomputing Tables ==="
	@mkdir -p $(OUT)
	@$(PYTHONPATH_CMD) $(PYTHON) scripts/recompute_final_tables.py --output $(OUT)

smoke: ## Run end-to-end smoke demo (no external API)
	@echo "=== Smoke Demo ==="
	@mkdir -p $(OUT)/smoke
	@$(PYTHONPATH_CMD) $(PYTHON) scripts/run_end_to_end_demo.py \
		--mode smoke \
		--max_text_files 1 --max_images 5 --max_exam_files 1 \
		--skip_training --skip_external_api \
		--output_dir $(OUT)/smoke \
		--planning-mode $(PLANNING_MODE)

dtcg-ablation: ## Run DTCG component ablation
	@echo "=== DTCG Ablation ==="
	@mkdir -p $(OUT)/dtcg_ablation
	@$(PYTHONPATH_CMD) $(PYTHON) scripts/run_dtcg_ablation.py --output $(OUT)/dtcg_ablation

freeze-manifest: ## Generate reproducibility manifest
	@echo "=== Freeze Manifest ==="
	@mkdir -p $(OUT)
	@$(PYTHONPATH_CMD) $(PYTHON) scripts/freeze_manifest.py --output $(OUT)/freeze_manifest.json

final-gate: ## Run final verification gate
	@echo "=== Final Verification Gate ==="
	@mkdir -p $(OUT)
	@$(PYTHONPATH_CMD) $(PYTHON) scripts/final_verification_gate.py --output $(OUT)

claim-check: ## Check claim registry for prohibited phrases
	@echo "=== Claim Check ==="
	@mkdir -p $(OUT)
	@$(PYTHONPATH_CMD) $(PYTHON) scripts/ingest_human_audit.py --validate-claims --paper-dir reports/paper_ready --output $(OUT)

human-audit-validate: ## Validate human audit status
	@echo "=== Human Audit Validation ==="
	@mkdir -p $(OUT)
	@$(PYTHONPATH_CMD) $(PYTHON) scripts/ingest_human_audit.py --validate-claims --output $(OUT)

cfbench-v1-validate: ## Validate CFBench v1.0
	@echo "=== CFBench v1.0 Validation ==="
	@mkdir -p $(OUT)
	@$(PYTHONPATH_CMD) $(PYTHON) scripts/validate_cfbench_v1.py --output $(OUT)/cfbench_v1_validation.json

all: compile test schema lineage sft-validate bench-validate recompute smoke dtcg-ablation freeze-manifest final-gate ## Run all validation steps

clean: ## Remove build artifacts
	rm -rf build/
