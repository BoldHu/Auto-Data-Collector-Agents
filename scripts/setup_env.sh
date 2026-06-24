#!/usr/bin/env bash
# setup_env.sh — One-command environment setup for AutoData project
# Usage: bash scripts/setup_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== AutoData Environment Setup ==="
echo "Project root: $PROJECT_ROOT"

# 1. Check Python version
echo ""
echo "[1/5] Checking Python version..."
python3 --version || { echo "ERROR: python3 not found"; exit 1; }
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ "$PY_VER" != "3.11" ]]; then
    echo "WARNING: Python $PY_VER detected. Recommended: 3.11"
fi

# 2. Create conda environment if not exists
echo ""
echo "[2/5] Setting up conda environment..."
if conda env list | grep -q "^autodata "; then
    echo "Conda env 'autodata' already exists. Skipping creation."
else
    echo "Creating conda env 'autodata' from environment.yml..."
    conda env create -f "$PROJECT_ROOT/environment.yml"
fi
echo "Activating conda env..."
eval "$(conda shell.bash hook)"
conda activate autodata

# 3. Install pip dependencies
echo ""
echo "[3/5] Installing pip dependencies..."
pip install -r "$PROJECT_ROOT/requirements.txt" --quiet

# 4. Verify critical imports
echo ""
echo "[4/5] Verifying critical imports..."
python3 -c "import openai; print(f'  openai {openai.__version__}')" || { echo "ERROR: openai import failed"; exit 1; }
python3 -c "import anthropic; print(f'  anthropic {anthropic.__version__}')" || echo "WARNING: anthropic import failed"
python3 -c "import networkx; print(f'  networkx {networkx.__version__}')"
python3 -c "import yaml; print('  pyyaml OK')"
python3 -c "import loguru; print('  loguru OK')"
python3 -c "import pandas; print(f'  pandas {pandas.__version__}')"
python3 -c "import langchain_openai; print(f'  langchain-openai OK')"

# 5. Verify project modules
echo ""
echo "[5/5] Verifying project modules..."
cd "$PROJECT_ROOT"
python3 -c "from src.autodata.utils.api_loader import load_xiaomi_config; cfg = load_xiaomi_config(); print(f'  Xiaomi API: model={cfg.default_model}, key_loaded={bool(cfg.api_key)}')"
python3 -c "from src.autodata.context_graph import DynamicTaskContextGraph, ContextSelector, LocalCache, MessageStore; print('  DTCG modules OK')"

echo ""
echo "=== Setup Complete ==="
echo "Run: conda activate autodata"
echo "Then: python scripts/run_phase_1_smoke_test.py"