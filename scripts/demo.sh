#!/usr/bin/env bash
# =====================================================================
# Local Jarvis — guided end-to-end demo
# Usage:  bash scripts/demo.sh
# =====================================================================
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CYAN='\033[0;36m'; GREEN='\033[0;32m'; NC='\033[0m'
step() { echo -e "\n${CYAN}==>${NC} $*"; }

# Activate venv if present.
if [[ -d .venv ]]; then source .venv/bin/activate; fi

step "1/6  Environment health check"
jarvis doctor || python -m jarvis.cli doctor

step "2/6  Generating sample data (so all deliverables exist instantly)"
python scripts/generate_sample_data.py

step "3/6  Building figures from data (Part A & B)"
jarvis plots || python -m jarvis.cli plots

step "4/6  RAG comparison (Part C) — requires Ollama + indexed corpus"
echo "    Skipping live RAG in demo. To run for real:"
echo "      jarvis rag download && jarvis rag index && jarvis rag compare"

step "5/6  Web-search tool demo (Part D) — requires Ollama + internet"
echo "    To run for real:  jarvis mcp-demo"

step "6/6  Analysis & reflection report (Part F)"
jarvis analyze || python -m jarvis.cli analyze

echo -e "\n${GREEN}Demo complete.${NC}"
echo "Open the report:  outputs/analysis.md"
echo "Figures:          benchmarks/plots/"
echo "For the FULL live pipeline (needs Ollama + models):  jarvis run-all"
