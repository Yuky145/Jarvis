#!/usr/bin/env bash
# =====================================================================
# Local Jarvis — one-command installer
# Usage:  bash scripts/install.sh
# Tested on Ubuntu 22.04 / macOS. CPU-only, 16 GB RAM target.
# =====================================================================
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[install]${NC} $*"; }
warn()  { echo -e "${YELLOW}[install]${NC} $*"; }
err()   { echo -e "${RED}[install]${NC} $*" >&2; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ---------------------------------------------------------------------
# 1. Python virtual environment + dependencies
# ---------------------------------------------------------------------
info "Creating Python virtual environment (.venv)..."
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel >/dev/null
info "Installing Python dependencies (this can take a few minutes)..."
pip install -r requirements.txt

# Install the package itself in editable mode.
pip install -e . >/dev/null 2>&1 || warn "Editable install skipped (pyproject optional)."

# ---------------------------------------------------------------------
# 2. Ollama
# ---------------------------------------------------------------------
if ! command -v ollama >/dev/null 2>&1; then
  warn "Ollama not found. Installing..."
  if [[ "$(uname)" == "Linux" ]]; then
    curl -fsSL https://ollama.com/install.sh | sh
  else
    err "Please install Ollama from https://ollama.com/download then re-run."
    exit 1
  fi
else
  info "Ollama already installed: $(ollama --version 2>/dev/null || echo unknown)"
fi

# Start the server if not running.
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  info "Starting Ollama server in background..."
  (ollama serve >/tmp/ollama.log 2>&1 &) || true
  sleep 3
fi

# ---------------------------------------------------------------------
# 3. Pull core + embedding models
# ---------------------------------------------------------------------
info "Pulling default model (qwen2.5:7b-instruct-q4_K_M)..."
ollama pull qwen2.5:7b-instruct-q4_K_M || warn "Model pull failed; pull manually later."
info "Pulling embedding model (nomic-embed-text)..."
ollama pull nomic-embed-text || warn "Embedding model pull failed; pull manually later."

info "Done!  Activate the env with:  source .venv/bin/activate"
info "Then try:  python -m jarvis.cli chat"
