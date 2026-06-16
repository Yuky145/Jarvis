<h1 align="center"> Local Jarvis</h1>

<p align="center">
  A fully <b>local, CPU-friendly</b> AI assistant for a 16&nbsp;GB RAM machine —
  with quantization benchmarking, KV-cache profiling, local RAG, MCP web-search
  tool use, and an automated evaluation harness.
</p>

<p align="center">
  <i>Applied Machine Learning assignment — Parts A–F, production-grade.</i>
</p>

---

##  What's inside

| Part | Topic | Entry point |
|------|-------|-------------|
| **A** | Quantization study (Q8_0 / Q4_K_M / Q3_K_M / Q2_K) | `jarvis bench-quant` |
| **B** | KV-cache & context-length profiling (512→16384) | `jarvis bench-kv` |
| **C** | Local RAG over a research-paper corpus (Chroma + `nomic-embed-text`) | `jarvis rag …` |
| **D** | MCP-style web-search tool + function calling | `jarvis mcp-demo` |
| **E** | Automated evaluation (22 prompts, 5 categories) | `jarvis eval` |
| **F** | Data-driven analysis & honest reflection | `jarvis analyze` |

Everything runs **offline** for core inference (Ollama). Only Part D reaches the
internet, and only to use the web-search tool.

---

##  Hardware requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 16 GB | 16 GB+ |
| CPU | 4 cores | 8+ cores |
| Disk | ~15 GB free (models + corpus) | 25 GB |
| GPU | **not required** | optional, speeds things up |
| OS | Linux / macOS (Windows via WSL2) | Ubuntu 22.04 |

> **Model choice.** The default is **Qwen 2.5 7B Instruct** — it offers the best
> quality-per-GB on 16 GB CPU machines. At `Q4_K_M` (~4.7 GB) it leaves plenty of
> headroom for the OS, the RAG vector store, and long contexts. **Llama 3.1 8B**
> is configured as a drop-in alternative (`models.alternative` in `config.yaml`).

---

##  One-command setup

```bash
git clone <your-repo-url> local-jarvis
cd local-jarvis
bash scripts/install.sh
```

The installer will:
1. create a Python virtual environment (`.venv`) and install pinned deps,
2. install **Ollama** (Linux) and start the server,
3. pull `qwen2.5:7b-instruct-q4_K_M` and `nomic-embed-text`.

Then activate the environment:

```bash
source .venv/bin/activate
jarvis doctor      # verify everything is wired up
```

---

##  Quick start

```bash
# Interactive local chat
jarvis chat

# --- Run individual assignment parts ---
jarvis bench-quant          # Part A: quantization benchmark  → measurements.csv
jarvis bench-kv             # Part B: KV-cache sweep           → kv_cache.csv
jarvis rag download         # Part C: fetch arXiv corpus
jarvis rag index            #         chunk + embed + store
jarvis rag compare          #         5 Qs: RAG vs no-RAG      → rag_comparison.json
jarvis mcp-demo             # Part D: 2 web-search tasks       → mcp_demo_results.json
jarvis eval                 # Part E: full test suite          → eval_results.{json,csv}
jarvis plots                # regenerate all figures           → benchmarks/plots/
jarvis analyze              # Part F: analysis & reflection    → outputs/analysis.md

# --- Or run the whole thing end-to-end (long!) ---
jarvis run-all
```

>  **Want to see all deliverables instantly** without waiting hours for the
> benchmarks? Generate representative **sample data**, then build the figures and
> report from it:
> ```bash
> python scripts/generate_sample_data.py
> jarvis plots && jarvis analyze
> ```

---

##  Project layout

```
local-jarvis/
├── config/config.yaml          # single source of truth for all knobs
├── scripts/
│   ├── install.sh              # one-command setup
│   ├── demo.sh                 # guided end-to-end demo
│   └── generate_sample_data.py # representative sample outputs
├── src/jarvis/
│   ├── core/                   # config, logging, Ollama client, metrics
│   ├── benchmarks/             # Part A (quantization) + Part B (KV cache) + plots
│   ├── rag/                    # Part C: corpus, chunking, vector store, pipeline
│   ├── mcp/                    # Part D: search server + function-calling agent
│   ├── eval/                   # Part E (runner) + Part F (analysis)
│   └── cli.py                  # unified `jarvis` command
├── data/
│   ├── corpus/                 # research papers (downloaded)
│   └── test_set.json           # 22 evaluation prompts
├── benchmarks/
│   ├── results/measurements.csv# Part A data  (+ kv_cache.csv for Part B)
│   └── plots/                  # PNG + PDF figures
├── outputs/                    # eval/rag/mcp JSON + analysis.md
├── docs/REPORT.md              # 4–6 page IEEE/ACM-style report template
└── tests/                      # pytest suite (no server needed)
```

---

## Configuration

All behaviour is controlled from [`config/config.yaml`](config/config.yaml):
swap the model family, change quantization levels, context lengths, chunk sizes,
`top_k`, the search backend (DuckDuckGo / Brave), and more — no code edits needed.

---

##  Tests

```bash
pytest            # runs offline unit tests (scoring, chunking, config, eval)
```

---

##  Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Ollama server is not reachable` | Run `ollama serve` (the installer starts it; on reboot start it again). |
| Model pull is slow / fails | `ollama pull qwen2.5:7b-instruct-q4_K_M` manually; check disk space. |
| `q8_0` KV cache has no effect (Part B) | Start the server with `OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve`. |
| Out of memory at 16k context | Use `Q4_K_M` or smaller, enable q8 KV cache, or lower `kv_cache.context_lengths`. |
| Web search returns nothing (Part D) | DuckDuckGo may rate-limit; retry, or set `BRAVE_API_KEY` and `mcp.search_backend: brave`. |
| `chromadb` import errors | `pip install -r requirements.txt` inside the venv; Python ≥ 3.10 required. |
| Embeddings fail | `ollama pull nomic-embed-text`. |

---

##  Deliverables checklist

- [x] Source code + configs (`src/`, `config/`)
- [x] One-command setup (`scripts/install.sh`)
- [x] `data/test_set.json` (22 prompts across 5 categories)
- [x] `benchmarks/results/measurements.csv` (+ `kv_cache.csv`)
- [x] Plotting / visualization (`jarvis plots`)
- [x] Technical report template (`docs/REPORT.md`)
- [x] Demo instructions (`scripts/demo.sh`, below)

---

##  Demo

```bash
bash scripts/demo.sh
```

This guided script checks your environment, runs a quick chat, a short
benchmark, a RAG comparison, and a web-search task, then opens the generated
report. See [`docs/DEMO.md`](docs/DEMO.md) for a narrated walk-through.

---

##  License

MIT — see [`LICENSE`](LICENSE).
