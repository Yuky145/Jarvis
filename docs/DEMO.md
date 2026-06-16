# 🎬 Local Jarvis — Demo Walk-through

A 5-minute narrated demo you can run live or record. Run the guided script:

```bash
bash scripts/demo.sh
```

…or follow the steps manually below.

---

### 0. Health check (10 s)
```bash
jarvis doctor
```
Shows machine RAM, Ollama status, and which models are installed. Use this to
confirm the environment is ready before demoing.

### 1. Local chat (30 s)
```bash
jarvis chat
```
Type a couple of questions to show fully-local, offline inference. `Ctrl-C` to exit.

### 2. Quantization snapshot — Part A (1 min)
For a quick demo, the committed `measurements.csv` already has results; just
show the figures:
```bash
jarvis plots
xdg-open benchmarks/plots/quant_size_vs_speed.png   # or open on macOS
```
Explain the size ↔ speed ↔ quality trade-off and why **Q4_K_M** is the sweet spot.

> Full run (slow): `jarvis bench-quant`

### 3. RAG vs no-RAG — Part C (1.5 min)
```bash
jarvis rag download    # first time only
jarvis rag index       # first time only
jarvis rag compare
```
Point out how the RAG answer cites a source paper while the no-RAG answer is
vaguer. Results saved to `outputs/rag_comparison.json`.

### 4. Web-search tool — Part D (1 min)
```bash
jarvis mcp-demo
```
Watch the model emit a `web_search` tool call, receive results, and synthesize an
answer with a citation. Saved to `outputs/mcp_demo_results.json`.

### 5. Evaluation + report — Parts E & F (1 min)
```bash
jarvis eval        # 22-prompt suite, category-wise success rates
jarvis analyze     # builds outputs/analysis.md from all artifacts
```
Open `outputs/analysis.md` to show the data-driven limits section and the
cloud-vs-local comparison.

---

### Talking points
- **Privacy & cost:** everything but Part D runs offline; no per-token fees.
- **The knee of the curve:** Q4_K_M ≈ best quality-per-GB on 16 GB.
- **Honest limits:** show a failed adversarial/multi-step case from the eval.
- **Biggest lever:** a small GPU would 5–10× throughput.
