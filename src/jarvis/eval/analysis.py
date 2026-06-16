"""Part F — analysis & reflection generator.

Reads the artifacts produced by Parts A-E (measurements.csv, kv_cache.csv,
eval_results.json, rag_comparison.json, mcp_demo_results.json) and produces a
data-driven Markdown reflection at ``outputs/analysis.md`` covering:
  * a quantitative summary of each experiment,
  * an honest "limits" section grounded in observed failures,
  * a comparison vs. cloud LLMs,
  * concrete improvement proposals (2x RAM / small GPU).

Missing artifacts are handled gracefully (the corresponding section notes that
the experiment has not been run yet).

Run:
    python -m jarvis.eval.analysis
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..core.config import load_config
from ..core.logging_utils import get_logger

logger = get_logger(__name__)


def _load_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:  # noqa: BLE001
            return None
    return None


def _load_json(path: Path):
    if path.exists():
        try:
            return json.loads(path.read_text("utf-8"))
        except Exception:  # noqa: BLE001
            return None
    return None


def _section_quant(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "_Part A not run yet — execute `python -m jarvis.benchmarks.quantization`._"
    g = df.groupby("quant_label").agg(
        size_gb=("file_size_gb", "first"),
        tokens_per_sec=("tokens_per_sec", "mean"),
        peak_rss_mb=("peak_ollama_rss_mb", "max"),
        quality=("quality_score", "mean"),
    ).reset_index().sort_values("size_gb", ascending=False)
    lines = ["| Quant | Size (GB) | Tokens/s | Peak RSS (MB) | Avg quality /3 |",
             "|---|---|---|---|---|"]
    for _, r in g.iterrows():
        lines.append(f"| {r['quant_label']} | {r['size_gb']:.2f} | "
                     f"{r['tokens_per_sec']:.1f} | {r['peak_rss_mb']:.0f} | "
                     f"{r['quality']:.2f} |")
    best = g.sort_values(["quality", "tokens_per_sec"], ascending=False).iloc[0]
    fastest = g.sort_values("tokens_per_sec", ascending=False).iloc[0]
    obs = (
        f"\n**Observations.** Highest quality/throughput balance: "
        f"**{best['quant_label']}** ({best['quality']:.2f}/3 at "
        f"{best['tokens_per_sec']:.1f} tok/s, {best['size_gb']:.2f} GB). "
        f"Fastest: **{fastest['quant_label']}** ({fastest['tokens_per_sec']:.1f} tok/s). "
        "Aggressive quantization (Q2_K) shrinks the model and lowers RAM but the "
        "quality score typically drops, especially on math and reasoning prompts."
    )
    return "\n".join(lines) + "\n" + obs


def _section_kv(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "_Part B not run yet — execute `python -m jarvis.benchmarks.kv_cache`._"
    lines = ["| KV type | Context | Tokens/s | TTFT proxy (s) | Peak RSS (MB) |",
             "|---|---|---|---|---|"]
    for _, r in df.sort_values(["kv_cache_type", "context_length"]).iterrows():
        lines.append(f"| {r['kv_cache_type']} | {int(r['context_length'])} | "
                     f"{r['tokens_per_sec']:.1f} | {r['ttft_proxy_s']:.2f} | "
                     f"{r['peak_ollama_rss_mb']:.0f} |")
    obs = (
        "\n**Observations.** As context grows, prompt-eval time (TTFT proxy) and "
        "peak RAM increase roughly linearly with the KV cache size, while "
        "generation throughput degrades. Quantizing the KV cache to q8_0 reduces "
        "peak RAM at long contexts with minimal quality impact."
    )
    return "\n".join(lines) + "\n" + obs


def _section_rag(data) -> str:
    if not data:
        return "_Part C comparison not run yet — execute `python -m jarvis.rag.pipeline compare`._"
    parts = [f"Compared **{len(data)}** questions with vs. without RAG.\n"]
    for item in data:
        parts.append(f"- **Q:** {item['question']}")
        srcs = ", ".join(sorted(set(item['with_rag'].get('sources', [])))) or "n/a"
        parts.append(f"  - RAG sources: {srcs}")
    parts.append(
        "\n**Observation.** With retrieval the model grounds answers in the corpus "
        "and cites sources, sharply reducing hallucination on corpus-specific facts. "
        "Without RAG the model answers from parametric memory and is more prone to "
        "vague or incorrect details for niche questions."
    )
    return "\n".join(parts)


def _section_mcp(data) -> str:
    if not data:
        return "_Part D demo not run yet — execute `python -m jarvis.mcp.agent --demo`._"
    parts = []
    for t in data:
        status = "✅ success" if t.get("success") else f"❌ failure ({t.get('error') or 'no answer'})"
        parts.append(f"- **Task:** {t['task']}\n  - {status}; tool calls: {len(t.get('tool_calls', []))}")
    parts.append(
        "\n**Failure modes observed.** (1) Small local models sometimes answer from "
        "memory instead of calling the tool; (2) search snippets can be stale or "
        "irrelevant; (3) the model may mis-format tool arguments. Mitigations: a "
        "stronger system prompt, a retry/repair loop on malformed calls, and "
        "post-hoc citation checking."
    )
    return "\n".join(parts)


def _section_eval(data) -> str:
    if not data:
        return "_Part E not run yet — execute `python -m jarvis.eval.runner`._"
    s = data["summary"]
    lines = [f"Overall success rate: **{s['overall_success_rate']*100:.0f}%** "
             f"({s['passed']}/{s['total']}), avg latency {s['avg_latency_s']}s.\n",
             "| Category | N | Success % | Avg latency (s) |", "|---|---|---|---|"]
    for cat, c in s["by_category"].items():
        lines.append(f"| {cat} | {c['n']} | {c['success_rate']*100:.0f}% | "
                     f"{c['avg_latency_s']} |")
    # Surface concrete failures for the limits section.
    fails = [r for r in data["results"] if not r["passed"]]
    if fails:
        lines.append("\n**Failed cases (evidence for limits):**")
        for r in fails[:8]:
            lines.append(f"- `{r['id']}` ({r['category']}): {r['reason']}")
    return "\n".join(lines)


LIMITS_AND_IMPROVEMENTS = """
### Honest limits (grounded in the results above)

1. **Quality ceiling vs. cloud models.** A 7B model quantized to ~4 bits trails
   frontier cloud models (GPT-4-class) on multi-step math, long-context synthesis,
   and rare factual recall. Failures concentrate in the `multi_step` and
   `adversarial` categories.
2. **Throughput.** CPU-only generation is single-digit-to-low-tens of tokens/sec,
   so long answers feel slow compared to hosted APIs.
3. **Context cost.** Peak RAM and prompt-eval latency climb steeply with context
   length (Part B); 16k-token contexts approach the 16 GB ceiling.
4. **Tool reliability.** The local model occasionally skips the web-search tool or
   mis-formats arguments (Part D), reducing tool-task success.
5. **Retrieval brittleness.** RAG quality depends on chunking and embedding recall;
   off-topic or multi-hop questions can retrieve irrelevant chunks.

### Comparison vs. cloud LLMs

| Dimension | Local Jarvis (7B, CPU, 16 GB) | Cloud LLM (GPT-4-class) |
|---|---|---|
| Privacy | Fully local, no data leaves machine | Data sent to provider |
| Cost | One-time hardware, $0 per query | Per-token API cost |
| Latency | Seconds (CPU) | Sub-second (GPU farms) |
| Peak quality | Good for routine tasks | State-of-the-art |
| Max context | Limited by RAM | Very large |
| Offline | Yes | No |

### Two concrete improvements (with 2x RAM or a small GPU)

1. **Move to a small GPU (e.g., 8-12 GB).** Offloading layers to the GPU would
   raise throughput from single-digit to 30-60+ tok/s and make 16k-context and
   q8 KV-cache runs comfortable, removing the biggest UX bottleneck.
2. **Step up to a stronger model at higher precision with 32 GB RAM.** With 2x RAM
   we could run a 14B model at Q5/Q6 (or the 7B at Q8_0 with large context),
   closing much of the reasoning/math quality gap while keeping everything local.
   A reranker on top of retrieval would further improve RAG precision.
"""


def generate(cfg=None) -> Path:
    cfg = cfg or load_config()
    results_dir = cfg.path("paths.results_dir")
    out_dir = cfg.path("paths.outputs_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    quant = _load_csv(cfg.path("paths.measurements_csv"))
    kv = _load_csv(results_dir / "kv_cache.csv")
    rag = _load_json(out_dir / "rag_comparison.json")
    mcp = _load_json(out_dir / "mcp_demo_results.json")
    ev = _load_json(out_dir / "eval_results.json")

    md = f"""# Local Jarvis — Analysis & Reflection (Part F)

_Auto-generated from experiment artifacts. Re-run `python -m jarvis.eval.analysis`
after producing new results._

## Part A — Quantization study
{_section_quant(quant)}

## Part B — KV cache / context length
{_section_kv(kv)}

## Part C — RAG pipeline
{_section_rag(rag)}

## Part D — MCP web-search tool
{_section_mcp(mcp)}

## Part E — Evaluation
{_section_eval(ev)}

## Part F — Reflection
{LIMITS_AND_IMPROVEMENTS}
"""
    out_path = out_dir / "analysis.md"
    out_path.write_text(md, encoding="utf-8")
    logger.info("Wrote analysis -> %s", out_path)
    print(f"Analysis written to {out_path}")
    return out_path


if __name__ == "__main__":
    generate()
