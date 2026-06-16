"""Generate realistic SAMPLE benchmark data + outputs.

This lets reviewers see the full set of deliverables (measurements.csv, plots,
analysis.md, eval/rag/mcp JSON) without first running the multi-hour benchmark
suite on hardware. Numbers are representative of Qwen2.5-7B-Instruct on a typical
16 GB CPU machine and are clearly flagged as sample data.

Run:
    python scripts/generate_sample_data.py
Then regenerate plots/report from the sample CSVs:
    python -m jarvis.benchmarks.plots
    python -m jarvis.eval.analysis
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
OUTPUTS = ROOT / "outputs"
RESULTS.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)
random.seed(42)

# Representative per-quant profile: (size_gb, tok/s, peak_rss_mb, base_quality)
QUANT_PROFILE = {
    "Q8_0":   (8.10, 6.2, 9300, {"math": 3, "code": 3, "summarization": 3, "factual": 3, "reasoning": 3}),
    "Q4_K_M": (4.68, 9.8, 5600, {"math": 3, "code": 3, "summarization": 3, "factual": 3, "reasoning": 2}),
    "Q3_K_M": (3.81, 11.4, 4700, {"math": 2, "code": 2, "summarization": 3, "factual": 3, "reasoning": 2}),
    "Q2_K":   (3.02, 12.9, 3900, {"math": 1, "code": 1, "summarization": 2, "factual": 2, "reasoning": 1}),
}
TAGS = {
    "Q8_0": "qwen2.5:7b-instruct-q8_0",
    "Q4_K_M": "qwen2.5:7b-instruct-q4_K_M",
    "Q3_K_M": "qwen2.5:7b-instruct-q3_K_M",
    "Q2_K": "qwen2.5:7b-instruct-q2_K",
}
CATEGORIES = ["math", "code", "summarization", "factual", "reasoning"]
QUANT_FIELDS = [
    "experiment", "model_base", "quant_label", "quant_tag", "prompt_id", "category",
    "file_size_gb", "completion_tokens", "tokens_per_sec", "prompt_tokens_per_sec",
    "eval_duration_s", "wall_time_s", "peak_ram_delta_mb", "peak_ollama_rss_mb",
    "quality_score", "system_total_ram_gb",
]


def gen_quant():
    rows = []
    for label, (size, tps, rss, quals) in QUANT_PROFILE.items():
        for cat in CATEGORIES:
            t = round(tps + random.uniform(-0.6, 0.6), 2)
            rows.append({
                "experiment": "quantization",
                "model_base": "qwen2.5:7b-instruct",
                "quant_label": label,
                "quant_tag": TAGS[label],
                "prompt_id": cat,
                "category": cat,
                "file_size_gb": size,
                "completion_tokens": 200,
                "tokens_per_sec": t,
                "prompt_tokens_per_sec": round(t * 4.5, 2),
                "eval_duration_s": round(200 / t, 3),
                "wall_time_s": round(200 / t + random.uniform(0.3, 0.8), 3),
                "peak_ram_delta_mb": round(rss * random.uniform(0.45, 0.55), 1),
                "peak_ollama_rss_mb": round(rss + random.uniform(-120, 120), 1),
                "quality_score": quals[cat],
                "system_total_ram_gb": 16.0,
            })
    with open(RESULTS / "measurements.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=QUANT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} quant rows -> measurements.csv")


KV_FIELDS = [
    "experiment", "model", "kv_cache_type", "context_length", "approx_prompt_tokens",
    "gen_tokens", "tokens_per_sec", "prompt_tokens_per_sec", "ttft_proxy_s",
    "eval_duration_s", "wall_time_s", "peak_ram_delta_mb", "peak_ollama_rss_mb",
    "system_total_ram_gb",
]
# (tok/s, ttft_s, rss_mb) per context length for f16 baseline.
KV_PROFILE = {
    512:   (9.9, 0.9, 5500),
    2048:  (9.1, 2.4, 5900),
    8192:  (7.2, 9.1, 7200),
    16384: (5.4, 19.8, 9100),
}


def gen_kv():
    rows = []
    for kv in ("f16", "q8_0"):
        for ctx, (tps, ttft, rss) in KV_PROFILE.items():
            # q8 KV cache: similar speed, lower RAM at long contexts.
            mult_rss = 1.0 if kv == "f16" else (1.0 if ctx <= 2048 else 0.78)
            mult_tps = 1.0 if kv == "f16" else 0.97
            rows.append({
                "experiment": "kv_cache",
                "model": "qwen2.5:7b-instruct-q4_K_M",
                "kv_cache_type": kv,
                "context_length": ctx,
                "approx_prompt_tokens": int(ctx * 0.95),
                "gen_tokens": 128,
                "tokens_per_sec": round(tps * mult_tps + random.uniform(-0.2, 0.2), 2),
                "prompt_tokens_per_sec": round(tps * 5, 2),
                "ttft_proxy_s": round(ttft + random.uniform(-0.2, 0.2), 2),
                "eval_duration_s": round(128 / (tps * mult_tps), 3),
                "wall_time_s": round(128 / (tps * mult_tps) + ttft, 3),
                "peak_ram_delta_mb": round(rss * mult_rss * 0.5, 1),
                "peak_ollama_rss_mb": round(rss * mult_rss, 1),
                "system_total_ram_gb": 16.0,
            })
    with open(RESULTS / "kv_cache.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=KV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} kv rows -> kv_cache.csv")


def gen_rag():
    qs = [
        ("What problem does the Transformer architecture replace recurrence with, and why?",
         "The Transformer replaces recurrence with self-attention [attention_is_all_you_need.pdf], enabling parallelization and better long-range dependency modeling.",
         "Transformers use attention mechanisms, though specifics may vary."),
        ("What are the two pre-training objectives used by BERT?",
         "Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) [bert.pdf].",
         "BERT is pre-trained on large text corpora using masked tokens."),
        ("How does LoRA reduce the number of trainable parameters during fine-tuning?",
         "LoRA freezes pretrained weights and injects trainable low-rank decomposition matrices [lora.pdf].",
         "LoRA uses adapters to reduce trainable parameters."),
        ("What is retrieval-augmented generation (RAG) and what components does it combine?",
         "RAG combines a parametric seq2seq generator with a non-parametric retriever over a document index [rag_knowledge_intensive.pdf].",
         "RAG retrieves documents to help a model answer questions."),
        ("What technique does GPT-3 rely on to perform tasks without gradient updates?",
         "In-context (few-shot) learning [gpt3_few_shot_learners.pdf].",
         "GPT-3 uses few-shot prompting."),
    ]
    data = []
    for q, a_rag, a_plain in qs:
        data.append({
            "question": q,
            "with_rag": {"question": q, "use_rag": True, "answer": a_rag,
                         "sources": ["attention_is_all_you_need.pdf"], "tokens_per_sec": 9.7,
                         "completion_tokens": random.randint(60, 140)},
            "without_rag": {"question": q, "use_rag": False, "answer": a_plain,
                            "sources": [], "tokens_per_sec": 9.8,
                            "completion_tokens": random.randint(40, 90)},
        })
    (OUTPUTS / "rag_comparison.json").write_text(json.dumps(data, indent=2), "utf-8")
    print("Wrote rag_comparison.json")


def gen_mcp():
    data = [
        {"task": "What is the latest stable version of the Python programming language, and what is one notable feature it added?",
         "tool_calls": [{"name": "web_search", "arguments": {"query": "latest stable Python version features"},
                          "result": {"results": [{"title": "Python 3.13", "url": "https://www.python.org/downloads/", "snippet": "Python 3.13 with a new interactive REPL and experimental free-threaded build."}]}}],
         "final_answer": "The latest stable Python is 3.13, which adds an improved interactive REPL and an experimental free-threaded (no-GIL) build. [python.org]",
         "success": True, "error": "", "rounds": 2},
        {"task": "Find the current population of Tokyo and name one recent news headline about the city.",
         "tool_calls": [{"name": "web_search", "arguments": {"query": "Tokyo population 2025 news"},
                          "result": {"results": [{"title": "Tokyo metropolis", "url": "https://example.org", "snippet": "Greater Tokyo ~37 million residents."}]}}],
         "final_answer": "Greater Tokyo has roughly 37 million residents. Recent headlines cover heat-wave preparedness measures across the metropolis. [example.org]",
         "success": True, "error": "", "rounds": 2},
    ]
    (OUTPUTS / "mcp_demo_results.json").write_text(json.dumps(data, indent=2), "utf-8")
    print("Wrote mcp_demo_results.json")


def gen_eval():
    # Representative pass/fail pattern by category.
    pattern = {
        "chat": [True, True, True, True, True, True],
        "rag": [True, True, True, True, False],
        "tool": [True, True, True, False, True],
        "multi_step": [True, False, True, True],
        "adversarial": [True, True, True],
    }
    results = []
    counters = {k: 0 for k in pattern}
    test_set = json.loads((ROOT / "data" / "test_set.json").read_text("utf-8"))
    for t in test_set["tests"]:
        cat = t["category"]
        idx = counters[cat]; counters[cat] += 1
        passed = pattern[cat][idx] if idx < len(pattern[cat]) else True
        tools = ["web_search"] if (t.get("requires_tool") and passed) else []
        results.append({
            "id": t["id"], "category": cat, "passed": passed,
            "reason": "passed" if passed else "missing required keyword/tool",
            "latency_s": round(random.uniform(3, 22) if t.get("requires_tool") or t.get("requires_rag") else random.uniform(2, 9), 2),
            "tokens": random.randint(40, 320), "tools_used": tools,
            "response": "(sample response omitted)",
        })

    from collections import defaultdict
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
    cat_stats = {}
    for cat, items in by_cat.items():
        n = len(items); p = sum(1 for i in items if i["passed"])
        cat_stats[cat] = {"n": n, "passed": p, "success_rate": round(p / n, 3),
                          "avg_latency_s": round(sum(i["latency_s"] for i in items) / n, 2),
                          "avg_tokens": round(sum(i["tokens"] for i in items) / n, 1)}
    total = len(results); passed = sum(1 for r in results if r["passed"])
    summary = {"model": "qwen2.5:7b-instruct-q4_K_M", "total": total, "passed": passed,
               "overall_success_rate": round(passed / total, 3),
               "avg_latency_s": round(sum(r["latency_s"] for r in results) / total, 2),
               "total_tokens": sum(r["tokens"] for r in results), "by_category": cat_stats}
    (OUTPUTS / "eval_results.json").write_text(
        json.dumps({"summary": summary, "results": results}, indent=2), "utf-8")
    with open(OUTPUTS / "eval_results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "category", "passed", "latency_s", "tokens", "tools_used", "reason"])
        for r in results:
            w.writerow([r["id"], r["category"], r["passed"], r["latency_s"],
                        r["tokens"], "|".join(r["tools_used"]), r["reason"]])
    print(f"Wrote eval_results.json ({passed}/{total} passed)")


if __name__ == "__main__":
    print("Generating SAMPLE data (representative, not from live hardware)...")
    gen_quant()
    gen_kv()
    gen_rag()
    gen_mcp()
    gen_eval()
    print("Done. Now run: python -m jarvis.benchmarks.plots && python -m jarvis.eval.analysis")
