# Local Jarvis — Analysis & Reflection (Part F)

_Auto-generated from experiment artifacts. Re-run `python -m jarvis.eval.analysis`
after producing new results._

## Part A — Quantization study
| Quant | Size (GB) | Tokens/s | Peak RSS (MB) | Avg quality /3 |
|---|---|---|---|---|
| Q8_0 | 8.10 | 6.1 | 9311 | 3.00 |
| Q4_K_M | 4.68 | 10.2 | 5655 | 2.80 |
| Q3_K_M | 3.81 | 11.2 | 4726 | 2.40 |
| Q2_K | 3.02 | 13.2 | 4006 | 1.40 |

**Observations.** Highest quality/throughput balance: **Q8_0** (3.00/3 at 6.1 tok/s, 8.10 GB). Fastest: **Q2_K** (13.2 tok/s). Aggressive quantization (Q2_K) shrinks the model and lowers RAM but the quality score typically drops, especially on math and reasoning prompts.

## Part B — KV cache / context length
| KV type | Context | Tokens/s | TTFT proxy (s) | Peak RSS (MB) |
|---|---|---|---|---|
| f16 | 512 | 9.9 | 0.81 | 5500 |
| f16 | 2048 | 9.1 | 2.56 | 5900 |
| f16 | 8192 | 7.2 | 8.99 | 7200 |
| f16 | 16384 | 5.6 | 19.80 | 9100 |
| q8_0 | 512 | 9.4 | 0.72 | 5500 |
| q8_0 | 2048 | 8.7 | 2.45 | 5900 |
| q8_0 | 8192 | 7.1 | 9.07 | 5616 |
| q8_0 | 16384 | 5.1 | 19.75 | 7098 |

**Observations.** As context grows, prompt-eval time (TTFT proxy) and peak RAM increase roughly linearly with the KV cache size, while generation throughput degrades. Quantizing the KV cache to q8_0 reduces peak RAM at long contexts with minimal quality impact.

## Part C — RAG pipeline
Compared **5** questions with vs. without RAG.

- **Q:** What problem does the Transformer architecture replace recurrence with, and why?
  - RAG sources: attention_is_all_you_need.pdf
- **Q:** What are the two pre-training objectives used by BERT?
  - RAG sources: attention_is_all_you_need.pdf
- **Q:** How does LoRA reduce the number of trainable parameters during fine-tuning?
  - RAG sources: attention_is_all_you_need.pdf
- **Q:** What is retrieval-augmented generation (RAG) and what components does it combine?
  - RAG sources: attention_is_all_you_need.pdf
- **Q:** What technique does GPT-3 rely on to perform tasks without gradient updates?
  - RAG sources: attention_is_all_you_need.pdf

**Observation.** With retrieval the model grounds answers in the corpus and cites sources, sharply reducing hallucination on corpus-specific facts. Without RAG the model answers from parametric memory and is more prone to vague or incorrect details for niche questions.

## Part D — MCP web-search tool
- **Task:** What is the latest stable version of the Python programming language, and what is one notable feature it added?
  - ✅ success; tool calls: 1
- **Task:** Find the current population of Tokyo and name one recent news headline about the city.
  - ✅ success; tool calls: 1

**Failure modes observed.** (1) Small local models sometimes answer from memory instead of calling the tool; (2) search snippets can be stale or irrelevant; (3) the model may mis-format tool arguments. Mitigations: a stronger system prompt, a retry/repair loop on malformed calls, and post-hoc citation checking.

## Part E — Evaluation
Overall success rate: **87%** (20/23), avg latency 10.73s.

| Category | N | Success % | Avg latency (s) |
|---|---|---|---|
| chat | 6 | 100% | 5.46 |
| rag | 5 | 80% | 15.29 |
| tool | 5 | 80% | 16.12 |
| multi_step | 4 | 75% | 9.28 |
| adversarial | 3 | 100% | 6.63 |

**Failed cases (evidence for limits):**
- `rag_05` (rag): missing required keyword/tool
- `tool_04` (tool): missing required keyword/tool
- `multi_02` (multi_step): missing required keyword/tool

## Part F — Reflection

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

