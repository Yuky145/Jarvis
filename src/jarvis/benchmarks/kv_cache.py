"""Part B — KV cache / context length experiment.

Using the best model from Part A we measure latency (tokens/sec, time-to-first
+ total) and peak RAM at several context lengths: 512, 2048, 8192, 16384.

We build a synthetic prompt that fills (most of) the requested context window,
then generate a fixed number of tokens. We optionally repeat the sweep with the
KV cache quantized to ``q8_0`` (requires a recent Ollama with flash attention:
set ``OLLAMA_FLASH_ATTENTION=1`` and ``OLLAMA_KV_CACHE_TYPE=q8_0`` on the server,
or pass it per-request where supported).

Results append to ``benchmarks/results/kv_cache.csv``.

Run:
    OLLAMA_FLASH_ATTENTION=1 python -m jarvis.benchmarks.kv_cache
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from ..core.config import load_config
from ..core.logging_utils import get_logger
from ..core.metrics import MemoryMonitor, total_ram_gb
from ..core.ollama_client import OllamaClient

logger = get_logger(__name__)
console = Console()

CSV_FIELDS = [
    "experiment",
    "model",
    "kv_cache_type",
    "context_length",
    "approx_prompt_tokens",
    "gen_tokens",
    "tokens_per_sec",
    "prompt_tokens_per_sec",
    "ttft_proxy_s",
    "eval_duration_s",
    "wall_time_s",
    "peak_ram_delta_mb",
    "peak_ollama_rss_mb",
    "system_total_ram_gb",
]

# A short, repeatable filler sentence (~16 tokens). We repeat it to fill ctx.
_FILLER = (
    "The quick brown fox jumps over the lazy dog while the engineer measures "
    "throughput and memory usage carefully. "
)


def _build_prompt(target_tokens: int) -> str:
    """Approximate ``target_tokens`` by word repetition (~0.75 words/token)."""
    words_needed = int(target_tokens * 0.75)
    filler_words = _FILLER.split()
    reps = max(1, words_needed // len(filler_words))
    body = (" ".join(filler_words) + " ") * reps
    return (
        "Read the following text and then write a one-paragraph summary of its "
        "general theme.\n\n" + body + "\n\nSummary:"
    )


def _ensure_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=CSV_FIELDS).writeheader()


def run(dry_run: bool = False) -> list[dict]:
    cfg = load_config()
    client = OllamaClient()
    model = cfg.get("kv_cache.model")
    ctx_lengths = cfg.get("kv_cache.context_lengths", [512, 2048, 8192, 16384])
    gen_tokens = cfg.get("kv_cache.gen_tokens", 128)
    kv_types = cfg.get("kv_cache.kv_cache_types", ["f16"])
    csv_path = cfg.path("paths.results_dir") / "kv_cache.csv"
    total_ram = total_ram_gb()

    console.rule(f"[bold cyan]Part B — KV cache study ({model})")
    console.print(f"Context lengths: {ctx_lengths} | gen tokens: {gen_tokens} | "
                  f"KV types: {kv_types} | machine RAM: {total_ram} GB")

    if dry_run:
        console.print("[yellow]Dry run — no generation performed.[/yellow]")
        return []

    if not client.is_up():
        raise RuntimeError(f"Ollama not reachable at {client.host}.")
    if not client.has_model(model):
        logger.info("Pulling %s ...", model)
        client.pull(model)

    _ensure_csv(csv_path)
    rows: list[dict] = []

    for kv in kv_types:
        # The KV cache type is set on the SERVER via env var. We record which
        # mode the run was performed under; a warning reminds the user.
        if kv != "f16":
            console.print(
                f"[yellow]Note:[/] for kv_cache_type='{kv}', start the server with "
                f"OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE={kv} ollama serve"
            )
        for ctx in tqdm(ctx_lengths, desc=f"kv={kv}", unit="ctx"):
            prompt = _build_prompt(ctx)
            with MemoryMonitor(interval=0.2) as mon:
                res = client.generate(
                    model,
                    prompt,
                    num_predict=gen_tokens,
                    num_ctx=ctx,
                )
            mem = mon.result
            # TTFT proxy: load + prompt-eval time (time before first new token).
            ttft = round(res.load_duration_s + res.prompt_eval_duration_s, 3)
            rows.append({
                "experiment": "kv_cache",
                "model": model,
                "kv_cache_type": kv,
                "context_length": ctx,
                "approx_prompt_tokens": res.prompt_tokens,
                "gen_tokens": res.completion_tokens,
                "tokens_per_sec": res.tokens_per_sec,
                "prompt_tokens_per_sec": res.prompt_tokens_per_sec,
                "ttft_proxy_s": ttft,
                "eval_duration_s": res.eval_duration_s,
                "wall_time_s": res.wall_time_s,
                "peak_ram_delta_mb": mem.delta_system_mb,
                "peak_ollama_rss_mb": mem.peak_ollama_rss_mb,
                "system_total_ram_gb": total_ram,
            })
        client.unload(model)

    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        for r in rows:
            writer.writerow(r)

    _print_summary(rows)
    console.print(f"\n[bold green]Saved {len(rows)} rows -> {csv_path}")
    return rows


def _print_summary(rows: list[dict]) -> None:
    table = Table(title="KV cache summary")
    for col in ["KV", "Ctx", "Tokens/s", "TTFT proxy s", "Peak RSS MB"]:
        table.add_column(col, justify="right")
    for r in rows:
        table.add_row(
            r["kv_cache_type"],
            str(r["context_length"]),
            f"{r['tokens_per_sec']:.1f}",
            f"{r['ttft_proxy_s']:.2f}",
            f"{r['peak_ollama_rss_mb']:.0f}",
        )
    console.print(table)


def main() -> None:
    ap = argparse.ArgumentParser(description="Part B KV cache benchmark")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
