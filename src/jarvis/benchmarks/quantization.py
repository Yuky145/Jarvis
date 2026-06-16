"""Part A — Quantization benchmarking.

For each configured quantization level of the same base model we measure:
  * file size (GB) on disk
  * peak RAM during a 200-token completion (delta + Ollama RSS)
  * generation throughput (tokens/sec)
  * quality score (0-3) on each of the 5 standardized prompts

Results are appended to ``benchmarks/results/measurements.csv``.

Run:
    python -m jarvis.benchmarks.quantization            # full run
    python -m jarvis.benchmarks.quantization --dry-run  # show plan only
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from ..core.config import load_config
from ..core.logging_utils import get_logger
from ..core.metrics import MemoryMonitor, total_ram_gb
from ..core.ollama_client import OllamaClient
from .prompts import PROMPTS, score_response

logger = get_logger(__name__)
console = Console()

CSV_FIELDS = [
    "experiment",
    "model_base",
    "quant_label",
    "quant_tag",
    "prompt_id",
    "category",
    "file_size_gb",
    "completion_tokens",
    "tokens_per_sec",
    "prompt_tokens_per_sec",
    "eval_duration_s",
    "wall_time_s",
    "peak_ram_delta_mb",
    "peak_ollama_rss_mb",
    "quality_score",
    "system_total_ram_gb",
]


def _ensure_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=CSV_FIELDS).writeheader()


def _append_rows(path: Path, rows: list[dict]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        for r in rows:
            writer.writerow(r)


def run(dry_run: bool = False, auto_pull: bool = True) -> list[dict]:
    cfg = load_config()
    client = OllamaClient()
    base = cfg.get("quantization.base")
    levels = cfg.get("quantization.levels", [])
    n_predict = cfg.get("quantization.completion_tokens", 200)
    repeats = cfg.get("quantization.repeats", 1)
    csv_path = cfg.path("paths.measurements_csv")
    total_ram = total_ram_gb()

    console.rule(f"[bold cyan]Part A — Quantization study ({base})")
    console.print(f"Levels: {[lv['label'] for lv in levels]}  | "
                  f"completion tokens: {n_predict} | repeats: {repeats} | "
                  f"machine RAM: {total_ram} GB")

    if dry_run:
        console.print("[yellow]Dry run — no generation performed.[/yellow]")
        return []

    if not client.is_up():
        raise RuntimeError(
            "Ollama server is not reachable at "
            f"{client.host}. Start it with `ollama serve`."
        )

    _ensure_csv(csv_path)
    all_rows: list[dict] = []

    for lv in levels:
        tag, label = lv["tag"], lv["label"]
        console.rule(f"[green]{label}  ({tag})")

        if not client.has_model(tag):
            if auto_pull:
                logger.info("Model %s not present — pulling.", tag)
                try:
                    client.pull(tag)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Could not pull %s: %s — skipping.", tag, exc)
                    continue
            else:
                logger.warning("Model %s missing and auto_pull disabled — skip.", tag)
                continue

        size_bytes = client.model_size_bytes(tag) or 0
        file_size_gb = round(size_bytes / (1024 ** 3), 3)

        for prompt in tqdm(PROMPTS, desc=f"{label} prompts", unit="prompt"):
            tps_runs, ram_runs, oll_runs, evald_runs, wall_runs = [], [], [], [], []
            ctok = 0
            ptps = 0.0
            response_text = ""
            for _ in range(repeats):
                with MemoryMonitor(interval=0.2) as mon:
                    res = client.generate(
                        tag,
                        prompt.prompt,
                        system=prompt.system,
                        num_predict=n_predict,
                    )
                mem = mon.result
                tps_runs.append(res.tokens_per_sec)
                evald_runs.append(res.eval_duration_s)
                wall_runs.append(res.wall_time_s)
                ram_runs.append(mem.delta_system_mb)
                oll_runs.append(mem.peak_ollama_rss_mb)
                ctok = res.completion_tokens
                ptps = res.prompt_tokens_per_sec
                response_text = res.text

            quality = score_response(prompt.id, response_text)
            row = {
                "experiment": "quantization",
                "model_base": base,
                "quant_label": label,
                "quant_tag": tag,
                "prompt_id": prompt.id,
                "category": prompt.category,
                "file_size_gb": file_size_gb,
                "completion_tokens": ctok,
                "tokens_per_sec": round(statistics.mean(tps_runs), 2),
                "prompt_tokens_per_sec": round(ptps, 2),
                "eval_duration_s": round(statistics.mean(evald_runs), 3),
                "wall_time_s": round(statistics.mean(wall_runs), 3),
                "peak_ram_delta_mb": round(statistics.mean(ram_runs), 1),
                "peak_ollama_rss_mb": round(max(oll_runs), 1),
                "quality_score": quality,
                "system_total_ram_gb": total_ram,
            }
            all_rows.append(row)

        # Free RAM before next quant level for a clean measurement.
        client.unload(tag)

    _append_rows(csv_path, all_rows)
    _print_summary(all_rows)
    console.print(f"\n[bold green]Saved {len(all_rows)} rows -> {csv_path}")
    return all_rows


def _print_summary(rows: list[dict]) -> None:
    if not rows:
        return
    table = Table(title="Quantization summary (averaged over prompts)")
    for col in ["Quant", "Size GB", "Tokens/s", "Peak RSS MB", "Avg quality /3"]:
        table.add_column(col, justify="right")
    by_label: dict[str, list[dict]] = {}
    for r in rows:
        by_label.setdefault(r["quant_label"], []).append(r)
    for label, group in by_label.items():
        tps = statistics.mean(g["tokens_per_sec"] for g in group)
        rss = max(g["peak_ollama_rss_mb"] for g in group)
        q = statistics.mean(g["quality_score"] for g in group)
        table.add_row(
            label,
            f"{group[0]['file_size_gb']:.2f}",
            f"{tps:.1f}",
            f"{rss:.0f}",
            f"{q:.2f}",
        )
    console.print(table)


def main() -> None:
    ap = argparse.ArgumentParser(description="Part A quantization benchmark")
    ap.add_argument("--dry-run", action="store_true", help="Show plan and exit")
    ap.add_argument("--no-pull", action="store_true", help="Do not auto-pull models")
    args = ap.parse_args()
    run(dry_run=args.dry_run, auto_pull=not args.no_pull)


if __name__ == "__main__":
    main()
