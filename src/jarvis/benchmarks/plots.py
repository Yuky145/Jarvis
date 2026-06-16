"""Plot generation for Parts A and B.

Reads the CSV artifacts and writes PNG + PDF figures into ``benchmarks/plots``:
  Part A:
    * quant_size_vs_speed.png      (file size vs tokens/sec, colored by quality)
    * quant_tradeoff.png           (size, speed, quality bars per quant level)
    * quant_quality_by_category.png
  Part B:
    * kv_context_vs_latency.png    (context length vs tokens/sec & TTFT)
    * kv_context_vs_ram.png        (context length vs peak RAM, per KV type)

Run:
    python -m jarvis.benchmarks.plots
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ..core.config import load_config
from ..core.logging_utils import get_logger

logger = get_logger(__name__)

# Order quant levels from highest to lowest precision for nice axes.
QUANT_ORDER = ["Q8_0", "Q4_K_M", "Q3_K_M", "Q2_K"]


def _save(fig, plots_dir: Path, name: str) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(plots_dir / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot %s", name)


def _order(df: pd.DataFrame, col: str, order: list[str]) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    return df.sort_values(col)


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------
def plot_quantization(csv_path: Path, plots_dir: Path) -> None:
    if not csv_path.exists():
        logger.warning("No measurements.csv at %s — skipping Part A plots.", csv_path)
        return
    df = pd.read_csv(csv_path)
    df = df[df["experiment"] == "quantization"]
    if df.empty:
        logger.warning("No quantization rows — skipping Part A plots.")
        return

    agg = df.groupby("quant_label").agg(
        size_gb=("file_size_gb", "first"),
        tps=("tokens_per_sec", "mean"),
        rss=("peak_ollama_rss_mb", "max"),
        quality=("quality_score", "mean"),
    ).reset_index()
    agg = _order(agg, "quant_label", QUANT_ORDER)

    # 1) Size vs speed, color = quality
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(agg["size_gb"], agg["tps"], c=agg["quality"],
                    cmap="viridis", s=220, vmin=0, vmax=3, edgecolor="k", zorder=3)
    for _, r in agg.iterrows():
        ax.annotate(r["quant_label"], (r["size_gb"], r["tps"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=10)
    ax.set_xlabel("Model file size (GB)")
    ax.set_ylabel("Generation throughput (tokens/sec)")
    ax.set_title("Part A: Size vs Speed vs Quality")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Avg quality score (0-3)")
    ax.grid(True, alpha=0.3)
    _save(fig, plots_dir, "quant_size_vs_speed")

    # 2) Grouped tradeoff bars (normalized)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(agg))
    width = 0.25
    norm_size = agg["size_gb"] / agg["size_gb"].max()
    norm_tps = agg["tps"] / agg["tps"].max()
    norm_q = agg["quality"] / 3.0
    ax.bar([i - width for i in x], norm_size, width, label="Size (norm)")
    ax.bar(list(x), norm_tps, width, label="Tokens/s (norm)")
    ax.bar([i + width for i in x], norm_q, width, label="Quality (norm)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(agg["quant_label"])
    ax.set_ylabel("Normalized value")
    ax.set_title("Part A: Quantization trade-offs (normalized)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, plots_dir, "quant_tradeoff")

    # 3) Quality by category heat-ish grouped bars
    pivot = df.pivot_table(index="category", columns="quant_label",
                           values="quality_score", aggfunc="mean")
    cols = [c for c in QUANT_ORDER if c in pivot.columns]
    pivot = pivot[cols]
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Quality score (0-3)")
    ax.set_ylim(0, 3.2)
    ax.set_title("Part A: Quality by prompt category and quantization")
    ax.legend(title="Quant")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, plots_dir, "quant_quality_by_category")


# ---------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------
def plot_kv_cache(csv_path: Path, plots_dir: Path) -> None:
    if not csv_path.exists():
        logger.warning("No kv_cache.csv at %s — skipping Part B plots.", csv_path)
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return

    # 1) Context vs latency (tokens/s and TTFT)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    for kv, grp in df.groupby("kv_cache_type"):
        grp = grp.sort_values("context_length")
        ax1.plot(grp["context_length"], grp["tokens_per_sec"], marker="o",
                 label=f"tok/s ({kv})")
        ax2.plot(grp["context_length"], grp["ttft_proxy_s"], marker="s",
                 linestyle="--", label=f"TTFT ({kv})")
    ax1.set_xlabel("Context length (tokens)")
    ax1.set_ylabel("Tokens/sec")
    ax2.set_ylabel("TTFT proxy (s)")
    ax1.set_xscale("log", base=2)
    ax1.set_title("Part B: Context length vs latency")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)
    _save(fig, plots_dir, "kv_context_vs_latency")

    # 2) Context vs RAM
    fig, ax = plt.subplots(figsize=(8, 5))
    for kv, grp in df.groupby("kv_cache_type"):
        grp = grp.sort_values("context_length")
        ax.plot(grp["context_length"], grp["peak_ollama_rss_mb"], marker="o",
                label=f"Peak RSS ({kv})")
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Peak Ollama RSS (MB)")
    ax.set_xscale("log", base=2)
    ax.set_title("Part B: Context length vs peak RAM")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, plots_dir, "kv_context_vs_ram")


def generate_all() -> None:
    cfg = load_config()
    plots_dir = cfg.path("paths.plots_dir")
    plot_quantization(cfg.path("paths.measurements_csv"), plots_dir)
    plot_kv_cache(cfg.path("paths.results_dir") / "kv_cache.csv", plots_dir)
    print(f"Plots written to {plots_dir}")


if __name__ == "__main__":
    generate_all()
