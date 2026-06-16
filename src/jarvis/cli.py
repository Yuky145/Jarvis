"""Unified command-line interface for Local Jarvis.

Examples
--------
    jarvis doctor                 # environment / Ollama health check
    jarvis chat                   # interactive local chat
    jarvis bench-quant            # Part A
    jarvis bench-kv               # Part B
    jarvis rag download|index|compare
    jarvis mcp-demo               # Part D demo tasks
    jarvis eval                   # Part E
    jarvis plots                  # regenerate figures
    jarvis analyze                # Part F report
    jarvis run-all                # full pipeline (long!)
"""
from __future__ import annotations

import click
from rich.console import Console

from .core.config import load_config
from .core.logging_utils import setup_logging
from .core.metrics import total_ram_gb
from .core.ollama_client import OllamaClient

console = Console()


@click.group()
@click.option("--log-level", default="INFO")
def cli(log_level: str) -> None:
    setup_logging(level=log_level)


@cli.command()
def doctor() -> None:
    """Check Ollama connectivity, RAM, and installed models."""
    cfg = load_config()
    client = OllamaClient()
    console.print(f"Machine RAM: [bold]{total_ram_gb()} GB[/]")
    up = client.is_up()
    console.print(f"Ollama server: {'[green]up[/]' if up else '[red]down[/]'} "
                  f"({client.host})")
    if up:
        names = [m.get("name") for m in client.list_models()]
        console.print(f"Installed models ({len(names)}):")
        for n in names:
            console.print(f"  • {n}")
        for key in ("models.primary", "models.embedding"):
            tag = cfg.get(key)
            ok = client.has_model(tag)
            console.print(f"  {key}: {tag} -> {'[green]present[/]' if ok else '[yellow]missing[/]'}")
    else:
        console.print("[yellow]Start it with `ollama serve`.[/]")


@cli.command()
@click.option("--model", default=None)
def chat(model: str | None) -> None:
    """Interactive local chat (Ctrl-C to exit)."""
    cfg = load_config()
    client = OllamaClient()
    model = model or cfg.get("models.primary")
    if not client.has_model(model):
        console.print(f"[yellow]Model {model} not found; trying default q4 tag.[/]")
        model = cfg.get("evaluation.model")
    console.print(f"[bold green]Jarvis[/] ready with [cyan]{model}[/]. Ctrl-C to quit.")
    history: list[dict[str, str]] = [
        {"role": "system", "content": "You are Jarvis, a concise local assistant."}
    ]
    try:
        while True:
            user = console.input("[bold blue]you ›[/] ")
            if not user.strip():
                continue
            history.append({"role": "user", "content": user})
            resp = client.chat(model, history)
            msg = resp.get("message", {}).get("content", "")
            history.append({"role": "assistant", "content": msg})
            console.print(f"[bold green]jarvis ›[/] {msg}")
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]bye[/]")


@cli.command("bench-quant")
@click.option("--dry-run", is_flag=True)
def bench_quant(dry_run: bool) -> None:
    """Part A: quantization benchmark."""
    from .benchmarks.quantization import run
    run(dry_run=dry_run)


@cli.command("bench-kv")
@click.option("--dry-run", is_flag=True)
def bench_kv(dry_run: bool) -> None:
    """Part B: KV cache benchmark."""
    from .benchmarks.kv_cache import run
    run(dry_run=dry_run)


@cli.command()
@click.argument("action", type=click.Choice(["download", "index", "compare"]))
def rag(action: str) -> None:
    """Part C: RAG pipeline operations."""
    from .rag.pipeline import RAGPipeline
    pipe = RAGPipeline()
    {"download": pipe.download, "index": pipe.index, "compare": pipe.compare}[action]()


@cli.command("mcp-demo")
def mcp_demo() -> None:
    """Part D: run the web-search demo tasks."""
    from .mcp.agent import run_demo
    run_demo()


@cli.command("eval")
@click.option("--category", default=None)
@click.option("--model", default=None)
def eval_cmd(category: str | None, model: str | None) -> None:
    """Part E: run the evaluation suite."""
    from .eval.runner import Evaluator
    Evaluator(model=model).run(category=category)


@cli.command()
def plots() -> None:
    """Regenerate all benchmark plots."""
    from .benchmarks.plots import generate_all
    generate_all()


@cli.command()
def analyze() -> None:
    """Part F: generate the analysis & reflection report."""
    from .eval.analysis import generate
    generate()


@cli.command("run-all")
def run_all() -> None:
    """Run the full pipeline end-to-end (Parts A-F). This is long."""
    from .benchmarks.quantization import run as run_quant
    from .benchmarks.kv_cache import run as run_kv
    from .benchmarks.plots import generate_all
    from .rag.pipeline import RAGPipeline
    from .mcp.agent import run_demo
    from .eval.runner import Evaluator
    from .eval.analysis import generate

    run_quant()
    run_kv()
    pipe = RAGPipeline(); pipe.download(); pipe.index(); pipe.compare()
    run_demo()
    Evaluator().run()
    generate_all()
    generate()
    console.print("[bold green]Full pipeline complete. See outputs/ and benchmarks/.[/]")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
