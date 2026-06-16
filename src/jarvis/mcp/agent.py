"""Function-calling agent that uses the web_search tool via Ollama.

The agent sends the tool schema to the model through ``/api/chat``. If the model
emits a tool call, we execute it locally (``search_server.call_tool``), append the
result, and ask the model to produce a final answer. A bounded loop prevents
runaway tool calls.

Run the two required end-to-end demo tasks:
    python -m jarvis.mcp.agent --demo
Ask your own:
    python -m jarvis.mcp.agent --task "Who won the 2022 FIFA World Cup final?"
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console

from ..core.config import load_config
from ..core.logging_utils import get_logger
from ..core.ollama_client import OllamaClient
from .search_server import TOOL_SCHEMA, call_tool

import time

logger = get_logger(__name__)
console = Console()

SYSTEM = (
    "You are Jarvis, a local AI assistant with access to a web_search tool. "
    "When a question needs current, niche, or post-training information, call "
    "web_search. After receiving results, synthesize a concise answer and cite "
    "the source URLs. If results are empty or irrelevant, say so honestly."
)

# Two tasks that genuinely require live web search.
DEMO_TASKS = [
    "What is the latest stable version of the Python programming language, and what is one notable feature it added?",
    "Find the current population of Tokyo and name one recent news headline about the city.",
]


@dataclass
class AgentTrace:
    task: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    success: bool = False
    error: str = ""
    rounds: int = 0


class WebSearchAgent:
    def __init__(self, model: str | None = None, max_rounds: int = 4):
        cfg = load_config()
        self.client = OllamaClient()
        self.model = model or cfg.get("mcp.model")
        self.max_rounds = max_rounds

    def run(self, task: str) -> AgentTrace:
        trace = AgentTrace(task=task)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": task},
        ]
        try:
            for _ in range(self.max_rounds):
                trace.rounds += 1
                resp = self.client.chat(self.model, messages, tools=TOOL_SCHEMA)
                msg = resp.get("message", {})
                tool_calls = msg.get("tool_calls") or []

                if not tool_calls:
                    trace.final_answer = (msg.get("content") or "").strip()
                    trace.success = bool(trace.final_answer)
                    break

                # Keep the assistant turn that requested the tool.
                messages.append(msg)
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    raw_args = fn.get("arguments", {})
                    args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args)
                    console.print(f"[cyan]→ tool call:[/] {name}({args})")
                    result = call_tool(name, args)
                    trace.tool_calls.append({"name": name, "arguments": args,
                                             "result": result})
                    messages.append({
                        "role": "tool",
                        "content": json.dumps(result)[:4000],
                    })
                    if name == "web_search":
                        time.sleep(2)
            else:
                trace.error = "max tool-call rounds exceeded"
        except Exception as exc:  # noqa: BLE001
            trace.error = str(exc)
            logger.error("Agent failed on task '%s': %s", task, exc)
        return trace


def _print_trace(trace: AgentTrace) -> None:
    console.rule(f"[bold]{trace.task}")
    for tc in trace.tool_calls:
        n = len(tc["result"].get("results", [])) if isinstance(tc["result"], dict) else 0
        console.print(f"  [dim]{tc['name']} -> {n} results[/dim]")
    status = "[green]SUCCESS" if trace.success else f"[red]FAIL ({trace.error})"
    console.print(f"Status: {status}")
    console.print(f"Answer: {trace.final_answer or '(none)'}\n")


def run_demo() -> list[AgentTrace]:
    agent = WebSearchAgent()
    traces = [agent.run(t) for t in DEMO_TASKS]
    for t in traces:
        _print_trace(t)
    cfg = load_config()
    out = cfg.path("paths.outputs_dir") / "mcp_demo_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([t.__dict__ for t in traces], indent=2), encoding="utf-8")
    console.print(f"[bold green]Saved -> {out}")
    return traces


def main() -> None:
    ap = argparse.ArgumentParser(description="Web-search function-calling agent")
    ap.add_argument("--demo", action="store_true", help="Run the 2 demo tasks")
    ap.add_argument("--task", help="Run a single custom task")
    args = ap.parse_args()
    if args.demo:
        run_demo()
    elif args.task:
        agent = WebSearchAgent()
        _print_trace(agent.run(args.task))
    else:
        ap.error("Pass --demo or --task")


if __name__ == "__main__":
    main()
