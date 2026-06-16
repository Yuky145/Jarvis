"""A minimal MCP-style web-search tool server.

This module exposes a single tool, ``Web Search``, plus a ``fetch_url`` helper.
It speaks a tiny JSON-RPC-like protocol over stdin/stdout so it behaves like an
MCP server (``list_tools`` / ``call_tool``), and it is *also* importable as a
plain Python function for the in-process function-calling agent in ``agent.py``.

Backends:
  * duckduckgo  — no API key required (default)
  * brave       — set BRAVE_API_KEY env var

Run as a standalone server (reads one JSON request per line on stdin):
    python -m jarvis.mcp.search_server
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from typing import Any

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

from ..core.config import load_config
from ..core.logging_utils import get_logger

logger = get_logger(__name__)

# MCP-style tool schema (also reused as the Ollama function-calling schema).
TOOL_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "Web Search",
            "description": (
                "Search the public web for up-to-date information. Returns a list "
                "of results with title, url, and snippet. Use for current events, "
                "facts you are unsure about, or anything after your training cutoff."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results (1-10).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    }
]


def _search_duckduckgo(query: str, max_results: int) -> list[dict[str, str]]:
    from ddgs import DDGS

    results: list[dict[str, str]] = []
    last_exc: Exception | None = None

    for attempt in range(4):
        try:
            # Eliminamos la lista de user_agents y la variable headers.
            # Simplemente llamamos a DDGS con el timeout.
            with DDGS(timeout=30) as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    })
            return results
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt + random.uniform(0, 1)
            logger.warning("DuckDuckGo intento %d falló: %s. Esperando %.1fs...",
                           attempt + 1, exc, wait)
            time.sleep(wait)

    logger.error("DuckDuckGo falló después de 4 intentos: %s", last_exc)
    return []

def _search_brave(query: str, max_results: int) -> list[dict[str, str]]:
    key = os.getenv(load_config().get("mcp.brave_api_key_env", "BRAVE_API_KEY"))
    if not key:
        raise RuntimeError("BRAVE_API_KEY not set.")
    resp = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={"X-Subscription-Token": key, "Accept": "application/json"},
        params={"q": query, "count": max_results},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    out = []
    for r in data.get("web", {}).get("results", [])[:max_results]:
        out.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("description", ""),
        })
    return out


def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Execute a web search using the configured backend with graceful fallback."""
    cfg = load_config()
    backend = cfg.get("mcp.search_backend", "duckduckgo")
    max_results = max(1, min(int(max_results), 10))
    try:
        if backend == "brave":
            return _search_brave(query, max_results)
        return _search_duckduckgo(query, max_results)
    except Exception as exc:
        logger.error("Search backend '%s' failed: %s", backend, exc)
        if backend != "duckduckgo":
            try:
                return _search_duckduckgo(query, max_results)
            except Exception as exc2:
                logger.error("Fallback search failed: %s", exc2)
        return []


def fetch_url(url: str, max_chars: int | None = None) -> str:
    """Fetch a URL and return cleaned visible text (truncated)."""
    cfg = load_config()
    max_chars = max_chars or cfg.get("mcp.fetch_page_chars", 2000)
    try:
        resp = requests.get(
            url, headers={"User-Agent": "Mozilla/5.0 (local-jarvis)"}, timeout=30
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        return text[:max_chars]
    except requests.RequestException as exc:
        logger.error("fetch_url failed for %s: %s", url, exc)
        return ""


# ---------------------------------------------------------------------------
# MCP-style dispatch
# ---------------------------------------------------------------------------
def list_tools() -> dict[str, Any]:
    return {"tools": TOOL_SCHEMA}


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name in ("Web Search", "Web Search"):
        results = web_search(
            arguments.get("query", ""), arguments.get("max_results", 5)
        )
        return {"results": results}
    if name == "fetch_url":
        return {"text": fetch_url(arguments.get("url", ""))}
    return {"error": f"unknown tool: {name}"}


def _serve_stdio() -> None:
    """Very small line-delimited JSON-RPC loop (MCP-ish)."""
    logger.info("MCP search server ready (stdio). Send one JSON request per line.")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            method = req.get("method")
            if method == "list_tools":
                resp = {"id": req.get("id"), "result": list_tools()}
            elif method == "call_tool":
                params = req.get("params", {})
                resp = {"id": req.get("id"),
                        "result": call_tool(params.get("name"), params.get("arguments", {}))}
            else:
                resp = {"id": req.get("id"), "error": f"unknown method: {method}"}
        except Exception as exc:
            resp = {"error": str(exc)}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    _serve_stdio()