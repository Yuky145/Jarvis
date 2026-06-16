"""A thin, well-instrumented wrapper around the Ollama HTTP API.

We use ``requests`` directly (rather than only the ``ollama`` python package)
so we get full access to timing fields returned by the server
(``prompt_eval_count``, ``eval_count``, ``eval_duration`` ...) which are needed
for accurate tokens/sec measurements.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Iterable

import requests

from .config import load_config
from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class GenerationResult:
    """Structured result from a single ``/api/generate`` call."""

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_duration_s: float
    load_duration_s: float
    prompt_eval_duration_s: float
    eval_duration_s: float
    wall_time_s: float

    @property
    def tokens_per_sec(self) -> float:
        """Generation throughput = completion tokens / eval duration."""
        if self.eval_duration_s <= 0:
            return 0.0
        return round(self.completion_tokens / self.eval_duration_s, 2)

    @property
    def prompt_tokens_per_sec(self) -> float:
        if self.prompt_eval_duration_s <= 0:
            return 0.0
        return round(self.prompt_tokens / self.prompt_eval_duration_s, 2)


class OllamaClient:
    """Client for talking to a local Ollama server."""

    def __init__(self, host: str | None = None, timeout: int | None = None):
        cfg = load_config()
        self.host = (host or cfg.get("ollama.host", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout or cfg.get("ollama.request_timeout", 600)
        self.keep_alive = cfg.get("ollama.keep_alive", "5m")
        gen = cfg.get("generation", {}) or {}
        self.default_options: dict[str, Any] = {
            "temperature": gen.get("temperature", 0.0),
            "top_p": gen.get("top_p", 1.0),
            "seed": gen.get("seed", 42),
        }

    # ------------------------------------------------------------------
    # Server / model management
    # ------------------------------------------------------------------
    def is_up(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[dict[str, Any]]:
        r = requests.get(f"{self.host}/api/tags", timeout=10)
        r.raise_for_status()
        return r.json().get("models", [])

    def has_model(self, tag: str) -> bool:
        names = {m.get("name", "") for m in self.list_models()}
        # Ollama tags default ":latest" suffix.
        return tag in names or f"{tag}:latest" in names or any(
            n.split(":")[0] == tag for n in names
        )

    def model_size_bytes(self, tag: str) -> int | None:
        for m in self.list_models():
            if m.get("name") in (tag, f"{tag}:latest"):
                return m.get("size")
        return None

    def pull(self, tag: str) -> None:
        """Pull a model, streaming progress to the log."""
        logger.info("Pulling model %s ...", tag)
        with requests.post(
            f"{self.host}/api/pull",
            json={"name": tag, "stream": True},
            stream=True,
            timeout=self.timeout,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if "error" in data:
                    raise RuntimeError(f"Pull failed for {tag}: {data['error']}")

    def unload(self, model: str) -> None:
        """Ask the server to evict a model from RAM (keep_alive=0)."""
        try:
            requests.post(
                f"{self.host}/api/generate",
                json={"model": model, "prompt": "", "keep_alive": 0},
                timeout=30,
            )
        except requests.RequestException:
            pass

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        num_predict: int | None = None,
        num_ctx: int | None = None,
        options: dict[str, Any] | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> GenerationResult:
        """Run a single non-streaming completion and capture timing metrics."""
        opts = dict(self.default_options)
        if options:
            opts.update(options)
        if num_predict is not None:
            opts["num_predict"] = num_predict
        if num_ctx is not None:
            opts["num_ctx"] = num_ctx

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": opts,
            "keep_alive": self.keep_alive,
        }
        if system:
            payload["system"] = system
        if extra_payload:
            payload.update(extra_payload)

        t0 = time.perf_counter()
        r = requests.post(
            f"{self.host}/api/generate", json=payload, timeout=self.timeout
        )
        wall = time.perf_counter() - t0
        r.raise_for_status()
        data = r.json()

        ns = 1e9  # durations are reported in nanoseconds
        return GenerationResult(
            text=data.get("response", ""),
            model=model,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            total_duration_s=round(data.get("total_duration", 0) / ns, 4),
            load_duration_s=round(data.get("load_duration", 0) / ns, 4),
            prompt_eval_duration_s=round(data.get("prompt_eval_duration", 0) / ns, 4),
            eval_duration_s=round(data.get("eval_duration", 0) / ns, 4),
            wall_time_s=round(wall, 4),
        )

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        tools: list[dict[str, Any]] | None = None,
        num_ctx: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call the ``/api/chat`` endpoint (used for tool/function calling)."""
        opts = dict(self.default_options)
        if options:
            opts.update(options)
        if num_ctx is not None:
            opts["num_ctx"] = num_ctx
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": opts,
            "keep_alive": self.keep_alive,
        }
        if tools:
            payload["tools"] = tools
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def embed(self, model: str, text: str | Iterable[str]) -> list[list[float]]:
        """Return embedding vectors for one or more input strings."""
        inputs = [text] if isinstance(text, str) else list(text)
        r = requests.post(
            f"{self.host}/api/embed",
            json={"model": model, "input": inputs},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json().get("embeddings", [])
