"""Unit tests that do NOT require a running Ollama server."""
from __future__ import annotations

from jarvis.benchmarks.prompts import PROMPTS, score_response
from jarvis.core.config import load_config
from jarvis.rag.chunking import chunk_text, count_tokens
from jarvis.eval.runner import _score


def test_config_loads():
    cfg = load_config()
    assert cfg.get("models.primary")
    assert cfg.get("quantization.completion_tokens") == 200


def test_prompts_present():
    cats = {p.category for p in PROMPTS}
    assert cats == {"math", "code", "summarization", "factual", "reasoning"}


def test_scoring_math():
    assert score_response("math", "The average speed is 42.86 km/h") == 3
    assert score_response("math", "It is about 100 km/h") == 0


def test_scoring_factual():
    assert score_response("factual", "Canberra; painted by Leonardo da Vinci") == 3
    assert score_response("factual", "Sydney; Picasso") == 0


def test_scoring_reasoning():
    assert score_response("reasoning", "Yes, all bloops are lazzies via razzies") == 3


def test_chunking_overlap():
    text = " ".join([f"Sentence number {i}." for i in range(200)])
    chunks = chunk_text(text, "doc.txt", chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(c.token_count <= 130 for c in chunks)  # size + small slack
    assert all(c.source == "doc.txt" for c in chunks)


def test_count_tokens():
    assert count_tokens("hello world") >= 2


def test_eval_scorer_refusal():
    test = {"eval": {"expect_refusal": True}}
    passed, _ = _score(test, "I cannot help with that request.", [])
    assert passed
    passed, _ = _score(test, "Sure, here is how...", [])
    assert not passed


def test_eval_scorer_keywords():
    test = {"eval": {"keywords_all": ["graphics", "processing", "unit"]}}
    passed, _ = _score(test, "Graphics Processing Unit", [])
    assert passed


def test_eval_scorer_tool():
    test = {"eval": {"expect_tool": "web_search"}}
    passed, _ = _score(test, "answer", ["web_search"])
    assert passed
    passed, _ = _score(test, "answer", [])
    assert not passed
