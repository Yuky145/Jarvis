"""The 5 standardized benchmark prompts and their automatic quality rubric.

Each prompt targets a distinct capability:
  1. math          - arithmetic / multi-step calculation
  2. code          - small programming task
  3. summarization - condense a passage
  4. factual       - factual recall
  5. reasoning     - logical / commonsense reasoning

Quality is scored 0-3 by :func:`score_response`:
  3 = fully correct & complete
  2 = mostly correct, minor error/omission
  1 = partially correct
  0 = wrong / no answer

The scorer uses deterministic keyword + structural heuristics so the whole
benchmark is reproducible without a human or a second "judge" LLM. The rubric
and reference answers are documented inline for transparency.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class BenchmarkPrompt:
    id: str
    category: str
    prompt: str
    system: str = "You are a concise, accurate assistant."
    # A function (response_text) -> int in [0, 3].
    scorer: Callable[[str], int] = field(repr=False, default=lambda r: 0)
    reference: str = ""


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _contains_all(text: str, needles: list[str]) -> int:
    t = _norm(text)
    return sum(1 for n in needles if n.lower() in t)


# ---- 1. Math --------------------------------------------------------------
# Q: A train travels 60 km in 1.5 h, then 90 km in 2 h. Average speed?
# (60+90)/(1.5+2) = 150/3.5 = 42.857... ≈ 42.86 km/h
def _score_math(resp: str) -> int:
    t = _norm(resp)
    # Accept 42.8, 42.86, 42.857, or the fraction 150/3.5.
    if re.search(r"42\.8[0-9]?", t) or "42.86" in t or "300/7" in t:
        return 3
    if "42.9" in t or re.search(r"\b43\b", t):  # rounded reasonably
        return 2
    if "150" in t and "3.5" in t:  # right setup, no final answer
        return 1
    return 0


# ---- 2. Code --------------------------------------------------------------
# Q: Write a Python function is_palindrome(s) ignoring case & non-alphanumerics.
def _score_code(resp: str) -> int:
    t = resp.lower()
    has_def = "def is_palindrome" in t.replace(" ", "") or "def is_palindrome(" in t
    has_reverse = "[::-1]" in resp or "reversed(" in t
    has_clean = any(k in t for k in ["isalnum", "re.sub", "lower()"])
    score = 0
    if has_def:
        score += 1
    if has_reverse or "==" in resp:
        score += 1
    if has_clean:
        score += 1
    return min(score, 3)


# ---- 3. Summarization -----------------------------------------------------
# Score: must be shorter than source, mention key concepts, single paragraph.
_SUMMARY_KEYWORDS = ["photosynthesis", "light", "glucose", "oxygen"]


def _score_summary(resp: str) -> int:
    hits = _contains_all(resp, _SUMMARY_KEYWORDS)
    words = len(resp.split())
    concise = words <= 60
    if hits >= 3 and concise:
        return 3
    if hits >= 2:
        return 2
    if hits >= 1:
        return 1
    return 0


# ---- 4. Factual recall ----------------------------------------------------
# Q: Capital of Australia + who painted the Mona Lisa.
def _score_factual(resp: str) -> int:
    t = _norm(resp)
    canberra = "canberra" in t
    davinci = "da vinci" in t or "leonardo" in t
    return (2 if canberra else 0) + (1 if davinci else 0) if (canberra or davinci) else 0


# ---- 5. Reasoning ---------------------------------------------------------
# Q: All bloops are razzies. All razzies are lazzies. Are all bloops lazzies?
def _score_reasoning(resp: str) -> int:
    t = _norm(resp)
    yes = bool(re.search(r"\b(yes|all bloops are lazzies|true)\b", t))
    explained = "razzie" in t and "lazzie" in t
    if yes and explained:
        return 3
    if yes:
        return 2
    if explained:
        return 1
    return 0


PROMPTS: list[BenchmarkPrompt] = [
    BenchmarkPrompt(
        id="math",
        category="math",
        prompt=(
            "A train travels 60 km in 1.5 hours, then 90 km in 2 hours. "
            "What is its average speed for the whole trip in km/h? "
            "Show your steps and give the final number."
        ),
        scorer=_score_math,
        reference="(60+90)/(1.5+2) = 150/3.5 ≈ 42.86 km/h",
    ),
    BenchmarkPrompt(
        id="code",
        category="code",
        prompt=(
            "Write a Python function `is_palindrome(s)` that returns True if the "
            "string is a palindrome, ignoring case, spaces, and punctuation. "
            "Return only the function."
        ),
        scorer=_score_code,
        reference="def is_palindrome(s): t=[c.lower() for c in s if c.isalnum()]; return t==t[::-1]",
    ),
    BenchmarkPrompt(
        id="summarization",
        category="summarization",
        prompt=(
            "Summarize the following in one sentence (max 40 words): "
            "'Photosynthesis is the process by which green plants, algae, and some "
            "bacteria convert light energy, usually from the sun, into chemical "
            "energy stored in glucose. During this process, carbon dioxide and water "
            "are used, and oxygen is released as a byproduct.'"
        ),
        scorer=_score_summary,
        reference="Plants use light to convert CO2 and water into glucose, releasing oxygen.",
    ),
    BenchmarkPrompt(
        id="factual",
        category="factual",
        prompt="What is the capital of Australia, and who painted the Mona Lisa?",
        scorer=_score_factual,
        reference="Canberra; Leonardo da Vinci.",
    ),
    BenchmarkPrompt(
        id="reasoning",
        category="reasoning",
        prompt=(
            "All bloops are razzies. All razzies are lazzies. "
            "Are all bloops definitely lazzies? Answer yes or no and explain briefly."
        ),
        scorer=_score_reasoning,
        reference="Yes — transitivity: bloops⊆razzies⊆lazzies.",
    ),
]


def score_response(prompt_id: str, response: str) -> int:
    """Score ``response`` for the prompt with ``prompt_id`` (0-3)."""
    for p in PROMPTS:
        if p.id == prompt_id:
            try:
                return int(max(0, min(3, p.scorer(response))))
            except Exception:
                return 0
    raise KeyError(f"Unknown prompt id: {prompt_id}")
