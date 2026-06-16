"""Corpus acquisition and loading for the RAG pipeline.

``download_arxiv`` fetches a small set of open-access arXiv PDFs (well over the
required 50 pages combined) into ``data/corpus``. ``load_documents`` reads every
PDF / .txt / .md file in the corpus directory and returns ``(source, text)``
pairs. PDF text is extracted with ``pypdf``.
"""
from __future__ import annotations

from pathlib import Path

import requests
from pypdf import PdfReader
from rich.console import Console
from tqdm import tqdm

from ..core.logging_utils import get_logger

logger = get_logger(__name__)
console = Console()

# Foundational, freely downloadable arXiv papers (collectively > 50 pages).
DEFAULT_ARXIV_PDFS: dict[str, str] = {
    "attention_is_all_you_need.pdf": "https://arxiv.org/pdf/1706.03762",
    "bert.pdf": "https://arxiv.org/pdf/1810.04805",
    "gpt3_few_shot_learners.pdf": "https://arxiv.org/pdf/2005.14165",
    "llama2.pdf": "https://arxiv.org/pdf/2307.09288",
    "rag_knowledge_intensive.pdf": "https://arxiv.org/pdf/2005.11401",
    "lora.pdf": "https://arxiv.org/pdf/2106.09685",
}


def download_arxiv(corpus_dir: Path, papers: dict[str, str] | None = None) -> list[Path]:
    """Download the default (or provided) arXiv PDFs into ``corpus_dir``."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    papers = papers or DEFAULT_ARXIV_PDFS
    saved: list[Path] = []
    headers = {"User-Agent": "Mozilla/5.0 (local-jarvis RAG corpus fetcher)"}

    for name, url in tqdm(papers.items(), desc="Downloading corpus", unit="paper"):
        dest = corpus_dir / name
        if dest.exists() and dest.stat().st_size > 10_000:
            saved.append(dest)
            continue
        try:
            resp = requests.get(url, headers=headers, timeout=60)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            saved.append(dest)
            logger.info("Saved %s (%.1f KB)", name, len(resp.content) / 1024)
        except requests.RequestException as exc:  # noqa: BLE001
            logger.error("Failed to download %s: %s", url, exc)
    return saved


def _read_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not parse PDF %s: %s", path.name, exc)
        return ""


def load_documents(corpus_dir: Path) -> list[tuple[str, str]]:
    """Load all PDF/txt/md documents -> list of (source_name, text)."""
    docs: list[tuple[str, str]] = []
    for path in sorted(corpus_dir.glob("**/*")):
        if path.suffix.lower() == ".pdf":
            text = _read_pdf(path)
        elif path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
        else:
            continue
        if text and text.strip():
            docs.append((path.name, text))
    logger.info("Loaded %d documents from %s", len(docs), corpus_dir)
    return docs
