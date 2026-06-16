"""End-to-end local RAG pipeline.

Commands (via ``python -m jarvis.rag.pipeline ...``):
  download  - fetch the arXiv corpus
  index     - chunk + embed + store the corpus in Chroma
  ask       - answer a single question (RAG)
  compare   - run the 5 test questions with vs. without RAG and save results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

from ..core.config import load_config
from ..core.logging_utils import get_logger
from ..core.ollama_client import OllamaClient
from .chunking import chunk_text
from .corpus import download_arxiv, load_documents
from .vectorstore import VectorStore

logger = get_logger(__name__)
console = Console()

RAG_SYSTEM = (
    "You are a research assistant. Answer the question using ONLY the provided "
    "context. If the context does not contain the answer, say you don't know. "
    "Cite the source filename(s) in square brackets."
)
PLAIN_SYSTEM = "You are a helpful research assistant. Answer concisely."

# 5 questions answerable from the default arXiv corpus.
TEST_QUESTIONS = [
    "What problem does the Transformer architecture replace recurrence with, and why?",
    "What are the two pre-training objectives used by BERT?",
    "How does LoRA reduce the number of trainable parameters during fine-tuning?",
    "What is retrieval-augmented generation (RAG) and what components does it combine?",
    "What technique does GPT-3 rely on to perform tasks without gradient updates?",
]


class RAGPipeline:
    def __init__(self):
        self.cfg = load_config()
        self.client = OllamaClient()
        self.corpus_dir = self.cfg.path("rag.corpus_dir")
        self.store = VectorStore(
            persist_dir=self.cfg.path("rag.vector_store_dir"),
            collection_name=self.cfg.get("rag.collection_name", "jarvis_papers"),
            embedding_model=self.cfg.get("rag.embedding_model", "nomic-embed-text"),
        )
        self.gen_model = self.cfg.get("rag.generation_model")
        self.top_k = self.cfg.get("rag.top_k", 4)

    # ----- ingestion -------------------------------------------------------
    def download(self) -> None:
        download_arxiv(self.corpus_dir)

    def index(self, reset: bool = True) -> int:
        if reset:
            self.store.reset()
        docs = load_documents(self.corpus_dir)
        if not docs:
            raise RuntimeError(
                f"No documents found in {self.corpus_dir}. Run `download` first."
            )
        size = self.cfg.get("rag.chunk_size", 800)
        overlap = self.cfg.get("rag.chunk_overlap", 120)
        ids, texts, metas = [], [], []
        for source, text in docs:
            for ch in chunk_text(text, source, size, overlap):
                ids.append(f"{source}::{ch.chunk_index}")
                texts.append(ch.text)
                metas.append({"source": source, "chunk": ch.chunk_index,
                              "tokens": ch.token_count})
        self.store.add(ids, texts, metas)
        console.print(f"[green]Indexed {len(ids)} chunks from {len(docs)} documents.")
        return len(ids)

    # ----- retrieval / generation -----------------------------------------
    def retrieve(self, question: str) -> list[dict]:
        return self.store.query(question, top_k=self.top_k)

    def answer(self, question: str, use_rag: bool = True) -> dict:
        if use_rag:
            hits = self.retrieve(question)
            context = "\n\n".join(
                f"[{h['metadata']['source']}] {h['text']}" for h in hits
            )
            prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            res = self.client.generate(self.gen_model, prompt, system=RAG_SYSTEM,
                                        num_predict=300)
            sources = [h["metadata"]["source"] for h in hits]
        else:
            res = self.client.generate(self.gen_model, question, system=PLAIN_SYSTEM,
                                        num_predict=300)
            sources = []
        return {
            "question": question,
            "use_rag": use_rag,
            "answer": res.text.strip(),
            "sources": sources,
            "tokens_per_sec": res.tokens_per_sec,
            "completion_tokens": res.completion_tokens,
        }

    def compare(self, questions: list[str] | None = None) -> list[dict]:
        questions = questions or TEST_QUESTIONS
        if self.store.count() == 0:
            raise RuntimeError("Vector store is empty — run `index` first.")
        results = []
        for q in questions:
            console.rule(f"[cyan]{q}")
            with_rag = self.answer(q, use_rag=True)
            without_rag = self.answer(q, use_rag=False)
            console.print(f"[green]RAG:[/] {with_rag['answer'][:300]}")
            console.print(f"[yellow]No-RAG:[/] {without_rag['answer'][:300]}")
            results.append({"question": q, "with_rag": with_rag,
                            "without_rag": without_rag})
        out = self.cfg.path("paths.outputs_dir") / "rag_comparison.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        console.print(f"[bold green]Saved comparison -> {out}")
        return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Local RAG pipeline")
    ap.add_argument("command", choices=["download", "index", "ask", "compare"])
    ap.add_argument("--question", "-q", help="Question for `ask`")
    ap.add_argument("--no-rag", action="store_true", help="Disable RAG for `ask`")
    args = ap.parse_args()

    pipe = RAGPipeline()
    if args.command == "download":
        pipe.download()
    elif args.command == "index":
        pipe.index()
    elif args.command == "ask":
        if not args.question:
            ap.error("--question is required for `ask`")
        res = pipe.answer(args.question, use_rag=not args.no_rag)
        console.print_json(data=res)
    elif args.command == "compare":
        pipe.compare()


if __name__ == "__main__":
    main()
