"""Chroma-backed vector store using local Ollama embeddings.

We implement a small Chroma ``EmbeddingFunction`` that calls the local
``nomic-embed-text`` model through Ollama, so no data leaves the machine.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from ..core.logging_utils import get_logger
from ..core.ollama_client import OllamaClient

logger = get_logger(__name__)


class OllamaEmbeddingFunction(EmbeddingFunction):
    """Chroma embedding function backed by a local Ollama embedding model."""

    def __init__(self, model: str = "nomic-embed-text", client: OllamaClient | None = None):
        self.model = model
        self.client = client or OllamaClient()

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002 (chroma API)
        return self.client.embed(self.model, list(input))


class VectorStore:
    """Persistent Chroma collection wrapper."""

    def __init__(
        self,
        persist_dir: str | Path,
        collection_name: str,
        embedding_model: str = "nomic-embed-text",
    ):
        self.persist_dir = str(persist_dir)
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._embed = OllamaEmbeddingFunction(embedding_model)
        self.collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embed,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        name = self.collection.name
        try:
            self._client.delete_collection(name)
        except Exception:  # noqa: BLE001
            pass
        self.collection = self._client.get_or_create_collection(
            name=name, embedding_function=self._embed,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids: list[str], texts: list[str], metadatas: list[dict[str, Any]]) -> None:
        # Batch to keep embedding requests reasonable in size.
        batch = 64
        for i in range(0, len(ids), batch):
            self.collection.add(
                ids=ids[i:i + batch],
                documents=texts[i:i + batch],
                metadatas=metadatas[i:i + batch],
            )

    def query(self, text: str, top_k: int = 4) -> list[dict[str, Any]]:
        res = self.collection.query(query_texts=[text], n_results=top_k)
        out: list[dict[str, Any]] = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            out.append({"text": doc, "metadata": meta, "distance": dist})
        return out
