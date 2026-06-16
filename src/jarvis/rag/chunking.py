"""Token-aware text chunking for the RAG corpus.

We split documents into overlapping chunks measured in tokens (via tiktoken's
``cl100k_base`` encoding, a good general proxy). Overlap preserves context across
chunk boundaries. The splitter tries to break on paragraph / sentence boundaries
for cleaner chunks before falling back to a hard token cut.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    token_count: int


def _split_sentences(text: str) -> list[str]:
    # Split on paragraph breaks first, then sentences.
    parts: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", para)
        parts.extend(s for s in sentences if s.strip())
    return parts


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Split ``text`` into overlapping token-bounded chunks."""
    sentences = _split_sentences(text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_tokens = 0
    idx = 0

    def flush(buffer: list[str]) -> None:
        nonlocal idx
        if not buffer:
            return
        body = " ".join(buffer).strip()
        if body:
            chunks.append(
                Chunk(text=body, source=source, chunk_index=idx,
                      token_count=count_tokens(body))
            )
            idx += 1

    for sent in sentences:
        st = count_tokens(sent)
        # Very long sentence: hard-split by tokens.
        if st > chunk_size:
            flush(current)
            current, current_tokens = [], 0
            ids = _ENC.encode(sent)
            for start in range(0, len(ids), chunk_size - chunk_overlap):
                piece = _ENC.decode(ids[start:start + chunk_size])
                chunks.append(
                    Chunk(text=piece, source=source, chunk_index=idx,
                          token_count=count_tokens(piece))
                )
                idx += 1
            continue

        if current_tokens + st > chunk_size:
            flush(current)
            # Start next chunk with an overlap tail from the previous one.
            tail: list[str] = []
            tail_tokens = 0
            for s in reversed(current):
                t = count_tokens(s)
                if tail_tokens + t > chunk_overlap:
                    break
                tail.insert(0, s)
                tail_tokens += t
            current = tail + [sent]
            current_tokens = tail_tokens + st
        else:
            current.append(sent)
            current_tokens += st

    flush(current)
    return chunks
