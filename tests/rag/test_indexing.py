"""Tests for the indexing orchestration, using DeterministicEmbedder +
InMemoryStore so the whole flow runs with zero external dependencies."""

from __future__ import annotations

from llm_playground.rag.embeddings import DeterministicEmbedder
from llm_playground.rag.indexing import (
    build_paper_index,
    index_documents,
)
from llm_playground.rag.models import Chunk
from llm_playground.rag.stores import InMemoryStore


def _one_chunk_per_doc(source: str, text: str) -> list[Chunk]:
    """Trivial chunk_fn: one chunk == whole doc. Isolates orchestration logic
    from chunking specifics."""
    return [Chunk(text=text, source=source, chunk_index=0)]


def test_index_documents_returns_chunk_count() -> None:
    store = InMemoryStore()
    n = index_documents(
        [("a", "alpha"), ("b", "beta")],
        embedder=DeterministicEmbedder(dim=8),
        store=store,
        chunk_fn=_one_chunk_per_doc,
    )
    assert n == 2


def test_index_documents_stores_are_retrievable() -> None:
    store = InMemoryStore()
    emb = DeterministicEmbedder(dim=8)
    index_documents(
        [("a", "alpha"), ("b", "beta")],
        embedder=emb,
        store=store,
        chunk_fn=_one_chunk_per_doc,
    )
    # Query with the exact text of doc "a" -> should come back first.
    results = store.query(emb.embed(["alpha"])[0], top_k=1)
    assert results[0].source == "a"


def test_index_documents_empty_returns_zero() -> None:
    store = InMemoryStore()
    n = index_documents(
        [], embedder=DeterministicEmbedder(), store=store, chunk_fn=_one_chunk_per_doc
    )
    assert n == 0


def test_index_documents_batches_embedding(monkeypatch) -> None:
    """All chunks across all docs are embedded in ONE call (throughput). Assert
    embed() is invoked once with every chunk text."""
    store = InMemoryStore()
    calls: list[list[str]] = []

    class _SpyEmbedder:
        dim = 8

        def embed(self, texts):
            calls.append(texts)
            return [[0.0] * 8 for _ in texts]

    index_documents(
        [("a", "alpha"), ("b", "beta"), ("c", "gamma")],
        embedder=_SpyEmbedder(),
        store=store,
        chunk_fn=_one_chunk_per_doc,
    )
    assert len(calls) == 1
    assert calls[0] == ["alpha", "beta", "gamma"]


def test_build_paper_index_uses_prose_chunker(monkeypatch) -> None:
    """build_paper_index should route through chunk_prose. We patch it to a
    stub so we don't need llama-index, and verify wiring."""
    from llm_playground.rag import indexing

    monkeypatch.setattr(
        indexing,
        "chunk_prose",
        lambda text, source, **k: [Chunk(text=text, source=source, chunk_index=0)],
    )
    store = InMemoryStore()
    n = build_paper_index(
        [("paper.pdf", "content")], embedder=DeterministicEmbedder(dim=8), store=store
    )
    assert n == 1


def test_build_code_index_uses_code_chunker(monkeypatch) -> None:
    """build_code_index should route through chunk_code (AST-aware). Patched to
    a stub to avoid tree-sitter; verifies wiring."""
    from llm_playground.rag import indexing
    from llm_playground.rag.indexing import build_code_index

    monkeypatch.setattr(
        indexing,
        "chunk_code",
        lambda text, source, **k: [Chunk(text=text, source=source, chunk_index=0)],
    )
    store = InMemoryStore()
    n = build_code_index(
        [("model.py", "def f(): pass")], embedder=DeterministicEmbedder(dim=8), store=store
    )
    assert n == 1
