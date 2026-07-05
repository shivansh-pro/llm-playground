"""Tests for vector stores: cosine math, the in-memory store, and Chroma (both
mocked-unit and real-integration)."""

from __future__ import annotations

import sys
import types

import pytest

from llm_playground.rag.models import Chunk
from llm_playground.rag.stores import (
    InMemoryStore,
    VectorStore,
    cosine_similarity,
)


# --- cosine_similarity (pure) ------------------------------------------------
def test_cosine_identical_vectors_is_one() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_clamped_to_zero() -> None:
    # Raw cosine of opposite vectors is -1; we clamp to 0 for a 0..1 relevance.
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == 0.0


def test_cosine_zero_vector_is_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_dim_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="dim mismatch"):
        cosine_similarity([1.0], [1.0, 2.0])


# --- InMemoryStore -----------------------------------------------------------
def _chunk(text: str, i: int) -> Chunk:
    return Chunk(text=text, source=f"doc{i}", chunk_index=i)


def test_inmemory_add_length_mismatch_raises() -> None:
    store = InMemoryStore()
    with pytest.raises(ValueError, match="same length"):
        store.add([_chunk("a", 0)], [[1.0], [2.0]])


def test_inmemory_returns_most_similar_first() -> None:
    store = InMemoryStore()
    chunks = [_chunk("apple", 0), _chunk("banana", 1), _chunk("cherry", 2)]
    embeddings = [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]
    store.add(chunks, embeddings)
    # Query aligned with chunk0's vector -> chunk0 should rank first, chunk2
    # (close to [1,0]) second, chunk1 (orthogonal) last.
    results = store.query([1.0, 0.0], top_k=3)
    assert [r.source for r in results] == ["doc0", "doc2", "doc1"]
    assert results[0].score == pytest.approx(1.0)


def test_inmemory_respects_top_k() -> None:
    store = InMemoryStore()
    store.add([_chunk(str(i), i) for i in range(5)], [[float(i), 1.0] for i in range(5)])
    assert len(store.query([1.0, 1.0], top_k=2)) == 2


def test_inmemory_satisfies_protocol() -> None:
    # Structural typing: InMemoryStore has add+query, so it IS a VectorStore.
    store: VectorStore = InMemoryStore()
    assert hasattr(store, "add") and hasattr(store, "query")


# --- ChromaStore (mocked unit) -----------------------------------------------
def _install_fake_chromadb(monkeypatch, query_return: dict) -> dict:
    """Inject a fake `chromadb` module. Records add() calls and returns a
    canned query result so we cover ChromaStore's translation logic (distance
    -> similarity, id/metadata assembly) without a real Chroma."""
    recorded: dict = {}

    class _FakeCollection:
        def add(self, **kwargs):
            recorded.update(kwargs)

        def query(self, **kwargs):
            recorded["query_kwargs"] = kwargs
            return query_return

    class _FakeClient:
        def get_or_create_collection(self, **kwargs):
            return _FakeCollection()

    fake = types.ModuleType("chromadb")
    fake.EphemeralClient = lambda: _FakeClient()
    fake.PersistentClient = lambda path: _FakeClient()
    monkeypatch.setitem(sys.modules, "chromadb", fake)
    return recorded


def test_chromastore_add_builds_ids_and_metadata(monkeypatch) -> None:
    recorded = _install_fake_chromadb(monkeypatch, query_return={})
    from llm_playground.rag.stores import ChromaStore

    store = ChromaStore(collection_name="c")
    store.add([Chunk(text="t", source="s.py", chunk_index=3, metadata={"k": "v"})], [[0.1, 0.2]])
    assert recorded["ids"] == ["s.py:3"]
    assert recorded["documents"] == ["t"]
    assert recorded["metadatas"][0]["source"] == "s.py"
    assert recorded["metadatas"][0]["k"] == "v"


def test_chromastore_query_converts_distance_to_similarity(monkeypatch) -> None:
    _install_fake_chromadb(
        monkeypatch,
        query_return={
            "documents": [["doc text"]],
            "metadatas": [[{"source": "s.py"}]],
            "distances": [[0.25]],  # cosine distance
        },
    )
    from llm_playground.rag.stores import ChromaStore

    store = ChromaStore()
    results = store.query([0.1, 0.2], top_k=1)
    assert results[0].source == "s.py"
    assert results[0].score == pytest.approx(0.75)  # 1 - 0.25


# --- Integration: real Chroma (ephemeral, no server) -------------------------
@pytest.mark.integration
def test_chromastore_real_roundtrip(  # pragma: no cover - integration
) -> None:
    from llm_playground.rag.embeddings import DeterministicEmbedder
    from llm_playground.rag.stores import ChromaStore

    emb = DeterministicEmbedder(dim=16)
    store = ChromaStore(collection_name="itest")
    chunks = [
        Chunk(text=t, source=f"d{i}", chunk_index=i) for i, t in enumerate(["cat", "dog", "car"])
    ]
    store.add(chunks, emb.embed([c.text for c in chunks]))
    results = store.query(emb.embed(["cat"])[0], top_k=1)
    assert results[0].text == "cat"
