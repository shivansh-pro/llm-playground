"""Tests for embedders: the deterministic test double + the ST wrapper (mocked)."""

from __future__ import annotations

import pytest

from llm_playground.rag.embeddings import (
    DeterministicEmbedder,
    Embedder,
    SentenceTransformerEmbedder,
)


def test_deterministic_embedder_dim_and_shape() -> None:
    emb = DeterministicEmbedder(dim=16)
    vecs = emb.embed(["hello", "world"])
    assert len(vecs) == 2
    assert all(len(v) == 16 for v in vecs)


def test_deterministic_is_deterministic() -> None:
    emb = DeterministicEmbedder(dim=16)
    assert emb.embed(["same"])[0] == emb.embed(["same"])[0]


def test_deterministic_distinct_texts_distinct_vectors() -> None:
    emb = DeterministicEmbedder(dim=16)
    assert emb.embed(["cat"])[0] != emb.embed(["dog"])[0]


def test_deterministic_vectors_are_unit_norm() -> None:
    emb = DeterministicEmbedder(dim=16)
    v = emb.embed(["normalize me"])[0]
    norm = sum(x * x for x in v) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-9)


def test_deterministic_satisfies_embedder_protocol() -> None:
    # @runtime_checkable Protocol: isinstance checks for an .embed method.
    assert isinstance(DeterministicEmbedder(), Embedder)


def test_st_embedder_reports_known_dim_without_loading() -> None:
    # Constructing must NOT load the model (no torch import cost); dim is known.
    emb = SentenceTransformerEmbedder()
    assert emb.dim == 384
    assert emb._model is None  # not loaded until first embed


def test_st_embedder_lazy_loads_and_converts(monkeypatch) -> None:
    """Mock the model so we cover the wrapper (lazy load + numpy->list) without
    downloading MiniLM. We patch _ensure_model to inject a fake encoder."""

    class _FakeVec(list):
        def tolist(self):  # emulate numpy .tolist()
            return list(self)

    class _FakeModel:
        def encode(self, texts, normalize_embeddings=True):
            return [_FakeVec([0.1, 0.2, 0.3]) for _ in texts]

    emb = SentenceTransformerEmbedder()
    monkeypatch.setattr(emb, "_ensure_model", lambda: setattr(emb, "_model", _FakeModel()))
    out = emb.embed(["a", "b"])
    assert out == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]


@pytest.mark.integration
def test_st_embedder_real_model(  # pragma: no cover - integration
) -> None:
    emb = SentenceTransformerEmbedder()
    vecs = emb.embed(["hello world"])
    assert len(vecs[0]) == 384
