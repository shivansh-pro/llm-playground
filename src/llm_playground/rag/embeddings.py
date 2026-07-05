"""Embeddings: turn text into vectors for similarity search.

An `Embedder` is anything with `.embed(texts) -> list[list[float]]`. Defining
it as a Protocol (structural typing) means callers depend on the *shape*, not a
concrete class — so tests can inject a deterministic fake and production can use
sentence-transformers, with zero changes to the code in between.

    - SentenceTransformerEmbedder: real model (all-MiniLM-L6-v2, 384-dim).
    - DeterministicEmbedder: hash-based fake for tests. Same text -> same vector,
      different text -> different vector, no model download, instant. It is NOT
      semantically meaningful (it can't tell "cat" is near "kitten"), but it is
      perfect for testing plumbing: storage roundtrips, dimensionality, that the
      right chunk comes back for an *exact* query.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Structural interface for embedding backends.

    @runtime_checkable lets `isinstance(x, Embedder)` work at runtime (checks
    for an `.embed` attribute). Useful in tests/asserts; note it only checks
    method NAMES, not signatures — structural typing is nominal-lite.
    """

    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector (len == dim) per text."""
        ...


class DeterministicEmbedder:
    """Deterministic, dependency-free embedder for unit tests.

    Maps each text to a fixed-length vector derived from its SHA-256 hash.
    Properties that make it test-friendly:
        - Deterministic: embed(["x"]) is always identical.
        - Collision-resistant: distinct texts get distinct vectors.
        - Normalized to unit length so cosine similarity is well-behaved.
    """

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        # Expand the 32-byte digest to `dim` floats by hashing with a counter.
        raw: list[float] = []
        counter = 0
        while len(raw) < self.dim:
            digest = hashlib.sha256(f"{counter}:{text}".encode()).digest()
            raw.extend(b / 255.0 for b in digest)
            counter += 1
        vec = raw[: self.dim]
        # L2-normalize so all vectors sit on the unit sphere.
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]


class SentenceTransformerEmbedder:
    """Real embedder backed by sentence-transformers (all-MiniLM-L6-v2, 384-d).

    The model is loaded lazily on first `.embed` so importing this module (and
    unit-testing code that merely references the type) doesn't pull in torch.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.dim = 384  # known for MiniLM-L6-v2
        self._model = None  # loaded on first use

    def _ensure_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure_model()
        assert self._model is not None
        vectors = self._model.encode(texts, normalize_embeddings=True)
        # sentence-transformers returns a numpy array; convert to plain lists so
        # the return type matches the Protocol and serializes cleanly.
        return [v.tolist() for v in vectors]
