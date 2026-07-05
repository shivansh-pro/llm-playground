"""Shared data models for the RAG layer.

We use Pydantic BaseModel (not plain dataclasses) here because these objects
cross boundaries: they are produced by chunkers, stored in vector DBs, returned
over the MCP server, and serialized into eval reports. Pydantic gives us
validation + `.model_dump()`/`.model_dump_json()` for free at those boundaries.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A unit of text produced by a chunker, ready to embed and store.

    `metadata` is deliberately open (dict) so different chunkers can attach
    different context: prose chunkers add nothing special; code chunkers add
    start/end line numbers so a retrieved snippet can be traced back to source.
    """

    text: str
    source: str  # file path or document id the chunk came from
    chunk_index: int  # position of this chunk within its source (0-based)
    metadata: dict = Field(default_factory=dict)


class SearchResult(BaseModel):
    """One hit from a vector-store similarity search.

    `score` is similarity in [0, 1] where higher = more similar. Different
    backends return different raw scores (cosine distance, L2, inner product);
    each store adapter normalizes to this convention so callers don't care
    which backend produced it. That normalization is the whole point of the
    VectorStore Protocol.
    """

    text: str
    source: str
    score: float
    metadata: dict = Field(default_factory=dict)
