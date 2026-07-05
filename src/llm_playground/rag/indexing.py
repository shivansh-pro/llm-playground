"""Indexing: the pipeline that turns raw documents into a searchable store.

    document text -> chunk -> embed -> store.add(...)

This module is pure orchestration with dependency injection: it receives an
Embedder and a VectorStore (both Protocols) plus a chunking function. That makes
the whole flow unit-testable with a DeterministicEmbedder + InMemoryStore, no
Chroma/Postgres/torch required. Production wires in the real implementations at
the composition root.
"""

from __future__ import annotations

from collections.abc import Callable

from llm_playground.rag.chunking import chunk_code, chunk_prose
from llm_playground.rag.embeddings import Embedder
from llm_playground.rag.models import Chunk
from llm_playground.rag.stores import VectorStore

# A "document" is just (source_path, text). A ChunkFn maps one document to chunks.
Document = tuple[str, str]
ChunkFn = Callable[[str, str], list[Chunk]]


def index_documents(
    documents: list[Document],
    *,
    embedder: Embedder,
    store: VectorStore,
    chunk_fn: ChunkFn,
) -> int:
    """Chunk, embed, and store a batch of documents. Returns #chunks indexed.

    Batched embedding: we collect ALL chunks across ALL documents and embed them
    in one call. Embedding is dominated by per-call overhead (model warmup, GPU
    transfer, network for API embedders), so batching is a real throughput win —
    an interview-relevant "why not embed one at a time" point.
    """
    all_chunks: list[Chunk] = []
    for source, text in documents:
        all_chunks.extend(chunk_fn(source, text))

    if not all_chunks:
        return 0

    embeddings = embedder.embed([c.text for c in all_chunks])
    store.add(all_chunks, embeddings)
    return len(all_chunks)


def build_paper_index(
    documents: list[Document],
    *,
    embedder: Embedder,
    store: VectorStore,
    chunk_size: int = 512,
    overlap: int = 64,
) -> int:
    """Index prose documents (papers) with sentence-aware chunking."""
    return index_documents(
        documents,
        embedder=embedder,
        store=store,
        chunk_fn=lambda source, text: chunk_prose(
            text, source, chunk_size=chunk_size, overlap=overlap
        ),
    )


def build_code_index(
    documents: list[Document],
    *,
    embedder: Embedder,
    store: VectorStore,
) -> int:
    """Index code documents with AST-aware chunking."""
    return index_documents(
        documents,
        embedder=embedder,
        store=store,
        chunk_fn=lambda source, text: chunk_code(text, source),
    )
