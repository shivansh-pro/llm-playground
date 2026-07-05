"""Vector stores: where embeddings live and how similarity search happens.

THE INTERVIEW POINT — Chroma vs pgvector (and the wider landscape):

    Chroma      In-process (or lightweight server), purpose-built for vectors.
                Zero-config to start (`EphemeralClient()` needs no server).
                Great for prototyping and single-service apps. Metadata filtering
                built in. Weaker story when you also need relational joins.

    pgvector    A Postgres extension. Your vectors live in a SQL table next to
                your relational data, so you can `JOIN` embeddings against users,
                permissions, timestamps, etc., and reuse Postgres ops/backup/
                replication you already run. THE data-engineer answer. Cost:
                you manage Postgres, and ANN indexing (HNSW/IVFFlat) is a config
                you must understand.

    FAISS       A LIBRARY, not a server. Meta's ANN toolkit; provides the HNSW/
                IVF index implementations that many DBs wrap. Reach for it when
                search runs *inside* one Python process with no network hop and
                you want maximum speed. No metadata/filtering out of the box.

    Qdrant      Dedicated vector DB (Rust). Rich payload filtering, named vectors
                (multiple vectors per point), horizontal scale. The modern choice
                when search is the primary workload and you've outgrown in-process.

    HNSW vs IVFFlat (the index question interviewers love):
        HNSW    graph-based; builds a navigable small-world graph. High recall,
                fast queries, higher memory + slower build. Default for quality.
        IVFFlat cluster-based; partitions vectors into lists, searches the
                nearest few. Lower memory, faster build, tune `nprobe` for the
                recall/speed tradeoff.

All three concrete stores below satisfy the same `VectorStore` Protocol, so the
indexing/query/eval layers are backend-agnostic. Swapping Chroma for pgvector is
a one-line change at the composition root.
"""

from __future__ import annotations

import re
from typing import Protocol

from llm_playground.rag.models import Chunk, SearchResult


class VectorStore(Protocol):
    """Structural interface every backend implements.

    `add` takes chunks + their embeddings (parallel lists). `query` takes a
    single query embedding and returns the top_k most similar chunks as
    normalized SearchResults (score in [0,1], higher = better).
    """

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    def query(self, embedding: list[float], top_k: int = 5) -> list[SearchResult]: ...


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors, clamped to [0, 1].

    Cosine = dot(a,b) / (|a||b|). For normalized vectors it's just the dot
    product. We clamp negatives to 0 so the score reads as a 0..1 relevance.
    Pure + tested — this is the math every vector DB does for you.
    """
    if len(a) != len(b):
        raise ValueError(f"dim mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    sim = dot / (na * nb)
    return max(0.0, sim)


class InMemoryStore:
    """Pure-Python VectorStore: brute-force cosine search over a list.

    This is a real, working store (O(n) per query) with no dependencies. It
    exists to (a) unit-test the indexing/query/eval layers end-to-end without
    Chroma/Postgres, and (b) make the retrieval math visible. Production uses
    an ANN index (HNSW) to avoid the O(n) scan; the *interface* is identical.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._embeddings: list[list[float]] = []

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        self._chunks.extend(chunks)
        self._embeddings.extend(embeddings)

    def query(self, embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        scored = [
            SearchResult(
                text=chunk.text,
                source=chunk.source,
                score=cosine_similarity(embedding, emb),
                metadata=chunk.metadata,
            )
            # strict=True: chunks and embeddings are kept parallel by add();
            # a length mismatch is a bug we want to surface, not silently zip short.
            for chunk, emb in zip(self._chunks, self._embeddings, strict=True)
        ]
        # Sort by score descending, return top_k. `key` + reverse is the idiom.
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


class ChromaStore:
    """VectorStore backed by Chroma.

    Uses an ephemeral in-process client by default (no server needed), or a
    persistent path if given. Chroma stores vectors + documents + metadata
    together and returns cosine *distance*; we convert distance -> similarity
    as `1 - distance` so the score matches our [0,1]-higher-is-better contract.

    Lazy import so this module stays importable without chromadb. Real behavior
    is covered by an integration test (`@pytest.mark.integration`).
    """

    def __init__(self, collection_name: str = "default", persist_dir: str | None = None) -> None:
        import chromadb

        self._client = (
            chromadb.PersistentClient(path=persist_dir)
            if persist_dir
            else chromadb.EphemeralClient()
        )
        # cosine space so distances are comparable across query/embeddings.
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self._collection.add(
            ids=[f"{c.source}:{c.chunk_index}" for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[
                {"source": c.source, "chunk_index": c.chunk_index, **c.metadata} for c in chunks
            ],
        )

    def query(self, embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        res = self._collection.query(query_embeddings=[embedding], n_results=top_k)
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        return [
            SearchResult(
                text=doc,
                source=str(meta.get("source", "")),
                score=1.0 - float(dist),  # cosine distance -> similarity
                metadata=dict(meta),
            )
            for doc, meta, dist in zip(docs, metas, dists, strict=True)
        ]


# A SQL identifier we're willing to interpolate into DDL/DML. Postgres table
# names can't be parameterized with %s (that's for VALUES only), so we MUST
# build them into the query string — which means we must validate them
# ourselves. This regex is the allowlist: letter/underscore start, then
# word chars. Anything else (spaces, quotes, semicolons) is rejected, closing
# the identity-injection hole (e.g. table='x; DROP TABLE users; --').
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_identifier(name: str) -> str:
    """Return `name` if it is a safe SQL identifier, else raise ValueError.

    Pure + unit-tested. Used to guard the interpolated table name in
    PgVectorStore. For untrusted identifiers in real systems, prefer
    psycopg2.sql.Identifier, which quotes properly; an allowlist is the
    belt-and-suspenders version and makes intent explicit.
    """
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"unsafe SQL identifier: {name!r}")
    return name


class PgVectorStore:
    """VectorStore backed by Postgres + the pgvector extension.

    Schema (created on init if missing):
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE <table> (
            id        SERIAL PRIMARY KEY,
            source    TEXT,
            chunk_index INT,
            content   TEXT,
            metadata  JSONB,
            embedding VECTOR(<dim>)
        );
        CREATE INDEX ... USING hnsw (embedding vector_cosine_ops);

    Query uses the `<=>` cosine-distance operator; similarity = 1 - distance.
    The HNSW index is what makes this scale past a brute-force scan.

    Lifecycle: holds one psycopg2 connection. Use as a context manager
    (`with PgVectorStore(...) as store:`) or call `.close()` explicitly so the
    connection is released — a long-lived service that leaks connections will
    exhaust the Postgres pool.

    Lazy import (psycopg2); requires a running Postgres. Integration-tested.
    """

    def __init__(self, dsn: str, dim: int, table: str = "chunks") -> None:
        import psycopg2

        # Validate BEFORE connecting so a bad identifier fails fast and cheap
        # (and so the validation is unit-testable without a live DB via the
        # standalone validate_identifier function).
        self._table = validate_identifier(table)
        self._dim = dim
        self._conn = psycopg2.connect(dsn)
        try:
            self._ensure_schema()
        except Exception:
            # Don't leak the connection if schema creation fails during init.
            self._conn.close()
            raise

    # Context-manager protocol: `with PgVectorStore(...) as s: ...` closes the
    # connection on exit even if the body raises. This is the idiomatic Python
    # way to tie a resource's lifetime to a scope (like Java try-with-resources).
    def __enter__(self) -> PgVectorStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the DB connection. Safe to call more than once."""
        if getattr(self, "_conn", None) is not None and not self._conn.closed:
            self._conn.close()

    def _ensure_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self._table} ("
                "id SERIAL PRIMARY KEY, source TEXT, chunk_index INT, "
                f"content TEXT, metadata JSONB, embedding VECTOR({self._dim}));"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self._table}_emb_idx ON {self._table} "
                "USING hnsw (embedding vector_cosine_ops);"
            )
        self._conn.commit()

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        import json

        # Rollback on failure so a bad batch doesn't leave the connection in an
        # aborted-transaction state (InFailedSqlTransaction) that poisons every
        # subsequent query on this instance.
        try:
            with self._conn.cursor() as cur:
                for chunk, emb in zip(chunks, embeddings, strict=True):
                    cur.execute(
                        f"INSERT INTO {self._table} "
                        "(source, chunk_index, content, metadata, embedding) "
                        "VALUES (%s, %s, %s, %s, %s);",
                        (
                            chunk.source,
                            chunk.chunk_index,
                            chunk.text,
                            json.dumps(chunk.metadata),
                            emb,
                        ),
                    )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def query(self, embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT content, source, metadata, 1 - (embedding <=> %s::vector) AS similarity "
                f"FROM {self._table} ORDER BY embedding <=> %s::vector LIMIT %s;",
                (embedding, embedding, top_k),
            )
            rows = cur.fetchall()
        return [
            SearchResult(text=r[0], source=r[1] or "", score=float(r[3]), metadata=r[2] or {})
            for r in rows
        ]
