"""Query: dual retrieval (papers + code) plus grounded answer synthesis.

This is the heart of the Paper<->Repo Mapper: a question may need the paper
index, the code index, or both. We retrieve from the requested sources, merge
the hits, and ask an LLM to answer using ONLY that context (grounding), citing
sources. The LLM call is injected as a `Generator` callable so the retrieval and
prompt-assembly logic is unit-testable without any API.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel

from llm_playground.rag.embeddings import Embedder
from llm_playground.rag.models import SearchResult
from llm_playground.rag.stores import VectorStore

Route = Literal["paper", "code", "both"]

# A Generator turns an assembled prompt into an answer string. In production this
# wraps Anthropic; in tests it's a lambda. Injecting it keeps this module free of
# API keys and network.
Generator = Callable[[str], str]


class Retriever:
    """Bundles an embedder with a vector store into a `search(query)` unit.

    Embedding the query and searching the store are always done together, so we
    package them. Composition over inheritance: a Retriever HAS-A store and
    embedder rather than IS-A store.
    """

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_vec = self._embedder.embed([query])[0]
        return self._store.query(query_vec, top_k=top_k)


class Answer(BaseModel):
    """Structured answer with provenance. Returned to callers and eval."""

    question: str
    route: Route
    answer: str
    sources: list[str]  # distinct source paths that grounded the answer


def retrieve(
    question: str,
    route: Route,
    *,
    paper_retriever: Retriever,
    code_retriever: Retriever,
    top_k: int = 5,
) -> list[SearchResult]:
    """Retrieve from the index(es) implied by `route`, merged and re-sorted.

    For "both" we take top_k from EACH then merge and re-sort by score, keeping
    the globally best top_k. (A more advanced system would use reciprocal-rank
    fusion; simple score merge is enough to demonstrate the pattern.)
    """
    hits: list[SearchResult] = []
    if route in ("paper", "both"):
        hits.extend(paper_retriever.search(question, top_k=top_k))
    if route in ("code", "both"):
        hits.extend(code_retriever.search(question, top_k=top_k))
    hits.sort(key=lambda r: r.score, reverse=True)
    return hits[:top_k]


def build_prompt(question: str, contexts: list[SearchResult]) -> str:
    """Assemble a grounded prompt. Numbered contexts let the model cite [1],[2].

    Grounding instruction is explicit: answer only from context, say "I don't
    know" otherwise. This is the single most effective anti-hallucination lever
    in RAG and a guaranteed interview talking point.
    """
    context_block = "\n\n".join(
        f"[{i + 1}] (source: {c.source})\n{c.text}" for i, c in enumerate(contexts)
    )
    return (
        "Answer the question using ONLY the context below. If the context is "
        "insufficient, say you don't know. Cite sources as [n].\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def answer_question(
    question: str,
    route: Route,
    *,
    paper_retriever: Retriever,
    code_retriever: Retriever,
    generate: Generator,
    top_k: int = 5,
) -> Answer:
    """Full RAG query: retrieve -> assemble prompt -> generate -> structure.

    Returns an Answer with de-duplicated source list for provenance/eval.
    """
    contexts = retrieve(
        question,
        route,
        paper_retriever=paper_retriever,
        code_retriever=code_retriever,
        top_k=top_k,
    )
    prompt = build_prompt(question, contexts)
    text = generate(prompt)
    # Preserve first-seen order while de-duplicating sources. dict.fromkeys is
    # the idiomatic ordered-dedupe in modern Python (dicts keep insertion order).
    sources = list(dict.fromkeys(c.source for c in contexts))
    return Answer(question=question, route=route, answer=text, sources=sources)


def make_anthropic_generator(model: str = "claude-sonnet-4-5") -> Generator:
    """Build a Generator backed by Anthropic. Lazy import; integration-only.

    Returned closure captures the client so callers just do generate(prompt).
    """
    import anthropic

    client = anthropic.Anthropic()

    def _generate(prompt: str) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        # Anthropic returns a list of content blocks; concatenate text blocks.
        return "".join(block.text for block in resp.content if block.type == "text")

    return _generate
