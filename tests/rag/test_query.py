"""Tests for dual retrieval + grounded answer synthesis. All LLM calls injected
as simple callables; retrieval uses InMemoryStore + DeterministicEmbedder."""

from __future__ import annotations

from typing import ClassVar

from llm_playground.rag.embeddings import DeterministicEmbedder
from llm_playground.rag.models import Chunk, SearchResult
from llm_playground.rag.query import (
    Answer,
    Retriever,
    answer_question,
    build_prompt,
    retrieve,
)
from llm_playground.rag.stores import InMemoryStore


def _make_retriever(docs: list[tuple[str, str]], emb: DeterministicEmbedder) -> Retriever:
    store = InMemoryStore()
    chunks = [Chunk(text=t, source=s, chunk_index=0) for s, t in docs]
    store.add(chunks, emb.embed([c.text for c in chunks]))
    return Retriever(emb, store)


def test_retriever_search_returns_results() -> None:
    emb = DeterministicEmbedder(dim=16)
    r = _make_retriever([("p1", "attention mechanism"), ("p2", "banana bread")], emb)
    results = r.search("attention mechanism", top_k=1)
    assert results[0].source == "p1"


def test_retrieve_paper_route_only_hits_papers() -> None:
    emb = DeterministicEmbedder(dim=16)
    papers = _make_retriever([("paper", "prose about transformers")], emb)
    code = _make_retriever([("code", "def forward(): pass")], emb)
    results = retrieve("transformers", "paper", paper_retriever=papers, code_retriever=code)
    assert all(r.source == "paper" for r in results)


def test_retrieve_both_route_merges_and_sorts() -> None:
    emb = DeterministicEmbedder(dim=16)
    papers = _make_retriever([("paper", "shared token")], emb)
    code = _make_retriever([("code", "different content")], emb)
    results = retrieve("shared token", "both", paper_retriever=papers, code_retriever=code, top_k=5)
    sources = {r.source for r in results}
    assert sources == {"paper", "code"}
    # Sorted by score descending — the exact-match "paper" ranks first.
    assert results[0].source == "paper"
    assert results == sorted(results, key=lambda r: r.score, reverse=True)


def test_retrieve_top_k_caps_merged_results() -> None:
    emb = DeterministicEmbedder(dim=16)
    papers = _make_retriever([(f"p{i}", f"text {i}") for i in range(5)], emb)
    code = _make_retriever([(f"c{i}", f"code {i}") for i in range(5)], emb)
    results = retrieve("text", "both", paper_retriever=papers, code_retriever=code, top_k=3)
    assert len(results) == 3


def test_build_prompt_numbers_and_grounds() -> None:
    contexts = [
        SearchResult(text="ctx A", source="a", score=0.9),
        SearchResult(text="ctx B", source="b", score=0.8),
    ]
    prompt = build_prompt("What is A?", contexts)
    assert "[1]" in prompt and "[2]" in prompt
    assert "ONLY the context" in prompt
    assert "What is A?" in prompt


def test_answer_question_end_to_end_with_fake_generator() -> None:
    emb = DeterministicEmbedder(dim=16)
    papers = _make_retriever([("paper.pdf", "the answer is 42")], emb)
    code = _make_retriever([("model.py", "def f(): pass")], emb)

    # Fake generator echoes a fixed answer; we assert plumbing, not model quality.
    def fake_generate(prompt: str) -> str:
        assert "42" in prompt  # retrieved context reached the prompt
        return "The answer is 42 [1]."

    result = answer_question(
        "what is the answer?",
        "paper",
        paper_retriever=papers,
        code_retriever=code,
        generate=fake_generate,
    )
    assert isinstance(result, Answer)
    assert result.answer == "The answer is 42 [1]."
    assert result.sources == ["paper.pdf"]
    assert result.route == "paper"


def test_answer_question_dedupes_sources() -> None:
    emb = DeterministicEmbedder(dim=16)
    # Two chunks from the SAME source -> sources list should collapse to one.
    store = InMemoryStore()
    chunks = [
        Chunk(text="chunk one", source="same.pdf", chunk_index=0),
        Chunk(text="chunk two", source="same.pdf", chunk_index=1),
    ]
    store.add(chunks, emb.embed([c.text for c in chunks]))
    r = Retriever(emb, store)
    result = answer_question(
        "chunk", "paper", paper_retriever=r, code_retriever=r, generate=lambda p: "ok", top_k=2
    )
    assert result.sources == ["same.pdf"]


def test_make_anthropic_generator_wraps_client(monkeypatch) -> None:
    """Cover make_anthropic_generator by injecting a fake anthropic client that
    returns content blocks; verify text-block concatenation."""
    import sys
    import types

    class _Block:
        def __init__(self, text, type="text"):
            self.text = text
            self.type = type

    class _Resp:
        # ClassVar annotation: this fake's content is a fixed class-level list,
        # not per-instance state (ruff RUF012 guards against accidental shared
        # mutable defaults; here it's intentional and read-only).
        content: ClassVar = [_Block("Hello "), _Block("world"), _Block("ignored", type="tool_use")]

    class _Messages:
        def create(self, **kwargs):
            return _Resp()

    class _Client:
        def __init__(self, *a, **k):
            self.messages = _Messages()

    fake = types.ModuleType("anthropic")
    fake.Anthropic = _Client
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    from llm_playground.rag.query import make_anthropic_generator

    gen = make_anthropic_generator()
    # Only text blocks concatenated; tool_use block ignored.
    assert gen("some prompt") == "Hello world"
