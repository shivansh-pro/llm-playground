"""Tests for chunking: strategy routing + the pure sliding-window chunker, plus
mocked and integration tests for the LlamaIndex-backed chunkers.

Coverage strategy:
    - Pure functions (select_strategy, language_for, chunk_text_fixed) get
      thorough unit tests.
    - chunk_prose / chunk_code call LlamaIndex. We cover them two ways:
        * unit: inject a FAKE llama_index submodule via sys.modules so the
          wrapper logic runs without the real lib (fast, hermetic, CI-safe).
        * integration: run the REAL splitter (@pytest.mark.integration; skipped
          in CI, run locally with `uv sync --all-extras`).
"""

from __future__ import annotations

import sys
import types

import pytest

from llm_playground.rag import chunking
from llm_playground.rag.chunking import (
    _chunk_code_by_lines,
    chunk_code,
    chunk_prose,
    chunk_text_fixed,
    language_for,
    select_strategy,
)


# --- select_strategy / language_for (pure) -----------------------------------
@pytest.mark.parametrize(
    "path,expected",
    [
        ("repos/nanoGPT/model.py", "code"),
        ("foo/Bar.JS", "code"),  # case-insensitive
        ("service.go", "code"),
        ("papers/attention.pdf", "prose"),
        ("notes/day1.md", "prose"),
        ("README", "prose"),  # no extension
    ],
)
def test_select_strategy(path: str, expected: str) -> None:
    assert select_strategy(path) == expected


@pytest.mark.parametrize(
    "path,lang",
    [("a.py", "python"), ("a.ts", "typescript"), ("a.rs", "rust"), ("a.pdf", None)],
)
def test_language_for(path: str, lang: str | None) -> None:
    assert language_for(path) == lang


# --- chunk_text_fixed (pure algorithm) ---------------------------------------
def test_fixed_empty_text_returns_empty() -> None:
    assert chunk_text_fixed("", "src") == []


def test_fixed_text_shorter_than_window_single_chunk() -> None:
    chunks = chunk_text_fixed("hello", "src", chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == "hello"
    assert chunks[0].chunk_index == 0


def test_fixed_window_and_overlap_math() -> None:
    # 20 chars, size=10, overlap=4 -> step=6. Windows start at 0, 6, 12.
    # At start=12 the window covers 12..20 and the loop breaks (end>=len), so
    # there is NO tiny trailing window at 18 — correct behavior, no runt chunks.
    text = "abcdefghijklmnopqrst"  # len 20
    chunks = chunk_text_fixed(text, "src", chunk_size=10, overlap=4)
    starts = [c.metadata["start_char"] for c in chunks]
    assert starts == [0, 6, 12]
    # Adjacent chunks overlap by exactly `overlap` characters.
    assert chunks[0].text[-4:] == chunks[1].text[:4]
    # chunk_index increments 0,1,2
    assert [c.chunk_index for c in chunks] == [0, 1, 2]


def test_fixed_last_chunk_may_be_short() -> None:
    text = "a" * 25
    chunks = chunk_text_fixed(text, "src", chunk_size=10, overlap=0)
    # step=10 -> windows 0-10,10-20,20-25 ; last chunk length 5
    assert len(chunks[-1].text) == 5


def test_fixed_overlap_ge_size_raises() -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunk_text_fixed("abc", "src", chunk_size=5, overlap=5)


# --- chunk_prose / chunk_code with a FAKE llama_index (mocked unit) -----------
def _install_fake_node_parser(monkeypatch, produced: list[str]) -> None:
    """Inject a fake `llama_index.core.node_parser` exposing SentenceSplitter
    and CodeSplitter, each returning `produced` from split_text. This exercises
    the wrapper logic (Chunk assembly, metadata) without the real library."""

    class _FakeSplitter:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def split_text(self, text: str) -> list[str]:
            return produced

    fake_mod = types.ModuleType("llama_index.core.node_parser")
    fake_mod.SentenceSplitter = _FakeSplitter
    fake_mod.CodeSplitter = _FakeSplitter
    monkeypatch.setitem(sys.modules, "llama_index.core.node_parser", fake_mod)


def test_chunk_prose_assembles_chunks(monkeypatch) -> None:
    _install_fake_node_parser(monkeypatch, ["piece one", "piece two"])
    chunks = chunk_prose("irrelevant text", "papers/x.pdf")
    assert [c.text for c in chunks] == ["piece one", "piece two"]
    assert [c.chunk_index for c in chunks] == [0, 1]
    assert all(c.metadata["strategy"] == "prose" for c in chunks)
    assert all(c.source == "papers/x.pdf" for c in chunks)


def test_chunk_code_infers_language_and_marks_metadata(monkeypatch) -> None:
    _install_fake_node_parser(monkeypatch, ["def f(): pass"])
    chunks = chunk_code("def f(): pass", "repos/nanoGPT/model.py")
    assert chunks[0].metadata["strategy"] == "code"
    assert chunks[0].metadata["language"] == "python"  # inferred from .py


# --- line-window fallback (pure) ---------------------------------------------
def test_chunk_code_by_lines_windows_and_overlap() -> None:
    text = "\n".join(f"line{i}" for i in range(10))  # 10 lines
    # chunk_lines=4, overlap=1 -> step=3. Windows start at lines 0,3,6,9.
    pieces = _chunk_code_by_lines(text, chunk_lines=4, overlap_lines=1)
    assert pieces[0] == "line0\nline1\nline2\nline3"
    # overlap: last line of piece0 == first line of piece1
    assert pieces[0].splitlines()[-1] == pieces[1].splitlines()[0]


def test_chunk_code_by_lines_empty() -> None:
    assert _chunk_code_by_lines("", 4, 1) == []


def test_chunk_code_by_lines_overlap_ge_chunk_raises() -> None:
    with pytest.raises(ValueError, match="overlap_lines"):
        _chunk_code_by_lines("a\nb", 2, 2)


def test_chunk_code_falls_back_when_ast_parser_fails(monkeypatch) -> None:
    """If CodeSplitter raises (e.g. tree-sitter version mismatch), chunk_code
    must degrade to line-window chunking and record the reason — never crash."""

    class _BrokenSplitter:
        def __init__(self, *a, **k):
            pass

        def split_text(self, text):
            raise TypeError("simulated tree-sitter binding mismatch")

    fake_mod = types.ModuleType("llama_index.core.node_parser")
    fake_mod.CodeSplitter = _BrokenSplitter
    fake_mod.SentenceSplitter = _BrokenSplitter
    monkeypatch.setitem(sys.modules, "llama_index.core.node_parser", fake_mod)

    code = "\n".join(f"line{i}" for i in range(50))
    chunks = chunk_code(code, "x.py")
    assert len(chunks) >= 1
    assert chunks[0].metadata["strategy"] == "code-fallback"
    assert chunks[0].metadata["fallback_reason"] == "TypeError"


def test_chunk_document_dispatches_by_extension(monkeypatch) -> None:
    # Patch the two chunkers to record which one was called.
    calls: list[str] = []
    monkeypatch.setattr(chunking, "chunk_prose", lambda *a, **k: calls.append("prose") or [])
    monkeypatch.setattr(chunking, "chunk_code", lambda *a, **k: calls.append("code") or [])
    chunking.chunk_document("text", "a.py")
    chunking.chunk_document("text", "a.pdf")
    assert calls == ["code", "prose"]


# --- Integration: real LlamaIndex splitters ----------------------------------
@pytest.mark.integration
def test_chunk_code_real_produces_chunks(  # pragma: no cover - integration
) -> None:
    """With the real libs installed, chunk_code produces chunks via EITHER the
    AST path ('code') or the graceful fallback ('code-fallback') depending on
    the environment's tree-sitter compatibility. Both are acceptable; the
    pipeline must not crash."""
    code = "def a():\n    return 1\n\n\ndef b():\n    return 2\n" * 10
    chunks = chunk_code(code, "x.py")
    assert len(chunks) >= 1
    assert all(c.metadata["language"] == "python" for c in chunks)
    assert all(c.metadata["strategy"] in ("code", "code-fallback") for c in chunks)


@pytest.mark.integration
def test_chunk_prose_real_splits_long_text(  # pragma: no cover - integration
) -> None:
    text = "This is a sentence. " * 500
    chunks = chunk_prose(text, "x.pdf", chunk_size=128, overlap=16)
    assert len(chunks) > 1
