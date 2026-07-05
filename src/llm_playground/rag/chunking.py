"""Chunking: splitting documents into retrievable passages.

THE INTERVIEW POINT — why chunk at all, and why strategy depends on content:

    1. Context windows are finite. You can't stuff a 30-page PDF into a prompt.
    2. Retrieval precision. One embedding per chunk. If a chunk is huge, its
       embedding averages many topics into mush and retrieval gets vague.
       Smaller, focused chunks -> sharper embeddings -> better recall of the
       *specific* passage that answers a question.
    3. Cost/latency. Fewer tokens in the prompt = cheaper, faster.

PROSE vs AST-AWARE (the "prose vs code" contrast this module demonstrates):

    - Prose (papers, docs): split on sentence boundaries with overlap. Breaking
      mid-sentence is fine-ish; the danger is splitting a *fact* across a
      boundary. Overlap (repeating a little text between adjacent chunks)
      mitigates that — a fact near a boundary appears whole in one of them.

    - Code: sentence splitting is wrong. Splitting a function in half yields
      two chunks that are each meaningless. AST-aware splitting (LlamaIndex's
      CodeSplitter, backed by tree-sitter) respects syntactic boundaries so a
      function/class stays intact in one chunk. It also lets us attach
      start/end line numbers as metadata for precise source attribution.

This module provides:
    - `select_strategy(path)`  : pure routing logic (extension -> strategy)
    - `chunk_text_fixed(...)`  : pure reference implementation of sliding-window
                                 char chunking WITH overlap — no dependencies, so
                                 the algorithm itself is unit-tested and visible.
    - `chunk_prose(...)`       : idiomatic LlamaIndex SentenceSplitter (lazy)
    - `chunk_code(...)`        : idiomatic LlamaIndex CodeSplitter / AST (lazy)
"""

from __future__ import annotations

from typing import Literal

from llm_playground.rag.models import Chunk

Strategy = Literal["prose", "code"]

# Extensions we treat as source code -> AST-aware chunking. Everything else
# (.pdf, .md, .txt, ...) is treated as prose.
_CODE_EXTENSIONS: frozenset[str] = frozenset(
    {".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c", ".rb"}
)

# Map file extension -> tree-sitter language name used by CodeSplitter.
_EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "c",
    ".rb": "ruby",
}


def select_strategy(path: str) -> Strategy:
    """Decide chunking strategy from a file path's extension. Pure function.

    This is deliberately its own tested function: routing logic is exactly the
    kind of thing that silently rots (someone adds .tsx, forgets to handle it).
    A unit test pins the behavior.
    """
    lower = path.lower()
    for ext in _CODE_EXTENSIONS:
        if lower.endswith(ext):
            return "code"
    return "prose"


def language_for(path: str) -> str | None:
    """Return the tree-sitter language name for a code path, else None."""
    lower = path.lower()
    for ext, lang in _EXTENSION_TO_LANGUAGE.items():
        if lower.endswith(ext):
            return lang
    return None


def chunk_text_fixed(
    text: str,
    source: str,
    *,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    """Pure reference chunker: fixed-size sliding window over characters.

    This is NOT what you'd ship for prose (it ignores sentence boundaries), but
    it makes the two core concepts — window size and OVERLAP — concrete and
    testable without any library. Read it to understand what SentenceSplitter
    is doing under the hood in spirit.

    The `*` in the signature forces chunk_size/overlap to be passed by keyword
    (keyword-only arguments). Idiom: prevents call-site bugs like
    chunk_text_fixed(t, s, 64, 512) where the numbers are silently swapped.

    Invariants (all asserted in tests):
        - Each chunk (except possibly the last) has length == chunk_size.
        - Consecutive chunks overlap by `overlap` characters.
        - Concatenating chunks with overlaps removed reconstructs the text.
        - Empty text -> empty list.
    """
    if overlap >= chunk_size:
        # Guard: overlap >= size would never advance the window -> infinite loop.
        raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")
    if not text:
        return []

    chunks: list[Chunk] = []
    step = chunk_size - overlap  # how far the window advances each iteration
    start = 0
    index = 0
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end]
        chunks.append(
            Chunk(
                text=piece,
                source=source,
                chunk_index=index,
                metadata={"start_char": start, "end_char": min(end, len(text))},
            )
        )
        index += 1
        if end >= len(text):
            break
        start += step
    return chunks


def chunk_prose(
    text: str,
    source: str,
    *,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    """Sentence-aware prose chunking via LlamaIndex SentenceSplitter.

    SentenceSplitter tries to break on sentence boundaries and only falls back
    to hard splits when a sentence exceeds chunk_size. `chunk_size` here is in
    TOKENS (LlamaIndex default tokenizer), not characters — a subtlety worth
    knowing: "512" means ~512 tokens, roughly 2000 characters of English.

    Lazy import: keeps this module importable (and unit-testable) without
    llama-index installed. The real call is covered by an integration test.
    """
    from llama_index.core.node_parser import SentenceSplitter

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    pieces = splitter.split_text(text)
    return [
        Chunk(text=piece, source=source, chunk_index=i, metadata={"strategy": "prose"})
        for i, piece in enumerate(pieces)
    ]


def _chunk_code_by_lines(text: str, chunk_lines: int, overlap_lines: int) -> list[str]:
    """Line-window fallback chunker for code. Pure, dependency-free.

    Not AST-aware, but it respects LINE boundaries (never splits mid-line) and
    keeps a configurable overlap so a function spanning a boundary still appears
    whole in one window. Used when the tree-sitter AST parser is unavailable or
    version-incompatible in the current environment.
    """
    if overlap_lines >= chunk_lines:
        raise ValueError("overlap_lines must be < chunk_lines")
    lines = text.splitlines()
    if not lines:
        return []
    step = chunk_lines - overlap_lines
    pieces: list[str] = []
    start = 0
    while start < len(lines):
        window = lines[start : start + chunk_lines]
        pieces.append("\n".join(window))
        if start + chunk_lines >= len(lines):
            break
        start += step
    return pieces


def chunk_code(
    text: str,
    source: str,
    *,
    language: str | None = None,
    chunk_lines: int = 40,
    chunk_lines_overlap: int = 15,
    max_chars: int = 1500,
) -> list[Chunk]:
    """AST-aware code chunking via LlamaIndex CodeSplitter (tree-sitter), with a
    graceful line-based fallback.

    Primary path: CodeSplitter parses the source into a syntax tree and splits
    along node boundaries, so functions/classes stay whole — the whole point of
    "code vs prose" chunking.

    Robustness: CodeSplitter depends on tree-sitter grammars whose Python-binding
    API has churned across versions. Rather than let the indexing pipeline
    hard-crash when the installed tree-sitter is incompatible, we catch the
    failure and fall back to line-window chunking, recording BOTH the fallback
    and the reason in metadata (observability). Production code that ingests
    arbitrary repos should never die because one environment shipped a mismatched
    parser build.

    If `language` is None we infer it from the source path's extension.
    Lazy import; both paths covered by tests (AST path via integration).
    """
    lang = language or language_for(source) or "python"
    try:
        from llama_index.core.node_parser import CodeSplitter

        splitter = CodeSplitter(
            language=lang,
            chunk_lines=chunk_lines,
            chunk_lines_overlap=chunk_lines_overlap,
            max_chars=max_chars,
        )
        pieces = splitter.split_text(text)
        base_meta = {"strategy": "code", "language": lang}
    except Exception as exc:
        # Degrade, don't die. Narrow-ish: any failure constructing/using the AST
        # splitter (ImportError for missing grammar, TypeError from a binding
        # mismatch, ValueError on unparseable input) routes to the fallback.
        pieces = _chunk_code_by_lines(text, chunk_lines, chunk_lines_overlap)
        base_meta = {
            "strategy": "code-fallback",
            "language": lang,
            "fallback_reason": type(exc).__name__,
        }
    return [
        Chunk(text=piece, source=source, chunk_index=i, metadata=dict(base_meta))
        for i, piece in enumerate(pieces)
    ]


def chunk_document(
    text: str,
    source: str,
    *,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    """Convenience dispatcher: pick prose vs code chunking from the source path.

    This is the function the indexing layer calls. It ties `select_strategy`
    (pure, tested) to the two library-backed chunkers.
    """
    if select_strategy(source) == "code":
        return chunk_code(text, source)
    return chunk_prose(text, source, chunk_size=chunk_size, overlap=overlap)
