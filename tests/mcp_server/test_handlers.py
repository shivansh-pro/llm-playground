"""Tests for MCP tool handlers + tool specs. Pure logic with injected deps —
no MCP runtime needed."""

from __future__ import annotations

import pytest

from llm_playground.mcp_server.handlers import (
    HANDLERS,
    TOOL_SPECS,
    MCPDeps,
    handle_ask,
    handle_search_code,
    handle_search_papers,
)
from llm_playground.rag.models import SearchResult
from llm_playground.rag.query import Answer


def _deps() -> MCPDeps:
    return MCPDeps(
        search_papers=lambda q, k: (
            [SearchResult(text=f"paper:{q}", source="p.pdf", score=0.9)][:k]
            or [SearchResult(text=f"paper:{q}", source="p.pdf", score=0.9)]
        ),
        search_code=lambda q, k: [SearchResult(text=f"code:{q}", source="m.py", score=0.8)],
        ask=lambda question, route: Answer(
            question=question, route=route, answer="grounded", sources=["p.pdf"]
        ),
    )


# --- tool specs (the discovery contract) -------------------------------------
def test_tool_specs_cover_expected_tools() -> None:
    names = {s.name for s in TOOL_SPECS}
    assert names == {"search_papers", "search_code", "ask"}


def test_every_spec_has_a_handler() -> None:
    # The discovery catalog (TOOL_SPECS) and the dispatch table (HANDLERS) must
    # stay in sync — this test fails loudly if someone adds one but not the other.
    assert {s.name for s in TOOL_SPECS} == set(HANDLERS.keys())


def test_specs_have_valid_json_schema_shape() -> None:
    for spec in TOOL_SPECS:
        assert spec.input_schema["type"] == "object"
        assert "properties" in spec.input_schema
        assert isinstance(spec.input_schema.get("required", []), list)


# --- search_papers / search_code ---------------------------------------------
def test_handle_search_papers_returns_payload() -> None:
    out = handle_search_papers({"query": "attention", "top_k": 5}, _deps())
    assert out[0]["source"] == "p.pdf"
    assert "score" in out[0] and "text" in out[0]


def test_handle_search_code_returns_payload() -> None:
    out = handle_search_code({"query": "forward"}, _deps())
    assert out[0]["source"] == "m.py"


def test_handle_search_papers_rejects_missing_query() -> None:
    with pytest.raises(ValueError, match="query"):
        handle_search_papers({}, _deps())


def test_handle_search_papers_rejects_blank_query() -> None:
    with pytest.raises(ValueError, match="query"):
        handle_search_papers({"query": "   "}, _deps())


# --- ask ---------------------------------------------------------------------
def test_handle_ask_returns_answer_dict() -> None:
    out = handle_ask({"question": "why?", "route": "both"}, _deps())
    assert out["answer"] == "grounded"
    assert out["sources"] == ["p.pdf"]
    assert out["route"] == "both"


def test_handle_ask_defaults_route_to_both() -> None:
    out = handle_ask({"question": "why?"}, _deps())
    assert out["route"] == "both"


def test_handle_ask_rejects_bad_route() -> None:
    with pytest.raises(ValueError, match="route"):
        handle_ask({"question": "why?", "route": "banana"}, _deps())


def test_handle_ask_rejects_missing_question() -> None:
    with pytest.raises(ValueError, match="question"):
        handle_ask({"route": "both"}, _deps())


def test_handlers_dispatch_table_callable() -> None:
    # HANDLERS is the single source of truth server.py uses to route calls.
    out = HANDLERS["search_code"]({"query": "x"}, _deps())
    assert isinstance(out, list)


def test_server_module_imports_and_exposes_build_server() -> None:
    """Import the server module (covers its top-level wiring imports) and
    confirm build_server is present. The FastMCP body itself is integration."""
    from llm_playground.mcp_server import server

    assert callable(server.build_server)
