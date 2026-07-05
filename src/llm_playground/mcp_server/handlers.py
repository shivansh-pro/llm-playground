"""MCP tool handlers + schemas (pure, dependency-injected).

WHY MCP vs PLAIN REST (the interview question):

    A REST API exposes endpoints that a human developer reads docs for, then
    writes bespoke client code against. Each new consumer re-integrates by hand.

    MCP (Model Context Protocol) standardizes how an LLM/agent DISCOVERS and
    CALLS capabilities at runtime. The differences that matter:

      1. Discovery. An MCP client calls `list_tools()` and gets back every tool
         with its JSON-Schema for arguments. The model learns what's available
         at runtime — no hardcoded endpoint list, no client regeneration when
         you add a tool. REST has no standard discovery (OpenAPI helps humans,
         but isn't a runtime contract the model consumes uniformly).

      2. Self-describing schemas. Each tool ships an input schema the model uses
         to form valid calls and that the server uses to validate them. One
         uniform envelope across all tools and all servers.

      3. Uniform invocation + transport. Every MCP tool is called the same way
         (`call_tool(name, args)`) over a standard transport (stdio/SSE). A
         host (Claude Desktop, an IDE, an agent) can consume ANY MCP server
         without custom glue. With REST, every API's auth/verbs/pagination/error
         shape differs, so every integration is bespoke.

      4. Capability negotiation + composition. A host can attach many MCP
         servers (files, git, your RAG, a database) and the model orchestrates
         across them through one interface. That's the "developer-facing
         capabilities / intelligent workflows" story a platform team wants.

    When REST is still the right call: public web APIs, browser clients, high-
    throughput CRUD, anything where the consumer is human-written software, not
    an LLM deciding at runtime which capability to invoke. MCP doesn't replace
    REST; it's the interface layer purpose-built for model-driven tool use.

This module keeps the tool LOGIC pure and injected (MCPDeps) so it is fully
unit-testable; server.py binds it to the actual MCP runtime.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel

from llm_playground.rag.models import SearchResult
from llm_playground.rag.query import Answer


class ToolSpec(BaseModel):
    """Description + JSON-Schema for one MCP tool. This is exactly the metadata
    an MCP client receives from `list_tools()` and uses to call the tool."""

    name: str
    description: str
    input_schema: dict


# The tool catalog this server advertises. Each schema is standard JSON-Schema:
# the same contract format across every MCP tool and every MCP server.
TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="search_papers",
        description="Semantic search over the indexed research papers. Returns top matching passages with source + score.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language search query"},
                "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="search_code",
        description="Semantic search over the indexed code (nanoGPT). Returns top matching code snippets with source + score.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language search query"},
                "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="ask",
        description="Ask a cross-cutting question answered with grounded RAG over papers and/or code. Returns an answer plus cited sources.",
        input_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "route": {
                    "type": "string",
                    "enum": ["paper", "code", "both"],
                    "default": "both",
                },
            },
            "required": ["question"],
        },
    ),
]


@dataclass
class MCPDeps:
    """Injected capabilities the tools delegate to. Real impls wrap the RAG
    retrievers + query pipeline; tests pass fakes."""

    search_papers: Callable[[str, int], list[SearchResult]]
    search_code: Callable[[str, int], list[SearchResult]]
    ask: Callable[[str, str], Answer]


def _results_to_payload(results: list[SearchResult]) -> list[dict]:
    """Normalize SearchResults into plain JSON-able dicts for the MCP wire."""
    return [{"text": r.text, "source": r.source, "score": round(r.score, 4)} for r in results]


def handle_search_papers(arguments: dict, deps: MCPDeps) -> list[dict]:
    """Validate args and delegate to the paper retriever.

    We validate defensively because tool arguments arrive from a model, which
    can and does produce malformed calls. Clear errors > silent misbehavior.
    """
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("'query' is required and must be a non-empty string")
    top_k = int(arguments.get("top_k", 5))
    return _results_to_payload(deps.search_papers(query, top_k))


def handle_search_code(arguments: dict, deps: MCPDeps) -> list[dict]:
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("'query' is required and must be a non-empty string")
    top_k = int(arguments.get("top_k", 5))
    return _results_to_payload(deps.search_code(query, top_k))


def handle_ask(arguments: dict, deps: MCPDeps) -> dict:
    question = arguments.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("'question' is required and must be a non-empty string")
    route = arguments.get("route", "both")
    if route not in ("paper", "code", "both"):
        raise ValueError(f"invalid route: {route!r}")
    answer = deps.ask(question, route)
    return answer.model_dump()


# Dispatch table: tool name -> handler. server.py uses this so there's a single
# source of truth for "what tools exist" (TOOL_SPECS) and "how they run" (here).
HANDLERS: dict[str, Callable[[dict, MCPDeps], object]] = {
    "search_papers": handle_search_papers,
    "search_code": handle_search_code,
    "ask": handle_ask,
}
