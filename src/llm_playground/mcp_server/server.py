"""MCP server wiring (FastMCP). Binds the pure handlers to the MCP runtime.

Lazy import of `mcp` so the handlers module stays unit-testable without the SDK.
This file is thin glue — the logic and validation live in handlers.py — so it's
covered by an integration test rather than unit tests.
"""

from __future__ import annotations

from llm_playground.mcp_server.handlers import (
    HANDLERS,
    TOOL_SPECS,
    MCPDeps,
)


def build_server(deps: MCPDeps, name: str = "paper-repo-mapper"):  # pragma: no cover
    """Construct a FastMCP server that advertises TOOL_SPECS and routes calls
    through HANDLERS. Returns the server; caller runs it (e.g. server.run()).

    Every tool is registered with the same envelope: name + JSON-Schema +
    handler. That uniformity is the MCP value proposition — a host can discover
    and call all three tools without knowing anything bespoke about them.
    """
    from mcp.server.fastmcp import FastMCP

    server = FastMCP(name)

    # Register each spec as an MCP tool whose implementation delegates to the
    # corresponding pure handler. We build the closure in a helper to avoid the
    # classic late-binding-loop bug (all closures capturing the last `spec`).
    def _register(spec_name: str) -> None:
        handler = HANDLERS[spec_name]
        spec = next(s for s in TOOL_SPECS if s.name == spec_name)

        @server.tool(name=spec.name, description=spec.description)
        def _tool(**arguments):
            return handler(arguments, deps)

    for spec in TOOL_SPECS:
        _register(spec.name)

    return server


if __name__ == "__main__":  # pragma: no cover
    # Minimal runnable entrypoint. In practice deps would wrap real retrievers;
    # here we fail fast to signal that wiring is required.
    raise SystemExit(
        "Configure MCPDeps with real retrievers before running the MCP server. "
        "See handlers.MCPDeps and rag.query.Retriever."
    )
