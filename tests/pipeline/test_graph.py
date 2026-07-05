"""Tests for the LangGraph state machine.

We test the node logic and the pure driver exhaustively (deterministic fakes),
then an integration test proves the real LangGraph build produces the same
result as the pure driver.
"""

from __future__ import annotations

import pytest

from llm_playground.pipeline.graph import (
    Critique,
    GraphDeps,
    RAGState,
    answer_node,
    critique_node,
    retrieve_node,
    route_node,
    run_pipeline,
    should_revise,
)
from llm_playground.rag.models import SearchResult


def _deps(
    *,
    route="both",
    critique_passes=True,
    critique_feedback="",
    max_iterations=2,
    generate=None,
) -> GraphDeps:
    """Build GraphDeps with deterministic fakes. `generate` can be overridden to
    inspect prompts / vary output across iterations."""
    return GraphDeps(
        classify=lambda q: route,
        retrieve=lambda q, r: [SearchResult(text=f"ctx for {q}", source="s", score=1.0)],
        generate=generate or (lambda prompt: "draft answer"),
        critique=lambda ans, ctx: Critique(passed=critique_passes, feedback=critique_feedback),
        max_iterations=max_iterations,
    )


# --- individual nodes --------------------------------------------------------
def test_route_node_sets_route() -> None:
    state = RAGState(question="q")
    assert route_node(state, _deps(route="code")) == {"route": "code"}


def test_retrieve_node_populates_contexts() -> None:
    state = RAGState(question="q", route="paper")
    update = retrieve_node(state, _deps())
    assert len(update["contexts"]) == 1
    assert update["contexts"][0].text == "ctx for q"


def test_answer_node_increments_iterations() -> None:
    state = RAGState(question="q", iterations=0)
    update = answer_node(state, _deps())
    assert update["iterations"] == 1
    assert update["draft_answer"] == "draft answer"


def test_answer_node_folds_in_critique_on_revision() -> None:
    """On a failed critique, the feedback must be injected into the prompt so
    the model actually revises. We capture the prompt to prove it."""
    seen: dict = {}

    def capture(prompt: str) -> str:
        seen["prompt"] = prompt
        return "revised"

    state = RAGState(
        question="q",
        contexts=[SearchResult(text="c", source="s", score=1.0)],
        critique=Critique(passed=False, feedback="missing the citation"),
        iterations=1,
    )
    answer_node(state, _deps(generate=capture))
    assert "missing the citation" in seen["prompt"]


def test_critique_node_sets_critique() -> None:
    state = RAGState(question="q", draft_answer="a")
    update = critique_node(state, _deps(critique_passes=False, critique_feedback="bad"))
    assert update["critique"].passed is False
    assert update["critique"].feedback == "bad"


# --- conditional edge (the loop guard) ---------------------------------------
def test_should_revise_finalizes_when_passed() -> None:
    state = RAGState(question="q", critique=Critique(passed=True), iterations=1)
    assert should_revise(state, _deps()) == "finalize"


def test_should_revise_loops_when_failed_and_under_budget() -> None:
    state = RAGState(question="q", critique=Critique(passed=False), iterations=1)
    assert should_revise(state, _deps(max_iterations=2)) == "answer"


def test_should_revise_finalizes_when_budget_exhausted() -> None:
    """Even a failing critique finalizes once we hit max_iterations — this is
    the infinite-loop guard."""
    state = RAGState(question="q", critique=Critique(passed=False), iterations=2)
    assert should_revise(state, _deps(max_iterations=2)) == "finalize"


# --- full pure driver --------------------------------------------------------
def test_run_pipeline_happy_path_single_iteration() -> None:
    state = run_pipeline("q", _deps(route="paper", critique_passes=True))
    assert state.route == "paper"
    assert state.final_answer == "draft answer"
    assert state.iterations == 1  # no revision needed


def test_run_pipeline_revises_then_finalizes() -> None:
    """Critique always fails -> loop should run exactly max_iterations times
    then finalize (budget guard), not forever."""
    state = run_pipeline("q", _deps(critique_passes=False, max_iterations=2))
    assert state.iterations == 2
    assert state.final_answer != ""


def test_run_pipeline_stops_when_critique_passes_second_time() -> None:
    """First draft fails, second passes -> exactly 2 iterations."""
    calls = {"n": 0}

    def critique_then_pass(ans, ctx):
        calls["n"] += 1
        return Critique(passed=calls["n"] >= 2, feedback="fix")

    deps = GraphDeps(
        classify=lambda q: "both",
        retrieve=lambda q, r: [],
        generate=lambda p: "answer",
        critique=critique_then_pass,
        max_iterations=5,
    )
    state = run_pipeline("q", deps)
    assert state.iterations == 2


# --- integration: real LangGraph matches the pure driver ---------------------
@pytest.mark.integration
def test_langgraph_build_matches_pure_driver(  # pragma: no cover - integration
) -> None:
    from llm_playground.pipeline.graph import build_graph

    deps = _deps(route="code", critique_passes=True)
    graph = build_graph(deps)
    result = graph.invoke(RAGState(question="q"))
    # LangGraph returns the final state (as dict-like); compare key fields.
    assert result["route"] == "code"
    assert result["final_answer"] == "draft answer"
