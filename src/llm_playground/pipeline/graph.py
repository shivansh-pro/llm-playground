"""The self-correcting RAG state machine.

Flow:

    route -> retrieve -> answer -> critique --(fail & under budget)--> answer
                                        |                                  ^
                                        |                                  |
                                        +--(pass, or budget hit)--> finalize -> END

WHY A STATE MACHINE (vs a plain function that calls things in order):
    - Branching: the critique step can send us BACK to answer, or forward to
      finalize. Linear code handles this with ad-hoc while-loops; a graph makes
      the control flow explicit and inspectable.
    - Bounded revision: we cap iterations so a model that never satisfies its
      own critic can't loop forever (a real failure mode of naive agent loops).
    - Observability: each node is a named step you can log, time, checkpoint,
      and replay. LangGraph gives streaming + persistence for free once you
      express the flow as a graph.

WHY LANGGRAPH vs LANGCHAIN (the interview question):
    LangChain composes CALLS (prompt | model | parser) — great for linear
    chains. LangGraph models STATE + TRANSITIONS — needed the moment you have
    loops, branches, or memory across steps. Use a chain for "A then B then C";
    reach for a graph for "do B, and depending on the result maybe redo A."

Design: node logic is written as pure `(state, deps) -> partial-update` funcs.
`run_pipeline` drives them by hand (unit-tested). `build_graph` wires the SAME
funcs into a LangGraph StateGraph (integration-tested). Dependencies (classify,
retrieve, generate, critique) are injected so no node touches an API directly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, Field

from llm_playground.rag.models import SearchResult
from llm_playground.rag.query import Route, build_prompt


class Critique(BaseModel):
    """Verdict from the critic node on a draft answer."""

    passed: bool
    feedback: str = ""


class RAGState(BaseModel):
    """The single object threaded through every node.

    Each node returns a PARTIAL update (a dict of just the fields it changed);
    the driver/LangGraph merges it into the running state. This "reducer"
    pattern keeps nodes decoupled — a node only declares what it produces.
    """

    question: str
    route: Route = "both"
    contexts: list[SearchResult] = Field(default_factory=list)
    draft_answer: str = ""
    critique: Critique | None = None
    final_answer: str = ""
    iterations: int = 0


@dataclass
class GraphDeps:
    """Injected capabilities. Real impls wrap retrievers + Anthropic; tests
    pass deterministic fakes.

    - classify:  question            -> Route ("paper"/"code"/"both")
    - retrieve:  (question, route)   -> retrieved contexts
    - generate:  prompt              -> answer text
    - critique:  (answer, contexts)  -> Critique(passed, feedback)
    """

    classify: Callable[[str], Route]
    retrieve: Callable[[str, Route], list[SearchResult]]
    generate: Callable[[str], str]
    critique: Callable[[str, list[SearchResult]], Critique]
    max_iterations: int = 2


# --- Node logic (pure: state in, partial-update dict out) --------------------


def route_node(state: RAGState, deps: GraphDeps) -> dict:
    return {"route": deps.classify(state.question)}


def retrieve_node(state: RAGState, deps: GraphDeps) -> dict:
    return {"contexts": deps.retrieve(state.question, state.route)}


def answer_node(state: RAGState, deps: GraphDeps) -> dict:
    """Generate a draft. On a revision pass, fold the critic's feedback into
    the prompt so the model actually improves rather than repeating itself.
    Increments the iteration counter — the budget guard depends on it."""
    prompt = build_prompt(state.question, state.contexts)
    if state.critique is not None and not state.critique.passed:
        prompt += (
            f"\n\nYour previous answer was rejected for this reason: "
            f"{state.critique.feedback}\nRevise and improve the answer."
        )
    return {"draft_answer": deps.generate(prompt), "iterations": state.iterations + 1}


def critique_node(state: RAGState, deps: GraphDeps) -> dict:
    return {"critique": deps.critique(state.draft_answer, state.contexts)}


def finalize_node(state: RAGState) -> dict:
    return {"final_answer": state.draft_answer}


def should_revise(state: RAGState, deps: GraphDeps) -> str:
    """Conditional-edge decision after critique. Returns the next node name.

    Revise (loop back to 'answer') only if the critic failed the draft AND we
    still have budget. Otherwise finalize. The budget check is what prevents an
    infinite answer<->critique loop.
    """
    failed = state.critique is not None and not state.critique.passed
    if failed and state.iterations < deps.max_iterations:
        return "answer"
    return "finalize"


# --- Pure driver (unit-tested; mirrors the graph exactly) --------------------


def run_pipeline(question: str, deps: GraphDeps) -> RAGState:
    """Execute the state machine by hand. Deterministic and dependency-free
    (given fake deps). This is what LangGraph does for you underneath."""
    state = RAGState(question=question)
    state = state.model_copy(update=route_node(state, deps))
    state = state.model_copy(update=retrieve_node(state, deps))
    while True:
        state = state.model_copy(update=answer_node(state, deps))
        state = state.model_copy(update=critique_node(state, deps))
        if should_revise(state, deps) == "finalize":
            break
    return state.model_copy(update=finalize_node(state))


# --- LangGraph wiring (integration-tested) -----------------------------------


def build_graph(deps: GraphDeps):
    """Wire the same node funcs into a compiled LangGraph StateGraph.

    Lazy import so importing this module (and unit-testing the pure driver)
    needs no langgraph. The lambdas adapt each pure `(state, deps)->dict` node
    to LangGraph's `state->dict` node signature by closing over `deps`.
    """
    from langgraph.graph import END, StateGraph

    graph = StateGraph(RAGState)
    graph.add_node("route", lambda s: route_node(s, deps))
    graph.add_node("retrieve", lambda s: retrieve_node(s, deps))
    graph.add_node("answer", lambda s: answer_node(s, deps))
    graph.add_node("critique", lambda s: critique_node(s, deps))
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "critique")
    # Conditional edge: critique -> {answer (revise) | finalize}, chosen by
    # should_revise. The mapping's values are the actual node names to jump to.
    graph.add_conditional_edges(
        "critique",
        lambda s: should_revise(s, deps),
        {"answer": "answer", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)
    return graph.compile()
