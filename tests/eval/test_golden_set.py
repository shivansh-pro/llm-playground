"""Tests for the golden set's integrity. A malformed golden set silently
corrupts every metric, so we validate its shape."""

from __future__ import annotations

from llm_playground.eval.golden_set import GOLDEN_SET


def test_golden_set_nonempty() -> None:
    assert len(GOLDEN_SET) >= 5


def test_every_example_well_formed() -> None:
    for ex in GOLDEN_SET:
        assert ex.question.strip()
        assert ex.expected_answer_fragment.strip()
        assert ex.expected_sources, f"{ex.question!r} has no expected sources"
        assert ex.route in ("paper", "code", "both")


def test_routes_are_diverse() -> None:
    # The set should exercise all three routes, or it can't catch routing bugs.
    routes = {ex.route for ex in GOLDEN_SET}
    assert routes == {"paper", "code", "both"}


def test_both_route_examples_cite_multiple_sources() -> None:
    # A cross-cutting ("both") question should expect sources from >1 place.
    for ex in GOLDEN_SET:
        if ex.route == "both":
            assert len(ex.expected_sources) >= 2
