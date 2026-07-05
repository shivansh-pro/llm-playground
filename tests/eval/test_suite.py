"""Tests for the 4-config evaluation suite. Fake config runners let us assert
the scoring, averaging, and gating logic deterministically — including the
expected ORDERING of configs (no-RAG worse than RAG)."""

from __future__ import annotations

from llm_playground.eval.golden_set import GOLDEN_SET, GoldenExample
from llm_playground.eval.suite import (
    DEFAULT_THRESHOLDS,
    EvalOutput,
    MetricScores,
    run_config,
    run_suite,
    score_example,
)


def _example() -> GoldenExample:
    return GoldenExample(
        question="Where is attention implemented?",
        expected_answer_fragment="CausalSelfAttention",
        expected_sources=["model.py"],
        route="code",
    )


# --- score_example -----------------------------------------------------------
def test_score_example_perfect() -> None:
    ex = _example()
    out = EvalOutput(
        answer="It is in CausalSelfAttention.",
        retrieved_sources=["model.py"],
        contexts=["the CausalSelfAttention class implements attention"],
    )
    scores = score_example(ex, out)
    assert scores.answer_relevancy == 1.0
    assert scores.contextual_precision == 1.0
    assert scores.contextual_recall == 1.0
    assert scores.faithfulness == 1.0


def test_score_example_no_retrieval_baseline() -> None:
    """Config A (no RAG): empty retrieval -> zero contextual scores and zero
    faithfulness (no context to ground against), even if the parametric answer
    happens to contain the fragment."""
    ex = _example()
    out = EvalOutput(answer="CausalSelfAttention", retrieved_sources=[], contexts=[])
    scores = score_example(ex, out)
    assert scores.contextual_precision == 0.0
    assert scores.faithfulness == 0.0
    assert scores.answer_relevancy == 1.0  # fragment still present


# --- run_config / gating -----------------------------------------------------
def test_run_config_averages_and_gates_pass() -> None:
    golden = [_example()]

    def good_runner(ex: GoldenExample) -> EvalOutput:
        return EvalOutput(
            answer="Found in CausalSelfAttention.",
            retrieved_sources=["model.py"],
            contexts=["CausalSelfAttention class code"],
        )

    report = run_config("B_rag", good_runner, golden)
    assert report.passed is True
    assert report.averaged.contextual_recall == 1.0
    assert len(report.per_example) == 1


def test_run_config_gates_fail_for_baseline() -> None:
    golden = [_example()]

    def no_rag_runner(ex: GoldenExample) -> EvalOutput:
        return EvalOutput(answer="I think it's somewhere.", retrieved_sources=[], contexts=[])

    report = run_config("A_baseline", no_rag_runner, golden)
    assert report.passed is False  # fails contextual thresholds


def test_metric_scores_mean() -> None:
    s = MetricScores(
        faithfulness=1.0, answer_relevancy=1.0, contextual_precision=0.0, contextual_recall=0.0
    )
    assert s.mean() == 0.5


def test_run_suite_ranks_rag_above_baseline() -> None:
    """The headline result: config B (RAG) should outscore config A (no RAG) on
    the mean metric. This is the comparison the eval REPORT.md is built to show."""
    golden = [_example()]

    configs = {
        "A_baseline": lambda ex: EvalOutput(answer="dunno", retrieved_sources=[], contexts=[]),
        "B_rag": lambda ex: EvalOutput(
            answer="In CausalSelfAttention.",
            retrieved_sources=["model.py"],
            contexts=["CausalSelfAttention implements attention"],
        ),
    }
    reports = run_suite(configs, golden)
    assert reports["B_rag"].averaged.mean() > reports["A_baseline"].averaged.mean()
    assert reports["B_rag"].passed and not reports["A_baseline"].passed


def test_run_suite_over_full_golden_set_with_oracle() -> None:
    """Smoke: an 'oracle' runner that always retrieves the expected sources and
    echoes the fragment should pass the gate across the entire real golden set."""

    def oracle(ex: GoldenExample) -> EvalOutput:
        return EvalOutput(
            answer=f"Answer mentioning {ex.expected_answer_fragment}.",
            retrieved_sources=list(ex.expected_sources),
            contexts=[f"context containing {ex.expected_answer_fragment} details"],
        )

    report = run_config("oracle", oracle, GOLDEN_SET)
    assert report.passed
    assert report.averaged.answer_relevancy == 1.0


def test_default_thresholds_are_sane() -> None:
    # Guard against someone zeroing the gate by accident.
    assert 0 < DEFAULT_THRESHOLDS.faithfulness <= 1
    assert 0 < DEFAULT_THRESHOLDS.contextual_recall <= 1
