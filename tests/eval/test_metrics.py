"""Tests for the transparent proxy metrics. These pin the METHODOLOGY: precision,
recall, faithfulness, and answer relevancy each behave exactly as defined."""

from __future__ import annotations

import pytest

from llm_playground.eval.metrics import (
    answer_relevancy,
    contextual_precision,
    contextual_recall,
    faithfulness,
)


# --- contextual precision ----------------------------------------------------
def test_precision_all_relevant_is_one() -> None:
    assert contextual_precision(["a", "b"], ["a", "b", "c"]) == 1.0


def test_precision_half_relevant() -> None:
    # retrieved {a, x}; relevant {a, b} -> intersection {a} / retrieved 2 = 0.5
    assert contextual_precision(["a", "x"], ["a", "b"]) == 0.5


def test_precision_empty_retrieval_is_zero() -> None:
    assert contextual_precision([], ["a"]) == 0.0


# --- contextual recall -------------------------------------------------------
def test_recall_retrieved_all_relevant_is_one() -> None:
    assert contextual_recall(["a", "b", "x"], ["a", "b"]) == 1.0


def test_recall_half() -> None:
    # relevant {a, b}; retrieved {a} -> 1/2
    assert contextual_recall(["a"], ["a", "b"]) == 0.5


def test_recall_no_relevant_defined_is_one() -> None:
    # Nothing was relevant, so nothing could be missed.
    assert contextual_recall(["a"], []) == 1.0


def test_precision_recall_tradeoff_illustrated() -> None:
    """Widening retrieval raises recall but can drop precision — the core RAG
    tradeoff, demonstrated numerically."""
    relevant = ["a", "b"]
    narrow = ["a"]  # precise but misses b
    wide = ["a", "b", "x", "y"]  # finds all, but adds junk
    assert contextual_precision(narrow, relevant) == 1.0
    assert contextual_recall(narrow, relevant) == 0.5
    assert contextual_precision(wide, relevant) == 0.5
    assert contextual_recall(wide, relevant) == 1.0


# --- answer relevancy --------------------------------------------------------
def test_answer_relevancy_fragment_present() -> None:
    assert answer_relevancy("The class is CausalSelfAttention.", "CausalSelfAttention") == 1.0


def test_answer_relevancy_case_insensitive() -> None:
    assert answer_relevancy("uses PRE-NORM here", "pre-norm") == 1.0


def test_answer_relevancy_fragment_absent() -> None:
    assert answer_relevancy("something unrelated", "RoFormer") == 0.0


def test_answer_relevancy_empty_expectation_is_one() -> None:
    assert answer_relevancy("anything", "") == 1.0


# --- faithfulness ------------------------------------------------------------
def test_faithfulness_fully_grounded() -> None:
    # Every answer sentence shares tokens with the context.
    contexts = ["attention uses queries keys values"]
    answer = "Attention uses queries and keys."
    assert faithfulness(answer, contexts) == 1.0


def test_faithfulness_hallucination_scores_low() -> None:
    contexts = ["transformers use attention"]
    # Answer about an unrelated topic shares no content tokens -> 0.
    answer = "Bananas grow in tropical climates worldwide."
    assert faithfulness(answer, contexts) == 0.0


def test_faithfulness_partial() -> None:
    contexts = ["attention uses softmax scaling"]
    # First sentence grounded, second invented -> 1/2.
    answer = "Attention uses softmax. Elephants roam the savannah."
    assert faithfulness(answer, contexts) == pytest.approx(0.5)


def test_faithfulness_empty_answer_is_one() -> None:
    # Nothing said -> nothing unsupported.
    assert faithfulness("", ["ctx"]) == 1.0


def test_faithfulness_no_context_is_zero() -> None:
    # A non-empty answer with no supporting context cannot be grounded.
    assert faithfulness("some claim here", []) == 0.0
