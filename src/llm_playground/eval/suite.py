"""The evaluation suite: score the golden set across configurations and gate.

This is "how you measured faithfulness across 4 configs" made concrete. Each
CONFIG is a black-box runner that answers a question and reports what it
retrieved. The suite scores every runner with the same metrics so the configs
are directly comparable.

    A  baseline      Claude only, NO retrieval.        Expect low faithfulness &
                                                       zero contextual scores —
                                                       nothing was retrieved, so
                                                       any correctness is the
                                                       model's parametric memory.
    B  rag           Claude + RAG.                     Retrieval lifts grounding.
    C  rag+critique  Claude + RAG + self-critique.     Revision loop should raise
                                                       faithfulness/relevancy.
    D  local+rag     Local fine-tuned model + RAG.     Compare a small local model
                                                       to the frontier one.

The suite is dependency-injected: a config is just a Callable[[GoldenExample],
EvalOutput]. Unit tests pass deterministic fake runners; production passes
runners that call the real pipeline. That's why the whole thing is testable
without any model.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from llm_playground.eval.golden_set import GoldenExample
from llm_playground.eval.metrics import (
    answer_relevancy,
    contextual_precision,
    contextual_recall,
    faithfulness,
)


class EvalOutput(BaseModel):
    """What a config produces for one question: the answer, the sources it
    retrieved, and the context texts (needed to score faithfulness)."""

    answer: str
    retrieved_sources: list[str] = []
    contexts: list[str] = []


# A config runner answers one golden example.
ConfigRunner = Callable[[GoldenExample], EvalOutput]


class MetricScores(BaseModel):
    faithfulness: float
    answer_relevancy: float
    contextual_precision: float
    contextual_recall: float

    def mean(self) -> float:
        """Simple average across the four metrics — a single headline number."""
        return (
            self.faithfulness
            + self.answer_relevancy
            + self.contextual_precision
            + self.contextual_recall
        ) / 4


class ConfigReport(BaseModel):
    config_name: str
    per_example: list[MetricScores]
    averaged: MetricScores
    passed: bool


DEFAULT_THRESHOLDS = MetricScores(
    faithfulness=0.5,
    answer_relevancy=0.5,
    contextual_precision=0.3,
    contextual_recall=0.5,
)


def score_example(example: GoldenExample, output: EvalOutput) -> MetricScores:
    """Apply all four metrics to a single (golden, output) pair. Pure."""
    return MetricScores(
        faithfulness=faithfulness(output.answer, output.contexts),
        answer_relevancy=answer_relevancy(output.answer, example.expected_answer_fragment),
        contextual_precision=contextual_precision(
            output.retrieved_sources, example.expected_sources
        ),
        contextual_recall=contextual_recall(output.retrieved_sources, example.expected_sources),
    )


def _average(scores: list[MetricScores]) -> MetricScores:
    """Element-wise mean across examples. Empty -> all zeros."""
    n = len(scores)
    if n == 0:
        return MetricScores(
            faithfulness=0.0,
            answer_relevancy=0.0,
            contextual_precision=0.0,
            contextual_recall=0.0,
        )
    return MetricScores(
        faithfulness=sum(s.faithfulness for s in scores) / n,
        answer_relevancy=sum(s.answer_relevancy for s in scores) / n,
        contextual_precision=sum(s.contextual_precision for s in scores) / n,
        contextual_recall=sum(s.contextual_recall for s in scores) / n,
    )


def _passes(averaged: MetricScores, thresholds: MetricScores) -> bool:
    """Gate: every averaged metric must meet or exceed its threshold. This is
    exactly the kind of check that lives in a CI/CD eval gate — block a deploy
    if faithfulness drops below the bar."""
    return (
        averaged.faithfulness >= thresholds.faithfulness
        and averaged.answer_relevancy >= thresholds.answer_relevancy
        and averaged.contextual_precision >= thresholds.contextual_precision
        and averaged.contextual_recall >= thresholds.contextual_recall
    )


def run_config(
    config_name: str,
    runner: ConfigRunner,
    golden_set: list[GoldenExample],
    thresholds: MetricScores = DEFAULT_THRESHOLDS,
) -> ConfigReport:
    """Run one config over the whole golden set and produce a gated report."""
    per_example = [score_example(ex, runner(ex)) for ex in golden_set]
    averaged = _average(per_example)
    return ConfigReport(
        config_name=config_name,
        per_example=per_example,
        averaged=averaged,
        passed=_passes(averaged, thresholds),
    )


def run_suite(
    configs: dict[str, ConfigRunner],
    golden_set: list[GoldenExample],
    thresholds: MetricScores = DEFAULT_THRESHOLDS,
) -> dict[str, ConfigReport]:
    """Run every config and return a name -> report mapping. This is the object
    you'd render into eval/REPORT.md as the 4-config comparison table."""
    return {
        name: run_config(name, runner, golden_set, thresholds) for name, runner in configs.items()
    }
