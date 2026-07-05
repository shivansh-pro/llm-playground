"""Evaluation metrics.

Two layers, deliberately:

  1. TRANSPARENT PROXY METRICS (this file, pure Python, unit-tested). They make
     the *methodology* explicit and debuggable. You can read exactly why a score
     is what it is. Great for CI gates and for explaining the numbers in an
     interview.

  2. LLM-AS-JUDGE METRICS (DeepEval adapter at the bottom, lazy import). The real
     faithfulness/answer-relevancy metrics prompt a strong model to judge each
     answer. More faithful to human judgment, but non-deterministic, costs money,
     and needs an API key — so they run as integration, not in CI.

THE FOUR METRICS (what each measures and its failure mode):

  faithfulness          Does the answer stay grounded in the retrieved context,
                        or does it hallucinate? Proxy: fraction of answer
                        sentences that share meaningful tokens with some context.
                        Low faithfulness = confident but unsupported claims.

  answer_relevancy      Does the answer actually address the question? Proxy:
                        did the answer contain the expected key fragment. Real
                        version compares question<->answer semantic similarity.

  contextual_precision  Of the sources we RETRIEVED, how many were relevant?
                        precision = |retrieved ∩ relevant| / |retrieved|.
                        Low precision = retriever pulls in junk that distracts
                        the model (a cause of hallucination).

  contextual_recall     Of the sources that ARE relevant, how many did we
                        retrieve? recall = |retrieved ∩ relevant| / |relevant|.
                        Low recall = the answer-bearing passage never reached
                        the model, so even a perfect model can't answer.

Precision vs recall is THE retrieval tradeoff: widen top_k to raise recall but
you usually drop precision. Tuning that balance is core RAG work.
"""

from __future__ import annotations

import re

_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "to",
        "in",
        "is",
        "are",
        "and",
        "or",
        "for",
        "with",
        "on",
        "at",
        "by",
        "as",
        "it",
        "this",
        "that",
        "be",
        "we",
        "you",
    }
)


def _content_tokens(text: str) -> set[str]:
    """Lowercase word tokens minus stopwords. Shared by faithfulness."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def contextual_precision(retrieved_sources: list[str], relevant_sources: list[str]) -> float:
    """|retrieved ∩ relevant| / |retrieved|. Empty retrieval -> 0.0."""
    if not retrieved_sources:
        return 0.0
    retrieved = set(retrieved_sources)
    relevant = set(relevant_sources)
    return len(retrieved & relevant) / len(retrieved)


def contextual_recall(retrieved_sources: list[str], relevant_sources: list[str]) -> float:
    """|retrieved ∩ relevant| / |relevant|. No relevant defined -> 1.0 (nothing
    to miss)."""
    if not relevant_sources:
        return 1.0
    retrieved = set(retrieved_sources)
    relevant = set(relevant_sources)
    return len(retrieved & relevant) / len(relevant)


def answer_relevancy(answer: str, expected_fragment: str) -> float:
    """Proxy: 1.0 if the answer contains the expected fragment (case-insensitive),
    else 0.0. Stand-in for DeepEval's semantic question<->answer relevancy."""
    if not expected_fragment:
        return 1.0
    return 1.0 if expected_fragment.lower() in answer.lower() else 0.0


def faithfulness(answer: str, contexts: list[str]) -> float:
    """Fraction of answer sentences 'supported' by the retrieved context.

    A sentence counts as supported if it shares at least one meaningful
    (non-stopword) token with the union of context tokens. Crude but honest: an
    answer invented from nowhere shares no vocabulary with the context and scores
    low. An empty answer scores 1.0 (nothing unsupported). This mirrors the SHAPE
    of DeepEval's faithfulness (claims supported by context) without an LLM.
    """
    sentences = [s for s in re.split(r"[.!?]+", answer) if s.strip()]
    if not sentences:
        return 1.0
    context_tokens: set[str] = set()
    for c in contexts:
        context_tokens |= _content_tokens(c)
    if not context_tokens:
        return 0.0
    supported = sum(1 for s in sentences if _content_tokens(s) & context_tokens)
    return supported / len(sentences)


def make_deepeval_faithfulness(threshold: float = 0.7, model: str = "claude-sonnet-4-5"):
    """Return a callable that scores faithfulness via DeepEval's LLM-judge.

    Lazy import; integration-only (needs an API key + network). The returned
    function has the same (answer, contexts, question) -> float shape as the
    proxy so the suite can swap judge<->proxy transparently.
    """
    from deepeval.metrics import FaithfulnessMetric  # pragma: no cover
    from deepeval.test_case import LLMTestCase  # pragma: no cover

    metric = FaithfulnessMetric(threshold=threshold, model=model)  # pragma: no cover

    def _score(answer: str, contexts: list[str], question: str) -> float:  # pragma: no cover
        case = LLMTestCase(input=question, actual_output=answer, retrieval_context=contexts)
        metric.measure(case)
        return float(metric.score)

    return _score  # pragma: no cover
