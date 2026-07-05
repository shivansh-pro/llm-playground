"""The golden set: hand-authored ground truth for evaluation.

A golden set is the single most important artifact in LLM evaluation. Metrics
are only as trustworthy as the examples you score against. Principles used here:
    - Each example has a KNOWN expected answer fragment and expected source(s),
      so retrieval AND generation can both be scored.
    - Examples span the routes (paper-only, code-only, cross-cutting) to catch
      routing regressions.
    - Kept small and high-quality. 8 sharp examples beat 100 vague ones; every
      example should be one a domain expert agrees on.
"""

from __future__ import annotations

from pydantic import BaseModel

from llm_playground.rag.query import Route


class GoldenExample(BaseModel):
    """One labeled evaluation case."""

    question: str
    expected_answer_fragment: str  # a substring the correct answer should contain
    expected_sources: list[str]  # source paths that SHOULD be retrieved
    route: Route


# Sources are referenced by short logical names; the real index would use file
# paths (e.g. "repos/nanoGPT/model.py"). Kept abstract so the set is stable.
GOLDEN_SET: list[GoldenExample] = [
    GoldenExample(
        question="Where is multi-head attention implemented in nanoGPT?",
        expected_answer_fragment="CausalSelfAttention",
        expected_sources=["model.py"],
        route="code",
    ),
    GoldenExample(
        question="What scaling factor is applied to attention scores in the transformer?",
        expected_answer_fragment="sqrt",
        expected_sources=["attention_is_all_you_need.pdf"],
        route="paper",
    ),
    GoldenExample(
        question="How does nanoGPT's normalization placement differ from the original transformer?",
        expected_answer_fragment="pre-norm",
        expected_sources=["model.py", "attention_is_all_you_need.pdf"],
        route="both",
    ),
    GoldenExample(
        question="Which paper introduces rotary position embeddings?",
        expected_answer_fragment="RoFormer",
        expected_sources=["roformer.pdf"],
        route="paper",
    ),
    GoldenExample(
        question="What does the MLP block do in a transformer layer?",
        expected_answer_fragment="feed-forward",
        expected_sources=["attention_is_all_you_need.pdf"],
        route="paper",
    ),
    GoldenExample(
        question="How does nanoGPT tie input and output embedding weights?",
        expected_answer_fragment="weight",
        expected_sources=["model.py"],
        route="code",
    ),
    GoldenExample(
        question="What memory optimization does FlashAttention provide?",
        expected_answer_fragment="memory",
        expected_sources=["flashattention.pdf"],
        route="paper",
    ),
    GoldenExample(
        question="Where does nanoGPT apply the causal mask, and why is it needed?",
        expected_answer_fragment="future",
        expected_sources=["model.py", "attention_is_all_you_need.pdf"],
        route="both",
    ),
]
