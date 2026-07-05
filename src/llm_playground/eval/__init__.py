"""Evaluation (Topic 4): measuring RAG quality across configurations.

golden_set.py  hand-written (question, expected answer fragment, expected
               sources) tuples — the ground truth.
metrics.py     transparent, unit-tested proxy metrics (faithfulness,
               answer relevancy, contextual precision/recall) PLUS a
               DeepEval adapter for the real LLM-as-judge versions.
suite.py       runs the golden set through 4 configs (A: no RAG, B: RAG,
               C: RAG + critique, D: local model + RAG) and gates on
               thresholds.
"""
