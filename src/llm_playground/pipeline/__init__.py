"""Orchestration pipeline (Topic 2): the route -> retrieve -> answer ->
critique -> revise -> done state machine, built on LangGraph.

We provide two equivalent drivers:
    - run_pipeline(): a hand-rolled pure-Python driver. Fully unit-tested,
      deterministic, and it makes explicit what a state machine actually is.
    - build_graph(): the idiomatic LangGraph StateGraph. Same nodes, same
      transitions; LangGraph adds streaming, checkpointing, persistence, and
      visualization on top. Integration-tested against the pure driver.
"""
