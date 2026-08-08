"""State contract for the LangGraph research workflow."""

from __future__ import annotations

from typing import Any, TypedDict


class ResearchState(TypedDict, total=False):
    """Shared state passed between research workflow nodes.

    The workflow initializes omitted optional fields in ``planner_node`` so a
    caller can start a run with only ``topic`` and ``max_iterations``.
    """

    topic: str
    search_queries: list[str]
    documents: list[dict[str, Any]]
    search_iterations: int
    max_iterations: int
    insights: list[str]
    verification_results: list[dict[str, Any]]
    output_file: str
