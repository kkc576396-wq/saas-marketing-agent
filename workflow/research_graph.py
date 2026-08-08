"""LangGraph workflow for AnySearch-backed SaaS marketing research."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from tools.anysearch_tool import anysearch_batch_search

from .state import ResearchState


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "research_output.json"


def planner_node(state: ResearchState) -> dict[str, Any]:
    """Initialize the run and create a small set of research queries.

    This first planner is intentionally deterministic. A model-backed planner
    can replace it later without changing the state or graph contract.
    """

    topic = state.get("topic", "").strip()
    if not topic:
        raise ValueError("ResearchState.topic must not be empty")

    configured_queries = [query.strip() for query in state.get("search_queries", []) if query.strip()]
    search_queries = configured_queries or [
        f"{topic} market overview and current trends",
        f"{topic} target customers pain points and use cases",
        f"{topic} competitors pricing and positioning",
    ]

    max_iterations = max(1, int(state.get("max_iterations", 5)))
    output_file = state.get("output_file") or str(DEFAULT_OUTPUT_FILE)

    return {
        "topic": topic,
        "search_queries": search_queries,
        "documents": list(state.get("documents", [])),
        "search_iterations": int(state.get("search_iterations", 0)),
        "max_iterations": max_iterations,
        "insights": list(state.get("insights", [])),
        "verification_results": list(state.get("verification_results", [])),
        "output_file": output_file,
    }


def search_node(state: ResearchState) -> dict[str, Any]:
    """Run the current query set through the existing AnySearch tool."""

    queries = state.get("search_queries", [])
    if not queries:
        raise ValueError("ResearchState.search_queries must not be empty")

    iteration = int(state.get("search_iterations", 0)) + 1
    query_items = [{"query": query, "max_results": 5} for query in queries]
    result = anysearch_batch_search.invoke({"queries": query_items})

    documents = list(state.get("documents", []))
    documents.append(
        {
            "iteration": iteration,
            "queries": list(queries),
            "content": result,
        }
    )
    return {
        "documents": documents,
        "search_iterations": iteration,
    }


def analyzer_node(state: ResearchState) -> dict[str, Any]:
    """Create lightweight, traceable insights from the retrieved text.

    This is a baseline analyzer that preserves source text and surfaces the
    first useful lines. It is deliberately not an LLM implementation yet.
    """

    insights: list[str] = []
    for document in state.get("documents", []):
        content = str(document.get("content", ""))
        lines = [line.strip(" -*#\t") for line in content.splitlines() if line.strip()]
        for line in lines[:5]:
            insights.append(f"Iteration {document.get('iteration', '?')}: {line[:500]}")

    return {"insights": insights[:50]}


def _topic_terms(topic: str) -> set[str]:
    """Return meaningful terms used for the baseline relevance check."""

    return {
        term.strip(".,:;!?()[]{}\"'").lower()
        for term in topic.split()
        if len(term.strip(".,:;!?()[]{}\"'")) >= 3
    }


def verifier_node(state: ResearchState) -> dict[str, Any]:
    """Verify insights before they are eligible for the final output.

    This is a deterministic first-pass verifier. It checks lexical relevance
    against the topic and confirms that the insight came from retrieved
    documents. A model-based verifier can replace these heuristics later while
    preserving the ``verification_results`` contract.
    """

    topic_terms = _topic_terms(state.get("topic", ""))
    documents = state.get("documents", [])
    verification_results: list[dict[str, Any]] = []
    verified_insights: list[str] = []

    for insight in state.get("insights", []):
        insight_terms = _topic_terms(insight)
        relevance_terms = sorted(topic_terms.intersection(insight_terms))
        relevance = bool(relevance_terms)
        evidence_available = bool(documents) and any(
            str(document.get("content", "")).strip() for document in documents
        )

        if relevance and evidence_available:
            confidence = 0.85
        elif evidence_available:
            confidence = 0.55
        elif relevance:
            confidence = 0.40
        else:
            confidence = 0.15

        passed = relevance and evidence_available and confidence >= 0.60
        verification_results.append(
            {
                "insight": insight,
                "relevant": relevance,
                "matching_terms": relevance_terms,
                "evidence_available": evidence_available,
                "confidence": confidence,
                "passed": passed,
            }
        )
        if passed:
            verified_insights.append(insight)

    return {
        "insights": verified_insights,
        "verification_results": verification_results,
    }


def save_output_node(state: ResearchState) -> dict[str, Any]:
    """Persist the complete research state as JSON."""

    output_path = Path(state.get("output_file") or DEFAULT_OUTPUT_FILE)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "topic": state.get("topic", ""),
        "search_queries": state.get("search_queries", []),
        "documents": state.get("documents", []),
        "search_iterations": state.get("search_iterations", 0),
        "max_iterations": state.get("max_iterations", 5),
        "insights": state.get("insights", []),
        "verification_results": state.get("verification_results", []),
        "output_file": str(output_path),
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"output_file": str(output_path)}


def should_continue_searching(state: ResearchState) -> str:
    """Route to another search iteration or finish the workflow."""

    search_iterations = state.get("search_iterations", 0)
    max_iterations = state.get("max_iterations", 5)
    if search_iterations >= max_iterations:
        return "save"
    return "search"


def build_research_graph():
    """Build and compile the AnySearch research graph."""

    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner_node)
    graph.add_node("search", search_node)
    graph.add_node("analyzer", analyzer_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("save", save_output_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "search")
    graph.add_edge("search", "analyzer")
    graph.add_edge("analyzer", "verifier")
    graph.add_conditional_edges(
        "verifier",
        should_continue_searching,
        {"search": "search", "save": "save"},
    )
    graph.add_edge("save", END)
    return graph.compile()


research_graph = build_research_graph()
