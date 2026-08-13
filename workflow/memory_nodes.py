"""LangGraph nodes for parallel memory retrieval and bounded memory commit."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .content_intent import deliverable_type
from .memory_manager import MemoryManager, get_default_memory_manager
from .state import MarketingState


def _enabled() -> bool:
    return os.getenv("MEMORY_ENABLED", "true").casefold() in {
        "1", "true", "yes", "on"
    }


def _scope(state: MarketingState) -> tuple[str, str | None, str]:
    brand_id = str(
        state.get("memory_brand_id")
        or os.getenv("MEMORY_DEFAULT_BRAND_ID", "smartpush")
    ).strip()
    user_id = str(state.get("memory_user_id") or "").strip() or None
    namespace = str(
        state.get("memory_namespace")
        or os.getenv("MEMORY_DEFAULT_NAMESPACE", "marketing")
    ).strip()
    return brand_id, user_id, namespace


def make_memory_prefetch_node(manager: MemoryManager | None = None):
    """Read medium-term task experience in parallel with Research and RAG."""

    def memory_prefetch_node(state: MarketingState) -> dict[str, Any]:
        if not _enabled():
            return {
                "memory_prefetch_status": "disabled",
                "memory_prefetch_results": [],
                "memory_prefetch_errors": [],
            }
        active_manager = manager or get_default_memory_manager()
        brand_id, user_id, namespace = _scope(state)
        query = str(
            state.get("raw_user_request")
            or state.get("research_objective")
            or state.get("original_query")
            or state.get("topic", "")
        ).strip()
        if not query:
            return {
                "memory_prefetch_status": "empty",
                "memory_prefetch_results": [],
                "memory_prefetch_errors": [],
            }
        try:
            payload = active_manager.search(
                query=query,
                brand_id=brand_id,
                user_id=user_id,
                namespaces=[namespace],
                memory_layers=["mid_term"],
                top_k=max(1, min(10, int(os.getenv("MEMORY_PREFETCH_TOP_K", "5")))),
            )
            results = list(payload.get("results", []))
            return {
                "memory_prefetch_status": "ready" if results else "empty",
                "memory_prefetch_results": results,
                "memory_prefetch_errors": [],
            }
        except Exception as exc:
            return {
                "memory_prefetch_status": "failed",
                "memory_prefetch_results": [],
                "memory_prefetch_errors": [f"{type(exc).__name__}: {exc}"],
            }

    return memory_prefetch_node


def _commit_content(state: MarketingState) -> str:
    request = str(
        state.get("raw_user_request")
        or state.get("original_query")
        or state.get("topic", "")
    ).strip()
    objective = str(state.get("research_objective", "")).strip()
    content_type = deliverable_type(state.get("content_intent", {})) or "research_only"
    outcome = str(state.get("executor_summary", "")).strip()
    if not outcome:
        eligible_count = len(state.get("eligible_insights", []))
        outcome = f"Produced {eligible_count} eligible research insights."
    parts = [f"Completed {content_type} task."]
    if request:
        parts.append(f"Original request: {request}")
    if objective and objective.casefold() != request.casefold():
        parts.append(f"Research objective: {objective}")
    if outcome:
        parts.append(f"Outcome: {outcome}")
    return " ".join(parts)[:2_000]


def _source_refs(state: MarketingState) -> list[str]:
    refs: list[str] = []
    for insight in state.get("eligible_insights", [])[:5]:
        if not isinstance(insight, dict):
            continue
        for source in insight.get("sources", [])[:3]:
            if isinstance(source, dict) and source.get("url"):
                refs.append(str(source["url"]))
    return list(dict.fromkeys(refs))[:10]


def make_memory_commit_node(manager: MemoryManager | None = None):
    """Persist a compact run episode without allowing it to become fact evidence."""

    def memory_commit_node(state: MarketingState) -> dict[str, Any]:
        if not _enabled():
            return {
                "memory_commit_status": "disabled",
                "memory_commit_ids": [],
                "memory_commit_errors": [],
            }
        if not state.get("final_content") and not state.get("eligible_insights"):
            return {
                "memory_commit_status": "skipped",
                "memory_commit_ids": [],
                "memory_commit_errors": [],
            }
        active_manager = manager or get_default_memory_manager()
        brand_id, user_id, namespace = _scope(state)
        output_file = str(state.get("output_file", "")).strip()
        source_run_id = Path(output_file).stem if output_file else None
        expiry = datetime.now(UTC) + timedelta(
            days=max(1, int(os.getenv("MEMORY_TASK_RETENTION_DAYS", "180")))
        )
        try:
            result = active_manager.add(
                content=_commit_content(state),
                memory_type="task_learning",
                brand_id=brand_id,
                user_id=user_id,
                namespace=namespace,
                memory_layer="mid_term",
                importance=0.6 if state.get("final_content") else 0.5,
                confidence=1.0,
                source_run_id=source_run_id,
                source_refs=_source_refs(state),
                metadata={
                    "content_intent": state.get("content_intent", {}),
                    "selected_sources": state.get("selected_sources", []),
                    "eligible_insight_ids": [
                        str(item.get("insight_id", ""))
                        for item in state.get("eligible_insights", [])
                        if isinstance(item, dict) and item.get("insight_id")
                    ],
                    "not_fact_evidence": True,
                },
                expires_at=expiry.isoformat(),
                actor="memory_commit_node",
            )
            memory = result.get("memory", {})
            memory_id = str(memory.get("memory_id", "")).strip()
            return {
                "memory_commit_status": "saved",
                "memory_commit_ids": [memory_id] if memory_id else [],
                "memory_commit_errors": [],
            }
        except Exception as exc:
            # Memory must never erase an otherwise usable final deliverable.
            return {
                "memory_commit_status": "failed",
                "memory_commit_ids": [],
                "memory_commit_errors": [f"{type(exc).__name__}: {exc}"],
            }

    return memory_commit_node
