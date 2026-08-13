"""LangChain tools and a Python dispatcher for controlled memory operations."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from .memory_manager import MemoryManager, get_default_memory_manager


class MemoryTool:
    """Small dispatcher matching ``memory_tool.execute(action, ...)`` calls."""

    def __init__(self, manager: MemoryManager | None = None):
        self.manager = manager or get_default_memory_manager()

    def execute(self, action: str, **kwargs: Any) -> dict[str, Any]:
        normalized = str(action or "").strip().casefold()
        if normalized == "add":
            return self.manager.add(**kwargs)
        if normalized == "search":
            return self.manager.search(**kwargs)
        if normalized == "forget":
            return self.manager.forget(**kwargs)
        raise ValueError("action must be add, search, or forget")


@tool
def memory_add(
    content: str,
    memory_type: str,
    brand_id: str = "smartpush",
    user_id: str | None = None,
    namespace: str = "marketing",
    memory_layer: str = "mid_term",
    importance: float = 0.5,
    confidence: float = 0.7,
    source_run_id: str | None = None,
    source_refs: list[str] | None = None,
    expires_at: str | None = None,
) -> dict[str, Any]:
    """Add one scoped memory.

    Long-term additions are always stored as candidates and require approval
    before they can enter the curated Brand RAG.
    """

    return get_default_memory_manager().add(
        content=content,
        memory_type=memory_type,
        brand_id=brand_id,
        user_id=user_id,
        namespace=namespace,
        memory_layer=memory_layer,
        importance=importance,
        confidence=confidence,
        source_run_id=source_run_id,
        source_refs=source_refs,
        expires_at=expires_at,
        actor="memory_add_tool",
    )


@tool
def memory_search(
    query: str,
    brand_id: str = "smartpush",
    user_id: str | None = None,
    namespaces: list[str] | None = None,
    memory_types: list[str] | None = None,
    memory_layers: list[str] | None = None,
    purpose: str = "content_generation",
    top_k: int = 5,
    include_candidates: bool = False,
) -> dict[str, Any]:
    """Search active scoped memories and approved long-term brand assets."""

    return get_default_memory_manager().search(
        query=query,
        brand_id=brand_id,
        user_id=user_id,
        namespaces=namespaces,
        memory_types=memory_types,
        memory_layers=memory_layers,
        purpose=purpose,
        top_k=top_k,
        include_candidates=include_candidates,
    )


@tool
def memory_forget(
    strategy: str,
    brand_id: str = "smartpush",
    user_id: str | None = None,
    namespace: str | None = None,
    memory_layer: str = "mid_term",
    threshold: float | None = None,
    max_age_days: int | None = None,
    max_records: int | None = None,
    memory_id: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Soft-forget scoped memories using importance, time, or capacity policy.

    Approved Brand RAG assets are protected; long-term forgetting only affects
    unapproved SQLite candidates.
    """

    return get_default_memory_manager().forget(
        strategy=strategy,
        brand_id=brand_id,
        user_id=user_id,
        namespace=namespace,
        memory_layer=memory_layer,
        threshold=threshold,
        max_age_days=max_age_days,
        max_records=max_records,
        memory_id=memory_id,
        dry_run=dry_run,
        actor="memory_forget_tool",
    )


MEMORY_TOOLS = [memory_add, memory_search, memory_forget]
