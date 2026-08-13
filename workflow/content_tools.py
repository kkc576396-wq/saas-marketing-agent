"""Allowlisted RAG tools and their LangGraph execution node."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from .rag_store import search_index
from .state import MarketingState


@tool
def brand_rag_search(
    query: str,
    corpora: list[str] | None = None,
    usage: str = "public_content",
    top_k: int = 6,
) -> str:
    """Search curated SmartPush product, audience, brand, platform, and compliance knowledge.

    Use ``public_content`` for facts or guidance that may appear in public copy.
    Use ``internal_strategy`` for private ICP and strategy context; private
    results must guide decisions only and must never be quoted or disclosed.
    ``corpora`` may include product, audience, brand, platform, or compliance.
    """

    return json.dumps(
        search_index(
            query,
            corpora=corpora,
            usage=usage,
            top_k=top_k,
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    )


CONTENT_TOOLS = [brand_rag_search]
CONTENT_TOOL_BY_NAME = {tool_item.name: tool_item for tool_item in CONTENT_TOOLS}

RAG_PREFETCH_SPECS = (
    {
        "name": "public_brand_platform",
        "corpora": ["brand", "platform", "compliance"],
        "usage": "public_content",
        "top_k": 6,
        "focus": "brand voice, platform-native structure, and compliance rules",
    },
    {
        "name": "public_product",
        "corpora": ["product"],
        "usage": "public_content",
        "top_k": 5,
        "focus": "approved product capabilities and externally usable product facts",
    },
    {
        "name": "internal_audience",
        "corpora": ["audience"],
        "usage": "internal_strategy",
        "top_k": 4,
        "focus": "audience needs, objections, and positioning guidance",
    },
)


def _prefetch_query(state: MarketingState, focus: str) -> str:
    intent = state.get("content_intent", {})
    return " | ".join(
        part
        for part in (
            str(
                state.get("raw_user_request")
                or state.get("original_query")
                or state.get("topic", "")
            ).strip(),
            str(state.get("research_objective", "")).strip(),
            json.dumps(intent, ensure_ascii=False, separators=(",", ":"))
            if isinstance(intent, dict)
            else "",
            focus,
        )
        if part
    )


def make_rag_prefetch_node(search_fn: Any | None = None):
    """Prefetch brand context concurrently with the external Research branch."""

    active_search = search_fn or search_index

    def rag_prefetch_node(state: MarketingState) -> dict[str, Any]:
        if not state.get("requires_content_generation", False):
            return {
                "rag_prefetch_status": "skipped",
                "rag_prefetch_results": [],
                "rag_prefetch_errors": [],
            }

        intent = state.get("content_intent", {})
        requires_brand_rag = bool(
            isinstance(intent, dict) and intent.get("requires_brand_rag", False)
        )
        specs = [RAG_PREFETCH_SPECS[0]]
        if requires_brand_rag:
            specs.extend(RAG_PREFETCH_SPECS[1:])

        batches: dict[str, dict[str, Any]] = {}
        errors: list[str] = []
        with ThreadPoolExecutor(
            max_workers=len(specs), thread_name_prefix="rag-prefetch"
        ) as executor:
            future_map = {
                executor.submit(
                    active_search,
                    _prefetch_query(state, str(spec["focus"])),
                    corpora=list(spec["corpora"]),
                    usage=str(spec["usage"]),
                    top_k=int(spec["top_k"]),
                ): spec
                for spec in specs
            }
            for future in as_completed(future_map):
                spec = future_map[future]
                name = str(spec["name"])
                try:
                    payload = future.result()
                    batches[name] = payload if isinstance(payload, dict) else {}
                except Exception as exc:
                    errors.append(f"{name}: {type(exc).__name__}: {exc}")

        results: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for spec in specs:
            payload = batches.get(str(spec["name"]), {})
            for item in payload.get("results", []):
                if not isinstance(item, dict):
                    continue
                chunk_id = str(item.get("chunk_id", "")).strip()
                if not chunk_id or chunk_id in seen_ids:
                    continue
                seen_ids.add(chunk_id)
                results.append(item)

        status = "ready" if results and not errors else "partial" if results else "failed"
        chunk_ids = [str(item["chunk_id"]) for item in results]
        history = [
            *state.get("rag_tool_history", []),
            {
                "step_index": -1,
                "executor_mode": "prefetch",
                "tool_name": "brand_rag_prefetch",
                "arguments": {
                    "batch_names": [str(spec["name"]) for spec in specs],
                },
                "chunk_ids": chunk_ids,
                "error": "; ".join(errors),
            },
        ]
        return {
            "rag_prefetch_status": status,
            "rag_prefetch_results": results,
            "rag_prefetch_errors": errors,
            "rag_tool_history": history,
        }

    return rag_prefetch_node


def _latest_tool_calls(messages: list[Any]) -> list[dict[str, Any]]:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return list(message.tool_calls or [])
    return []


def content_tools_node(state: MarketingState) -> dict[str, Any]:
    """Execute one Executor-selected RAG call batch and persist observations."""

    messages = list(state.get("content_messages", []))
    calls = _latest_tool_calls(messages)
    if not calls:
        return {
            "executor_status": "failed",
            "executor_summary": "Content tools node received no tool call.",
        }

    current_index = int(state.get("current_step_index", 0))
    executor_mode = (
        "revision" if state.get("executor_mode") == "revision" else "plan"
    )
    if executor_mode == "revision":
        current_index = int(state.get("current_revision_step_index", 0))
    history = list(state.get("rag_tool_history", []))
    tool_messages: list[ToolMessage] = []
    for call in calls:
        name = str(call.get("name", ""))
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        tool_item = CONTENT_TOOL_BY_NAME.get(name)
        error = ""
        content = ""
        if tool_item is None:
            error = f"Unsupported Content tool: {name}"
        else:
            try:
                content = str(tool_item.invoke(args))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

        chunk_ids: list[str] = []
        if content and not error:
            try:
                parsed = json.loads(content)
                chunk_ids = [
                    str(item.get("chunk_id", ""))
                    for item in parsed.get("results", [])
                    if isinstance(item, dict) and item.get("chunk_id")
                ]
            except (json.JSONDecodeError, AttributeError):
                error = "RAG tool returned malformed JSON"
        observation = (
            content
            if not error
            else json.dumps({"status": "error", "error": error}, ensure_ascii=False)
        )
        tool_messages.append(
            ToolMessage(
                content=observation,
                tool_call_id=str(call.get("id", "")),
                name=name,
                status="error" if error else "success",
            )
        )
        history.append(
            {
                "step_index": current_index,
                "executor_mode": executor_mode,
                "tool_call_id": str(call.get("id", "")),
                "tool_name": name,
                "arguments": args,
                "chunk_ids": chunk_ids,
                "error": error,
            }
        )

    return {
        "content_messages": [*messages, *tool_messages],
        "rag_tool_history": history,
        "executor_status": "tool_results_ready",
        "executor_summary": "Content RAG observations are ready for the current step.",
    }
