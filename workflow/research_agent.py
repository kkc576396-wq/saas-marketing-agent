"""Bounded ReAct research layer with LLM-selected external tools."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import BoundedSemaphore
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from tools.agent_reach_tool import agent_reach_search
from tools.anysearch_tool import anysearch_batch_search, anysearch_search

from .normalizer import deduplicate_documents, normalize_tool_result
from .content_intent import deliverable_type
from .model_config import load_chat_model
from .router import AGENT_REACH_REDDIT, AGENT_REACH_RSS, AGENT_REACH_WEB, ANYSEARCH
from .state import ResearchState


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = PROJECT_ROOT / "prompts" / "research_agent.md"
RESEARCH_TOOLS = [anysearch_search, anysearch_batch_search, agent_reach_search]
TOOL_BY_NAME = {tool.name: tool for tool in RESEARCH_TOOLS}
MAX_PARALLEL_TOOL_CALLS = 5
MAX_OBSERVATION_DOCUMENTS = 5
MAX_OBSERVATION_SUMMARY_CHARS = 500
REDDIT_SEMAPHORE = BoundedSemaphore(2)
DEFAULT_RESEARCH_FAST_MODEL = "qwen3.7-flash"
DEFAULT_RESEARCH_BROAD_MODEL = "qwen3.7-plus"
BROAD_RESEARCH_FACETS = {
    "market_intelligence",
    "competitor_monitoring",
    "competitor_pricing",
    "trend_research",
    "product_update_research",
    "competitor_content_analysis",
}


def select_research_model_name(state: ResearchState) -> str:
    """Route narrow community work to Flash and broad research to Plus."""

    content_type = deliverable_type(state.get("content_intent", {}))
    if content_type in {"reddit_reply", "reddit_promotion"}:
        return os.getenv("RESEARCH_FAST_MODEL", DEFAULT_RESEARCH_FAST_MODEL)
    facets = {str(item) for item in state.get("intent_facets", [])}
    if content_type == "competitor_report" or facets.intersection(
        BROAD_RESEARCH_FACETS
    ):
        return os.getenv("RESEARCH_BROAD_MODEL", DEFAULT_RESEARCH_BROAD_MODEL)
    return os.getenv("RESEARCH_FAST_MODEL", DEFAULT_RESEARCH_FAST_MODEL)


def load_research_model(model_name: str | None = None):
    """Load the configured OpenAI-compatible model for ReAct tool calling."""

    selected = model_name or os.getenv(
        "RESEARCH_FAST_MODEL", DEFAULT_RESEARCH_FAST_MODEL
    )
    return load_chat_model(
        model_env="RESEARCH_FAST_MODEL",
        default_model=DEFAULT_RESEARCH_FAST_MODEL,
        model_override=selected,
        timeout_env="RESEARCH_AGENT_TIMEOUT_SECONDS",
        default_timeout=45,
        retries_env="RESEARCH_AGENT_MAX_RETRIES",
        default_retries=0,
        tokens_env="RESEARCH_AGENT_MAX_TOKENS",
        default_tokens=1000,
        json_mode=False,
    )


def _initial_messages(state: ResearchState) -> list[BaseMessage]:
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    context = {
        "topic": state.get("topic", ""),
        "raw_user_request": state.get("raw_user_request", ""),
        "research_objective": state.get("research_objective", ""),
        "downstream_content_intent": state.get("content_intent", {}),
        "translated_query": state.get("translated_query", ""),
        "research_date": state.get("research_date", ""),
        "freshness_required": state.get("freshness_required", False),
        "freshness_window_days": state.get("freshness_window_days", 365),
        "detected_entities": state.get("detected_entities", []),
        "intent_facets": state.get("intent_facets", []),
        "router_recommendation": state.get("recommended_sources", []),
        "router_reasoning": state.get("source_plan_reasoning", ""),
        "suggested_queries": state.get("source_queries", {}),
        "max_search_rounds": state.get("max_iterations", 5),
    }
    return [
        SystemMessage(content=prompt),
        HumanMessage(
            content=(
                "Research this goal using the available tools. Initial context:\n"
                + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
            )
        ),
    ]


def make_research_agent_node(model: Any | None = None):
    """Create a model node so tests can inject a deterministic fake model."""

    bound_models: dict[str, Any] = {}

    def research_agent_node(state: ResearchState) -> dict[str, Any]:
        model_name = (
            "injected" if model is not None else select_research_model_name(state)
        )
        bound_model = bound_models.get(model_name)
        if bound_model is None:
            active_model = model or load_research_model(model_name)
            # qwen3.7-plus thinking mode supports automatic tool selection but
            # rejects forced/required tool_choice, so intentionally omit it.
            bound_model = active_model.bind_tools(RESEARCH_TOOLS)
            bound_models[model_name] = bound_model

        history = list(state.get("messages", []))
        new_base_messages: list[BaseMessage] = []
        if not history:
            new_base_messages = _initial_messages(state)
            history.extend(new_base_messages)

        iteration = int(state.get("search_iterations", 0))
        max_iterations = int(state.get("max_iterations", 5))
        control = SystemMessage(
            content=(
                f"Search budget: {iteration}/{max_iterations} rounds used; "
                f"{max(0, max_iterations - iteration)} remain. "
                "Use tools only when another evidence-gathering round is useful."
            )
        )
        response = bound_model.invoke(history + [control])
        tool_calls = list(getattr(response, "tool_calls", []) or [])
        status = "tool_calls_requested" if tool_calls else "complete"
        reasoning = str(getattr(response, "content", "") or "").strip()
        return {
            "messages": [*new_base_messages, response],
            "research_agent_status": status,
            "research_agent_reasoning": reasoning,
        }

    return research_agent_node


def _source_for_call(name: str, args: dict[str, Any]) -> str:
    if name.startswith("anysearch_"):
        return ANYSEARCH
    channel = str(args.get("channel", "")).casefold()
    return {
        "reddit": AGENT_REACH_REDDIT,
        "web": AGENT_REACH_WEB,
        "rss": AGENT_REACH_RSS,
    }.get(channel, f"Agent-Reach {channel.title()}" if channel else "Agent-Reach")


def _queries_for_call(name: str, args: dict[str, Any]) -> list[str]:
    if name == "anysearch_search":
        return [str(args.get("query", ""))]
    if name == "anysearch_batch_search":
        return [
            str(item.get("query", ""))
            for item in args.get("queries", [])
            if isinstance(item, dict) and item.get("query")
        ]
    return [str(query) for query in args.get("queries", []) if str(query).strip()]


def _execute_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    name = str(call.get("name", ""))
    args = call.get("args") if isinstance(call.get("args"), dict) else {}
    tool = TOOL_BY_NAME.get(name)
    if tool is None:
        return {
            "id": str(call.get("id", "")),
            "name": name,
            "args": args,
            "content": "",
            "error": f"Unsupported research tool: {name}",
        }
    try:
        if name == "agent_reach_search" and args.get("channel") == "reddit":
            # OpenCLI Reddit calls share one browser bridge. Preserve the
            # established two-call safety limit inside otherwise parallel runs.
            with REDDIT_SEMAPHORE:
                content = str(tool.invoke(args))
        else:
            content = str(tool.invoke(args))
        error = ""
    except Exception as exc:  # one failed provider must not erase other evidence
        content = ""
        error = f"{type(exc).__name__}: {exc}"
    return {
        "id": str(call.get("id", "")),
        "name": name,
        "args": args,
        "content": content,
        "error": error,
    }


def _latest_tool_calls(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return list(message.tool_calls or [])
    return []


def _compact_observation(
    result: dict[str, Any], documents: list[dict[str, Any]]
) -> str:
    payload = {
        "status": "error" if result.get("error") else "ok",
        "source": result["source"],
        "queries": result["queries"],
        "retrieved_count": len(documents),
        "results": [
            {
                "title": document.get("title", ""),
                "url": document.get("url"),
                "published_at": document.get("published_at"),
                "date_status": document.get("date_status", "unknown"),
                "summary": str(document.get("summary", ""))[
                    :MAX_OBSERVATION_SUMMARY_CHARS
                ],
            }
            for document in documents[:MAX_OBSERVATION_DOCUMENTS]
        ],
    }
    if result.get("error"):
        payload["error"] = result["error"]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def research_tools_node(state: ResearchState) -> dict[str, Any]:
    """Execute one LLM-selected tool batch concurrently and record evidence."""

    calls = _latest_tool_calls(list(state.get("messages", [])))
    if not calls:
        return {
            "research_agent_status": "complete",
            "research_agent_reasoning": "The model requested no research tools.",
        }

    ordered_results: list[dict[str, Any] | None] = [None] * len(calls)
    with ThreadPoolExecutor(
        max_workers=min(MAX_PARALLEL_TOOL_CALLS, len(calls))
    ) as executor:
        futures = {
            executor.submit(_execute_tool_call, call): index
            for index, call in enumerate(calls)
        }
        for future in as_completed(futures):
            ordered_results[futures[future]] = future.result()

    iteration = int(state.get("search_iterations", 0)) + 1
    documents = list(state.get("documents", []))
    tool_results = list(state.get("tool_results", []))
    tool_history = list(state.get("research_tool_history", []))
    selected_sources = list(state.get("selected_sources", []))
    tool_messages: list[ToolMessage] = []
    round_queries: dict[str, list[str]] = {}

    for raw_result in ordered_results:
        if raw_result is None:
            continue
        source = _source_for_call(raw_result["name"], raw_result["args"])
        queries = _queries_for_call(raw_result["name"], raw_result["args"])
        result = {
            "iteration": iteration,
            "source": source,
            "tool_name": raw_result["name"],
            "queries": queries,
            "content": raw_result["content"],
        }
        if raw_result.get("error"):
            result["errors"] = [raw_result["error"]]
        parsed_documents = normalize_tool_result(result)
        documents.extend(parsed_documents)
        tool_results.append(result)
        selected_sources.append(source)
        round_queries.setdefault(source, []).extend(queries)
        tool_messages.append(
            ToolMessage(
                content=_compact_observation(
                    {**result, "error": raw_result.get("error", "")},
                    parsed_documents,
                ),
                tool_call_id=raw_result["id"],
                name=raw_result["name"],
                status="error" if raw_result.get("error") else "success",
            )
        )

    tool_history.append(
        {
            "iteration": iteration,
            "source_queries": {
                source: list(dict.fromkeys(queries))
                for source, queries in round_queries.items()
            },
            "tool_calls": [
                {
                    "name": result["name"],
                    "args": result["args"],
                    "error": result.get("error", ""),
                }
                for result in ordered_results
                if result is not None
            ],
        }
    )
    max_iterations = int(state.get("max_iterations", 5))
    reached_limit = iteration >= max_iterations
    return {
        "messages": tool_messages,
        "documents": deduplicate_documents(documents),
        "tool_results": tool_results,
        "research_tool_history": tool_history,
        "query_history": tool_history,
        "search_iterations": iteration,
        "selected_sources": list(dict.fromkeys(selected_sources)),
        "source_reasoning": "Sources selected dynamically by the Research Agent LLM.",
        "research_agent_status": "max_iterations_reached" if reached_limit else "observing",
        "research_agent_reasoning": (
            f"Maximum search iterations reached ({iteration}/{max_iterations})."
            if reached_limit
            else f"Research Agent completed tool round {iteration}/{max_iterations}."
        ),
    }


def route_after_research_agent(state: ResearchState) -> str:
    """Run tools only when the latest model response requested them."""

    if int(state.get("search_iterations", 0)) >= int(
        state.get("max_iterations", 5)
    ):
        return "evaluation"
    return "tools" if _latest_tool_calls(list(state.get("messages", []))) else "evaluation"


def route_after_research_tools(state: ResearchState) -> str:
    """Return observations to the model unless the round budget is exhausted."""

    if int(state.get("search_iterations", 0)) >= int(
        state.get("max_iterations", 5)
    ):
        return "evaluation"
    return "research_agent"
