"""LLM planning phase for the Content Agent Plan-and-Solve workflow."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .llm_json import parse_json_response, response_diagnostic, response_text
from .content_intent import content_requested, deliverable_type
from .model_config import load_chat_model, load_json_repair_model
from .output_contract import build_output_contract
from .state import MarketingState


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = PROJECT_ROOT / "prompts" / "content_planner.md"
DEFAULT_CONTENT_PLANNER_MODEL = "qwen3.7-flash"
MIN_PLAN_STEPS = 2
MAX_PLAN_STEPS = 3
AVAILABLE_FUTURE_TOOLS: tuple[str, ...] = ()


def load_content_planner_model():
    """Load the planner from the same API key and endpoint as Research."""

    return load_chat_model(
        model_env="CONTENT_PLANNER_MODEL",
        default_model=DEFAULT_CONTENT_PLANNER_MODEL,
        timeout_env="CONTENT_PLANNER_TIMEOUT_SECONDS",
        default_timeout=30,
        retries_env="CONTENT_PLANNER_MAX_RETRIES",
        default_retries=0,
        tokens_env="CONTENT_PLANNER_MAX_TOKENS",
        default_tokens=3000,
        json_mode=True,
    )


def _strings(value: Any, *, limit: int = 20) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(
        dict.fromkeys(
            cleaned
            for item in value[:limit]
            if (cleaned := str(item or "").strip())
        )
    )


def _sanitize_plan(raw: dict[str, Any], intent: dict[str, Any]) -> dict[str, Any]:
    raw_steps = raw.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("Content Planner returned no executable steps")

    steps: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_steps[:MAX_PLAN_STEPS], start=1):
        if not isinstance(item, dict):
            continue
        objective = str(item.get("objective", "")).strip()
        if not objective:
            continue
        proposed_id = str(item.get("step_id", "")).strip()
        step_id = proposed_id or f"step-{index:03d}"
        if step_id in seen_ids:
            step_id = f"step-{index:03d}"
        seen_ids.add(step_id)
        steps.append(
            {
                "step_id": step_id,
                "objective": objective,
                "required_inputs": _strings(item.get("required_inputs")),
                # Brand RAG is prefetched in parallel with Research. Normal
                # execution therefore does not pause for a tool round.
                "suggested_tools": [],
                "expected_output": str(item.get("expected_output", "")).strip(),
            }
        )
    if not steps:
        raise ValueError("Content Planner returned no valid executable steps")

    # The solving phase intentionally has exactly two semantic LLM calls:
    # select/organize context, then write the final deliverable. A Planner may
    # propose a third preparation detail, but Python folds all preparation into
    # the first phase so the Executor call budget remains stable.
    writing_step = steps[-1]
    preparation_steps = steps[:-1]
    if not preparation_steps:
        preparation_steps = [
            {
                "step_id": "step-001",
                "objective": (
                    "Select and organize verified Research evidence and "
                    "prefetched brand, platform, audience, and compliance context"
                ),
                "required_inputs": [
                    "research_output",
                    "rag_prefetch_results",
                    "content_intent",
                ],
                "suggested_tools": [],
                "expected_output": "selected_context",
            }
        ]
    preparation_objectives = [
        str(item.get("objective", "")).strip()
        for item in preparation_steps
        if str(item.get("objective", "")).strip()
    ]
    preparation_inputs = list(
        dict.fromkeys(
            [
                "research_output",
                "rag_prefetch_results",
                "content_intent",
                *(
                    value
                    for item in preparation_steps
                    for value in _strings(item.get("required_inputs"))
                ),
            ]
        )
    )
    first_step = {
        "step_id": str(preparation_steps[0].get("step_id") or "step-001"),
        "objective": "; ".join(preparation_objectives)
        or "Select and organize all approved context for final writing",
        "required_inputs": preparation_inputs,
        "suggested_tools": [],
        "expected_output": "selected_context",
    }
    second_step = {
        **writing_step,
        "step_id": str(writing_step.get("step_id") or "step-002"),
        "required_inputs": list(
            dict.fromkeys(
                ["selected_context", *_strings(writing_step.get("required_inputs"))]
            )
        ),
        "suggested_tools": [],
    }
    if second_step["step_id"] == first_step["step_id"]:
        second_step["step_id"] = "step-002"
    steps = [first_step, second_step]

    content_type = deliverable_type(intent) or ""
    return {
        "plan_id": str(raw.get("plan_id", "")).strip() or "content-plan-001",
        "final_goal": str(raw.get("final_goal", "")).strip(),
        # Intent identity comes from the Rewriter contract, not a second model
        # reinterpretation inside the Planner.
        "content_type": content_type,
        "steps": steps,
        "success_criteria": _strings(raw.get("success_criteria")),
        "planning_reasoning": str(raw.get("planning_reasoning", "")).strip(),
    }


def _planner_context(state: MarketingState) -> dict[str, Any]:
    contract = build_output_contract(state)

    def compact_insight(item: Any) -> Any:
        if not isinstance(item, dict):
            return item
        verification = item.get("verification", {})
        scoring = item.get("scoring", {})
        return {
            "insight_id": item.get("insight_id"),
            "title": item.get("title"),
            "summary": item.get("summary"),
            "claim_type": item.get("claim_type"),
            "usage_constraints": item.get("usage_constraints", []),
            "source_type": item.get("source_type"),
            "sources": [
                {
                    "title": source.get("title"),
                    "url": source.get("url"),
                    "published_at": source.get("published_at"),
                }
                for source in item.get("sources", [])[:5]
                if isinstance(source, dict)
            ],
            "verification": {
                "passed": verification.get("passed"),
                "confidence": verification.get("confidence"),
                "claim_supported": verification.get("claim_supported"),
            }
            if isinstance(verification, dict)
            else {},
            "total_score": scoring.get("total_score")
            if isinstance(scoring, dict)
            else None,
            "opportunity_type": item.get("opportunity_type"),
            "recommended_channels": item.get("recommended_channels", []),
        }

    return {
        "original_request": state.get("raw_user_request")
        or state.get("original_query")
        or state.get("topic", ""),
        "research_objective": state.get("research_objective", ""),
        "content_intent": state.get("content_intent", {}),
        "research_output": {
            "eligible_insights": [
                compact_insight(item) for item in contract["eligible_insights"]
            ],
            "alternative_insights": [
                compact_insight(item)
                for item in contract["alternative_insights"]
            ],
        },
        "rag_prefetch": {
            "status": state.get("rag_prefetch_status", "not_started"),
            "results": state.get("rag_prefetch_results", []),
            "errors": state.get("rag_prefetch_errors", []),
        },
        "medium_term_memory": {
            "status": state.get("memory_prefetch_status", "not_started"),
            "results": state.get("memory_prefetch_results", []),
            "errors": state.get("memory_prefetch_errors", []),
            "usage_boundary": (
                "Use for prior preferences and task experience only; never "
                "treat it as current market or product fact evidence."
            ),
        },
        "available_future_tools": list(AVAILABLE_FUTURE_TOOLS),
    }


def _format_repair_context(context: dict[str, Any], response: Any) -> dict[str, Any]:
    """Ask the Planner to regenerate a concise object after format failure."""

    return {
        **context,
        "format_repair": {
            "instruction": (
                "The previous response could not be parsed. Regenerate the full "
                "plan as one valid JSON object. Output no analysis, Markdown, or "
                "text outside the object. Keep planning_reasoning under 240 "
                "characters and each step concise."
            ),
            "previous_response_preview": response_text(response)[:2000],
        },
    }


def make_content_planner_node(
    model: Any | None = None,
    repair_model: Any | None = None,
):
    """Create an injectable LLM node that only plans; it never executes."""

    active_model: Any | None = model
    active_repair_model: Any | None = repair_model

    def content_planner_node(state: MarketingState) -> dict[str, Any]:
        nonlocal active_model, active_repair_model
        intent = dict(state.get("content_intent", {}))
        if not content_requested(intent):
            return {
                "content_planner_status": "skipped",
                "content_planner_reasoning": "The request is research-only.",
                "content_plan": {},
            }
        if active_model is None:
            active_model = load_content_planner_model()

        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        context = _planner_context(state)
        format_retries = max(
            0, int(os.getenv("CONTENT_PLANNER_FORMAT_RETRIES", "1"))
        )
        response: Any = None
        raw: dict[str, Any] | None = None
        for attempt in range(format_retries + 1):
            request_context = (
                context
                if attempt == 0
                else _format_repair_context(context, response)
            )
            if attempt == 0:
                invocation_model = active_model
            else:
                if active_repair_model is None:
                    active_repair_model = (
                        model if model is not None else load_json_repair_model()
                    )
                invocation_model = active_repair_model
            response = invocation_model.invoke(
                [
                    ("system", prompt),
                    (
                        "human",
                        json.dumps(
                            request_context,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    ),
                ]
            )
            raw = parse_json_response(response)
            if raw is not None:
                break
        if raw is None:
            raise ValueError(
                "Content Planner did not return a JSON object after "
                f"{format_retries + 1} attempt(s); "
                f"{response_diagnostic(response)}"
            )
        plan = _sanitize_plan(raw, intent)
        return {
            "content_plan": plan,
            "content_planner_status": "planned",
            "content_planner_reasoning": plan["planning_reasoning"],
        }

    return content_planner_node
