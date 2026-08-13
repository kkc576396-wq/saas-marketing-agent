"""Single-step LLM Executor for the Content Agent Solving phase."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .llm_json import parse_json_response, response_diagnostic
from .content_intent import deliverable_type
from .model_config import load_chat_model, load_json_repair_model
from .output_contract import build_output_contract
from .state import MarketingState


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = PROJECT_ROOT / "prompts" / "content_executor.md"
DEFAULT_CONTENT_EXECUTOR_MODEL = "qwen3.6-plus"
DEFAULT_CONTENT_EXECUTOR_PREP_MODEL = "qwen3.7-flash"
DEFAULT_CONTENT_EXECUTOR_PREMIUM_MODEL = "qwen3.6-plus"
DEFAULT_MAX_EXECUTOR_ITERATIONS = 20
EXECUTOR_STATUSES = {"completed", "blocked", "failed"}
REDDIT_REPLY_TARGET_COUNT = 5
ASTERISK_BULLET_PATTERN = re.compile(r"(?m)^([ \t]*)\*+[ \t]+")


def load_content_executor_model():
    """Load the default final-writing Executor model."""

    return load_chat_model(
        model_env="CONTENT_EXECUTOR_MODEL",
        default_model=DEFAULT_CONTENT_EXECUTOR_MODEL,
        timeout_env="CONTENT_EXECUTOR_TIMEOUT_SECONDS",
        default_timeout=90,
        retries_env="CONTENT_EXECUTOR_MAX_RETRIES",
        default_retries=0,
        tokens_env="CONTENT_EXECUTOR_MAX_TOKENS",
        default_tokens=8000,
        json_mode=False,
    )


def load_content_executor_prep_model():
    """Load the small model used for evidence and context preparation."""

    return load_chat_model(
        model_env="CONTENT_EXECUTOR_PREP_MODEL",
        default_model=DEFAULT_CONTENT_EXECUTOR_PREP_MODEL,
        timeout_env="CONTENT_EXECUTOR_PREP_TIMEOUT_SECONDS",
        default_timeout=30,
        retries_env="CONTENT_EXECUTOR_PREP_MAX_RETRIES",
        default_retries=0,
        tokens_env="CONTENT_EXECUTOR_PREP_MAX_TOKENS",
        default_tokens=3000,
        json_mode=False,
    )


def load_content_executor_premium_model():
    """Load Pro only for homepage and competitor long-form final writing."""

    return load_chat_model(
        model_env="CONTENT_EXECUTOR_PREMIUM_MODEL",
        default_model=DEFAULT_CONTENT_EXECUTOR_PREMIUM_MODEL,
        timeout_env="CONTENT_EXECUTOR_PREMIUM_TIMEOUT_SECONDS",
        default_timeout=120,
        retries_env="CONTENT_EXECUTOR_PREMIUM_MAX_RETRIES",
        default_retries=0,
        tokens_env="CONTENT_EXECUTOR_PREMIUM_MAX_TOKENS",
        default_tokens=10000,
        json_mode=False,
    )


def select_executor_model_role(
    state: MarketingState,
    *,
    current_step: dict[str, Any],
    current_index: int,
    step_count: int,
    executor_mode: str,
) -> str:
    """Route preparation to Qwen Flash and final writing by content risk."""

    expected = str(current_step.get("expected_output", "")).casefold()
    objective = str(current_step.get("objective", "")).casefold()
    writing_markers = {
        "draft",
        "report",
        "article",
        "post",
        "reply",
        "final_content",
        "revised_content",
    }
    is_writing = (
        executor_mode == "revision"
        or current_index >= max(0, step_count - 1)
        or expected in writing_markers
        or any(
            marker in objective
            for marker in ("write", "draft", "generate", "撰写", "生成")
        )
    )
    if not is_writing:
        return "prep"
    content_type = deliverable_type(state.get("content_intent", {}))
    return (
        "premium"
        if content_type in {"homepage_promotion", "competitor_report"}
        else "writing"
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


def _research_context(state: MarketingState) -> dict[str, Any]:
    contract = build_output_contract(state)
    return {
        "eligible_insights": contract["eligible_insights"],
        "alternative_insights": contract["alternative_insights"],
    }


def _reddit_reply_targets(research_output: dict[str, Any]) -> list[dict[str, Any]]:
    """Select the first five distinct verified Reddit posts deterministically."""

    targets: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for collection_name in ("eligible_insights", "alternative_insights"):
        for insight in research_output.get(collection_name, []):
            if not isinstance(insight, dict):
                continue
            source = next(
                (
                    item
                    for item in insight.get("sources", [])
                    if isinstance(item, dict)
                    and "reddit.com" in str(item.get("url", "")).casefold()
                ),
                None,
            )
            if source is None:
                continue
            url = str(source.get("url", "")).strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            targets.append(
                {
                    "insight_id": str(insight.get("insight_id", "")),
                    "post_title": str(insight.get("title", "")).strip(),
                    "post_url": url,
                    "published_at": source.get("published_at"),
                    "post_summary": str(insight.get("summary", "")).strip(),
                    "usage_constraints": insight.get("usage_constraints", []),
                }
            )
            if len(targets) >= REDDIT_REPLY_TARGET_COUNT:
                return targets
    return targets


def _contains_cjk(value: Any) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", json.dumps(value, ensure_ascii=False)))


def _reddit_reply_contract_violations(
    raw: dict[str, Any], targets: list[dict[str, Any]]
) -> list[str]:
    result = raw.get("result")
    replies = result.get("replies") if isinstance(result, dict) else None
    if not isinstance(replies, list):
        return ["result.replies must be an array"]
    expected_urls = [str(item["post_url"]) for item in targets]
    actual_urls = [
        str(item.get("post_url", ""))
        for item in replies
        if isinstance(item, dict)
    ]
    violations: list[str] = []
    if len(replies) != len(targets):
        violations.append(
            f"expected {len(targets)} replies, received {len(replies)}"
        )
    if set(actual_urls) != set(expected_urls):
        violations.append("reply URLs must exactly match the selected Top posts")
    if any(
        not isinstance(item, dict) or not str(item.get("reply", "")).strip()
        for item in replies
    ):
        violations.append("every reply requires non-empty reply text")
    return violations


def _sanitize_reddit_reply_result(
    result: Any, targets: list[dict[str, Any]]
) -> dict[str, Any]:
    if not isinstance(result, dict) or not isinstance(result.get("replies"), list):
        raise ValueError("Reddit reply output must contain result.replies")
    replies_by_url = {
        str(item.get("post_url", "")): item
        for item in result["replies"]
        if isinstance(item, dict)
    }
    normalized: list[dict[str, Any]] = []
    for target in targets:
        item = replies_by_url.get(str(target["post_url"]), {})
        reply = str(item.get("reply", "")).strip()
        if not reply:
            raise ValueError("Reddit reply output is missing a Top-post reply")
        normalized.append(
            {
                "insight_id": target["insight_id"],
                "post_title": target["post_title"],
                "post_url": target["post_url"],
                "published_at": target.get("published_at"),
                "reply": reply,
            }
        )
    return {"replies": normalized, "reply_count": len(normalized)}


def _sanitize_publish_content(value: Any) -> Any:
    """Remove asterisk Markdown from final public-facing content recursively."""

    if isinstance(value, str):
        value = ASTERISK_BULLET_PATTERN.sub(r"\1- ", value)
        return value.replace("*", "")
    if isinstance(value, list):
        return [_sanitize_publish_content(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _sanitize_publish_content(item)
            for key, item in value.items()
        }
    return value


def _validate_publish_content(value: Any) -> None:
    """Reject a final deliverable if deterministic cleanup missed an asterisk."""

    if isinstance(value, str) and "*" in value:
        raise ValueError("Final content contains forbidden character '*'")
    if isinstance(value, list):
        for item in value:
            _validate_publish_content(item)
    elif isinstance(value, dict):
        for item in value.values():
            _validate_publish_content(item)


def _final_output_contract_violations(
    raw: dict[str, Any],
    *,
    reddit_reply_targets: list[dict[str, Any]],
) -> list[str]:
    if str(raw.get("status", "")).casefold() != "completed":
        return []
    violations: list[str] = []
    if _contains_cjk(raw.get("result")):
        violations.append("all final generated content must be English-only")
    if reddit_reply_targets:
        violations.extend(
            _reddit_reply_contract_violations(raw, reddit_reply_targets)
        )
    return violations


def _allowed_evidence_ids(research_output: dict[str, Any]) -> set[str]:
    return {
        str(item.get("insight_id", "")).strip()
        for collection in (
            research_output.get("eligible_insights", []),
            research_output.get("alternative_insights", []),
        )
        for item in collection
        if isinstance(item, dict) and item.get("insight_id")
    }


def _sanitize_step_result(
    raw: dict[str, Any],
    *,
    current_step: dict[str, Any],
    research_output: dict[str, Any],
    allowed_rag_chunk_ids: set[str] | None = None,
    allowed_evidence_ids: set[str] | None = None,
    reddit_reply_targets: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    status = str(raw.get("status", "")).strip().casefold()
    if status not in EXECUTOR_STATUSES:
        raise ValueError("Content Executor returned an invalid status")

    result = raw.get("result", {})
    if reddit_reply_targets:
        result = _sanitize_reddit_reply_result(result, reddit_reply_targets)
    try:
        json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Content Executor result is not JSON serializable") from exc

    allowed_ids = _allowed_evidence_ids(research_output)
    if allowed_evidence_ids is not None:
        allowed_ids.intersection_update(allowed_evidence_ids)
    used_evidence_ids = [
        insight_id
        for insight_id in _strings(raw.get("used_evidence_ids"))
        if insight_id in allowed_ids
    ]
    if reddit_reply_targets:
        used_evidence_ids = list(
            dict.fromkeys(
                [
                    *used_evidence_ids,
                    *(
                        str(item.get("insight_id", ""))
                        for item in reddit_reply_targets
                        if item.get("insight_id") in allowed_ids
                    ),
                ]
            )
        )
    allowed_rag_ids = allowed_rag_chunk_ids or set()
    used_rag_chunk_ids = [
        chunk_id
        for chunk_id in _strings(raw.get("used_rag_chunk_ids"))
        if chunk_id in allowed_rag_ids
    ]
    if isinstance(result, dict) and isinstance(
        result.get("selected_insight_ids"), list
    ):
        result = dict(result)
        result["selected_insight_ids"] = [
            insight_id
            for insight_id in _strings(result["selected_insight_ids"])
            if insight_id in allowed_ids
        ]
    if isinstance(result, dict) and isinstance(
        result.get("selected_rag_chunk_ids"), list
    ):
        result = dict(result)
        result["selected_rag_chunk_ids"] = [
            chunk_id
            for chunk_id in _strings(result["selected_rag_chunk_ids"])
            if chunk_id in allowed_rag_ids
        ]

    blocking_reason = str(raw.get("blocking_reason", "")).strip()
    missing_inputs = _strings(raw.get("missing_inputs"))
    if status == "completed" and result in ({}, [], "", None):
        raise ValueError("Completed Content Executor step has no result")
    if status == "blocked" and not blocking_reason:
        raise ValueError("Blocked Content Executor step has no blocking reason")

    return {
        # The model cannot advance or switch steps; the graph owns identity.
        "step_id": str(current_step.get("step_id", "")),
        "status": status,
        "result_type": str(
            current_step.get("expected_output")
            or raw.get("result_type", "")
        ).strip(),
        "result": result,
        "used_evidence_ids": used_evidence_ids,
        "used_rag_chunk_ids": used_rag_chunk_ids,
        "execution_summary": str(raw.get("execution_summary", "")).strip(),
        "blocking_reason": blocking_reason,
        "missing_inputs": missing_inputs,
    }


def _executor_context(
    state: MarketingState,
    *,
    current_step: dict[str, Any],
    research_output: dict[str, Any],
    executor_mode: str,
) -> dict[str, Any]:
    reddit_targets = _reddit_reply_targets(research_output)
    return {
        "executor_mode": executor_mode,
        "original_request": state.get("raw_user_request")
        or state.get("original_query")
        or state.get("topic", ""),
        "full_plan": state.get("content_plan", {}),
        "execution_history": state.get("execution_history", []),
        "current_step": current_step,
        "content_intent": state.get("content_intent", {}),
        "research_output": research_output,
        "reddit_reply_targets": reddit_targets,
        "execution_artifacts": state.get("execution_artifacts", {}),
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
                "Use for preferences and prior task strategy only. These "
                "records are not evidence for current factual claims."
            ),
        },
        "current_draft": state.get("final_content"),
        "reflection_verification_results": state.get(
            "reflection_verification_results", []
        ),
        "revision_history": state.get("revision_history", []),
        "available_tools": [],
    }


def _allowed_rag_chunk_ids(
    state: MarketingState, step_index: int, executor_mode: str
) -> set[str]:
    return {
        str(chunk_id)
        for record in state.get("rag_tool_history", [])
        if isinstance(record, dict)
        and (
            (
                int(record.get("step_index", -1)) == step_index
                and str(record.get("executor_mode", "plan")) == executor_mode
            )
            or str(record.get("executor_mode", "")) == "prefetch"
        )
        for chunk_id in record.get("chunk_ids", [])
        if str(chunk_id).strip()
    }


def make_content_executor_node(model: Any | None = None):
    """Create one LLM node that is re-entered once per solving step."""

    active_models: dict[str, Any] = {}
    bound_models: dict[str, Any] = {}
    repair_model: Any | None = None

    def content_executor_node(state: MarketingState) -> dict[str, Any]:
        nonlocal repair_model
        executor_mode = (
            "revision" if state.get("executor_mode") == "revision" else "plan"
        )
        if executor_mode == "revision":
            steps = state.get("revision_steps", [])
            current_index = max(
                0, int(state.get("current_revision_step_index", 0))
            )
            completion_status = "revision_completed"
        else:
            plan = state.get("content_plan", {})
            steps = plan.get("steps", []) if isinstance(plan, dict) else []
            current_index = max(0, int(state.get("current_step_index", 0)))
            completion_status = "plan_completed"
        if not isinstance(steps, list) or not steps:
            raise ValueError(
                "Content Executor requires non-empty plan or revision steps"
            )

        if current_index >= len(steps):
            return {"executor_status": completion_status}

        iterations = int(state.get("executor_iterations", 0))
        max_iterations = max(
            1,
            int(
                state.get(
                    "max_executor_iterations", DEFAULT_MAX_EXECUTOR_ITERATIONS
                )
            ),
        )
        if iterations >= max_iterations:
            return {
                "executor_status": "iteration_limit_reached",
                "executor_summary": "Content Executor reached its iteration limit.",
            }

        current_step = steps[current_index]
        if not isinstance(current_step, dict):
            raise ValueError("Content plan contains an invalid step")
        model_role = select_executor_model_role(
            state,
            current_step=current_step,
            current_index=current_index,
            step_count=len(steps),
            executor_mode=executor_mode,
        )
        cache_key = "injected" if model is not None else model_role
        active_model = active_models.get(cache_key)
        bound_model = bound_models.get(cache_key)
        if active_model is None or bound_model is None:
            if model is not None:
                active_model = model
            elif model_role == "prep":
                active_model = load_content_executor_prep_model()
            elif model_role == "premium":
                active_model = load_content_executor_premium_model()
            else:
                active_model = load_content_executor_model()
            # RAG is prefetched before planning, so normal solving never adds a
            # tool-call round. This keeps the initial Executor at two semantic
            # LLM calls: context selection/organization and final writing.
            bound_model = active_model
            active_models[cache_key] = active_model
            bound_models[cache_key] = bound_model

        research_output = _research_context(state)
        current_step_id = (
            f"{executor_mode}:{str(current_step.get('step_id', ''))}"
        )
        messages = list(state.get("content_messages", []))
        if state.get("content_active_step_id") != current_step_id or not messages:
            prompt = PROMPT_PATH.read_text(encoding="utf-8")
            messages = [
                ("system", prompt),
                (
                    "human",
                    json.dumps(
                        _executor_context(
                            state,
                            current_step=current_step,
                            research_output=research_output,
                            executor_mode=executor_mode,
                        ),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                ),
            ]
        response = bound_model.invoke(messages)
        iterations += 1
        tool_calls = list(getattr(response, "tool_calls", []) or [])
        if tool_calls:
            return {
                "executor_iterations": iterations,
                "executor_status": "blocked",
                "executor_summary": (
                    "Content Executor cannot call tools after the parallel RAG "
                    "prefetch phase."
                ),
                "content_messages": [],
                "content_active_step_id": "",
            }

        raw = parse_json_response(response)
        format_retries = max(
            0, int(os.getenv("CONTENT_EXECUTOR_FORMAT_RETRIES", "1"))
        )
        format_attempts = 0
        while (
            raw is None
            and format_attempts < format_retries
            and iterations < max_iterations
        ):
            format_attempts += 1
            # Keep the original step context and any current-step tool results.
            # Use the unbound model so a formatting repair cannot trigger a new
            # tool call or accidentally mutate the plan.
            if repair_model is None:
                repair_model = (
                    model if model is not None else load_json_repair_model()
                )
            response = repair_model.invoke(
                [
                    *messages,
                    response,
                    (
                        "human",
                        "Your previous response could not be parsed as one complete "
                        "JSON object. Regenerate the complete result for the same "
                        "current step. Do not execute another step and do not call "
                        "tools. Return only one valid JSON object matching the "
                        "required Executor schema. Put long-form copy inside "
                        "result.content, escape JSON newlines correctly, keep all "
                        "metadata fields concise, and write no Markdown fence or "
                        "analysis outside the object.",
                    ),
                ]
            )
            iterations += 1
            raw = parse_json_response(response)
        if raw is None:
            raise ValueError(
                "Content Executor did not return a JSON object after "
                f"{format_attempts + 1} attempt(s) for {current_step_id}; "
                f"{response_diagnostic(response)}"
            )
        intent = state.get("content_intent", {})
        content_type = (
            str(intent.get("type", "")) if isinstance(intent, dict) else ""
        )
        final_delivery_step = (
            executor_mode == "revision" or current_index == len(steps) - 1
        )
        reddit_reply_targets = (
            _reddit_reply_targets(research_output)
            if final_delivery_step and content_type == "reddit_reply"
            else []
        )
        contract_violations = (
            _final_output_contract_violations(
                raw,
                reddit_reply_targets=reddit_reply_targets,
            )
            if final_delivery_step
            else []
        )
        contract_retries = max(
            0, int(os.getenv("CONTENT_EXECUTOR_CONTRACT_RETRIES", "1"))
        )
        contract_attempts = 0
        while (
            contract_violations
            and contract_attempts < contract_retries
            and iterations < max_iterations
        ):
            contract_attempts += 1
            response = active_model.invoke(
                [
                    *messages,
                    response,
                    (
                        "human",
                        "Your previous final deliverable violated the hard output "
                        "contract: "
                        + "; ".join(contract_violations)
                        + ". Regenerate the complete result for the same step as "
                        "one JSON object. All public-facing text must be English. "
                        "For reddit_reply, return result.replies with exactly one "
                        "reply for every supplied reddit_reply_targets item, using "
                        "each exact post_url. Do not omit, merge, or invent posts.",
                    ),
                ]
            )
            iterations += 1
            raw = parse_json_response(response)
            if raw is None:
                contract_violations = ["corrected response was not valid JSON"]
            else:
                contract_violations = _final_output_contract_violations(
                    raw,
                    reddit_reply_targets=reddit_reply_targets,
                )
        if contract_violations:
            raise ValueError(
                "Content Executor violated the final output contract after "
                f"{contract_attempts + 1} attempt(s): "
                + "; ".join(contract_violations)
            )
        allowed_rag_ids = (
            set(_strings(current_step.get("allowed_rag_chunk_ids")))
            if executor_mode == "revision"
            else _allowed_rag_chunk_ids(state, current_index, executor_mode)
        )
        allowed_revision_evidence = (
            set(_strings(current_step.get("allowed_evidence_ids")))
            if executor_mode == "revision"
            else None
        )
        record = _sanitize_step_result(
            raw,
            current_step=current_step,
            research_output=research_output,
            allowed_rag_chunk_ids=allowed_rag_ids,
            allowed_evidence_ids=allowed_revision_evidence,
            reddit_reply_targets=reddit_reply_targets,
        )
        if final_delivery_step and record["status"] == "completed":
            record["result"] = _sanitize_publish_content(record["result"])
            _validate_publish_content(record["result"])

        history_key = (
            "revision_history" if executor_mode == "revision" else "execution_history"
        )
        history = [*state.get(history_key, []), record]
        updates: dict[str, Any] = {
            history_key: history,
            "executor_iterations": iterations,
            "executor_summary": record["execution_summary"],
            "content_messages": [],
            "content_active_step_id": "",
        }
        if record["status"] == "completed":
            artifacts = dict(state.get("execution_artifacts", {}))
            result_type = record["result_type"] or f"step_{current_index + 1}"
            artifacts[result_type] = record["result"]
            next_index = current_index + 1
            if executor_mode == "revision":
                artifacts["revised_content"] = record["result"]
                updates.update(
                    {
                        "execution_artifacts": artifacts,
                        "final_content": record["result"],
                        "current_revision_step_index": next_index,
                        "current_step_attempt": 0,
                        "executor_status": (
                            "revision_completed"
                            if next_index >= len(steps)
                            else "revision_step_completed"
                        ),
                    }
                )
                if next_index >= len(steps) and state.get(
                    "save_after_revision", False
                ):
                    updates["reflection_status"] = "revision_applied_at_limit"
            else:
                updates.update(
                    {
                        "execution_artifacts": artifacts,
                        "current_step_index": next_index,
                        "current_step_attempt": 0,
                        "executor_status": (
                            "plan_completed"
                            if next_index >= len(steps)
                            else "step_completed"
                        ),
                    }
                )
                if next_index >= len(steps):
                    updates["final_content"] = record["result"]
        else:
            index_key = (
                "current_revision_step_index"
                if executor_mode == "revision"
                else "current_step_index"
            )
            updates.update(
                {
                    index_key: current_index,
                    "current_step_attempt": int(
                        state.get("current_step_attempt", 0)
                    )
                    + 1,
                    "executor_status": record["status"],
                }
            )
        return updates

    return content_executor_node


def route_after_content_executor(state: MarketingState) -> str:
    """Self-loop one completed step at a time; stop safely on blockers."""

    status = state.get("executor_status")
    if status in {"step_completed", "revision_step_completed"}:
        return "content_executor"
    if status == "plan_completed":
        return "draft_checkpoint"
    if status == "revision_completed":
        return (
            "save"
            if state.get("save_after_revision", False)
            else "draft_checkpoint"
        )
    return "save"
