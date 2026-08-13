"""Content Agent intake and evidence selection.

The intake stage is deliberately separate from the Research Agent graph. It
can consume the persisted research contract without exposing raw documents or
tool output to a content-writing model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, TypedDict

from .llm_json import parse_json_response

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = PROJECT_ROOT / "prompts" / "content_intake.md"
MAX_SELECTED_INSIGHTS = 5


class ContentIntakeState(TypedDict, total=False):
    """Input and output contract for the first Content Agent stage."""

    research_output: dict[str, Any]
    research_output_file: str
    content_goal: str
    audience: str
    channel: str
    language: str
    selected_insight_ids: list[str]
    rejected_insights: list[dict[str, str]]
    content_angle: str
    evidence_map: list[dict[str, Any]]
    risk_flags: list[str]
    requires_more_research: bool


def load_research_output(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate the frozen Research Agent contract."""

    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Research output must be a JSON object.")
    if not isinstance(payload.get("eligible_insights", []), list):
        raise ValueError("Research output eligible_insights must be a list.")
    if not isinstance(payload.get("alternative_insights", []), list):
        raise ValueError("Research output alternative_insights must be a list.")
    return payload


def _insight_id(item: Any) -> str:
    return str(item.get("insight_id", "")).strip() if isinstance(item, dict) else ""


def _candidate_index(research_output: dict[str, Any]) -> dict[str, dict[str, Any]]:
    candidates = [
        *research_output.get("eligible_insights", []),
        *research_output.get("alternative_insights", []),
    ]
    return {
        insight_id: item
        for item in candidates
        if isinstance(item, dict) and (insight_id := _insight_id(item))
    }


def _load_model() -> Any | None:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except Exception:
        pass
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": os.getenv("CONTENT_AGENT_MODEL")
            or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "temperature": 0,
            "timeout": float(os.getenv("CONTENT_AGENT_TIMEOUT_SECONDS", "45")),
            "max_retries": int(os.getenv("CONTENT_AGENT_MAX_RETRIES", "0")),
        }
        base_url = os.getenv("OPENAI_BASE_URL", "").strip()
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)
    except Exception:
        return None


def _fallback_selection(
    candidates: dict[str, dict[str, Any]],
    *,
    content_goal: str,
    audience: str,
    channel: str,
    language: str,
) -> dict[str, Any]:
    """Safe offline fallback: use only already eligible evidence."""

    eligible = [
        item for item in candidates.values()
        if item.get("verification", {}).get("passed") is True
        and float(item.get("scoring", {}).get("total_score", 0)) >= 60
        and item in candidates.values()
    ]
    eligible.sort(key=lambda item: float(item.get("scoring", {}).get("total_score", 0)), reverse=True)
    selected = [_insight_id(item) for item in eligible[:MAX_SELECTED_INSIGHTS]]
    return {
        "selected_insight_ids": selected,
        "rejected_insights": [],
        "content_angle": content_goal or "Evidence-led email marketing guidance",
        "audience": audience,
        "channel": channel,
        "language": language,
        "evidence_map": [],
        "risk_flags": [],
        "requires_more_research": not bool(selected),
    }


def _model_selection(
    model: Any,
    *,
    research_output: dict[str, Any],
    content_goal: str,
    audience: str,
    channel: str,
    language: str,
) -> dict[str, Any] | None:
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    context = {
        "content_goal": content_goal,
        "audience": audience,
        "channel": channel,
        "language": language,
        "research": {
            "original_query": research_output.get("original_query", ""),
            "eligible_insights": research_output.get("eligible_insights", []),
            "alternative_insights": research_output.get("alternative_insights", []),
        },
    }
    try:
        response = model.invoke(
            [("system", prompt), ("human", json.dumps(context, ensure_ascii=False))]
        )
        return parse_json_response(response)
    except Exception:
        return None


def _sanitize_selection(raw: dict[str, Any], *, candidates: dict[str, dict[str, Any]], defaults: dict[str, Any]) -> dict[str, Any]:
    requested = raw.get("selected_insight_ids", [])
    selected: list[str] = []
    if isinstance(requested, list):
        for value in requested:
            insight_id = str(value).strip()
            item = candidates.get(insight_id)
            if not item or insight_id in selected:
                continue
            if item.get("verification", {}).get("passed") is not True:
                continue
            if float(item.get("scoring", {}).get("total_score", 0)) < 60:
                continue
            selected.append(insight_id)
            if len(selected) >= MAX_SELECTED_INSIGHTS:
                break

    rejected: list[dict[str, str]] = []
    if isinstance(raw.get("rejected_insights"), list):
        for entry in raw["rejected_insights"]:
            if not isinstance(entry, dict):
                continue
            insight_id = str(entry.get("insight_id", "")).strip()
            reason = str(entry.get("reason", "")).strip()
            if insight_id in candidates and insight_id not in selected and reason:
                rejected.append({"insight_id": insight_id, "reason": reason})

    result = {
        "selected_insight_ids": selected,
        "rejected_insights": rejected,
        "content_angle": str(raw.get("content_angle") or defaults["content_angle"]).strip(),
        "audience": str(raw.get("audience") or defaults["audience"]).strip(),
        "channel": str(raw.get("channel") or defaults["channel"]).strip(),
        "language": str(raw.get("language") or defaults["language"]).strip(),
        "evidence_map": [],
        "risk_flags": [str(flag).strip() for flag in raw.get("risk_flags", []) if str(flag).strip()]
        if isinstance(raw.get("risk_flags"), list) else [],
        "requires_more_research": bool(raw.get("requires_more_research", not selected)) or not selected,
    }
    for entry in raw.get("evidence_map", []) if isinstance(raw.get("evidence_map"), list) else []:
        if not isinstance(entry, dict):
            continue
        insight_id = str(entry.get("insight_id", "")).strip()
        if insight_id in selected:
            candidate = candidates[insight_id]
            source_urls = [
                str(source.get("url")).strip()
                for source in candidate.get("sources", [])
                if isinstance(source, dict) and str(source.get("url", "")).startswith(("http://", "https://"))
            ]
            constraints = [
                str(value).strip()
                for value in candidate.get("usage_constraints", [])
                if str(value).strip()
            ]
            result["evidence_map"].append({
                "insight_id": insight_id,
                "supports": str(entry.get("supports", "")).strip(),
                "source_urls": source_urls,
                "usage_constraints": constraints,
            })
    mapped_ids = {entry["insight_id"] for entry in result["evidence_map"]}
    for insight_id in selected:
        if insight_id in mapped_ids:
            continue
        candidate = candidates[insight_id]
        result["evidence_map"].append({
            "insight_id": insight_id,
            "supports": str(candidate.get("summary", "")).strip(),
            "source_urls": [
                str(source.get("url")).strip()
                for source in candidate.get("sources", [])
                if isinstance(source, dict) and str(source.get("url", "")).startswith(("http://", "https://"))
            ],
            "usage_constraints": list(candidate.get("usage_constraints", [])),
        })
    return result


def content_intake_node(state: ContentIntakeState) -> dict[str, Any]:
    """Select a bounded, citation-preserving evidence set for later drafting."""

    research_output = state.get("research_output")
    if research_output is None:
        path = state.get("research_output_file")
        if not path:
            raise ValueError("Content intake requires research_output or research_output_file.")
        research_output = load_research_output(path)
    candidates = _candidate_index(research_output)
    defaults = _fallback_selection(
        candidates,
        content_goal=str(state.get("content_goal", "")).strip(),
        audience=str(state.get("audience", "")).strip(),
        channel=str(state.get("channel", "")).strip(),
        language=str(state.get("language", "English")).strip() or "English",
    )
    model = _load_model()
    raw = _model_selection(
        model,
        research_output=research_output,
        content_goal=str(state.get("content_goal", "")).strip(),
        audience=str(state.get("audience", "")).strip(),
        channel=str(state.get("channel", "")).strip(),
        language=str(state.get("language", "English")).strip() or "English",
    ) if model is not None else None
    return _sanitize_selection(raw or defaults, candidates=candidates, defaults=defaults)
