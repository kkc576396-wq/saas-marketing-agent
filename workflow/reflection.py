"""Two-model Reflection/CoVe phase for Content Agent outputs."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .llm_json import parse_json_response, response_diagnostic
from .domain_context import all_competitor_aliases
from .content_intent import deliverable_type
from .model_config import load_chat_model, load_json_repair_model
from .output_contract import build_output_contract
from .rag_store import get_chunks_by_ids
from .state import MarketingState


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTION_PROMPT_PATH = PROJECT_ROOT / "prompts" / "reflection_question_planner.md"
VERIFICATION_PROMPT_PATH = PROJECT_ROOT / "prompts" / "reflection_verification.md"
DEFAULT_REFLECTION_QUESTION_MODEL = "deepseek-v4-flash"
DEFAULT_VERIFICATION_MODEL = "qwen3.7-plus"
DEFAULT_MAX_REFLECTION_ITERATIONS = 1
MAX_CLAIM_CHECKS = 20
MAX_QUALITY_CHECKS = 20
MAX_REVISION_STEPS = 8
CLAIM_TYPES = {
    "product_capability",
    "market_trend",
    "competitor_fact",
    "pricing_or_plan",
    "performance_metric",
    "customer_or_community_claim",
    "date_or_recency",
    "platform_or_policy",
    "other_factual_claim",
}
RISK_LEVELS = {"low", "medium", "high"}
QUALITY_CATEGORIES = {
    "intent_fidelity",
    "missing_requirement",
    "platform_style",
    "brand_voice",
    "compliance",
    "structure",
    "clarity",
}
CLAIM_VERDICTS = {"supported", "contradicted", "insufficient_evidence"}
QUALITY_VERDICTS = {"confirmed", "dismissed"}
REVISION_ACTIONS = {
    "remove",
    "replace",
    "qualify",
    "add_missing_requirement",
    "restructure",
    "style_fix",
    "compliance_fix",
}
PRICING_RISK_PATTERN = re.compile(
    r"(?:[$€£¥]\s?\d|\bprice|\bpricing|\bcost|\btier|\bdiscount|"
    r"\bdollars?\b|\bmonthly\b|\bannual(?:ly)?\b|\bsubscription\b|"
    r"价格|定价|费用|套餐|折扣)",
    re.IGNORECASE,
)
METRIC_RISK_PATTERN = re.compile(
    r"(?:\d|%|\bpercent\b|\broi\b|\bconversion rate\b|\bopen rate\b|"
    r"\bclick rate\b|百分比|百分之|转化率|打开率|点击率|投资回报|"
    r"[零一二三四五六七八九十百千万两]+(?:个|倍|元|美元|天|月|年))",
    re.IGNORECASE,
)
PRODUCT_REFERENCE_PATTERN = re.compile(
    r"(?:smartpush|shopline|our product|our platform|the product|the platform|"
    r"\bthis tool\b|\bthe tool\b|\bit can\b|"
    r"我们的产品|我们的平台|该产品|该平台|该工具)",
    re.IGNORECASE,
)
CAPABILITY_PATTERN = re.compile(
    r"(?:\bcan\b|\bsupports?\b|\boffers?\b|\benables?\b|\bautomates?\b|"
    r"\bintegrates?\b|功能|支持|能够|可以|自动化|集成|提供)",
    re.IGNORECASE,
)
MARKET_TREND_PATTERN = re.compile(
    r"(?:\btrend|\bmarket\b|\bindustry\b|\badoption\b|\bgrowing\b|"
    r"\bincreasingly\b|\bmore (?:brands|merchants|companies)\b|"
    r"市场趋势|行业趋势|市场增长|行业增长|采用率|越来越多|最新趋势)",
    re.IGNORECASE,
)


def _draft_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(
            str(value.get(key, ""))
            for key in ("title", "content", "report", "draft", "reply", "article")
            if value.get(key)
        ) or json.dumps(value, ensure_ascii=False)
    return str(value or "")


def assess_reflection_risk(state: MarketingState) -> dict[str, Any]:
    """Choose lightweight Reddit review or the complete two-model CoVe."""

    content_type = deliverable_type(state.get("content_intent", {}))
    draft = _draft_text(_final_content(state))
    normalized = draft.casefold()
    reasons: list[str] = []
    if content_type in {"homepage_promotion", "competitor_report"}:
        reasons.append("content_type_requires_full_cove")
    elif content_type != "reddit_reply":
        reasons.append("not_a_reddit_advisory_reply")
    if METRIC_RISK_PATTERN.search(draft):
        reasons.append("number_or_metric")
    if PRICING_RISK_PATTERN.search(draft):
        reasons.append("pricing_or_plan")
    if any(
        re.search(
            rf"(?<![a-z0-9]){re.escape(alias.casefold())}(?![a-z0-9])",
            normalized,
        )
        for alias in all_competitor_aliases()
    ):
        reasons.append("competitor_fact")
    elif re.search(r"\bcompetitor(?:s)?\b|竞品|竞争对手", draft, re.IGNORECASE):
        reasons.append("competitor_fact")
    if PRODUCT_REFERENCE_PATTERN.search(draft) and CAPABILITY_PATTERN.search(draft):
        reasons.append("product_capability")
    if MARKET_TREND_PATTERN.search(draft):
        reasons.append("market_trend")
    return {
        "reflection_mode": "full" if reasons else "light",
        "reflection_risk_level": "high" if reasons else "low",
        "reflection_risk_reasons": list(dict.fromkeys(reasons)),
        "reflection_status": "risk_assessed",
    }


def _load_chat_model(
    *,
    model_env: str,
    default_model: str,
    timeout_env: str,
    default_timeout: str,
    retries_env: str,
    tokens_env: str,
    default_tokens: str,
):
    return load_chat_model(
        model_env=model_env,
        default_model=default_model,
        timeout_env=timeout_env,
        default_timeout=float(default_timeout),
        retries_env=retries_env,
        default_retries=0,
        tokens_env=tokens_env,
        default_tokens=int(default_tokens),
        # Qwen Verification supports native JSON mode. DeepSeek on the current
        # Model Studio compatibility layer is kept prompt-structured and uses
        # the Qwen JSON repair fallback only when needed.
        json_mode=default_model.startswith("qwen"),
    )


def load_reflection_question_model():
    """Load the DeepSeek Reflection Question Planner."""

    return _load_chat_model(
        model_env="REFLECTION_QUESTION_MODEL",
        default_model=DEFAULT_REFLECTION_QUESTION_MODEL,
        timeout_env="REFLECTION_QUESTION_TIMEOUT_SECONDS",
        default_timeout="45",
        retries_env="REFLECTION_QUESTION_MAX_RETRIES",
        tokens_env="REFLECTION_QUESTION_MAX_TOKENS",
        default_tokens="5000",
    )


def load_verification_model():
    """Load the independent Qwen evidence verifier."""

    return _load_chat_model(
        model_env="VERIFICATION_MODEL",
        default_model=DEFAULT_VERIFICATION_MODEL,
        timeout_env="VERIFICATION_TIMEOUT_SECONDS",
        default_timeout="60",
        retries_env="VERIFICATION_MAX_RETRIES",
        tokens_env="VERIFICATION_MAX_TOKENS",
        default_tokens="6000",
    )


def _invoke_reflection_json(
    model: Any,
    messages: list[Any],
    *,
    node_name: str,
    repair_model: Any | None = None,
) -> dict[str, Any]:
    """Invoke a Reflection model with one bounded schema-repair retry."""

    format_retries = max(
        0, int(os.getenv("REFLECTION_FORMAT_RETRIES", "1"))
    )
    response: Any = None
    for attempt in range(format_retries + 1):
        request_messages = messages
        if attempt > 0:
            request_messages = [
                *messages,
                response,
                (
                    "human",
                    "Your previous response could not be parsed as one complete "
                    "JSON object. Regenerate the full answer for the same task. "
                    "Return exactly one valid JSON object matching the required "
                    "schema, with no analysis, preface, or Markdown fence. Keep "
                    "descriptive fields concise and preserve every required item.",
                ),
            ]
        invocation_model = model if attempt == 0 else (repair_model or model)
        response = invocation_model.invoke(request_messages)
        raw = parse_json_response(response)
        if raw is not None:
            return raw
    raise ValueError(
        f"{node_name} did not return JSON after "
        f"{format_retries + 1} attempt(s); {response_diagnostic(response)}"
    )


def _is_timeout_error(exc: Exception) -> bool:
    """Recognize SDK and built-in timeout exceptions without hiding others."""

    return isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.casefold()


def _reflection_timeout_updates(*, node_name: str) -> dict[str, Any]:
    return {
        "reflection_question_status": "timeout"
        if node_name == "Reflection Question Planner"
        else "questions_ready",
        "reflection_status": "completed_with_review_warning",
        "verification_summary": (
            f"{node_name} timed out. The completed draft was preserved and "
            "saved without a completed automated review."
        ),
    }


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


def _final_content(state: MarketingState) -> Any:
    if state.get("final_content") not in (None, "", {}, []):
        return state["final_content"]
    artifacts = state.get("execution_artifacts", {})
    if isinstance(artifacts, dict):
        for key in ("revised_content", "draft", "report", "final_content"):
            if artifacts.get(key) not in (None, "", {}, []):
                return artifacts[key]
    for record in reversed(state.get("revision_history", [])):
        if isinstance(record, dict) and record.get("result") not in (
            None,
            "",
            {},
            [],
        ):
            return record["result"]
    for record in reversed(state.get("execution_history", [])):
        if isinstance(record, dict) and record.get("status") == "completed":
            if record.get("result") not in (None, "", {}, []):
                return record["result"]
    return None


def _research_evidence(state: MarketingState) -> dict[str, Any]:
    contract = build_output_contract(state)
    return {
        "eligible_insights": contract["eligible_insights"],
        "alternative_insights": contract["alternative_insights"],
    }


def _allowed_evidence_ids(research: dict[str, Any]) -> set[str]:
    return {
        str(item.get("insight_id", "")).strip()
        for collection in (
            research.get("eligible_insights", []),
            research.get("alternative_insights", []),
        )
        for item in collection
        if isinstance(item, dict) and item.get("insight_id")
    }


def _retrieved_rag_ids(state: MarketingState) -> list[str]:
    ids: list[str] = []
    for record in state.get("rag_tool_history", []):
        if not isinstance(record, dict):
            continue
        ids.extend(_strings(record.get("chunk_ids")))
    for history_name in ("execution_history", "revision_history"):
        for record in state.get(history_name, []):
            if isinstance(record, dict):
                ids.extend(_strings(record.get("used_rag_chunk_ids")))
    return list(dict.fromkeys(ids))


def _sanitize_question_plan(raw: dict[str, Any]) -> dict[str, Any]:
    claims: list[dict[str, Any]] = []
    seen_claim_ids: set[str] = set()
    raw_claims = raw.get("claim_checks", [])
    if not isinstance(raw_claims, list):
        raw_claims = []
    for index, item in enumerate(
        raw_claims[:MAX_CLAIM_CHECKS], start=1
    ):
        if not isinstance(item, dict):
            continue
        excerpt = str(item.get("draft_excerpt", "")).strip()
        question = str(item.get("verification_question", "")).strip()
        statement = str(item.get("claim", "")).strip() or excerpt
        if not excerpt or not question or not statement:
            continue
        claim_id = str(item.get("claim_id", "")).strip() or f"claim-{index:03d}"
        if claim_id in seen_claim_ids:
            claim_id = f"claim-{index:03d}"
        seen_claim_ids.add(claim_id)
        claim_type = str(item.get("claim_type", "")).strip().casefold()
        risk = str(item.get("risk_level", "")).strip().casefold()
        claims.append(
            {
                "claim_id": claim_id,
                "draft_excerpt": excerpt,
                "claim": statement,
                "claim_type": (
                    claim_type if claim_type in CLAIM_TYPES else "other_factual_claim"
                ),
                "risk_level": risk if risk in RISK_LEVELS else "medium",
                "verification_question": question,
                "required_evidence_type": str(
                    item.get("required_evidence_type", "")
                ).strip(),
            }
        )

    quality: list[dict[str, Any]] = []
    seen_issue_ids: set[str] = set()
    raw_quality = raw.get("quality_checks", [])
    if not isinstance(raw_quality, list):
        raw_quality = []
    for index, item in enumerate(
        raw_quality[:MAX_QUALITY_CHECKS], start=1
    ):
        if not isinstance(item, dict):
            continue
        problem = str(item.get("problem", "")).strip()
        instruction = str(item.get("revision_instruction", "")).strip()
        if not problem or not instruction:
            continue
        issue_id = str(item.get("issue_id", "")).strip() or f"issue-{index:03d}"
        if issue_id in seen_issue_ids:
            issue_id = f"issue-{index:03d}"
        seen_issue_ids.add(issue_id)
        category = str(item.get("category", "")).strip().casefold()
        severity = str(item.get("severity", "")).strip().casefold()
        quality.append(
            {
                "issue_id": issue_id,
                "category": (
                    category if category in QUALITY_CATEGORIES else "clarity"
                ),
                "severity": severity if severity in RISK_LEVELS else "medium",
                "draft_excerpt": str(item.get("draft_excerpt", "")).strip(),
                "problem": problem,
                "revision_instruction": instruction,
            }
        )
    return {
        "claim_checks": claims,
        "quality_checks": quality,
        "review_summary": str(raw.get("review_summary", "")).strip(),
    }


def _question_context(state: MarketingState, draft: Any) -> dict[str, Any]:
    return {
        "original_request": state.get("raw_user_request")
        or state.get("original_query")
        or state.get("topic", ""),
        "content_intent": state.get("content_intent", {}),
        "content_plan": state.get("content_plan", {}),
        "draft_to_review": draft,
        "execution_history": state.get("execution_history", []),
        "revision_history": state.get("revision_history", []),
        "used_research_evidence": _research_evidence(state),
        "retrieved_rag_chunk_ids": _retrieved_rag_ids(state),
        "review_mode": state.get("reflection_mode", "full"),
        "review_mode_instruction": (
            "Run one lightweight intent, platform-style, brand-voice, clarity, "
            "and compliance review. Return no claim checks unless you detect an "
            "actual factual assertion missed by the deterministic risk gate."
            if state.get("reflection_mode") == "light"
            else "Extract factual claims and quality issues for full two-model CoVe."
        ),
        "reflection_round": int(state.get("reflection_iterations", 0)) + 1,
    }


def _light_review_updates(
    state: MarketingState,
    plan: dict[str, Any],
    *,
    reflection_iteration: int,
) -> dict[str, Any]:
    """Convert one lightweight quality review directly into bounded revisions."""

    steps = [
        {
            "step_id": f"revision-{index:03d}",
            "target_ids": [str(issue["issue_id"])],
            "action": {
                "compliance": "compliance_fix",
                "platform_style": "style_fix",
                "brand_voice": "style_fix",
                "structure": "restructure",
                "missing_requirement": "add_missing_requirement",
                "intent_fidelity": "add_missing_requirement",
            }.get(str(issue.get("category", "")), "style_fix"),
            "instruction": str(issue.get("revision_instruction", "")),
            "allowed_evidence_ids": [],
            "allowed_rag_chunk_ids": [],
            "expected_output": "revised_content",
        }
        for index, issue in enumerate(
            plan.get("quality_checks", [])[:MAX_REVISION_STEPS], start=1
        )
        if isinstance(issue, dict)
        and issue.get("issue_id")
        and issue.get("revision_instruction")
    ]
    status = "revision_required" if steps else "passed"
    summary = plan.get("review_summary") or (
        "Lightweight Reddit review requested a bounded revision."
        if steps
        else "Lightweight Reddit review passed."
    )
    history = [
        *state.get("reflection_history", []),
        {
            "round": reflection_iteration,
            "mode": "light",
            "question_plan": plan,
            "verification_results": [],
            "verification_summary": summary,
            "revision_steps": steps,
            "status": status,
        },
    ]
    return {
        "reflection_question_plan": {
            **plan,
            "claim_checks": [],
        },
        "reflection_question_status": "light_review_completed",
        "reflection_verification_results": [],
        "reflection_history": history,
        "verification_summary": str(summary),
        "revision_steps": steps,
        "current_revision_step_index": 0,
        "reflection_status": status,
        "reflection_iterations": reflection_iteration,
        "executor_mode": "revision" if steps else "plan",
        "save_after_revision": bool(steps),
    }


def make_reflection_question_node(model: Any | None = None):
    """Create the DeepSeek node that finds claims and review questions only."""

    active_model: Any | None = model
    active_repair_model: Any | None = model

    def reflection_question_node(state: MarketingState) -> dict[str, Any]:
        nonlocal active_model, active_repair_model
        draft = _final_content(state)
        if draft in (None, "", {}, []):
            return {
                "reflection_question_status": "blocked",
                "reflection_status": "blocked",
                "verification_summary": "No final content is available for Reflection.",
            }
        if active_model is None:
            active_model = load_reflection_question_model()
        messages = [
            ("system", QUESTION_PROMPT_PATH.read_text(encoding="utf-8")),
            (
                "human",
                json.dumps(
                    _question_context(state, draft),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            ),
        ]
        if active_repair_model is None:
            active_repair_model = load_json_repair_model()
        try:
            raw = _invoke_reflection_json(
                active_model,
                messages,
                node_name="Reflection Question Planner",
                repair_model=active_repair_model,
            )
        except Exception as exc:
            if _is_timeout_error(exc):
                return _reflection_timeout_updates(
                    node_name="Reflection Question Planner"
                )
            raise
        plan = _sanitize_question_plan(raw)
        reflection_iteration = int(state.get("reflection_iterations", 0)) + 1
        if state.get("reflection_mode") == "light":
            # If the reviewer discovers a factual assertion missed by the
            # deterministic gate, fail safe by upgrading to full CoVe.
            if not plan.get("claim_checks"):
                return _light_review_updates(
                    state,
                    plan,
                    reflection_iteration=reflection_iteration,
                )
            risk_reasons = [
                *state.get("reflection_risk_reasons", []),
                "reviewer_detected_factual_claim",
            ]
            return {
                "reflection_question_plan": plan,
                "reflection_question_status": "questions_ready",
                "reflection_mode": "full",
                "reflection_risk_level": "high",
                "reflection_risk_reasons": list(dict.fromkeys(risk_reasons)),
                "reflection_status": "verifying",
                "reflection_iterations": reflection_iteration,
            }
        return {
            "reflection_question_plan": plan,
            "reflection_question_status": "questions_ready",
            "reflection_status": "verifying",
            "reflection_iterations": reflection_iteration,
        }

    return reflection_question_node


def route_after_reflection_questions(state: MarketingState) -> str:
    if state.get("reflection_question_status") == "questions_ready":
        return "reflection_verification"
    if state.get("reflection_status") == "revision_required":
        return "content_executor"
    return "save"


def _verification_context(state: MarketingState) -> dict[str, Any]:
    research = _research_evidence(state)
    rag_ids = _retrieved_rag_ids(state)
    rag_evidence = [
        chunk
        for chunk in get_chunks_by_ids(rag_ids)
        if chunk.get("approved_for_external_use") is True
    ]
    public_rag_ids = [str(chunk["chunk_id"]) for chunk in rag_evidence]
    return {
        "original_request": state.get("raw_user_request")
        or state.get("original_query")
        or state.get("topic", ""),
        "content_intent": state.get("content_intent", {}),
        # Deliberately pass atomic excerpts/questions, not the complete draft,
        # to reduce anchoring on the writer's conclusion.
        "reflection_question_plan": state.get("reflection_question_plan", {}),
        "research_evidence": research,
        "rag_evidence": rag_evidence,
        "evidence_rules": {
            "research_ids_are_allowlisted": sorted(_allowed_evidence_ids(research)),
            "rag_ids_are_allowlisted": public_rag_ids,
            "missing_evidence_means_insufficient": True,
            "internal_rag_is_not_public_evidence": True,
        },
        "reflection_round": int(state.get("reflection_iterations", 1)),
    }


def _sanitize_verification(
    raw: dict[str, Any], state: MarketingState
) -> dict[str, Any]:
    question_plan = state.get("reflection_question_plan", {})
    claims = {
        str(item.get("claim_id", "")): item
        for item in question_plan.get("claim_checks", [])
        if isinstance(item, dict) and item.get("claim_id")
    }
    issues = {
        str(item.get("issue_id", "")): item
        for item in question_plan.get("quality_checks", [])
        if isinstance(item, dict) and item.get("issue_id")
    }
    research = _research_evidence(state)
    allowed_evidence = _allowed_evidence_ids(research)
    rag_evidence = get_chunks_by_ids(_retrieved_rag_ids(state))
    # The reviewed deliverable is public-facing. Internal-only audience
    # strategy can inform writing, but it cannot prove a public claim.
    allowed_rag = {
        str(chunk.get("chunk_id", ""))
        for chunk in rag_evidence
        if chunk.get("approved_for_external_use") is True
        and chunk.get("chunk_id")
    }

    raw_claim_items = raw.get("claim_results", [])
    if not isinstance(raw_claim_items, list):
        raw_claim_items = []
    raw_claim_results = {
        str(item.get("claim_id", "")): item
        for item in raw_claim_items
        if isinstance(item, dict) and item.get("claim_id") in claims
    }
    claim_results: list[dict[str, Any]] = []
    for claim_id, claim in claims.items():
        item = raw_claim_results.get(claim_id, {})
        verdict = str(item.get("verdict", "")).strip().casefold()
        if verdict not in CLAIM_VERDICTS:
            verdict = "insufficient_evidence"
        evidence_ids = [
            value
            for value in _strings(item.get("evidence_ids"))
            if value in allowed_evidence
        ]
        rag_chunk_ids = [
            value
            for value in _strings(item.get("rag_chunk_ids"))
            if value in allowed_rag
        ]
        # A factual claim cannot be supported by an uncited model answer.
        if verdict == "supported" and not evidence_ids and not rag_chunk_ids:
            verdict = "insufficient_evidence"
        claim_results.append(
            {
                "claim_id": claim_id,
                "verdict": verdict,
                "answer": str(item.get("answer", "")).strip(),
                "evidence_ids": evidence_ids,
                "rag_chunk_ids": rag_chunk_ids,
                "replacement_guidance": str(
                    item.get("replacement_guidance", "")
                ).strip(),
                "risk_level": claim.get("risk_level", "medium"),
            }
        )

    raw_quality_items = raw.get("quality_results", [])
    if not isinstance(raw_quality_items, list):
        raw_quality_items = []
    raw_quality_results = {
        str(item.get("issue_id", "")): item
        for item in raw_quality_items
        if isinstance(item, dict) and item.get("issue_id") in issues
    }
    quality_results: list[dict[str, Any]] = []
    for issue_id in issues:
        item = raw_quality_results.get(issue_id, {})
        verdict = str(item.get("verdict", "")).strip().casefold()
        if verdict not in QUALITY_VERDICTS:
            verdict = "confirmed"
        quality_results.append(
            {
                "issue_id": issue_id,
                "verdict": verdict,
                "explanation": str(item.get("explanation", "")).strip(),
            }
        )

    required_targets = {
        item["claim_id"]
        for item in claim_results
        if item["verdict"] != "supported"
    } | {
        item["issue_id"]
        for item in quality_results
        if item["verdict"] == "confirmed"
    }
    steps: list[dict[str, Any]] = []
    covered_targets: set[str] = set()
    raw_revision_steps = raw.get("revision_steps", [])
    if not isinstance(raw_revision_steps, list):
        raw_revision_steps = []
    for index, item in enumerate(
        raw_revision_steps[:MAX_REVISION_STEPS], start=1
    ):
        if not isinstance(item, dict):
            continue
        targets = [
            target
            for target in _strings(item.get("target_ids"))
            if target in required_targets
        ]
        instruction = str(item.get("instruction", "")).strip()
        if not targets or not instruction:
            continue
        action = str(item.get("action", "")).strip().casefold()
        steps.append(
            {
                "step_id": f"revision-{len(steps) + 1:03d}",
                "target_ids": targets,
                "action": action if action in REVISION_ACTIONS else "replace",
                "instruction": instruction,
                "allowed_evidence_ids": [
                    value
                    for value in _strings(item.get("allowed_evidence_ids"))
                    if value in allowed_evidence
                ],
                "allowed_rag_chunk_ids": [
                    value
                    for value in _strings(item.get("allowed_rag_chunk_ids"))
                    if value in allowed_rag
                ],
                "expected_output": "revised_content",
            }
        )
        covered_targets.update(targets)

    for target in sorted(required_targets.difference(covered_targets)):
        if len(steps) >= MAX_REVISION_STEPS:
            break
        if target in claims:
            result = next(
                item for item in claim_results if item["claim_id"] == target
            )
            guidance = result["replacement_guidance"] or (
                "Remove this claim or qualify it so it does not assert more than "
                "the approved evidence supports."
            )
            action = "replace" if result["verdict"] == "contradicted" else "qualify"
            evidence_ids = result["evidence_ids"]
            rag_ids = result["rag_chunk_ids"]
        else:
            issue = issues[target]
            guidance = str(issue.get("revision_instruction", "")).strip()
            category = issue.get("category")
            action = {
                "compliance": "compliance_fix",
                "platform_style": "style_fix",
                "brand_voice": "style_fix",
                "structure": "restructure",
                "missing_requirement": "add_missing_requirement",
                "intent_fidelity": "add_missing_requirement",
            }.get(category, "style_fix")
            evidence_ids = []
            rag_ids = []
        steps.append(
            {
                "step_id": f"revision-{len(steps) + 1:03d}",
                "target_ids": [target],
                "action": action,
                "instruction": guidance,
                "allowed_evidence_ids": evidence_ids,
                "allowed_rag_chunk_ids": rag_ids,
                "expected_output": "revised_content",
            }
        )

    max_rounds = max(
        1,
        int(
            state.get(
                "max_reflection_iterations", DEFAULT_MAX_REFLECTION_ITERATIONS
            )
        ),
    )
    status = "revision_required" if steps else "passed"
    combined_results = [*claim_results, *quality_results]
    verification_summary = str(raw.get("verification_summary", "")).strip()
    reflection_history = [
        *state.get("reflection_history", []),
        {
            "round": int(state.get("reflection_iterations", 1)),
            "question_plan": question_plan,
            "verification_results": combined_results,
            "verification_summary": verification_summary,
            "revision_steps": steps,
            "status": status,
        },
    ]
    return {
        "reflection_verification_results": combined_results,
        "reflection_history": reflection_history,
        "verification_summary": verification_summary,
        "revision_steps": steps,
        "current_revision_step_index": 0,
        "reflection_status": status,
        "executor_mode": "revision" if steps else "plan",
        "save_after_revision": bool(
            steps and int(state.get("reflection_iterations", 1)) >= max_rounds
        ),
    }


def make_reflection_verification_node(model: Any | None = None):
    """Create the Qwen node that answers checks from supplied evidence only."""

    active_model: Any | None = model
    active_repair_model: Any | None = model

    def reflection_verification_node(state: MarketingState) -> dict[str, Any]:
        nonlocal active_model, active_repair_model
        if state.get("reflection_question_status") != "questions_ready":
            return {
                "reflection_status": "blocked",
                "verification_summary": "Reflection questions are unavailable.",
            }
        if active_model is None:
            active_model = load_verification_model()
        messages = [
            ("system", VERIFICATION_PROMPT_PATH.read_text(encoding="utf-8")),
            (
                "human",
                json.dumps(
                    _verification_context(state),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            ),
        ]
        if active_repair_model is None:
            active_repair_model = load_json_repair_model()
        try:
            raw = _invoke_reflection_json(
                active_model,
                messages,
                node_name="Reflection Verification",
                repair_model=active_repair_model,
            )
        except Exception as exc:
            if _is_timeout_error(exc):
                return _reflection_timeout_updates(
                    node_name="Reflection Verification"
                )
            raise
        return _sanitize_verification(raw, state)

    return reflection_verification_node


def route_after_reflection_verification(state: MarketingState) -> str:
    if state.get("reflection_status") == "revision_required":
        return "content_executor"
    return "save"
