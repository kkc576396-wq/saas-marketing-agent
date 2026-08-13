"""Automatic translation and platform-specific query rewriting."""

from __future__ import annotations

import json
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .llm_json import parse_json_response
from .model_config import load_chat_model
from .content_intent import DELIVERABLE_TYPES, normalize_deliverable_type

from .domain_context import (
    DEFAULT_RESEARCH_COMPETITORS,
    competitor_entity_records,
    detect_intent_facets,
    known_competitor_names,
    match_known_competitors,
    validate_intent_facets,
)
from .router import AGENT_REACH_REDDIT, AGENT_REACH_RSS, AGENT_REACH_WEB, ANYSEARCH
from .state import ResearchState


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = PROJECT_ROOT / "prompts" / "query_rewriter.md"
MAX_QUERIES = 5
DEFAULT_FRESHNESS_WINDOW_DAYS = 365
TIME_SENSITIVE_TERMS = (
    "latest",
    "current",
    "recent",
    "trend",
    "trends",
    "update",
    "updates",
    "news",
    "pricing",
    "最新",
    "趋势",
    "动态",
    "更新",
    "价格",
)
HISTORICAL_TERMS = ("historical", "history", "archive", "过去", "历史", "回顾")
COMPETITOR_INTENT_TERMS = (
    "competitor",
    "competitors",
    "competitive",
    "comparison",
    "竞品",
    "竞争对手",
)

# These are fallback patterns, not an allow-list for LLM-generated queries.
# The model may use any relevant vocabulary as long as the result is a short,
# searchable Reddit phrase.
COMPETITOR_REDDIT_TEMPLATES = (
    "{competitor} pricing complaint",
    "{competitor} alternative Shopify",
    "{competitor} merchant experience",
    "{competitor} migration",
    "{competitor} expensive",
)
REDDIT_FORBIDDEN_TERMS = {
    "analyze",
    "analysis",
    "can",
    "could",
    "how",
    "please",
    "research",
    "summarize",
    "what",
    "why",
    "would",
}

TRANSLATION_FALLBACKS = (
    ("竞品最新发布的产业动态", "latest competitor industry developments and releases"),
    ("最新发布", "latest releases"),
    ("产业动态", "industry developments"),
    ("功能升级", "feature upgrades"),
    ("功能更新", "feature updates"),
    ("产品更新", "product updates"),
    ("产品发布", "product launches"),
    ("版本更新", "version updates"),
    ("关于竞品动态的讨论", "discussions about competitor updates"),
    ("竞品动态", "competitor updates"),
    ("竞争对手动态", "competitor updates"),
    ("竞争对手", "competitors"),
    ("竞品", "competitor"),
    ("讨论", "discussions"),
    ("关于", "about"),
    ("邮件营销", "email marketing"),
    ("营销自动化", "marketing automation"),
    ("客户细分", "customer segmentation"),
    ("生命周期营销", "lifecycle marketing"),
    ("客户留存", "customer retention"),
    ("最新动态", "latest developments and updates"),
    ("最新消息", "latest news and updates"),
    ("最新", "latest"),
    ("动态", "updates"),
    ("企业", "companies"),
    ("公司", "companies"),
    ("企业级", "enterprise"),
    ("定价", "pricing"),
    ("价格", "pricing"),
    ("投诉", "complaints"),
    ("抱怨", "complaints"),
    ("替代方案", "alternatives"),
    ("替代", "alternative"),
    ("商户", "merchants"),
    ("客户", "customers"),
    ("趋势", "trends"),
    ("人工智能", "AI"),
    ("产业", "industry"),
    ("发布", "releases"),
    ("升级", "upgrades"),
    ("上线", "launches"),
    ("功能", "features"),
    ("痛点", "pain points"),
    ("研究", "research"),
    ("比较", "comparison"),
)


def _has_cjk(value: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", value))


def _clean_query(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" \t\n,.;")


def requires_freshness(query: str) -> bool:
    normalized = query.casefold()
    return any(term in normalized for term in TIME_SENSITIVE_TERMS)


def parse_explicit_freshness_window_days(query: str) -> int | None:
    """Parse user-specified day/week/month/year windows deterministically."""

    normalized = _clean_query(query).casefold()
    english = re.search(
        r"(?:last|past|previous|within|recent)\s+(\d{1,4})\s*"
        r"(days?|weeks?|months?|years?)\b",
        normalized,
    )
    chinese = re.search(
        r"(?:最近|过去|近|搜索)?\s*(\d{1,4})\s*"
        r"(天|日|周|星期|个月|月|年)\s*(?:内|以内|之内)?",
        normalized,
    )
    match = english or chinese
    if match is None:
        return None
    amount = int(match.group(1))
    unit = match.group(2)
    multiplier = 1
    if unit.startswith("week") or unit in {"周", "星期"}:
        multiplier = 7
    elif unit.startswith("month") or unit in {"个月", "月"}:
        multiplier = 30
    elif unit.startswith("year") or unit == "年":
        multiplier = 365
    return max(1, min(3650, amount * multiplier))


def _parse_research_date(value: str | None) -> date:
    try:
        return date.fromisoformat(str(value or ""))
    except ValueError:
        return date.today()


def _sanitize_temporal_query(
    query: str,
    *,
    research_date: date,
    freshness_window_days: int,
    freshness_required: bool,
    allow_historical: bool,
    add_time_anchor: bool,
) -> str:
    cleaned = _clean_query(query)
    if not freshness_required or allow_historical:
        return cleaned
    cutoff = research_date - timedelta(days=freshness_window_days)

    def replace_old_year(match: re.Match[str]) -> str:
        year = int(match.group(0))
        return str(research_date.year) if year < cutoff.year else match.group(0)

    cleaned = re.sub(r"\b(?:19|20)\d{2}\b", replace_old_year, cleaned)
    if add_time_anchor and not re.search(
        r"\b(?:since|after|from)\s+(?:19|20)\d{2}-\d{2}-\d{2}\b",
        cleaned,
        re.IGNORECASE,
    ):
        cleaned = f"{cleaned} published since {cutoff.isoformat()}"
    return _clean_query(cleaned)


def _reddit_tokens(query: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", query)


def is_valid_reddit_query(value: Any) -> bool:
    """Return whether a rewritten query is a compact Reddit search phrase.

    This deliberately validates shape and safety rather than vocabulary. New
    merchant language produced by an LLM should remain usable without adding
    every possible phrase to a static template list.
    """

    query = _clean_query(value)
    if not query or _has_cjk(query) or "?" in query or len(query) > 80:
        return False
    tokens = _reddit_tokens(query)
    if not 2 <= len(tokens) <= 8:
        return False
    lowered = {token.casefold() for token in tokens}
    return not lowered.intersection(REDDIT_FORBIDDEN_TERMS)


def _sanitize_reddit_queries(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    cleaned: list[str] = []
    for value in values:
        query = _clean_query(value)
        if is_valid_reddit_query(query) and query not in cleaned:
            cleaned.append(query)
        if len(cleaned) >= MAX_QUERIES:
            break
    return cleaned


def _fallback_translate(query: str) -> str:
    translated = re.sub(r"[、，。；：！？（）【】]", " ", query)
    for source, target in TRANSLATION_FALLBACKS:
        translated = translated.replace(source, f" {target} ")
    translated = re.sub(r"[\u3400-\u9fff]+", " ", translated)
    translated = _clean_query(translated)
    return translated or "latest email marketing industry updates"


def _fallback_content_intent(query: str) -> dict[str, Any]:
    """Infer a conservative content intent when the rewriter is unavailable."""

    normalized = query.casefold()
    reddit = "reddit" in normalized
    reply = any(term in normalized for term in ("reply", "comment", "回复", "评论"))
    homepage = any(
        term in normalized
        for term in ("homepage", "landing page", "主页", "落地页", "首页")
    )
    generation = any(
        term in normalized
        for term in (
            "write",
            "generate",
            "draft",
            "article",
            "post",
            "report",
            "copy",
            "写",
            "生成",
            "撰写",
            "文章",
            "帖子",
            "报告",
            "文案",
        )
    ) or reply
    competitor_report = generation and any(
        term in normalized
        for term in ("competitor", "competitive", "竞品", "竞争对手")
    ) and any(term in normalized for term in ("article", "report", "文章", "报告"))

    if reddit and reply:
        content_type = "reddit_reply"
    elif reddit and generation:
        content_type = "reddit_promotion"
    elif homepage and generation:
        content_type = "homepage_promotion"
    elif competitor_report:
        content_type = "competitor_report"
    else:
        content_type = None

    requested = content_type is not None

    return {
        "requested": requested,
        "deliverable_type": content_type,
        "deliverable_description": query if requested else "",
        "request_evidence": query if requested else "",
        "platform": (
            "reddit"
            if content_type in {"reddit_promotion", "reddit_reply"}
            else "website"
            if content_type == "homepage_promotion"
            else "report"
            if content_type == "competitor_report"
            else ""
        ),
        "language": "English" if requested else "",
        "audience": "",
        "tone": [],
        "constraints": [],
        "requires_post_selection": content_type == "reddit_reply",
        "requires_brand_rag": requested,
        "requires_content_generation": requested,
    }


def _string_list(value: Any, *, limit: int = 10) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(
        dict.fromkeys(
            cleaned
            for item in value[:limit]
            if (cleaned := _clean_query(item))
        )
    )


def _sanitize_content_intent(value: Any, original: str) -> dict[str, Any]:
    fallback = _fallback_content_intent(original)
    if not isinstance(value, dict):
        return fallback
    content_type = normalize_deliverable_type(
        value.get("deliverable_type", value.get("type"))
    )
    requested = bool(value.get("requested", content_type is not None))
    if not requested:
        content_type = None
    return {
        "requested": requested,
        "deliverable_type": content_type,
        "deliverable_description": _clean_query(
            value.get("deliverable_description")
        ),
        "request_evidence": _clean_query(value.get("request_evidence")),
        "platform": (
            _clean_query(value.get("platform")) or fallback["platform"]
            if requested
            else ""
        ),
        # Public content is always generated in English. Input language only
        # affects intent recognition and Research query translation.
        "language": "English" if requested else "",
        "audience": _clean_query(value.get("audience")),
        "tone": _string_list(value.get("tone")),
        "constraints": _string_list(value.get("constraints")),
        "requires_post_selection": requested and (
            content_type == "reddit_reply"
            or bool(value.get("requires_post_selection", False))
        ),
        # Every downstream deliverable receives the shared brand/platform/
        # compliance context. This is a workflow invariant, not a semantic
        # classification delegated to the model.
        "requires_brand_rag": requested,
        "requires_content_generation": requested,
    }


def _content_intent_violations(value: Any, original: str) -> list[str]:
    """Validate structure and internal consistency without classifying words."""

    if not isinstance(value, dict):
        return ["content_intent must be a JSON object"]
    violations: list[str] = []
    if not isinstance(value.get("requested"), bool):
        violations.append("requested must be a boolean")
    requested = value.get("requested") is True
    content_type = normalize_deliverable_type(value.get("deliverable_type"))
    evidence = _clean_query(value.get("request_evidence"))
    description = _clean_query(value.get("deliverable_description"))
    platform = _clean_query(value.get("platform"))

    if requested:
        if content_type not in DELIVERABLE_TYPES:
            violations.append("requested content requires a valid deliverable_type")
        if not description:
            violations.append("requested content requires deliverable_description")
        if not evidence or evidence not in original:
            violations.append(
                "request_evidence must be an exact non-empty excerpt of the original request"
            )
    else:
        if content_type is not None:
            violations.append("unrequested content cannot have deliverable_type")
        if description or evidence or platform:
            violations.append(
                "unrequested content cannot contain deliverable, evidence, or platform fields"
            )
    return violations


def _upgrade_legacy_model_content_intent(
    value: Any, original: str
) -> Any:
    """Translate the retired type-only schema without making a new judgment."""

    if not isinstance(value, dict) or "requested" in value:
        return value
    upgraded = dict(value)
    old_type = _clean_query(value.get("type")).casefold()
    content_type = normalize_deliverable_type(old_type)
    requested = content_type is not None
    upgraded.update(
        {
            "requested": requested,
            "deliverable_type": content_type,
            "deliverable_description": original if requested else "",
            "request_evidence": original if requested else "",
        }
    )
    return upgraded


def _load_openai_rewriter():
    """Create an optional model client; deterministic fallback remains available."""

    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except Exception:
        pass
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        return load_chat_model(
            model_env="QUERY_REWRITER_MODEL",
            default_model="qwen3.7-flash",
            timeout_env="QUERY_REWRITER_TIMEOUT_SECONDS",
            default_timeout=15,
            retries_env="QUERY_REWRITER_MAX_RETRIES",
            default_retries=0,
            tokens_env="QUERY_REWRITER_MAX_TOKENS",
            default_tokens=1800,
            json_mode=True,
        )
    except Exception:
        return None


def _load_intent_repair_model():
    """Reuse the Rewriter model for one bounded structural repair call."""

    return _load_openai_rewriter()


def _repair_content_intent(
    original: str,
    previous: Any,
    violations: list[str],
) -> dict[str, Any] | None:
    """Resolve a structural contradiction without adding another graph node."""

    model = _load_intent_repair_model()
    if model is None:
        return None
    prompt = (
        "You repair one Content intent classification. Interpret the original "
        "request semantically; do not use keyword matching and do not weaken "
        "the request. `requested` says whether the user wants a downstream "
        "artifact after research. `deliverable_type` must be null or one of: "
        "homepage_promotion, reddit_promotion, reddit_reply, competitor_report. "
        "competitor_report means any reader-facing artifact that synthesizes "
        "competitor research, regardless of how the user phrases or formats it. "
        "When requested is true, request_evidence must quote an exact contiguous "
        "excerpt from the original request. When false, deliverable_type, "
        "deliverable_description, request_evidence, and platform must be null or "
        "empty. All generated content uses English. Return only the corrected "
        "content_intent JSON object."
    )
    payload = {
        "original_request": original,
        "previous_content_intent": previous,
        "validation_errors": violations,
    }
    try:
        response = model.invoke(
            [("system", prompt), ("human", json.dumps(payload, ensure_ascii=False))]
        )
        repaired = parse_json_response(response)
    except Exception:
        return None
    if isinstance(repaired, dict) and isinstance(
        repaired.get("content_intent"), dict
    ):
        repaired = repaired["content_intent"]
    return repaired if isinstance(repaired, dict) else None


def _model_rewrite(query: str) -> dict[str, Any] | None:
    model = _load_openai_rewriter()
    if model is None:
        return None
    try:
        today = date.today()
        cutoff = today - timedelta(days=DEFAULT_FRESHNESS_WINDOW_DAYS)
        prompt = PROMPT_PATH.read_text(encoding="utf-8") + (
            "\n\nCanonical known competitor registry: "
            f"{', '.join(known_competitor_names())}."
            f"\n\nRuntime date: {today.isoformat()}. For latest, trend, update, "
            f"news, and pricing research, prioritize material published since "
            f"{cutoff.isoformat()} and do not target older years unless the user "
            "explicitly requests historical research."
        )
        response = model.invoke([("system", prompt), ("human", query)])
        return parse_json_response(response)
    except Exception:
        return None


def _fallback_plan(translated_query: str, *, current_year: int | None = None) -> dict[str, Any]:
    current_year = current_year or date.today().year
    normalized = translated_query.casefold()
    competitors = match_known_competitors(translated_query)
    fallback_competitors = competitors or list(DEFAULT_RESEARCH_COMPETITORS)
    competitor_text = " ".join(competitors) or "email marketing companies"
    anysearch = [
        translated_query,
        f"{translated_query} official announcements product updates {current_year}",
        f"{competitor_text} latest pricing AI automation and market trends",
    ]
    reddit = [
        template.format(competitor=competitor)
        for competitor in fallback_competitors
        for template in COMPETITOR_REDDIT_TEMPLATES
    ]
    if not competitors:
        reddit.extend(
            [
                "email marketing merchant experience",
                "email tool complaint Shopify",
                "AI email automation problem",
                "customer retention tool review",
                "email platform alternative",
            ]
        )
    web = [
        f"{translated_query} official company updates",
        f"{competitor_text} product updates blog",
    ]
    rss = [
        "email marketing industry AI automation trends",
        "customer retention lifecycle marketing updates",
    ]
    return {
        "research_objective": translated_query,
        "translated_query": translated_query,
        "detected_entities": competitor_entity_records(translated_query),
        "intent_facets": detect_intent_facets(translated_query),
        "source_queries": {
            ANYSEARCH: anysearch[:MAX_QUERIES],
            AGENT_REACH_REDDIT: _sanitize_reddit_queries(reddit),
            AGENT_REACH_WEB: web[:MAX_QUERIES],
            AGENT_REACH_RSS: rss[:MAX_QUERIES],
        },
        "hyde_terms": [term for term in ("email marketing", "automation", "AI", "pricing", "retention") if term in normalized or term == "email marketing"],
        "reasoning": "Deterministic fallback generated platform-specific English queries.",
    }


def rewrite_query(
    query: str,
    *,
    research_date: str | None = None,
    freshness_window_days: int = DEFAULT_FRESHNESS_WINDOW_DAYS,
    freshness_required: bool | None = None,
) -> dict[str, Any]:
    """Translate and rewrite one user query, using an LLM when configured."""

    original = _clean_query(query)
    if not original:
        raise ValueError("query must not be empty")
    effective_date = _parse_research_date(research_date)
    explicit_freshness_window = parse_explicit_freshness_window_days(original)
    freshness_window_days = (
        explicit_freshness_window
        if explicit_freshness_window is not None
        else max(1, int(freshness_window_days))
    )
    freshness_required = bool(
        explicit_freshness_window is not None
        or (
            requires_freshness(original)
            if freshness_required is None
            else freshness_required
        )
    )
    explicit_year = bool(re.search(r"\b(?:19|20)\d{2}\b", original))
    allow_historical = explicit_year or any(
        term in original.casefold() for term in HISTORICAL_TERMS
    )
    if allow_historical and explicit_freshness_window is None:
        freshness_required = False
    translated = original if not _has_cjk(original) else _fallback_translate(original)
    # Intent parsing must also work for English requests, so use the configured
    # rewriter whenever credentials are available. The loader still returns
    # ``None`` for fully offline runs.
    plan = _model_rewrite(original)
    model_plan_available = bool(plan)
    if not plan:
        plan = _fallback_plan(translated, current_year=effective_date.year)
    else:
        raw_content_intent = _upgrade_legacy_model_content_intent(
            plan.get("content_intent"), original
        )
        plan["content_intent"] = raw_content_intent
        intent_violations = _content_intent_violations(
            raw_content_intent, original
        )
        if intent_violations:
            repaired_intent = _repair_content_intent(
                original, raw_content_intent, intent_violations
            )
            repaired_violations = _content_intent_violations(
                repaired_intent, original
            )
            if repaired_violations:
                raise ValueError(
                    "Content intent could not be resolved consistently: "
                    + "; ".join(repaired_violations)
                )
            plan["content_intent"] = repaired_intent

    translated_query = _clean_query(plan.get("translated_query")) or translated
    research_objective = (
        _clean_query(plan.get("research_objective")) or translated_query
    )
    fallback = _fallback_plan(translated_query, current_year=effective_date.year)
    source_queries: dict[str, list[str]] = {}
    for channel in (ANYSEARCH, AGENT_REACH_REDDIT, AGENT_REACH_WEB, AGENT_REACH_RSS):
        values = plan.get("source_queries", {}).get(channel, []) if isinstance(plan.get("source_queries"), dict) else []
        if channel == AGENT_REACH_REDDIT:
            cleaned = _sanitize_reddit_queries(values)
        else:
            cleaned = list(
                dict.fromkeys(
                    _sanitize_temporal_query(
                        value,
                        research_date=effective_date,
                        freshness_window_days=freshness_window_days,
                        freshness_required=bool(freshness_required),
                        allow_historical=allow_historical,
                        add_time_anchor=channel in {ANYSEARCH, AGENT_REACH_WEB, AGENT_REACH_RSS},
                    )
                    for value in values
                    if _clean_query(value)
                )
            )[:MAX_QUERIES]
        for fallback_query in fallback["source_queries"][channel]:
            if len(cleaned) >= MAX_QUERIES:
                break
            sanitized_fallback = (
                fallback_query
                if channel == AGENT_REACH_REDDIT
                else _sanitize_temporal_query(
                    fallback_query,
                    research_date=effective_date,
                    freshness_window_days=freshness_window_days,
                    freshness_required=bool(freshness_required),
                    allow_historical=allow_historical,
                    add_time_anchor=channel in {ANYSEARCH, AGENT_REACH_WEB, AGENT_REACH_RSS},
                )
            )
            if sanitized_fallback not in cleaned:
                cleaned.append(sanitized_fallback)
        source_queries[channel] = cleaned[:MAX_QUERIES]

    combined_intent = f"{original} {translated_query}".casefold()
    # Entity identity is always derived from the deterministic SmartPush
    # registry. The model may propose entity records, but cannot invent a
    # product-category relationship that changes routing or verification.
    detected_entities = competitor_entity_records(combined_intent)
    model_facets = validate_intent_facets(plan.get("intent_facets", []))
    fallback_facets = detect_intent_facets(combined_intent)
    intent_facets = list(dict.fromkeys(model_facets + fallback_facets))
    if "product_update_research" in intent_facets:
        benchmark_competitors = list(DEFAULT_RESEARCH_COMPETITORS)
        update_queries = [
            f"{competitor} latest product releases and feature updates {effective_date.year}"
            for competitor in benchmark_competitors
        ] + [
            (
                "email marketing automation competitor product launches and "
                f"industry developments {effective_date.year}"
            ),
            (
                "ecommerce marketing platform AI feature upgrades and release "
                f"notes {effective_date.year}"
            ),
        ]
        web_queries = [
            f"{competitor} official release notes and product updates"
            for competitor in benchmark_competitors
        ] + [
            "email marketing automation official product changelogs"
        ]
        rss_queries = [
            f"{competitor} official product update feed"
            for competitor in benchmark_competitors
        ] + [
            "email marketing automation product launch news"
        ]
        source_queries[ANYSEARCH] = list(
            dict.fromkeys(update_queries + source_queries[ANYSEARCH])
        )[:MAX_QUERIES]
        source_queries[AGENT_REACH_WEB] = list(
            dict.fromkeys(web_queries + source_queries[AGENT_REACH_WEB])
        )[:MAX_QUERIES]
        source_queries[AGENT_REACH_RSS] = list(
            dict.fromkeys(rss_queries + source_queries[AGENT_REACH_RSS])
        )[:MAX_QUERIES]
    if (
        any(term in combined_intent for term in COMPETITOR_INTENT_TERMS)
        and "product_update_research" not in intent_facets
    ):
        cutoff = effective_date - timedelta(days=freshness_window_days)
        benchmark_competitors = " ".join(DEFAULT_RESEARCH_COMPETITORS)
        competitor_queries = [
            (
                f"{benchmark_competitors} competitor product pricing AI updates "
                f"{effective_date.year} published since {cutoff.isoformat()}"
            ),
            (
                f"{' vs '.join(DEFAULT_RESEARCH_COMPETITORS)} "
                "email automation segmentation "
                f"market positioning {effective_date.year} published since {cutoff.isoformat()}"
            ),
        ]
        source_queries[ANYSEARCH] = list(
            dict.fromkeys(competitor_queries + source_queries[ANYSEARCH])
        )[:MAX_QUERIES]

    return {
        "original_query": original,
        "raw_user_request": original,
        "research_objective": research_objective,
        "translated_query": translated_query,
        "detected_entities": detected_entities,
        "intent_facets": intent_facets,
        "source_queries": source_queries,
        "hyde_terms": list(plan.get("hyde_terms", []))[:MAX_QUERIES],
        "query_reasoning": _clean_query(plan.get("reasoning")) or "Platform-specific query rewriting completed.",
        "research_date": effective_date.isoformat(),
        "freshness_window_days": freshness_window_days,
        "freshness_required": bool(freshness_required),
        "freshness_window_explicit": explicit_freshness_window is not None,
        "content_intent": _sanitize_content_intent(
            plan.get("content_intent") if model_plan_available else None,
            original,
        ),
    }


def query_rewriter_node(state: ResearchState) -> dict[str, Any]:
    """Create platform-specific English queries before source execution."""

    original = str(state.get("original_query") or state.get("topic") or "")
    plan = rewrite_query(
        original,
        research_date=state.get("research_date"),
        freshness_window_days=int(
            state.get("freshness_window_days", DEFAULT_FRESHNESS_WINDOW_DAYS)
        ),
        freshness_required=state.get("freshness_required"),
    )
    return {
        # Preserve the full request while replacing ``topic`` with the pure
        # research objective used by routing, retrieval, and verification.
        "raw_user_request": plan["raw_user_request"],
        "topic": plan["research_objective"],
        "original_query": plan["original_query"],
        "research_objective": plan["research_objective"],
        "translated_query": plan["translated_query"],
        "detected_entities": plan["detected_entities"],
        "intent_facets": plan["intent_facets"],
        "source_queries": plan["source_queries"],
        "hyde_terms": plan["hyde_terms"],
        "query_reasoning": plan["query_reasoning"],
        "research_date": plan["research_date"],
        "freshness_window_days": plan["freshness_window_days"],
        "freshness_required": plan["freshness_required"],
        "freshness_window_explicit": plan["freshness_window_explicit"],
        "content_intent": plan["content_intent"],
        "requires_content_generation": bool(
            plan["content_intent"].get("requires_content_generation", False)
        ),
        # Keep this field populated for backwards compatibility and for
        # AnySearch callers that inspect the planner output.
        "search_queries": plan["source_queries"][ANYSEARCH],
    }
