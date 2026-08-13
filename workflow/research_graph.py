"""LangGraph workflow for AnySearch-backed SaaS marketing research."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from langgraph.graph import END, START, StateGraph

from .state import ResearchState
from .domain_context import all_competitor_aliases, match_known_competitors
from .entity_relevance import analyze_entity_importance, entity_roles
from .router import source_router_node
from .scoring import (
    calculate_freshness_score,
    detect_product_update_evidence,
    scoring_node,
)
from .opportunity_classifier import opportunity_classifier_node
from .output_contract import build_output_contract
from .normalizer import deduplicate_documents
from .semantic_dedup import semantic_deduplicate_documents
from .query_rewriter import query_rewriter_node
from .query_rewriter import DEFAULT_FRESHNESS_WINDOW_DAYS, requires_freshness
from .research_agent import (
    make_research_agent_node,
    research_tools_node,
    route_after_research_agent,
    route_after_research_tools,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "research_output.json"
VERIFIER_PRODUCT_TERMS = (
    "email marketing",
    "email automation",
    "marketing automation",
    "customer segmentation",
    "segmentation",
    "lifecycle marketing",
    "retention marketing",
    "email flow",
    "deliverability",
    "abandoned cart",
    "ecommerce crm",
    "e-commerce crm",
)
VERIFIER_TARGET_TERMS = (
    "ecommerce",
    "e-commerce",
    "shopify",
    "dtc",
    "merchant",
    "online retailer",
    "saas",
)
VERIFIER_COMPETITOR_TERMS = all_competitor_aliases()
VERIFIER_COMPETITIVE_MARKET_TERMS = (
    "competitive landscape",
    "market share",
    "platform comparison",
    "software comparison",
    "vendor comparison",
    "email service providers",
    "email marketing platforms",
)
VERIFIER_BROAD_CLAIM_TERMS = (
    "trend",
    "market",
    "industry",
    "report",
    "state of",
    "growth",
    "adoption",
    "benchmark",
)
VERIFIER_PROMOTIONAL_TERMS = (
    "i built",
    "my product",
    "our product",
    "free beta",
    "sign up",
    "book a demo",
    "affiliate",
    "promo code",
)
COMMUNITY_OBSERVATION_USAGE_CONSTRAINTS = (
    "Present as an individual community observation, not a market-wide fact.",
    "Use scoped wording such as 'a merchant reported' or 'a user mentioned'.",
    "Do not infer prevalence unless multiple independent user reports are available.",
)
GENERIC_TOPIC_TERMS = {
    "about",
    "analysis",
    "and",
    "companies",
    "company",
    "competitor",
    "competitors",
    "discussion",
    "discussions",
    "email",
    "for",
    "latest",
    "marketing",
    "market",
    "research",
    "the",
    "trend",
    "trends",
}


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
    research_date = state.get("research_date") or date.today().isoformat()
    freshness_window_days = max(
        1,
        int(state.get("freshness_window_days", DEFAULT_FRESHNESS_WINDOW_DAYS)),
    )
    freshness_required = state.get("freshness_required")
    if freshness_required is None:
        freshness_required = requires_freshness(topic)

    return {
        "topic": topic,
        "research_date": research_date,
        "freshness_window_days": freshness_window_days,
        "freshness_required": bool(freshness_required),
        "original_query": state.get("original_query") or topic,
        "translated_query": state.get("translated_query", ""),
        "search_queries": search_queries,
        "source_queries": dict(state.get("source_queries", {})),
        "detected_entities": list(state.get("detected_entities", [])),
        "intent_facets": list(state.get("intent_facets", [])),
        "query_reasoning": state.get("query_reasoning", ""),
        "hyde_terms": list(state.get("hyde_terms", [])),
        "query_history": list(state.get("query_history", [])),
        "messages": list(state.get("messages", [])),
        "research_agent_status": state.get("research_agent_status", "pending"),
        "research_agent_reasoning": state.get("research_agent_reasoning", ""),
        "research_tool_history": list(state.get("research_tool_history", [])),
        "documents": list(state.get("documents", [])),
        "search_iterations": int(state.get("search_iterations", 0)),
        "max_iterations": max_iterations,
        "recommended_sources": list(state.get("recommended_sources", [])),
        "source_plan_reasoning": state.get("source_plan_reasoning", ""),
        "selected_sources": list(state.get("selected_sources", [])),
        "source_reasoning": state.get("source_reasoning", ""),
        "tool_results": list(state.get("tool_results", [])),
        "insights": list(state.get("insights", [])),
        "candidate_insights": list(state.get("candidate_insights", [])),
        "verification_results": list(state.get("verification_results", [])),
        "insight_scores": list(state.get("insight_scores", [])),
        "opportunity_types": list(state.get("opportunity_types", [])),
        "recommended_channels": list(state.get("recommended_channels", [])),
        "eligible_insights": list(state.get("eligible_insights", [])),
        "alternative_insights": list(state.get("alternative_insights", [])),
        "rejected_insights": list(state.get("rejected_insights", [])),
        "output_file": output_file,
    }


def planning_node(state: ResearchState) -> dict[str, Any]:
    """Build the complete initial research plan in one graph transition."""

    working: dict[str, Any] = dict(state)
    updates: dict[str, Any] = {}
    for stage in (planner_node, query_rewriter_node, source_router_node):
        stage_updates = stage(working)
        working.update(stage_updates)
        updates.update(stage_updates)
    # The rules-based router now seeds the LLM with advice; only calls made by
    # the Research Agent populate ``selected_sources``.
    updates["recommended_sources"] = list(updates.get("selected_sources", []))
    updates["source_plan_reasoning"] = str(updates.get("source_reasoning", ""))
    updates["selected_sources"] = []
    updates["source_reasoning"] = ""
    combined = {**state, **working, **updates}
    intent = combined.get("content_intent", {})
    is_reddit = (
        isinstance(intent, dict)
        and str(intent.get("platform", "")).casefold() == "reddit"
    ) or "reddit" in str(
        combined.get("raw_user_request")
        or combined.get("original_query")
        or combined.get("topic", "")
    ).casefold()
    if is_reddit:
        updates["max_iterations"] = min(
            2, max(1, int(updates.get("max_iterations", 2)))
        )
    return updates


def _insight_id(insight: Any) -> str:
    if isinstance(insight, dict):
        return str(insight.get("insight_id") or insight.get("document_id") or insight.get("title") or "")
    return str(insight)


def _insight_text(insight: Any) -> str:
    if isinstance(insight, dict):
        return " ".join(
            str(insight.get(field, ""))
            for field in ("title", "summary")
            if insight.get(field)
        )
    return str(insight)


def analyzer_node(state: ResearchState) -> dict[str, Any]:
    """Create one structured insight candidate per normalized document."""

    candidates: list[dict[str, Any]] = []
    unique_documents = deduplicate_documents(list(state.get("documents", [])))
    for document in semantic_deduplicate_documents(unique_documents):
        document_id = str(document.get("document_id", "")).strip()
        title = str(document.get("title", "")).strip()
        summary = str(document.get("summary", "")).strip()
        url = document.get("url")
        if not document_id or not title or not summary or not url:
            continue
        candidates.append(
            {
                "insight_id": document_id,
                "title": title,
                "summary": summary,
                "source_type": document.get("source_type", "unknown"),
                "source_document_id": document_id,
                "source_bias": document.get("source_bias", "unknown"),
                "date_status": document.get("date_status", "unknown"),
                "entity_mentions": analyze_entity_importance(title, summary),
                "duplicate_count": int(document.get("duplicate_count", 1)),
                "duplicate_document_ids": list(
                    document.get("duplicate_document_ids", [])
                ),
                "duplicate_sources": list(document.get("duplicate_sources", [])),
                "sources": [
                    {
                        "title": title,
                        "url": url,
                        "published_at": document.get("published_at"),
                        "retrieved_at": document.get("retrieved_at"),
                        "date_status": document.get("date_status", "unknown"),
                        "date_confidence": document.get("date_confidence", 0.0),
                        "page_type": document.get("page_type", "article"),
                        "source_bias": document.get("source_bias", "unknown"),
                    }
                ],
            }
        )

    return {"insights": candidates, "candidate_insights": candidates}


def _topic_terms(topic: str) -> set[str]:
    """Return meaningful terms used for the baseline relevance check."""

    return {
        term.strip(".,:;!?()[]{}\"'").lower()
        for term in topic.split()
        if len(term.strip(".,:;!?()[]{}\"'")) >= 3
        and term.strip(".,:;!?()[]{}\"'").lower() not in GENERIC_TOPIC_TERMS
    }


def _phrase_matches(text: str, phrases: tuple[str, ...]) -> list[str]:
    normalized = text.casefold()
    return [phrase for phrase in phrases if phrase in normalized]


def _valid_source_url(value: Any) -> bool:
    return str(value or "").startswith(("http://", "https://"))


def _parse_published_date(value: Any) -> date | None:
    cleaned = str(value or "").strip()
    if not cleaned:
        return None
    try:
        return datetime.fromisoformat(cleaned.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(cleaned[:10])
        except ValueError:
            return None


def _within_requested_freshness_window(
    insight: Any,
    *,
    research_date: str | None,
    window_days: int,
) -> tuple[bool, int | None, str | None]:
    """Apply the user's exact date window instead of a broad score band."""

    if not isinstance(insight, dict):
        return False, None, None
    effective_date = _parse_published_date(research_date) or date.today()
    published_dates = [
        parsed
        for source in insight.get("sources", [])
        if isinstance(source, dict)
        and (parsed := _parse_published_date(source.get("published_at")))
    ]
    if not published_dates:
        return False, None, None
    newest = max(published_dates)
    age_days = max(0, (effective_date - newest).days)
    cutoff = effective_date - timedelta(days=max(1, int(window_days)))
    return newest >= cutoff, age_days, newest.isoformat()


def _source_domain(value: Any) -> str:
    return urlsplit(str(value or "")).netloc.casefold()


def _supporting_sources(
    insight: dict[str, Any],
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Find bounded independent corroboration for broad trend claims."""

    insight_text = _insight_text(insight).casefold()
    signals = _phrase_matches(insight_text, VERIFIER_COMPETITOR_TERMS)
    if not signals:
        signals = _phrase_matches(insight_text, VERIFIER_PRODUCT_TERMS)
    original_urls = {
        str(source.get("url"))
        for source in insight.get("sources", [])
        if isinstance(source, dict) and source.get("url")
    }
    supporting: list[dict[str, Any]] = []
    seen_domains = {_source_domain(url) for url in original_urls}
    for document in documents:
        url = str(document.get("url") or "")
        domain = _source_domain(url)
        if not url or url in original_urls or not domain or domain in seen_domains:
            continue
        if document.get("source_bias") == "vendor_source":
            continue
        document_text = f"{document.get('title', '')} {document.get('summary', '')}".casefold()
        if signals and not any(signal in document_text for signal in signals):
            continue
        seen_domains.add(domain)
        supporting.append(
            {
                "title": str(document.get("title", "")),
                "url": url,
                "published_at": document.get("published_at"),
            }
        )
        if len(supporting) >= 3:
            break
    return supporting


def verifier_node(state: ResearchState) -> dict[str, Any]:
    """Apply relevance, evidence, freshness, bias, and corroboration gates."""

    topic = state.get("translated_query") or state.get("topic", "")
    topic_terms = _topic_terms(topic)
    product_required = bool(_phrase_matches(topic, VERIFIER_PRODUCT_TERMS))
    competitor_required = "competitor" in topic.casefold() or bool(
        match_known_competitors(topic)
    )
    freshness_required = bool(state.get("freshness_required", False))
    freshness_window_days = max(
        1, int(state.get("freshness_window_days", DEFAULT_FRESHNESS_WINDOW_DAYS))
    )
    product_update_required = "product_update_research" in state.get(
        "intent_facets", []
    )
    documents = state.get("documents", [])
    verification_results: list[dict[str, Any]] = []
    verified_insights: list[Any] = []

    candidate_insights = state.get("candidate_insights", state.get("insights", []))
    for insight in candidate_insights:
        insight_id = _insight_id(insight)
        insight_text = _insight_text(insight)
        insight_terms = _topic_terms(insight_text)
        relevance_terms = sorted(topic_terms.intersection(insight_terms))
        product_matches = _phrase_matches(insight_text, VERIFIER_PRODUCT_TERMS)
        roles = entity_roles(insight)
        primary_competitors = roles["primary"]
        supporting_competitors = roles["supporting"]
        incidental_competitors = roles["incidental"]
        competitor_matches = list(
            dict.fromkeys(
                primary_competitors
                + supporting_competitors
                + incidental_competitors
            )
        )
        product_relevant = bool(product_matches or primary_competitors)
        if product_matches:
            product_relevance_basis = "explicit_product_term"
        elif primary_competitors:
            # Preserve the established output value; role detail is exposed in
            # the dedicated primary/supporting/incidental fields below.
            product_relevance_basis = "known_competitor_entity"
        else:
            product_relevance_basis = "none"
        competitive_market_matches = _phrase_matches(
            insight_text, VERIFIER_COMPETITIVE_MARKET_TERMS
        )
        competitor_evidence = bool(
            primary_competitors
            or supporting_competitors
            or competitive_market_matches
        )
        competitor_relevance_points = (
            35.0
            if primary_competitors
            else 15.0
            if supporting_competitors
            else 5.0
            if incidental_competitors
            else 0.0
        )
        target_matches = _phrase_matches(insight_text, VERIFIER_TARGET_TERMS)
        relevance_score = min(
            100.0,
            (40.0 if product_relevant else 0.0)
            + competitor_relevance_points
            + (20.0 if target_matches else 0.0)
            + min(15.0, 5.0 * len(relevance_terms)),
        )

        if isinstance(insight, dict):
            sources = insight.get("sources", [])
            valid_urls = [
                source.get("url")
                for source in sources
                if isinstance(source, dict) and _valid_source_url(source.get("url"))
            ]
            summary = str(insight.get("summary", "")).strip()
            title = str(insight.get("title", "")).strip()
            title_support_terms = _topic_terms(title).intersection(_topic_terms(summary))
            claim_supported = bool(
                title_support_terms
                or primary_competitors
                or supporting_competitors
                or product_matches
            )
            source_bias = str(insight.get("source_bias") or "unknown")
            source_quality_score = {
                "independent_source": 85.0,
                "community_source": 75.0,
                "vendor_source": 65.0,
            }.get(source_bias, 50.0)
            evidence_score = (
                (25.0 if insight.get("source_document_id") else 0.0)
                + (25.0 if valid_urls else 0.0)
                + (25.0 if len(summary) >= 80 else 0.0)
                + (15.0 if claim_supported else 0.0)
                + 0.1 * source_quality_score
            )
            promotional_matches = _phrase_matches(
                f"{title} {summary}", VERIFIER_PROMOTIONAL_TERMS
            )
            promotional_content = len(set(promotional_matches)) >= 2
            freshness_score, freshness_signals = calculate_freshness_score(
                insight,
                research_date=state.get("research_date"),
            )
            broad_claim = bool(
                _phrase_matches(f"{title} {summary}", VERIFIER_BROAD_CLAIM_TERMS)
            )
            community_observation = source_bias == "community_source"
            if community_observation:
                claim_type = "community_observation"
                claim_scope = "single_user_experience"
                cross_domain_required = False
                usage_constraints = list(COMMUNITY_OBSERVATION_USAGE_CONSTRAINTS)
            elif broad_claim:
                claim_type = "market_or_industry_claim"
                claim_scope = "market_level"
                cross_domain_required = True
                usage_constraints = []
            else:
                claim_type = "source_based_finding"
                claim_scope = "source_level"
                cross_domain_required = False
                usage_constraints = []
            supporting_sources = (
                _supporting_sources(insight, documents)
                if cross_domain_required
                else []
            )
            original_domains = {
                _source_domain(url) for url in valid_urls if _source_domain(url)
            }
            supporting_domains = {
                _source_domain(source.get("url"))
                for source in supporting_sources
                if _source_domain(source.get("url"))
            }
            independent_corroboration = bool(supporting_domains)
        else:
            evidence_score = 75.0 if bool(documents) and any(
                str(document.get("raw_content") or document.get("content", "")).strip()
                for document in documents
            ) else 0.0
            claim_supported = evidence_score > 0
            source_bias = "unknown"
            source_quality_score = 50.0
            promotional_matches = []
            promotional_content = False
            freshness_score, freshness_signals = calculate_freshness_score(
                insight,
                research_date=state.get("research_date"),
            )
            broad_claim = False
            community_observation = False
            claim_type = "source_based_finding"
            claim_scope = "source_level"
            cross_domain_required = False
            usage_constraints = []
            supporting_sources = []
            original_domains = set()
            supporting_domains = set()
            independent_corroboration = False

        product_update_evidence = (
            detect_product_update_evidence(insight)
            if product_update_required
            else {
                "passed": True,
                "rejection_reasons": [],
            }
        )
        within_window, source_age_days, newest_published_at = (
            _within_requested_freshness_window(
                insight,
                research_date=state.get("research_date"),
                window_days=freshness_window_days,
            )
            if freshness_required
            else (True, None, None)
        )
        rejection_reasons: list[str] = []
        if relevance_score < 60.0:
            rejection_reasons.append("low_topic_relevance")
        if product_required and not product_relevant:
            rejection_reasons.append("missing_email_marketing_relevance")
        if competitor_required and not competitor_evidence:
            rejection_reasons.append("missing_competitor_evidence")
        if evidence_score < 70.0:
            rejection_reasons.append("insufficient_evidence")
        if not claim_supported:
            rejection_reasons.append("claim_not_supported_by_source")
        if freshness_required and not within_window:
            rejection_reasons.append(
                "missing_published_at"
                if source_age_days is None
                else "source_outside_freshness_window"
            )
        if promotional_content:
            rejection_reasons.append("promotional_content")
        if product_update_required:
            rejection_reasons.extend(
                reason
                for reason in product_update_evidence.get(
                    "rejection_reasons", []
                )
                if reason not in rejection_reasons
            )
        if cross_domain_required and not independent_corroboration:
            rejection_reasons.append("insufficient_independent_sources")

        confidence = round(
            (
                0.35 * relevance_score
                + 0.30 * evidence_score
                + 0.20 * source_quality_score
                + 0.15 * freshness_score
            )
            / 100.0,
            2,
        )
        passed = not rejection_reasons and confidence >= 0.65
        verification_results.append(
            {
                "insight": insight_id,
                "title": insight.get("title") if isinstance(insight, dict) else insight_id,
                "relevant": relevance_score >= 60.0,
                "matching_terms": relevance_terms,
                "matched_product_terms": sorted(set(product_matches)),
                "matched_competitors": sorted(set(competitor_matches)),
                "primary_competitors": sorted(set(primary_competitors)),
                "supporting_competitors": sorted(set(supporting_competitors)),
                "incidental_competitors": sorted(set(incidental_competitors)),
                "entity_mentions": (
                    list(insight.get("entity_mentions", []))
                    if isinstance(insight, dict)
                    else analyze_entity_importance(str(insight), "")
                ),
                "product_relevant": product_relevant,
                "product_relevance_basis": product_relevance_basis,
                "matched_competitive_market_terms": sorted(
                    set(competitive_market_matches)
                ),
                "matched_target_terms": sorted(set(target_matches)),
                "relevance_score": relevance_score,
                "evidence_available": evidence_score >= 70.0,
                "evidence_score": round(evidence_score, 2),
                "claim_supported": claim_supported,
                "source_quality_score": source_quality_score,
                "source_bias": source_bias,
                "freshness_score": freshness_score,
                "freshness_signals": freshness_signals,
                "freshness_window_days": freshness_window_days,
                "within_requested_freshness_window": within_window,
                "source_age_days": source_age_days,
                "newest_published_at": newest_published_at,
                "broad_claim": broad_claim,
                "claim_type": claim_type,
                "claim_scope": claim_scope,
                "cross_domain_required": cross_domain_required,
                "usage_constraints": usage_constraints,
                "supporting_sources": supporting_sources,
                "independent_domain_count": len(
                    original_domains.union(supporting_domains)
                ),
                "promotional_signals": sorted(set(promotional_matches)),
                "product_update_evidence": product_update_evidence,
                "confidence": confidence,
                "passed": passed,
                "rejection_reasons": rejection_reasons,
            }
        )
        if passed:
            verified_insights.append(insight)

    return {
        "insights": verified_insights,
        "candidate_insights": list(candidate_insights),
        "verification_results": verification_results,
    }


def evaluation_node(state: ResearchState) -> dict[str, Any]:
    """Run the deterministic insight quality pipeline as one graph stage."""

    working: dict[str, Any] = dict(state)
    updates: dict[str, Any] = {}
    for stage in (
        analyzer_node,
        verifier_node,
        scoring_node,
        opportunity_classifier_node,
    ):
        stage_updates = stage(working)
        working.update(stage_updates)
        updates.update(stage_updates)
    return updates


def save_output_node(state: ResearchState) -> dict[str, Any]:
    """Persist only the ranked Top 5 and five verified alternatives."""

    output_path = Path(state.get("output_file") or DEFAULT_OUTPUT_FILE)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    contract = build_output_contract(state)
    verification_passed = sum(
        1 for item in state.get("verification_results", []) if item.get("passed")
    )
    score_ge_60 = sum(
        1
        for item in state.get("insight_scores", [])
        if float(item.get("total_score", 0.0)) >= 60.0
    )
    source_call_counts: dict[str, int] = {}
    for item in state.get("tool_results", []):
        source = str(item.get("source", "unknown"))
        source_call_counts[source] = source_call_counts.get(source, 0) + 1
    date_status_counts: dict[str, int] = {}
    source_bias_counts: dict[str, int] = {}
    for document in state.get("documents", []):
        date_status = str(document.get("date_status", "unknown"))
        source_bias = str(document.get("source_bias", "unknown"))
        date_status_counts[date_status] = date_status_counts.get(date_status, 0) + 1
        source_bias_counts[source_bias] = source_bias_counts.get(source_bias, 0) + 1
    rejection_reason_counts: dict[str, int] = {}
    for verification in state.get("verification_results", []):
        for reason in verification.get("rejection_reasons", []):
            rejection_reason_counts[reason] = rejection_reason_counts.get(reason, 0) + 1
    payload = {
        "topic": state.get("topic", ""),
        "raw_user_request": state.get("raw_user_request")
        or state.get("original_query")
        or state.get("topic", ""),
        "research_objective": state.get("research_objective")
        or state.get("translated_query")
        or state.get("topic", ""),
        "research_date": state.get("research_date", date.today().isoformat()),
        "freshness_window_days": state.get(
            "freshness_window_days", DEFAULT_FRESHNESS_WINDOW_DAYS
        ),
        "freshness_required": state.get("freshness_required", False),
        "freshness_window_explicit": state.get(
            "freshness_window_explicit", False
        ),
        "original_query": state.get("original_query") or state.get("topic", ""),
        "translated_query": state.get("translated_query", ""),
        "search_queries": state.get("search_queries", []),
        "source_queries": state.get("source_queries", {}),
        "detected_entities": state.get("detected_entities", []),
        "intent_facets": state.get("intent_facets", []),
        "content_intent": state.get("content_intent", {}),
        "requires_content_generation": state.get(
            "requires_content_generation", False
        ),
        "query_reasoning": state.get("query_reasoning", ""),
        "hyde_terms": state.get("hyde_terms", []),
        "query_history": state.get("query_history", []),
        "research_agent_status": state.get("research_agent_status", "complete"),
        "research_agent_reasoning": state.get("research_agent_reasoning", ""),
        "research_tool_history": state.get("research_tool_history", []),
        "search_iterations": state.get("search_iterations", 0),
        "max_iterations": state.get("max_iterations", 5),
        "recommended_sources": state.get("recommended_sources", []),
        "source_plan_reasoning": state.get("source_plan_reasoning", ""),
        "selected_sources": state.get("selected_sources", []),
        "source_reasoning": state.get("source_reasoning", ""),
        "retrieval_summary": {
            "source_calls": source_call_counts,
            "document_count": len(state.get("documents", [])),
            "candidate_insight_count": len(state.get("candidate_insights", [])),
            "verification_passed_count": verification_passed,
            "score_ge_60_count": score_ge_60,
            "date_status_counts": date_status_counts,
            "source_bias_counts": source_bias_counts,
            "rejection_reason_counts": rejection_reason_counts,
        },
        "verification_policy": {
            "minimum_relevance_score": 60,
            "minimum_evidence_score": 70,
            "minimum_confidence": 0.65,
            "minimum_freshness_score_when_required": 60,
            "explicit_freshness_windows_are_hard_filters": True,
            "broad_claims_require_independent_corroboration": True,
            "community_observations_require_cross_domain_corroboration": False,
            "community_observations_must_remain_scoped": True,
            "product_updates_require_explicit_release_evidence": True,
            "product_updates_require_published_at": True,
            "minimum_product_update_topic_alignment_score": 60,
        },
        # ``insights`` is the backwards-compatible alias for the downstream
        # eligible collection. Future Content Agents must consume only this
        # eligible Top 5 list, never ``rejected_insights``.
        "insights": contract["eligible_insights"],
        "eligible_insights": contract["eligible_insights"],
        "alternative_insights": contract["alternative_insights"],
        "rejected_insights": contract["rejected_insights"],
        "content_plan": state.get("content_plan", {}),
        "content_planner_status": state.get(
            "content_planner_status", "not_requested"
        ),
        "content_planner_reasoning": state.get(
            "content_planner_reasoning", ""
        ),
        "current_step_index": state.get("current_step_index", 0),
        "execution_history": state.get("execution_history", []),
        "execution_artifacts": state.get("execution_artifacts", {}),
        "executor_iterations": state.get("executor_iterations", 0),
        "max_executor_iterations": state.get("max_executor_iterations", 20),
        "executor_status": state.get("executor_status", "not_started"),
        "executor_summary": state.get("executor_summary", ""),
        "rag_tool_history": state.get("rag_tool_history", []),
        "rag_prefetch_status": state.get("rag_prefetch_status", "not_started"),
        "rag_prefetch_chunk_ids": [
            str(item.get("chunk_id", ""))
            for item in state.get("rag_prefetch_results", [])
            if isinstance(item, dict) and item.get("chunk_id")
        ],
        "rag_prefetch_errors": state.get("rag_prefetch_errors", []),
        "memory_prefetch_status": state.get(
            "memory_prefetch_status", "not_started"
        ),
        "memory_prefetch_ids": [
            str(item.get("memory_id", ""))
            for item in state.get("memory_prefetch_results", [])
            if isinstance(item, dict) and item.get("memory_id")
        ],
        "memory_prefetch_errors": state.get("memory_prefetch_errors", []),
        "memory_commit_status": state.get("memory_commit_status", "not_started"),
        "memory_commit_ids": state.get("memory_commit_ids", []),
        "memory_commit_errors": state.get("memory_commit_errors", []),
        "final_content": state.get("final_content"),
        "draft_checkpoint_status": state.get(
            "draft_checkpoint_status", "not_saved"
        ),
        "reflection_mode": state.get("reflection_mode", "not_assessed"),
        "reflection_risk_level": state.get(
            "reflection_risk_level", "not_assessed"
        ),
        "reflection_risk_reasons": state.get("reflection_risk_reasons", []),
        "reflection_question_plan": state.get("reflection_question_plan", {}),
        "reflection_question_status": state.get(
            "reflection_question_status", "not_started"
        ),
        "reflection_verification_results": state.get(
            "reflection_verification_results", []
        ),
        "reflection_history": state.get("reflection_history", []),
        "verification_summary": state.get("verification_summary", ""),
        "reflection_status": state.get("reflection_status", "not_started"),
        "reflection_iterations": state.get("reflection_iterations", 0),
        "max_reflection_iterations": state.get("max_reflection_iterations", 1),
        "revision_steps": state.get("revision_steps", []),
        "current_revision_step_index": state.get(
            "current_revision_step_index", 0
        ),
        "revision_history": state.get("revision_history", []),
        "executor_mode": state.get("executor_mode", "plan"),
        "output_file": str(output_path),
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "eligible_insights": contract["eligible_insights"],
        "alternative_insights": contract["alternative_insights"],
        "rejected_insights": contract["rejected_insights"],
        "content_plan": state.get("content_plan", {}),
        "content_planner_status": state.get(
            "content_planner_status", "not_requested"
        ),
        "content_planner_reasoning": state.get(
            "content_planner_reasoning", ""
        ),
        "current_step_index": state.get("current_step_index", 0),
        "execution_history": state.get("execution_history", []),
        "execution_artifacts": state.get("execution_artifacts", {}),
        "executor_iterations": state.get("executor_iterations", 0),
        "max_executor_iterations": state.get("max_executor_iterations", 20),
        "executor_status": state.get("executor_status", "not_started"),
        "executor_summary": state.get("executor_summary", ""),
        "rag_tool_history": state.get("rag_tool_history", []),
        "rag_prefetch_status": state.get("rag_prefetch_status", "not_started"),
        "rag_prefetch_results": state.get("rag_prefetch_results", []),
        "rag_prefetch_errors": state.get("rag_prefetch_errors", []),
        "memory_prefetch_status": state.get(
            "memory_prefetch_status", "not_started"
        ),
        "memory_prefetch_results": state.get("memory_prefetch_results", []),
        "memory_prefetch_errors": state.get("memory_prefetch_errors", []),
        "memory_commit_status": state.get("memory_commit_status", "not_started"),
        "memory_commit_ids": state.get("memory_commit_ids", []),
        "memory_commit_errors": state.get("memory_commit_errors", []),
        "final_content": state.get("final_content"),
        "draft_checkpoint_status": state.get(
            "draft_checkpoint_status", "not_saved"
        ),
        "reflection_mode": state.get("reflection_mode", "not_assessed"),
        "reflection_risk_level": state.get(
            "reflection_risk_level", "not_assessed"
        ),
        "reflection_risk_reasons": state.get("reflection_risk_reasons", []),
        "reflection_question_plan": state.get("reflection_question_plan", {}),
        "reflection_question_status": state.get(
            "reflection_question_status", "not_started"
        ),
        "reflection_verification_results": state.get(
            "reflection_verification_results", []
        ),
        "reflection_history": state.get("reflection_history", []),
        "verification_summary": state.get("verification_summary", ""),
        "reflection_status": state.get("reflection_status", "not_started"),
        "reflection_iterations": state.get("reflection_iterations", 0),
        "max_reflection_iterations": state.get("max_reflection_iterations", 1),
        "revision_steps": state.get("revision_steps", []),
        "current_revision_step_index": state.get(
            "current_revision_step_index", 0
        ),
        "revision_history": state.get("revision_history", []),
        "executor_mode": state.get("executor_mode", "plan"),
        "output_file": str(output_path),
    }


def build_research_graph(model: Any | None = None):
    """Build the bounded ReAct research graph and deterministic evaluation."""

    graph = StateGraph(ResearchState)
    graph.add_node("planning", planning_node)
    graph.add_node("research_agent", make_research_agent_node(model))
    graph.add_node("tools", research_tools_node)
    graph.add_node("evaluation", evaluation_node)
    graph.add_node("save", save_output_node)

    graph.add_edge(START, "planning")
    graph.add_edge("planning", "research_agent")
    graph.add_conditional_edges(
        "research_agent",
        route_after_research_agent,
        {"tools": "tools", "evaluation": "evaluation"},
    )
    graph.add_conditional_edges(
        "tools",
        route_after_research_tools,
        {"research_agent": "research_agent", "evaluation": "evaluation"},
    )
    graph.add_edge("evaluation", "save")
    graph.add_edge("save", END)
    return graph.compile()


research_graph = build_research_graph()
