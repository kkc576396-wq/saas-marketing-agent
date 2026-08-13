"""SmartPush-specific insight scoring."""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any

from .domain_context import detect_intent_facets, match_known_competitors
from .entity_relevance import entity_roles
from .state import ResearchState


SMARTPUSH_RELEVANCE_TERMS = (
    "smartpush",
    "email marketing",
    "email automation",
    "marketing automation",
    "segmentation",
    "customer segment",
    "email flow",
    "automated flow",
    "lifecycle marketing",
    "e-commerce crm",
    "ecommerce crm",
    "retention marketing",
    "shopify",
    "dtc",
    "merchant",
    "online retailer",
)
PAIN_TERMS = (
    "problem",
    "issue",
    "pain",
    "frustrat",
    "complaint",
    "complain",
    "hate",
    "expensive",
    "too expensive",
    "bad experience",
    "migration",
    "migrate",
    "switch",
    "stuck",
    "difficult",
    "hard to",
)
CONTENT_OPPORTUNITY_TERMS = (
    "pricing",
    "shopify",
    "merchant",
    "price",
    "alternative",
    "competitor",
    "comparison",
    "compare",
    "feature",
    "segmentation",
    "flow",
    "case study",
    "review",
    "benchmark",
    "how to",
)
FRESHNESS_TERMS = (
    "latest",
    "recent",
    "current",
    "new",
    "update",
    "updated",
    "launch",
    "release",
    "roadmap",
    "news",
    "2025",
    "2026",
)
COMPETITOR_SCORE_FACETS = {
    "competitor_monitoring",
    "competitor_pricing",
    "alternative_research",
    "competitor_content_analysis",
    "product_update_research",
}
TOPIC_GENERIC_TERMS = {
    "about",
    "analysis",
    "and",
    "company",
    "companies",
    "email",
    "for",
    "latest",
    "marketing",
    "research",
    "the",
}
INSIGHT_FACET_TERMS = {
    "market_intelligence": (
        "market share",
        "market size",
        "market change",
        "market changes",
        "market dynamics",
        "industry",
        "landscape",
        "benchmark",
        "report",
        "adoption",
        "growth",
        "forecast",
        "outlook",
    ),
    "trend_research": (
        "trend",
        "trends",
        "emerging",
        "forecast",
        "outlook",
        "adoption",
        "benchmark",
        "industry shift",
        "market growth",
        "innovation",
    ),
    "community_intelligence": (
        "discussion",
        "experience",
        "review",
        "complaint",
        "frustrated",
        "merchant feedback",
        "user feedback",
    ),
    "competitor_pricing": (
        "pricing",
        "price",
        "cost",
        "expensive",
        "plan",
        "tier",
    ),
    "alternative_research": (
        "alternative",
        "switch",
        "migration",
        "migrate",
        "replacement",
    ),
    "product_feedback": (
        "feature request",
        "missing feature",
        "integration issue",
        "product feedback",
        "does not support",
        "problem",
        "issue",
    ),
    "competitor_content_analysis": (
        "comparison",
        "compare",
        "case study",
        "pricing page",
        "documentation",
        "review",
    ),
    "product_update_research": (
        "product update",
        "product release",
        "feature update",
        "feature release",
        "feature upgrade",
        "latest release",
        "product launch",
        "release notes",
        "released",
        "launched",
        "rollout",
        "changelog",
        "new feature",
        "what's new",
        "what’s new",
        "product enhancement",
        "new capabilities",
        "shipped",
        "ships",
        "unveils",
        "introduces",
    ),
}
PRODUCT_UPDATE_EVIDENCE_TERMS = (
    "product update",
    "product release",
    "feature update",
    "feature release",
    "feature upgrade",
    "latest release",
    "release notes",
    "changelog",
    "what's new",
    "what’s new",
    "new feature",
    "new capabilities",
    "product enhancement",
    "launches",
    "launched",
    "released",
    "rollout",
    "rolled out",
    "ships",
    "shipped",
    "unveils",
    "unveiled",
    "introduces",
    "introduced",
)
PRODUCT_UPDATE_NON_RELEASE_CONTENT_TERMS = (
    "review",
    "comparison",
    "alternatives",
    "alternative",
    "migration",
    "moving to",
    "best email marketing tools",
    "best sms marketing apps",
    "pricing guide",
)
SCORE_WEIGHTS = {
    "topic_alignment_score": 0.25,
    "business_relevance_score": 0.20,
    "customer_pain_score": 0.25,
    "content_opportunity_score": 0.20,
    "freshness_score": 0.10,
}


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _matched_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if term in text]


def _contains_phrase(text: str, phrase: str) -> bool:
    return bool(
        re.search(
            rf"(?<![a-z0-9]){re.escape(phrase.casefold())}(?![a-z0-9])",
            text.casefold(),
        )
    )


def _topic_tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", value.casefold()):
        if token in TOPIC_GENERIC_TERMS:
            continue
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.add(token)
    return tokens


def detect_product_update_evidence(insight: Any) -> dict[str, Any]:
    """Detect explicit, dated release evidence without trusting generic reviews."""

    if isinstance(insight, dict):
        title = _normalize(str(insight.get("title", "")))
        summary = _normalize(str(insight.get("summary", "")))
        sources = [
            source
            for source in insight.get("sources", [])
            if isinstance(source, dict)
        ]
    else:
        title = _normalize(str(insight))
        summary = ""
        sources = []

    primary_source = sources[0] if sources else {}
    url = _normalize(str(primary_source.get("url", "")))
    title_signals = sorted(
        term for term in PRODUCT_UPDATE_EVIDENCE_TERMS if term in title
    )
    summary_signals = sorted(
        term for term in PRODUCT_UPDATE_EVIDENCE_TERMS if term in summary
    )
    url_signals = sorted(
        term
        for term in ("release-notes", "release_notes", "changelog", "whats-new")
        if term in url
    )
    non_release_signals = sorted(
        term for term in PRODUCT_UPDATE_NON_RELEASE_CONTENT_TERMS if term in title
    )
    strong_title_or_url_evidence = bool(title_signals or url_signals)
    summary_only_evidence = bool(summary_signals) and not non_release_signals
    release_signal_available = strong_title_or_url_evidence or summary_only_evidence
    published_at = primary_source.get("published_at")
    published_at_available = bool(_parse_date(published_at))

    rejection_reasons: list[str] = []
    if not release_signal_available:
        rejection_reasons.append("missing_product_update_evidence")
    if not published_at_available:
        rejection_reasons.append("missing_product_update_published_at")
    if non_release_signals and not strong_title_or_url_evidence:
        rejection_reasons.append("product_update_review_or_comparison")

    return {
        "passed": not rejection_reasons,
        "release_signal_available": release_signal_available,
        "published_at_available": published_at_available,
        "published_at": published_at,
        "title_signals": title_signals,
        "summary_signals": summary_signals,
        "url_signals": url_signals,
        "non_release_signals": non_release_signals,
        "rejection_reasons": rejection_reasons,
    }


def calculate_topic_alignment_score(
    insight: Any,
    *,
    topic: str,
    intent_facets: list[str] | None = None,
    source: str = "",
) -> tuple[float, dict[str, Any]]:
    """Measure whether an insight answers this run's specific research goal."""

    insight_text = _normalize(_insight_text(insight))
    topic_text = _normalize(topic)
    expected_facets = list(
        dict.fromkeys(intent_facets or detect_intent_facets(topic_text))
    )
    query_competitors = match_known_competitors(topic_text)
    insight_competitors = match_known_competitors(insight_text)

    matched_facets: list[str] = []
    facet_signals: dict[str, list[str]] = {}
    for facet in expected_facets:
        if facet == "competitor_monitoring":
            if query_competitors:
                matched = sorted(
                    set(query_competitors).intersection(insight_competitors)
                )
            else:
                matched = list(insight_competitors)
            signals = [f"competitor:{name}" for name in matched]
        else:
            signals = [
                term
                for term in INSIGHT_FACET_TERMS.get(facet, ())
                if _contains_phrase(insight_text, term)
            ]
            if (
                facet == "community_intelligence"
                and "reddit" in _normalize(source)
            ):
                signals.append("reddit_source")
        if signals:
            matched_facets.append(facet)
            facet_signals[facet] = sorted(set(signals))

    facet_score = (
        100.0 * len(matched_facets) / len(expected_facets)
        if expected_facets
        else 50.0
    )
    topic_tokens = _topic_tokens(topic_text)
    insight_tokens = _topic_tokens(insight_text)
    matched_topic_terms = sorted(topic_tokens.intersection(insight_tokens))
    lexical_score = (
        100.0 * len(matched_topic_terms) / len(topic_tokens)
        if topic_tokens
        else 50.0
    )

    if query_competitors:
        matched_entities = sorted(
            set(query_competitors).intersection(insight_competitors)
        )
        entity_score = 100.0 if matched_entities else 0.0
        score = 0.50 * facet_score + 0.25 * lexical_score + 0.25 * entity_score
    else:
        matched_entities = []
        entity_score = None
        score = 0.65 * facet_score + 0.35 * lexical_score

    return round(min(100.0, max(0.0, score)), 2), {
        "expected_facets": expected_facets,
        "matched_facets": matched_facets,
        "facet_signals": facet_signals,
        "matched_topic_terms": matched_topic_terms,
        "query_competitors": query_competitors,
        "matched_query_competitors": matched_entities,
        "facet_score": round(facet_score, 2),
        "lexical_score": round(lexical_score, 2),
        "entity_score": entity_score,
    }


def _bounded_score(match_count: int, *, max_score: float = 100.0) -> float:
    """Convert a number of matching signals into a bounded 0–100 score."""

    return min(max_score, float(match_count * 25))


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.date()


def calculate_freshness_score(
    insight: Any,
    *,
    research_date: str | None,
) -> tuple[float, list[str]]:
    """Score freshness from source metadata instead of freshness words."""

    if not isinstance(insight, dict):
        text = _normalize(_insight_text(insight))
        matches = _matched_terms(text, FRESHNESS_TERMS)
        return _bounded_score(len(set(matches))), matches

    source_items = [
        source for source in insight.get("sources", []) if isinstance(source, dict)
    ]
    source_item = source_items[0] if source_items else {}
    date_status = str(
        source_item.get("date_status") or insight.get("date_status") or "unknown"
    )
    if date_status == "current_page":
        return 80.0, ["current_page"]

    published = _parse_date(source_item.get("published_at"))
    raw_date_confidence = source_item.get("date_confidence")
    date_confidence = (
        float(raw_date_confidence)
        if raw_date_confidence is not None
        else 1.0
    )
    effective_research_date = _parse_date(research_date) or date.today()
    if published is None:
        return 20.0, ["date_unknown"]

    age_days = max(0, (effective_research_date - published).days)
    if age_days <= 30:
        score = 100.0
    elif age_days <= 90:
        score = 90.0
    elif age_days <= 180:
        score = 75.0
    elif age_days <= 365:
        score = 60.0
    elif age_days <= 730:
        score = 30.0
    else:
        score = 0.0
    if date_confidence < 0.5 and published.year != effective_research_date.year:
        score = min(score, 40.0)
    return score, [
        f"age_days:{age_days}",
        f"published_at:{published.isoformat()}",
        f"date_confidence:{date_confidence}",
    ]


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


def score_insight(
    insight: Any,
    *,
    topic: str = "",
    source: str = "",
    research_date: str | None = None,
    intent_facets: list[str] | None = None,
) -> dict[str, Any]:
    """Score one verified insight for SmartPush research prioritization."""

    insight_text = _normalize(_insight_text(insight))
    relevance_matches = _matched_terms(insight_text, SMARTPUSH_RELEVANCE_TERMS)
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
    relevance_matches.extend(
        f"known_competitor:{name}" for name in competitor_matches
    )
    pain_matches = _matched_terms(insight_text, PAIN_TERMS)
    content_matches = _matched_terms(insight_text, CONTENT_OPPORTUNITY_TERMS)
    content_matches.extend(
        f"known_competitor:{name}" for name in competitor_matches
    )
    freshness_score, freshness_matches = calculate_freshness_score(
        insight,
        research_date=research_date,
    )
    effective_facets = list(
        dict.fromkeys(intent_facets or detect_intent_facets(topic))
    )
    topic_alignment_score, topic_alignment_signals = (
        calculate_topic_alignment_score(
            insight,
            topic=topic,
            intent_facets=effective_facets,
            source=(
                str(insight.get("source_type", ""))
                if isinstance(insight, dict)
                else source
            ),
        )
    )
    product_update_evidence = detect_product_update_evidence(insight)

    # Only normalized Reddit candidates with a real source URL receive the
    # additional community-evidence signal. Legacy string fixtures retain the
    # previous behavior for backwards compatibility.
    has_reddit_evidence = "reddit" in _normalize(
        str(insight.get("source_type", "")) if isinstance(insight, dict) else source
    ) and (
        not isinstance(insight, dict)
        or any(source_item.get("url") for source_item in insight.get("sources", []))
    )
    if has_reddit_evidence and "pain" not in pain_matches:
        pain_matches.append("reddit evidence")

    competitor_intent = bool(set(effective_facets).intersection(COMPETITOR_SCORE_FACETS))
    if competitor_intent:
        # Entity relevance is intentionally capped. Mentioning multiple known
        # competitors should not overpower topic fit or evidence quality.
        entity_relevance_bonus = (
            15.0
            if primary_competitors
            else 5.0
            if supporting_competitors
            else 0.0
        )
        entity_content_bonus = (
            5.0
            if primary_competitors
            else 2.0
            if supporting_competitors
            else 0.0
        )
    else:
        # In broad market/trend research, a competitor mention is only an
        # auxiliary domain clue rather than a content-opportunity shortcut.
        entity_relevance_bonus = 3.0 if primary_competitors else 1.0 if supporting_competitors else 0.0
        entity_content_bonus = 1.0 if primary_competitors else 0.0
    non_entity_relevance_matches = [
        match for match in relevance_matches if not match.startswith("known_competitor:")
    ]
    non_entity_content_matches = [
        match for match in content_matches if not match.startswith("known_competitor:")
    ]
    business_relevance_score = min(
        100.0,
        _bounded_score(len(set(non_entity_relevance_matches)))
        + entity_relevance_bonus,
    )
    customer_pain_score = _bounded_score(len(set(pain_matches)))
    content_opportunity_score = min(
        100.0,
        _bounded_score(len(set(non_entity_content_matches)))
        + entity_content_bonus,
    )
    total_score = round(
        SCORE_WEIGHTS["topic_alignment_score"] * topic_alignment_score
        + SCORE_WEIGHTS["business_relevance_score"] * business_relevance_score
        + SCORE_WEIGHTS["customer_pain_score"] * customer_pain_score
        + SCORE_WEIGHTS["content_opportunity_score"] * content_opportunity_score
        + SCORE_WEIGHTS["freshness_score"] * freshness_score,
        2,
    )

    return {
        "insight": _insight_id(insight),
        "topic_alignment_score": topic_alignment_score,
        "business_relevance_score": business_relevance_score,
        "customer_pain_score": customer_pain_score,
        "content_opportunity_score": content_opportunity_score,
        "freshness_score": freshness_score,
        "total_score": total_score,
        "product_update_evidence": product_update_evidence,
        "matched_signals": {
            "topic_alignment": topic_alignment_signals,
            "relevance": sorted(set(relevance_matches)),
            "pain": sorted(set(pain_matches)),
            "content_opportunity": sorted(set(content_matches)),
            "freshness": sorted(set(freshness_matches)),
            "competitors": sorted(set(competitor_matches)),
            "entity_roles": roles,
            "competitor_intent_bonus_enabled": competitor_intent,
            "entity_relevance_bonus": entity_relevance_bonus,
            "entity_content_bonus": entity_content_bonus,
        },
        "score_weights": dict(SCORE_WEIGHTS),
    }


def scoring_node(state: ResearchState) -> dict[str, Any]:
    """Score only insights that survived the verifier node."""

    source = " ".join(state.get("selected_sources", []))
    topic = state.get("translated_query") or state.get("topic", "")
    insight_scores = [
        score_insight(
            insight,
            topic=topic,
            source=source,
            research_date=state.get("research_date"),
            intent_facets=list(state.get("intent_facets", [])),
        )
        for insight in state.get("insights", [])
    ]
    return {"insight_scores": insight_scores}
