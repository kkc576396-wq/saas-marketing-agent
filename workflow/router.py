"""SmartPush-specific source classification and routing decisions."""

from __future__ import annotations

import re
from typing import Any

from .domain_context import all_competitor_aliases, match_known_competitors
from .state import ResearchState


ANYSEARCH = "AnySearch"
AGENT_REACH_REDDIT = "Agent-Reach Reddit"
AGENT_REACH_WEB = "Agent-Reach Web"
AGENT_REACH_RSS = "Agent-Reach RSS"

INTENT_FACET_SOURCES = {
    "market_intelligence": (ANYSEARCH, AGENT_REACH_RSS),
    "community_intelligence": (AGENT_REACH_REDDIT,),
    "competitor_monitoring": (ANYSEARCH,),
    "competitor_pricing": (ANYSEARCH,),
    "alternative_research": (AGENT_REACH_REDDIT,),
    "product_feedback": (AGENT_REACH_REDDIT,),
    "trend_research": (ANYSEARCH, AGENT_REACH_RSS),
    "product_update_research": (ANYSEARCH, AGENT_REACH_WEB, AGENT_REACH_RSS),
    "competitor_content_analysis": (ANYSEARCH, AGENT_REACH_WEB),
}

SMARTPUSH_PRODUCT_CONTEXT = {
    "product": [
        "email marketing automation",
        "customer segmentation",
        "automated email flows",
        "customer lifecycle marketing",
        "e-commerce crm",
        "retention marketing",
    ],
    "target_market": [
        "north american e-commerce merchants",
        "shopify merchants",
        "dtc brands",
        "smb online retailers",
    ],
    "competitors": list(all_competitor_aliases()),
}

COMPETITOR_TRIGGERS = (
    "pricing",
    "product update",
    "ai",
    "segmentation",
    "flows",
    "benchmark",
    "alternative",
    "competitor",
    "competitors",
    "automation",
)
TREND_TRIGGERS = (
    "email marketing trends",
    "marketing automation trends",
    "ai marketing",
    "ai email automation",
    "customer segmentation trends",
    "crm trends",
    "retention marketing",
    "lifecycle marketing",
    "e-commerce automation",
)
# These phrases describe market-news intent specifically. They are kept
# separate from the broader trend triggers so a plain "email marketing
# trends" query keeps its existing AnySearch-only route, while a compound
# market-dynamics query gets industry-feed coverage as well.
MARKET_DYNAMIC_TRIGGERS = (
    "market dynamics",
    "market dynamic",
    "market update",
    "market updates",
    "market news",
    "industry dynamics",
    "industry dynamic",
    "industry update",
    "industry updates",
    "industry news",
    "software market",
    "market landscape",
    "latest developments",
    "recent developments",
    "市场动态",
    "市场更新",
    "行业动态",
)
CONTENT_TRIGGERS = (
    "blog",
    "case study",
    "pricing page",
    "feature comparison",
    "release notes",
    "product roadmap",
    "documentation",
    "compare",
    "comparison",
    "analysis",
)
PRODUCT_UPDATE_TRIGGERS = (
    "product update",
    "product updates",
    "product release",
    "product releases",
    "feature update",
    "feature updates",
    "feature release",
    "feature releases",
    "feature upgrade",
    "feature upgrades",
    "latest release",
    "latest releases",
    "product launch",
    "product launches",
    "release notes",
    "changelog",
    "产品更新",
    "功能更新",
    "功能升级",
    "最新发布",
    "产品发布",
    "版本更新",
)
COMMUNITY_TRIGGERS = (
    "problem",
    "issue",
    "frustrated",
    "alternative",
    "recommendation",
    "experience",
    "review",
    "migration",
    "migrate",
    "switch",
    "worth it",
    "hate",
    "expensive",
    "too expensive",
    "bad experience",
    "complaint",
    "complaining",
    "discussion",
    "discussions",
)
REDDIT_COMMUNITIES = (
    "r/ecommerce",
    "r/shopify",
    "r/saas",
    "r/entrepreneur",
    "r/marketing",
    "r/emailmarketing",
    "r/dtc",
)
COMMUNITY_TOPIC_TRIGGERS = (
    "email marketing tool",
    "email automation",
    "marketing automation",
    "shopify email app",
    "klaviyo alternative",
    "omnisend alternative",
    "email segmentation",
    "abandoned cart email",
    "customer retention",
)
INDUSTRY_SOURCES = (
    "klaviyo blog",
    "omnisend blog",
    "shopify blog",
    "hubspot marketing blog",
    "marketing brew",
    "emarketer",
    "search engine journal",
)
INDUSTRY_TOPICS = (
    "email privacy",
    "first-party data",
    "customer retention",
    "crm",
)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _contains_any(query: str, phrases: tuple[str, ...]) -> list[str]:
    return [phrase for phrase in phrases if phrase in query]


def classify_query(query: str) -> dict[str, Any]:
    """Classify one SmartPush research query into provider sources."""

    normalized = _normalize(query)
    if not normalized:
        raise ValueError("research queries must not be empty")

    competitor_names = match_known_competitors(normalized)
    competitor_triggers = _contains_any(normalized, COMPETITOR_TRIGGERS)
    trend_triggers = _contains_any(normalized, TREND_TRIGGERS)
    market_dynamic_triggers = _contains_any(normalized, MARKET_DYNAMIC_TRIGGERS)
    content_triggers = _contains_any(normalized, CONTENT_TRIGGERS)
    product_update_triggers = _contains_any(normalized, PRODUCT_UPDATE_TRIGGERS)
    community_triggers = _contains_any(normalized, COMMUNITY_TRIGGERS)
    reddit_communities = _contains_any(normalized, REDDIT_COMMUNITIES)
    community_topics = _contains_any(normalized, COMMUNITY_TOPIC_TRIGGERS)
    industry_sources = _contains_any(normalized, INDUSTRY_SOURCES)
    industry_topics = _contains_any(normalized, INDUSTRY_TOPICS)

    selected_sources: list[str] = []
    reasons: list[str] = []

    if (competitor_names and competitor_triggers) or trend_triggers or (
        "competitor" in normalized and competitor_triggers
    ):
        selected_sources.append(ANYSEARCH)
        reasons.append("official competitor and SaaS market intelligence")

    if market_dynamic_triggers:
        if ANYSEARCH not in selected_sources:
            selected_sources.append(ANYSEARCH)
            reasons.append("market dynamics and software landscape discovery")
        selected_sources.append(AGENT_REACH_RSS)
        reasons.append("industry publications and source feeds for market updates")

    if competitor_names and (content_triggers or "competitor" in normalized):
        if ANYSEARCH not in selected_sources:
            selected_sources.append(ANYSEARCH)
        selected_sources.append(AGENT_REACH_WEB)
        reasons.append("discover competitor pages, then extract detailed content")

    if product_update_triggers:
        if ANYSEARCH not in selected_sources:
            selected_sources.append(ANYSEARCH)
        selected_sources.extend((AGENT_REACH_WEB, AGENT_REACH_RSS))
        reasons.append(
            "discover and extract official product releases, feature updates, and feeds"
        )

    if community_triggers or reddit_communities or community_topics:
        selected_sources.append(AGENT_REACH_REDDIT)
        reasons.append("authentic merchant and practitioner discussions")

    if industry_sources or industry_topics:
        selected_sources.append(AGENT_REACH_RSS)
        reasons.append("industry publications and source feeds")

    if not selected_sources:
        selected_sources.append(ANYSEARCH)
        reasons.append("default SmartPush market-intelligence route")

    # Preserve order while removing duplicates from multi-signal queries.
    selected_sources = list(dict.fromkeys(selected_sources))
    return {
        "query": query,
        "selected_sources": selected_sources,
        "reasoning": "; ".join(reasons),
        "signals": {
            "competitors": competitor_names,
            "competitor_triggers": competitor_triggers,
            "trend_triggers": trend_triggers,
            "market_dynamic_triggers": market_dynamic_triggers,
            "content_triggers": content_triggers,
            "product_update_triggers": product_update_triggers,
            "community_triggers": community_triggers,
            "reddit_communities": reddit_communities,
            "industry_sources": industry_sources,
            "industry_topics": industry_topics,
        },
    }


def source_router_node(state: ResearchState) -> dict[str, Any]:
    """Create a multi-source plan for the current SmartPush research run."""

    translated_query = state.get("translated_query", "").strip()
    if translated_query:
        # Route from the single translated user intent. Platform-specific
        # query variants are recommendations for the ReAct Research Agent.
        queries = [translated_query]
    else:
        queries = [query.strip() for query in state.get("search_queries", []) if query.strip()]
        if not queries:
            topic = state.get("topic", "").strip()
            queries = [topic] if topic else []
    if not queries:
        raise ValueError("ResearchState must contain topic or search_queries")

    decisions = [classify_query(query) for query in queries]
    selected_sources = list(
        dict.fromkeys(
            source
            for decision in decisions
            for source in decision["selected_sources"]
        )
    )
    intent_facets = list(state.get("intent_facets", []))
    for facet in intent_facets:
        selected_sources.extend(INTENT_FACET_SOURCES.get(facet, ()))
    selected_sources = list(dict.fromkeys(selected_sources))
    reasoning = " | ".join(
        f"{decision['query']}: {decision['reasoning']}" for decision in decisions
    )
    if intent_facets:
        reasoning = (
            f"{reasoning} | structured intent facets: "
            f"{', '.join(intent_facets)}"
        )

    return {
        "search_queries": queries,
        "selected_sources": selected_sources,
        "source_reasoning": reasoning,
    }
