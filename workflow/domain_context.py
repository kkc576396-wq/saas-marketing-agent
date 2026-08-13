"""Shared SmartPush product and competitor knowledge.

This module contains deterministic domain relationships, not research facts.
It lets the workflow understand that an explicitly named competitor belongs
to the email-marketing category without adding another model call.
"""

from __future__ import annotations

import re
from typing import Any


COMPETITOR_ENTITY_RULES: tuple[dict[str, Any], ...] = (
    {
        "name": "klaviyo",
        "aliases": ("klaviyo",),
        "category": "email_marketing_automation",
        "context_required": False,
    },
    {
        "name": "omnisend",
        "aliases": ("omnisend",),
        "category": "email_marketing_automation",
        "context_required": False,
    },
    {
        "name": "mailchimp",
        "aliases": ("mailchimp", "intuit mailchimp"),
        "category": "email_marketing_automation",
        "context_required": False,
    },
    {
        "name": "hubspot marketing hub",
        "aliases": ("hubspot marketing hub", "hubspot marketing"),
        "category": "marketing_automation",
        "context_required": False,
    },
    {
        "name": "hubspot marketing hub",
        "aliases": ("hubspot",),
        "category": "marketing_automation",
        "context_required": True,
    },
    {
        "name": "yotpo email & sms",
        "aliases": ("yotpo email & sms", "yotpo email and sms", "yotpo email"),
        "category": "email_marketing_automation",
        "context_required": False,
    },
    {
        "name": "drip",
        "aliases": ("drip",),
        "category": "email_marketing_automation",
        "context_required": True,
    },
    {
        "name": "attentive",
        "aliases": ("attentive",),
        "category": "email_and_sms_marketing",
        "context_required": True,
    },
)

# Used only when the user requests general competitor research without naming
# a brand. Keeping the benchmark set here prevents query-rewriter drift.
DEFAULT_RESEARCH_COMPETITORS = ("klaviyo", "omnisend", "mailchimp")

AMBIGUOUS_COMPETITOR_CONTEXT_TERMS = (
    "email",
    "sms marketing",
    "marketing automation",
    "campaign",
    "segmentation",
    "shopify",
    "ecommerce",
    "e-commerce",
    "merchant",
    "subscriber",
    "deliverability",
    "abandoned cart",
)

ALLOWED_INTENT_FACETS = (
    "market_intelligence",
    "community_intelligence",
    "competitor_monitoring",
    "competitor_pricing",
    "alternative_research",
    "product_feedback",
    "trend_research",
    "product_update_research",
    "competitor_content_analysis",
)

INTENT_FACET_SIGNALS = {
    "market_intelligence": (
        "market",
        "industry",
        "landscape",
        "industry developments",
        "市场",
        "行业",
        "产业",
        "产业动态",
    ),
    "community_intelligence": (
        "reddit",
        "discussion",
        "discussions",
        "experience",
        "review",
        "complaint",
        "complaints",
        "merchant feedback",
        "用户讨论",
        "用户体验",
        "评价",
        "投诉",
    ),
    "competitor_monitoring": (
        "competitor",
        "competitors",
        "competitive",
        "product update",
        "release notes",
        "竞品",
        "竞争对手",
    ),
    "competitor_pricing": (
        "pricing",
        "price",
        "expensive",
        "cost",
        "价格",
        "定价",
        "昂贵",
    ),
    "alternative_research": (
        "alternative",
        "alternatives",
        "switch",
        "switching",
        "migration",
        "migrate",
        "替代",
        "迁移",
        "更换",
    ),
    "product_feedback": (
        "feature request",
        "missing feature",
        "integration issue",
        "product feedback",
        "功能请求",
        "缺少功能",
        "集成问题",
        "产品反馈",
    ),
    "trend_research": (
        "trend",
        "trends",
        "latest developments",
        "industry developments",
        "emerging",
        "趋势",
        "最新动态",
        "新趋势",
        "产业动态",
    ),
    "product_update_research": (
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
        "上线",
    ),
    "competitor_content_analysis": (
        "comparison",
        "compare",
        "case study",
        "pricing page",
        "documentation",
        "对比",
        "比较",
        "案例研究",
        "文档",
    ),
}


def _contains_phrase(text: str, phrase: str) -> bool:
    """Match an entity alias on token boundaries."""

    return bool(
        re.search(
            rf"(?<![a-z0-9]){re.escape(phrase.casefold())}(?![a-z0-9])",
            text.casefold(),
        )
    )


def all_competitor_aliases() -> tuple[str, ...]:
    """Return stable, de-duplicated aliases for routing and retrieval."""

    return tuple(
        dict.fromkeys(
            alias
            for rule in COMPETITOR_ENTITY_RULES
            for alias in rule["aliases"]
        )
    )


def known_competitor_names() -> tuple[str, ...]:
    """Return canonical competitor names from the shared registry."""

    return tuple(dict.fromkeys(str(rule["name"]) for rule in COMPETITOR_ENTITY_RULES))


def competitor_aliases(name: str) -> tuple[str, ...]:
    """Return all configured aliases for one canonical competitor name."""

    return tuple(
        dict.fromkeys(
            alias
            for rule in COMPETITOR_ENTITY_RULES
            if str(rule["name"]) == name
            for alias in rule["aliases"]
        )
    )


def match_known_competitors(text: str) -> list[str]:
    """Return known competitor entities found in text.

    Ambiguous names such as Drip and Attentive require email/e-commerce
    context in the same source text. This prevents unrelated phrases such as
    "drip coffee" from satisfying SmartPush product relevance.
    """

    normalized = str(text or "").casefold()
    has_product_context = any(
        _contains_phrase(normalized, term)
        for term in AMBIGUOUS_COMPETITOR_CONTEXT_TERMS
    )
    matches: list[str] = []
    for rule in COMPETITOR_ENTITY_RULES:
        if rule["context_required"] and not has_product_context:
            continue
        if any(_contains_phrase(normalized, alias) for alias in rule["aliases"]):
            matches.append(str(rule["name"]))
    return list(dict.fromkeys(matches))


def competitor_category(name: str) -> str | None:
    """Return the configured product category for one canonical entity."""

    for rule in COMPETITOR_ENTITY_RULES:
        if rule["name"] == name:
            return str(rule["category"])
    return None


def competitor_entity_records(text: str) -> list[dict[str, str]]:
    """Return normalized entity records safe to expose in workflow state."""

    return [
        {
            "name": name,
            "canonical_name": name,
            "entity_type": "known_competitor",
            "product_category": competitor_category(name)
            or "email_marketing_automation",
        }
        for name in match_known_competitors(text)
    ]


def detect_intent_facets(text: str) -> list[str]:
    """Infer stable research facets as a deterministic model fallback."""

    normalized = str(text or "").casefold()
    facets = [
        facet
        for facet, signals in INTENT_FACET_SIGNALS.items()
        if any(signal.casefold() in normalized for signal in signals)
    ]
    if match_known_competitors(normalized) and "competitor_monitoring" not in facets:
        facets.append("competitor_monitoring")
    return facets or ["market_intelligence"]


def validate_intent_facets(values: Any) -> list[str]:
    """Keep only known intent enums from an untrusted model response."""

    if not isinstance(values, (list, tuple)):
        return []
    return list(
        dict.fromkeys(
            str(value).strip()
            for value in values
            if str(value).strip() in ALLOWED_INTENT_FACETS
        )
    )
