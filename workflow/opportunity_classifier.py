"""SmartPush opportunity classification for verified, scored insights."""

from __future__ import annotations

import re
from typing import Any

from .entity_relevance import entity_roles
from .state import ResearchState


COMPETITOR_OPPORTUNITY = "competitor_opportunity"
EDUCATIONAL_CONTENT = "educational_content"
PRODUCT_FEEDBACK = "product_feedback"

COMPETITOR_OPPORTUNITY_TERMS = (
    "pricing",
    "price",
    "complaint",
    "complain",
    "alternative",
    "migration",
    "migrate",
    "switch",
    "competitor",
    "compare",
    "comparison",
)
EDUCATIONAL_CONTENT_TERMS = (
    "tutorial",
    "guide",
    "best practice",
    "best practices",
    "how to",
    "trend",
    "trends",
    "documentation",
    "learn",
    "strategy",
)
PRODUCT_FEEDBACK_TERMS = (
    "feature request",
    "request",
    "integration",
    "missing capability",
    "missing feature",
    "doesn't support",
    "does not support",
    "cannot",
    "can't",
    "problem",
    "issue",
    "feedback",
    "wishlist",
)

RECOMMENDED_CHANNELS = {
    COMPETITOR_OPPORTUNITY: [
        "competitor comparison page",
        "competitive landing page",
        "sales enablement",
    ],
    EDUCATIONAL_CONTENT: [
        "SEO blog",
        "email newsletter",
        "help center",
    ],
    PRODUCT_FEEDBACK: [
        "product roadmap",
        "customer research",
        "in-app feedback",
    ],
}


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _matches(text: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if term in text]


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


def classify_insight(insight: Any) -> dict[str, Any]:
    """Classify one insight into the primary SmartPush opportunity type."""

    normalized = _normalize(_insight_text(insight))
    if not normalized:
        raise ValueError("insight must not be empty")

    roles = entity_roles(insight)
    competitor_names = list(
        dict.fromkeys(roles["primary"] + roles["supporting"])
    )
    competitor_signals = _matches(normalized, COMPETITOR_OPPORTUNITY_TERMS)
    educational_signals = _matches(normalized, EDUCATIONAL_CONTENT_TERMS)
    product_signals = _matches(normalized, PRODUCT_FEEDBACK_TERMS)

    # Product requests take precedence over general pain language, while
    # explicit competitor pricing/switching signals take precedence over the
    # educational interpretation of the same insight.
    if competitor_names and competitor_signals:
        opportunity_type = COMPETITOR_OPPORTUNITY
    elif product_signals:
        opportunity_type = PRODUCT_FEEDBACK
    elif educational_signals:
        opportunity_type = EDUCATIONAL_CONTENT
    elif competitor_names:
        opportunity_type = COMPETITOR_OPPORTUNITY
    else:
        # Keep every scored insight actionable even when it lacks a strong
        # taxonomy signal; generic research guidance is the safest fallback.
        opportunity_type = EDUCATIONAL_CONTENT

    all_matches = {
        COMPETITOR_OPPORTUNITY: sorted(set(competitor_names + competitor_signals)),
        EDUCATIONAL_CONTENT: sorted(set(educational_signals)),
        PRODUCT_FEEDBACK: sorted(set(product_signals)),
    }
    return {
        "insight": _insight_id(insight),
        "opportunity_type": opportunity_type,
        "matched_signals": all_matches,
        "entity_roles": roles,
        "recommended_channels": RECOMMENDED_CHANNELS[opportunity_type],
    }


def opportunity_classifier_node(state: ResearchState) -> dict[str, Any]:
    """Classify the insights that were retained by scoring."""

    classifications = [
        classify_insight(insight) for insight in state.get("insights", [])
    ]
    return {
        "opportunity_types": classifications,
        "recommended_channels": [
            {
                "insight": classification["insight"],
                "channels": classification["recommended_channels"],
            }
            for classification in classifications
        ],
    }
