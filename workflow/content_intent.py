"""Shared structural helpers for normalized Content intent state."""

from __future__ import annotations

from typing import Any


DELIVERABLE_TYPES = frozenset(
    {
        "homepage_promotion",
        "reddit_promotion",
        "reddit_reply",
        "competitor_report",
    }
)

LEGACY_DELIVERABLE_ALIASES = {
    "competitor_research": "competitor_report",
    "research_only": None,
}


def normalize_deliverable_type(value: Any) -> str | None:
    """Normalize a current or legacy deliverable enum without inferring intent."""

    normalized = str(value or "").strip().casefold()
    if normalized in LEGACY_DELIVERABLE_ALIASES:
        return LEGACY_DELIVERABLE_ALIASES[normalized]
    return normalized if normalized in DELIVERABLE_TYPES else None


def deliverable_type(intent: Any) -> str | None:
    """Read the new field first and accept old persisted state during migration."""

    if not isinstance(intent, dict):
        return None
    return normalize_deliverable_type(
        intent.get("deliverable_type", intent.get("type"))
    )


def content_requested(intent: Any) -> bool:
    """Read explicit request state, falling back only for legacy persisted data."""

    if not isinstance(intent, dict):
        return False
    requested = intent.get("requested")
    if isinstance(requested, bool):
        return requested
    return deliverable_type(intent) is not None
