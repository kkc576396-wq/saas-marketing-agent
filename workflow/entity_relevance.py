"""Explainable competitor-entity importance for individual insights.

Entity detection answers whether a brand is mentioned. This module answers a
different question: whether the brand is central to the source claim. Keeping
the two decisions separate prevents a competitor in a vendor list or an aside
from receiving the same weight as the subject of a complaint.
"""

from __future__ import annotations

import re
from typing import Any

from .domain_context import competitor_aliases, match_known_competitors


ENTITY_CONTEXT_SIGNALS = (
    "pricing",
    "price",
    "expensive",
    "cost",
    "alternative",
    "switch",
    "switching",
    "migration",
    "migrate",
    "complaint",
    "complain",
    "problem",
    "issue",
    "feature",
    "integration",
    "automation",
    "segmentation",
    "deliverability",
    "shopify",
    "ecommerce",
    "e-commerce",
    "email marketing",
    "sms marketing",
)
INCIDENTAL_CUES = (
    "including",
    "included",
    "such as",
    "for example",
    "e.g.",
    "among others",
    "alongside",
)
LEAD_CHARACTER_LIMIT = 320


def _contains_phrase(text: str, phrase: str) -> bool:
    return bool(
        re.search(
            rf"(?<![a-z0-9]){re.escape(phrase.casefold())}(?![a-z0-9])",
            text.casefold(),
        )
    )


def _occurrence_count(text: str, aliases: tuple[str, ...]) -> int:
    matches: list[tuple[int, int]] = []
    for alias in sorted(aliases, key=len, reverse=True):
        matches.extend(
            (match.start(), match.end())
            for match in re.finditer(
                rf"(?<![a-z0-9]){re.escape(alias.casefold())}(?![a-z0-9])",
                text.casefold(),
            )
        )
    # Avoid counting an overlapping short alias inside a longer configured one.
    non_overlapping: list[tuple[int, int]] = []
    for start, end in sorted(matches, key=lambda item: (item[0], -(item[1] - item[0]))):
        if any(start < kept_end and end > kept_start for kept_start, kept_end in non_overlapping):
            continue
        non_overlapping.append((start, end))
    return len(non_overlapping)


def _sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?。！？])\s+|[\r\n]+", text)
        if sentence.strip()
    ]


def analyze_entity_importance(title: str, summary: str = "") -> list[dict[str, Any]]:
    """Classify each competitor mention as primary, supporting, or incidental.

    A title mention, a context-rich lead mention, or repeated discussion makes
    an entity primary. A contextual mention elsewhere is supporting. A single
    list/example mention is incidental and cannot establish product relevance.
    """

    clean_title = str(title or "").strip()
    clean_summary = str(summary or "").strip()
    combined = " ".join(part for part in (clean_title, clean_summary) if part)
    if not combined:
        return []

    sentence_items = _sentences(combined)
    lead = clean_summary[:LEAD_CHARACTER_LIMIT] or clean_title[:LEAD_CHARACTER_LIMIT]
    all_competitors = match_known_competitors(combined)
    records: list[dict[str, Any]] = []

    for competitor in all_competitors:
        aliases = competitor_aliases(competitor)
        occurrence_count = _occurrence_count(combined, aliases)
        appears_in_title = any(_contains_phrase(clean_title, alias) for alias in aliases)
        appears_in_lead = any(_contains_phrase(lead, alias) for alias in aliases)
        mention_sentences = [
            sentence
            for sentence in sentence_items
            if any(_contains_phrase(sentence, alias) for alias in aliases)
        ]
        nearby_signals = sorted(
            {
                signal
                for sentence in mention_sentences
                for signal in ENTITY_CONTEXT_SIGNALS
                if signal in sentence.casefold()
            }
        )
        listed_with_other_competitors = any(
            len(match_known_competitors(sentence)) >= 2
            for sentence in mention_sentences
        )
        incidental_cue = any(
            cue in sentence.casefold()
            for sentence in mention_sentences
            for cue in INCIDENTAL_CUES
        )

        if appears_in_title:
            role = "primary"
            importance_score = 100.0
            basis = "title_subject"
        elif len(mention_sentences) >= 2 or occurrence_count >= 2:
            role = "primary"
            importance_score = 85.0
            basis = "repeated_discussion"
        elif appears_in_lead and nearby_signals:
            role = "primary"
            importance_score = 75.0
            basis = "core_summary_with_product_context"
        elif incidental_cue and occurrence_count == 1:
            role = "incidental"
            importance_score = 10.0
            basis = "single_or_list_mention"
        elif nearby_signals and not (listed_with_other_competitors and incidental_cue):
            role = "supporting"
            importance_score = 45.0
            basis = "contextual_supporting_mention"
        elif appears_in_lead and not listed_with_other_competitors:
            role = "supporting"
            importance_score = 35.0
            basis = "core_summary_mention"
        else:
            role = "incidental"
            importance_score = 10.0
            basis = "single_or_list_mention"

        records.append(
            {
                "competitor": competitor,
                "entity_role": role,
                "importance_score": importance_score,
                "basis": basis,
                "occurrence_count": occurrence_count,
                "appears_in_title": appears_in_title,
                "appears_in_lead": appears_in_lead,
                "nearby_signals": nearby_signals,
            }
        )

    return records


def entity_roles(insight: Any) -> dict[str, list[str]]:
    """Return entity names grouped by role, reusing Analyzer metadata."""

    if isinstance(insight, dict):
        records = insight.get("entity_mentions")
        if not isinstance(records, list):
            records = analyze_entity_importance(
                str(insight.get("title", "")), str(insight.get("summary", ""))
            )
    else:
        records = analyze_entity_importance(str(insight), "")

    grouped = {"primary": [], "supporting": [], "incidental": []}
    for record in records:
        if not isinstance(record, dict):
            continue
        role = str(record.get("entity_role", "incidental"))
        name = str(record.get("competitor", "")).strip()
        if role in grouped and name and name not in grouped[role]:
            grouped[role].append(name)
    return grouped
