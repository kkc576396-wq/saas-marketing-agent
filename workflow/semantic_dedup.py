"""High-precision semantic deduplication for Analyzer candidates."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any


def _normalize_text(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\br/[a-z0-9_]+\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(value: Any) -> list[str]:
    return [token for token in _normalize_text(value).split() if len(token) > 1]


def _shingles(value: Any, size: int = 4) -> set[tuple[str, ...]]:
    tokens = _tokens(value)
    if len(tokens) < size:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def _jaccard(left: set[Any], right: set[Any]) -> float:
    if not left or not right:
        return 0.0
    return len(left.intersection(right)) / len(left.union(right))


def semantic_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Return title-and-summary similarity without external model calls."""

    left_title = _normalize_text(left.get("title"))
    right_title = _normalize_text(right.get("title"))
    title_sequence = SequenceMatcher(None, left_title, right_title).ratio()
    title_tokens = _jaccard(set(_tokens(left_title)), set(_tokens(right_title)))
    title_similarity = max(title_sequence, title_tokens)
    summary_similarity = _jaccard(
        _shingles(left.get("summary")), _shingles(right.get("summary"))
    )
    return round(0.3 * title_similarity + 0.7 * summary_similarity, 4)


def _is_semantic_duplicate(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_summary = _normalize_text(left.get("summary"))
    right_summary = _normalize_text(right.get("summary"))
    if min(len(_tokens(left_summary)), len(_tokens(right_summary))) < 8:
        return False
    if left_summary == right_summary:
        return True

    summary_similarity = _jaccard(
        _shingles(left_summary), _shingles(right_summary)
    )
    title_similarity = max(
        SequenceMatcher(
            None,
            _normalize_text(left.get("title")),
            _normalize_text(right.get("title")),
        ).ratio(),
        _jaccard(set(_tokens(left.get("title"))), set(_tokens(right.get("title")))),
    )
    combined = 0.3 * title_similarity + 0.7 * summary_similarity
    return summary_similarity >= 0.9 or (
        combined >= 0.85 and title_similarity >= 0.65
    )


def _representative_quality(document: dict[str, Any]) -> tuple[float, float, int]:
    source_type = str(document.get("source_type", "")).casefold()
    direct_source = 2.0 if "agent-reach" in source_type else 1.0
    date_confidence = float(document.get("date_confidence") or 0.0)
    summary_length = len(_tokens(document.get("summary")))
    return direct_source, date_confidence, summary_length


def _source_record(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_id": document.get("document_id"),
        "title": document.get("title"),
        "url": document.get("url"),
        "source_type": document.get("source_type", "unknown"),
        "published_at": document.get("published_at"),
    }


def semantic_deduplicate_documents(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Cluster near-identical content and keep one candidate per cluster.

    The strict threshold targets copied/cross-posted text. Independently
    worded user experiences remain separate evidence even when their topics
    are similar.
    """

    clusters: list[list[dict[str, Any]]] = []
    for document in documents:
        for cluster in clusters:
            if any(_is_semantic_duplicate(document, member) for member in cluster):
                cluster.append(document)
                break
        else:
            clusters.append([document])

    representatives: list[dict[str, Any]] = []
    for cluster in clusters:
        representative = max(cluster, key=_representative_quality)
        duplicates = [item for item in cluster if item is not representative]
        enriched = dict(representative)
        enriched["duplicate_count"] = len(cluster)
        enriched["duplicate_document_ids"] = [
            str(item.get("document_id", "")) for item in duplicates
        ]
        enriched["duplicate_sources"] = [_source_record(item) for item in duplicates]
        representatives.append(enriched)
    return representatives
