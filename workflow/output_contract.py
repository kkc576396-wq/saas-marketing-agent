"""Final Research Agent output contract and downstream eligibility rules."""

from __future__ import annotations

import re
from typing import Any

from .state import ResearchState


MIN_ELIGIBLE_SCORE = 60.0
MIN_PRODUCT_UPDATE_ALIGNMENT_SCORE = 60.0
MAX_ELIGIBLE_INSIGHTS = 5
MAX_ALTERNATIVE_INSIGHTS = 5


def _insight_key(value: Any) -> str:
    """Return a stable key for structured insights and legacy strings."""

    if isinstance(value, dict):
        return str(
            value.get("insight_id")
            or value.get("document_id")
            or value.get("title")
            or value.get("summary")
            or ""
        ).strip()
    return re.sub(r"^Iteration\s+[^:]+:\s*", "", str(value)).strip()


def _title_and_summary(insight: Any) -> tuple[str, str]:
    if isinstance(insight, dict):
        title = str(insight.get("title", "")).strip()
        summary = str(insight.get("summary", "")).strip()
        return title[:160], summary
    summary = _insight_key(insight)
    title = summary.splitlines()[0].strip() if summary else "Untitled insight"
    return title[:160], summary


def _urls_from_text(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"https?://[^\s)\]>]+", text)))


def _source_records(insight: Any, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build normalized source metadata, preferring candidate-level sources."""

    if isinstance(insight, dict):
        candidate_sources = insight.get("sources")
        if isinstance(candidate_sources, list):
            normalized = []
            for source in candidate_sources:
                if not isinstance(source, dict):
                    continue
                normalized.append(
                    {
                        "title": str(source.get("title", "")).strip(),
                        "url": source.get("url"),
                        "published_at": source.get("published_at"),
                    }
                )
            if normalized:
                return normalized

    key = _insight_key(insight)
    matching_documents = [
        document
        for document in documents
        if key and key in str(document.get("content", ""))
    ]
    if not matching_documents:
        matching_documents = documents

    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None]] = set()
    for document in matching_documents:
        content = str(document.get("raw_content") or document.get("content", ""))
        urls = document.get("urls") or _urls_from_text(content)
        url = urls[0] if urls else document.get("url")
        source_type = str(document.get("source_type") or document.get("source") or "unknown")
        title = str(document.get("title") or (document.get("queries") or [source_type])[0] or source_type)
        published_at = document.get("published_at")
        identity = (title, url)
        if identity in seen:
            continue
        seen.add(identity)
        sources.append({"title": title, "url": url, "published_at": published_at})

    return sources or [{"title": "Unknown source", "url": None, "published_at": None}]


def _source_type(insight: Any, documents: list[dict[str, Any]]) -> str:
    if isinstance(insight, dict) and insight.get("source_type"):
        return str(insight["source_type"])
    key = _insight_key(insight)
    matching = [
        str(document.get("source_type") or document.get("source") or "unknown")
        for document in documents
        if key and key in str(document.get("content", ""))
    ]
    types = list(dict.fromkeys(matching)) or [
        str(document.get("source_type") or document.get("source") or "unknown")
        for document in documents
    ]
    return ", ".join(dict.fromkeys(types)) if types else "unknown"


def _index_by_insight(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        reference = item.get("insight", item.get("insight_id", ""))
        key = _insight_key(reference)
        if key:
            indexed[key] = item
    return indexed


def _build_record(
    insight: Any,
    state: ResearchState,
    verification: dict[str, Any],
    scoring: dict[str, Any],
    opportunity: dict[str, Any],
    channels: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    title, summary = _title_and_summary(insight)
    insight_id = (
        str(insight.get("insight_id"))
        if isinstance(insight, dict) and insight.get("insight_id")
        else f"insight-{index:03d}"
    )
    sources = _source_records(insight, state.get("documents", []))
    seen_urls = {source.get("url") for source in sources if source.get("url")}
    for source in verification.get("supporting_sources", []):
        if not isinstance(source, dict) or not source.get("url") or source.get("url") in seen_urls:
            continue
        seen_urls.add(source.get("url"))
        sources.append(
            {
                "title": str(source.get("title", "")).strip(),
                "url": source.get("url"),
                "published_at": source.get("published_at"),
            }
        )

    return {
        "insight_id": insight_id,
        "title": title,
        "summary": summary,
        "entity_mentions": (
            list(insight.get("entity_mentions", []))
            if isinstance(insight, dict)
            else []
        ),
        "duplicate_count": (
            int(insight.get("duplicate_count", 1))
            if isinstance(insight, dict)
            else 1
        ),
        "duplicate_sources": (
            list(insight.get("duplicate_sources", []))
            if isinstance(insight, dict)
            else []
        ),
        "claim_type": verification.get("claim_type", "source_based_finding"),
        "claim_scope": verification.get("claim_scope", "source_level"),
        "usage_constraints": list(verification.get("usage_constraints", [])),
        "source_type": _source_type(insight, state.get("documents", [])),
        "sources": sources,
        "verification": {
            "passed": bool(verification.get("passed", False)),
            "confidence": float(verification.get("confidence", 0.0)),
            "relevance_score": float(verification.get("relevance_score", 0.0)),
            "evidence_score": float(verification.get("evidence_score", 0.0)),
            "source_quality_score": float(
                verification.get("source_quality_score", 0.0)
            ),
            "freshness_score": float(verification.get("freshness_score", 0.0)),
            "source_bias": verification.get("source_bias", "unknown"),
            "product_relevant": bool(
                verification.get("product_relevant", False)
            ),
            "product_relevance_basis": verification.get(
                "product_relevance_basis", "none"
            ),
            "primary_competitors": list(
                verification.get("primary_competitors", [])
            ),
            "supporting_competitors": list(
                verification.get("supporting_competitors", [])
            ),
            "incidental_competitors": list(
                verification.get("incidental_competitors", [])
            ),
            "claim_supported": bool(verification.get("claim_supported", False)),
            "cross_domain_required": bool(
                verification.get("cross_domain_required", False)
            ),
            "independent_domain_count": int(
                verification.get("independent_domain_count", 0)
            ),
            "rejection_reasons": list(
                verification.get("rejection_reasons", [])
            ),
        },
        "scoring": {
            "topic_alignment_score": float(scoring.get("topic_alignment_score", 0.0)),
            "business_relevance_score": float(scoring.get("business_relevance_score", 0.0)),
            "customer_pain_score": float(scoring.get("customer_pain_score", 0.0)),
            "content_opportunity_score": float(scoring.get("content_opportunity_score", 0.0)),
            "freshness_score": float(scoring.get("freshness_score", 0.0)),
            "total_score": float(scoring.get("total_score", 0.0)),
            "product_update_evidence": dict(
                scoring.get(
                    "product_update_evidence",
                    verification.get("product_update_evidence", {}),
                )
            ),
        },
        "opportunity_type": opportunity.get("opportunity_type", "unclassified"),
        "matched_signals": opportunity.get("matched_signals", {}),
        "recommended_channels": channels.get("channels", opportunity.get("recommended_channels", [])),
    }


def _quality_valid(record: dict[str, Any], insight: Any) -> bool:
    """Prevent raw provider payloads from becoming downstream insights."""

    if not isinstance(insight, dict):
        return True  # backwards-compatible support for existing unit fixtures
    if not record["title"] or not record["summary"]:
        return False
    if record["title"].lstrip().startswith(("[", "{")) or record["summary"].lstrip().startswith(("[", "{")):
        return False
    return any(source.get("url") for source in record["sources"])


def _downstream_rejection_reasons(
    record: dict[str, Any], state: ResearchState
) -> list[str]:
    """Return hard downstream gates in addition to the global score threshold."""

    reasons: list[str] = []
    if record["scoring"]["total_score"] < MIN_ELIGIBLE_SCORE:
        reasons.append("score_below_60")

    if "product_update_research" in state.get("intent_facets", []):
        if (
            record["scoring"]["topic_alignment_score"]
            < MIN_PRODUCT_UPDATE_ALIGNMENT_SCORE
        ):
            reasons.append("product_update_topic_alignment_below_60")
        evidence = record["scoring"].get("product_update_evidence", {})
        for reason in evidence.get("rejection_reasons", []):
            if reason not in reasons:
                reasons.append(reason)
        if not evidence:
            reasons.append("missing_product_update_evidence")
    return reasons


def build_output_contract(state: ResearchState) -> dict[str, list[dict[str, Any]]]:
    """Build the Top 5 eligible insights and up to five verified alternatives."""

    verifications = _index_by_insight(state.get("verification_results", []))
    scores = _index_by_insight(state.get("insight_scores", []))
    opportunities = _index_by_insight(state.get("opportunity_types", []))
    channels = _index_by_insight(
        [
            {"insight": item.get("insight", ""), "channels": item.get("channels", [])}
            for item in state.get("recommended_channels", [])
        ]
    )

    record_entries: list[tuple[dict[str, Any], Any]] = []
    candidate_insights = state.get("candidate_insights", state.get("insights", []))
    seen_keys: set[str] = set()
    for index, insight in enumerate(candidate_insights, start=1):
        key = _insight_key(insight)
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        record_entries.append(
            (
                _build_record(
                    insight,
                    state,
                    verifications.get(key, {}),
                    scores.get(key, {}),
                    opportunities.get(key, {}),
                    channels.get(key, {}),
                    index,
                ),
                insight,
            )
        )

    eligible_candidates = [
        record
        for record, insight in record_entries
        if record["verification"]["passed"]
        and not _downstream_rejection_reasons(record, state)
        and _quality_valid(record, insight)
    ]
    eligible_candidates.sort(key=lambda record: record["scoring"]["total_score"], reverse=True)
    eligible = eligible_candidates[:MAX_ELIGIBLE_INSIGHTS]
    eligible_ids = {record["insight_id"] for record in eligible}

    alternatives: list[dict[str, Any]] = []
    for record, insight in record_entries:
        if record["insight_id"] in eligible_ids:
            continue
        if not record["verification"]["passed"] or not _quality_valid(record, insight):
            continue
        reasons = _downstream_rejection_reasons(record, state)
        if not reasons:
            reasons.append("outside_top_5")
        record["rejection_reasons"] = reasons
        alternatives.append(record)

    alternatives.sort(
        key=lambda record: record["scoring"]["total_score"],
        reverse=True,
    )
    alternatives = alternatives[:MAX_ALTERNATIVE_INSIGHTS]

    return {
        "eligible_insights": eligible,
        "alternative_insights": alternatives,
        # Backwards-compatible alias. This collection is now intentionally
        # bounded and contains verified alternatives only.
        "rejected_insights": alternatives,
    }
