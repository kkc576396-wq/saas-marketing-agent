"""Tests for Research Agent downstream output eligibility."""

from workflow.output_contract import build_output_contract


def _state(insights, *, passed=True, scores=None):
    scores = scores or [80.0 for _ in insights]
    return {
        "topic": "SmartPush research",
        "candidate_insights": list(insights),
        "insights": list(insights),
        "documents": [
            {
                "source": "AnySearch",
                "title": "SmartPush research source",
                "url": "https://example.com/source",
                "published_at": "2026-01-01",
                "content": "\n".join(str(insight) for insight in insights),
            }
        ],
        "verification_results": [
            {"insight": insight, "passed": passed, "confidence": 0.9}
            for insight in insights
        ],
        "insight_scores": [
            {"insight": insight, "total_score": score}
            for insight, score in zip(insights, scores)
        ],
        "opportunity_types": [
            {
                "insight": insight,
                "opportunity_type": "educational_content",
                "matched_signals": {},
                "recommended_channels": ["SEO blog"],
            }
            for insight in insights
        ],
        "recommended_channels": [
            {"insight": insight, "channels": ["SEO blog"]}
            for insight in insights
        ],
    }


def test_verified_high_score_insight_is_eligible():
    result = build_output_contract(_state(["Verified insight"], scores=[80]))

    assert len(result["eligible_insights"]) == 1
    assert result["eligible_insights"][0]["verification"]["passed"] is True
    assert result["eligible_insights"][0]["scoring"]["total_score"] == 80.0
    assert result["eligible_insights"][0]["sources"][0]["url"] == "https://example.com/source"


def test_unverified_insight_is_rejected():
    result = build_output_contract(_state(["Unverified insight"], passed=False, scores=[95]))

    assert result["eligible_insights"] == []
    assert result["alternative_insights"] == []
    assert result["rejected_insights"] == []


def test_score_below_sixty_is_rejected():
    result = build_output_contract(_state(["Low score insight"], scores=[59]))

    assert result["eligible_insights"] == []
    assert "score_below_60" in result["rejected_insights"][0]["rejection_reasons"]


def test_only_top_five_eligible_insights_are_returned():
    insights = [f"Insight {index}" for index in range(7)]
    result = build_output_contract(
        _state(insights, scores=[60, 70, 80, 90, 100, 110, 120])
    )

    eligible = result["eligible_insights"]
    assert len(eligible) == 5
    assert [item["scoring"]["total_score"] for item in eligible] == [
        120.0,
        110.0,
        100.0,
        90.0,
        80.0,
    ]
    assert len(result["rejected_insights"]) == 2
    assert all(
        "outside_top_5" in item["rejection_reasons"]
        for item in result["rejected_insights"]
    )


def test_structured_duplicate_insights_are_emitted_once():
    insight = {
        "insight_id": "reddit-post-123",
        "title": "Klaviyo alternatives for ecommerce",
        "summary": "Merchants compare Klaviyo with lower-cost alternatives.",
        "source_type": "Agent-Reach Reddit",
        "sources": [
            {
                "title": "Klaviyo alternatives for ecommerce",
                "url": "https://www.reddit.com/r/Emailmarketing/comments/post-123/example",
                "published_at": None,
            }
        ],
    }
    state = _state([insight, dict(insight)])
    state["verification_results"] = [
        {"insight": "reddit-post-123", "passed": True, "confidence": 0.85}
    ]
    state["insight_scores"] = [
        {"insight": "reddit-post-123", "total_score": 80}
    ]
    state["opportunity_types"] = [
        {
            "insight": "reddit-post-123",
            "opportunity_type": "competitor_opportunity",
            "matched_signals": {},
            "recommended_channels": ["competitor comparison page"],
        }
    ]
    state["recommended_channels"] = [
        {"insight": "reddit-post-123", "channels": ["competitor comparison page"]}
    ]

    result = build_output_contract(state)

    assert len(result["eligible_insights"]) == 1
    assert result["eligible_insights"][0]["insight_id"] == "reddit-post-123"


def test_raw_json_title_is_not_eligible():
    raw = {
        "insight_id": "raw-1",
        "title": '[{"channel":"reddit"}]',
        "summary": "Raw provider payload",
        "source_type": "Agent-Reach Reddit",
        "sources": [{"title": "raw", "url": "https://example.com/raw"}],
    }
    state = _state([raw])
    state["verification_results"] = [{"insight": "raw-1", "passed": True, "confidence": 0.9}]
    state["insight_scores"] = [{"insight": "raw-1", "total_score": 90}]
    state["opportunity_types"] = [{"insight": "raw-1", "opportunity_type": "educational_content"}]
    state["recommended_channels"] = [{"insight": "raw-1", "channels": ["SEO blog"]}]

    result = build_output_contract(state)

    assert result["eligible_insights"] == []
    assert result["alternative_insights"] == []


def test_only_five_highest_scoring_verified_alternatives_are_retained():
    insights = [f"Insight {index}" for index in range(12)]
    result = build_output_contract(
        _state(insights, scores=[120, 110, 100, 90, 80, 79, 78, 77, 76, 75, 74, 73])
    )

    alternatives = result["alternative_insights"]
    assert len(alternatives) == 5
    assert [item["scoring"]["total_score"] for item in alternatives] == [
        79.0,
        78.0,
        77.0,
        76.0,
        75.0,
    ]
    assert result["rejected_insights"] == alternatives


def test_product_update_requires_alignment_and_release_evidence():
    valid = {
        "insight_id": "valid-release",
        "title": "What’s new: Omnisend’s June 2026 updates",
        "summary": "Omnisend shipped Dark Mode and new reporting capabilities.",
        "source_type": "AnySearch",
        "sources": [
            {
                "title": "What’s new: Omnisend’s June 2026 updates",
                "url": "https://www.omnisend.com/blog/whats-new-june-2026",
                "published_at": "2026-06-01T00:00:00+00:00",
            }
        ],
    }
    low_alignment = {
        **valid,
        "insight_id": "low-alignment",
        "title": "Klaviyo migration discussion",
    }
    missing_evidence = {
        **valid,
        "insight_id": "missing-evidence",
        "title": "Attentive Review 2026",
    }
    state = _state([valid, low_alignment, missing_evidence])
    state["intent_facets"] = [
        "product_update_research",
        "competitor_monitoring",
    ]
    state["verification_results"] = [
        {"insight": item["insight_id"], "passed": True, "confidence": 0.9}
        for item in (valid, low_alignment, missing_evidence)
    ]
    state["insight_scores"] = [
        {
            "insight": "valid-release",
            "total_score": 75,
            "topic_alignment_score": 85,
            "product_update_evidence": {"passed": True, "rejection_reasons": []},
        },
        {
            "insight": "low-alignment",
            "total_score": 90,
            "topic_alignment_score": 39.5,
            "product_update_evidence": {"passed": True, "rejection_reasons": []},
        },
        {
            "insight": "missing-evidence",
            "total_score": 90,
            "topic_alignment_score": 85,
            "product_update_evidence": {
                "passed": False,
                "rejection_reasons": ["missing_product_update_evidence"],
            },
        },
    ]

    result = build_output_contract(state)

    assert [item["insight_id"] for item in result["eligible_insights"]] == [
        "valid-release"
    ]
    rejection_map = {
        item["insight_id"]: item["rejection_reasons"]
        for item in result["alternative_insights"]
    }
    assert "product_update_topic_alignment_below_60" in rejection_map[
        "low-alignment"
    ]
    assert "missing_product_update_evidence" in rejection_map[
        "missing-evidence"
    ]
