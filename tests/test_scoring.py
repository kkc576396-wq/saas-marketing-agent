"""SmartPush insight-scoring tests."""

from workflow.scoring import detect_product_update_evidence, score_insight


def test_klaviyo_pricing_complaints_from_shopify_merchants_score_high():
    result = score_insight(
        "Klaviyo pricing complaints from Shopify merchants",
        topic="Klaviyo pricing complaints from Shopify merchants",
        source="Agent-Reach Reddit",
    )

    assert result["total_score"] >= 70
    assert result["business_relevance_score"] >= 60
    assert result["customer_pain_score"] >= 75
    assert result["topic_alignment_score"] >= 90


def test_general_ai_news_scores_lower_for_smartpush():
    result = score_insight(
        "General AI news",
        topic="General AI news",
    )

    assert result["total_score"] < 50


def test_freshness_uses_actual_publication_date_not_freshness_words():
    base = {
        "insight_id": "dated-insight",
        "title": "Klaviyo email marketing pricing update",
        "summary": "Klaviyo changed email marketing pricing for ecommerce merchants.",
        "source_type": "AnySearch",
        "sources": [
            {
                "url": "https://example.com/klaviyo-update",
                "date_status": "confirmed",
            }
        ],
    }
    fresh = dict(base)
    fresh["sources"] = [{**base["sources"][0], "published_at": "2026-08-01T00:00:00+00:00"}]
    stale = dict(base)
    stale["sources"] = [{**base["sources"][0], "published_at": "2023-01-01T00:00:00+00:00"}]

    fresh_result = score_insight(
        fresh,
        topic="email marketing competitor trends",
        research_date="2026-08-09",
    )
    stale_result = score_insight(
        stale,
        topic="email marketing competitor trends",
        research_date="2026-08-09",
    )

    assert fresh_result["freshness_score"] == 100.0
    assert stale_result["freshness_score"] == 0.0
    assert fresh_result["total_score"] > stale_result["total_score"]


def test_scoring_uses_shared_competitor_entity_knowledge():
    result = score_insight(
        "Klaviyo pricing increased as our list grew",
        topic="merchant software discussion",
    )

    assert result["matched_signals"]["competitors"] == ["klaviyo"]
    assert "known_competitor:klaviyo" in result["matched_signals"]["relevance"]


def test_scoring_does_not_treat_drip_coffee_as_a_competitor():
    result = score_insight(
        "Drip coffee equipment and brewing temperature",
        topic="coffee market discussion",
    )

    assert result["matched_signals"]["competitors"] == []


def test_primary_competitor_bonus_is_lower_and_capped():
    insight = {
        "insight_id": "competitor-only",
        "title": "Klaviyo company profile",
        "summary": "A short Klaviyo company overview.",
        "entity_mentions": [
            {
                "competitor": "klaviyo",
                "entity_role": "primary",
                "importance_score": 100,
            }
        ],
    }

    result = score_insight(
        insight,
        topic="Klaviyo competitor analysis",
        intent_facets=["competitor_monitoring"],
    )

    assert result["matched_signals"]["entity_relevance_bonus"] == 15.0
    assert result["matched_signals"]["entity_content_bonus"] == 5.0
    assert result["business_relevance_score"] == 15.0
    assert result["content_opportunity_score"] == 5.0


def test_topic_alignment_prioritizes_market_trend_over_brand_pricing_page():
    trend = {
        "insight_id": "market-trend",
        "title": "Email automation industry trends and adoption benchmark",
        "summary": (
            "An industry report tracks emerging segmentation automation, "
            "market growth, and lifecycle marketing adoption."
        ),
        "sources": [{"published_at": "2026-08-01T00:00:00+00:00"}],
    }
    pricing = {
        "insight_id": "brand-pricing",
        "title": "Klaviyo pricing guide",
        "summary": (
            "Klaviyo email marketing plans for Shopify merchants, including "
            "features, segmentation, flows, and current pricing."
        ),
        "entity_mentions": [
            {
                "competitor": "klaviyo",
                "entity_role": "primary",
                "importance_score": 100,
            }
        ],
        "sources": [{"published_at": "2026-08-01T00:00:00+00:00"}],
    }

    trend_result = score_insight(
        trend,
        topic="email marketing market trends",
        research_date="2026-08-10",
        intent_facets=["market_intelligence", "trend_research"],
    )
    pricing_result = score_insight(
        pricing,
        topic="email marketing market trends",
        research_date="2026-08-10",
        intent_facets=["market_intelligence", "trend_research"],
    )

    assert trend_result["topic_alignment_score"] >= 80
    assert pricing_result["topic_alignment_score"] < 40
    assert trend_result["total_score"] > pricing_result["total_score"]


def test_product_release_intent_ranks_feature_launch_over_alternative_page():
    launch = {
        "insight_id": "feature-launch",
        "title": "Klaviyo launches new AI segmentation feature",
        "summary": "The product update rollout adds a new automated segmentation feature.",
        "entity_mentions": [
            {"competitor": "klaviyo", "entity_role": "primary"}
        ],
    }
    alternative = {
        "insight_id": "alternative-list",
        "title": "Klaviyo alternatives for ecommerce",
        "summary": "A comparison of Klaviyo, Omnisend, and Mailchimp pricing.",
        "entity_mentions": [
            {"competitor": "klaviyo", "entity_role": "primary"}
        ],
    }
    facets = ["competitor_monitoring", "product_update_research"]
    topic = "latest competitor product releases and feature upgrades"

    launch_result = score_insight(launch, topic=topic, intent_facets=facets)
    alternative_result = score_insight(
        alternative,
        topic=topic,
        intent_facets=facets,
    )

    assert launch_result["topic_alignment_score"] >= 65
    assert alternative_result["topic_alignment_score"] < 40
    assert launch_result["total_score"] > alternative_result["total_score"]


def test_product_update_evidence_requires_dated_release_not_generic_review():
    release = {
        "title": "What’s new: Omnisend’s June 2026 updates",
        "summary": "Omnisend shipped Dark Mode and new reporting capabilities.",
        "sources": [
            {
                "url": "https://www.omnisend.com/blog/whats-new-june-2026",
                "published_at": "2026-06-01T00:00:00+00:00",
            }
        ],
    }
    review = {
        "title": "Best Email Marketing Tools in 2026: honest breakdown",
        "summary": "A review comparing platforms and mentioning their new features.",
        "sources": [
            {
                "url": "https://example.com/best-email-tools",
                "published_at": "2026-07-01T00:00:00+00:00",
            }
        ],
    }

    assert detect_product_update_evidence(release)["passed"] is True
    review_evidence = detect_product_update_evidence(review)
    assert review_evidence["passed"] is False
    assert "product_update_review_or_comparison" in review_evidence["rejection_reasons"]


def test_whats_new_signal_raises_product_update_topic_alignment():
    insight = {
        "title": "What’s new: Omnisend’s June 2026 updates",
        "summary": "Omnisend shipped Dark Mode and new reporting capabilities.",
        "entity_mentions": [
            {"competitor": "omnisend", "entity_role": "primary"}
        ],
        "sources": [
            {
                "url": "https://www.omnisend.com/blog/whats-new-june-2026",
                "published_at": "2026-06-01T00:00:00+00:00",
            }
        ],
    }

    result = score_insight(
        insight,
        topic="latest feature upgrades released by competitors",
        intent_facets=["product_update_research", "competitor_monitoring"],
        research_date="2026-08-10",
    )

    assert result["topic_alignment_score"] >= 60
    assert result["product_update_evidence"]["passed"] is True
