"""Tests for shared SmartPush competitor entity knowledge."""

from workflow.domain_context import (
    competitor_entity_records,
    detect_intent_facets,
    match_known_competitors,
    validate_intent_facets,
)


def test_high_confidence_competitors_match_without_category_words():
    assert match_known_competitors("Klaviyo pricing is getting expensive") == [
        "klaviyo"
    ]
    assert match_known_competitors("Switching from Mailchimp") == ["mailchimp"]
    assert match_known_competitors("Omnisend alternatives") == ["omnisend"]


def test_ambiguous_competitor_requires_product_context():
    assert match_known_competitors("Drip coffee market discussion") == []
    assert match_known_competitors("Drip email automation for Shopify") == ["drip"]


def test_entity_records_include_product_category():
    records = competitor_entity_records("Klaviyo and Omnisend pricing")

    assert [record["canonical_name"] for record in records] == [
        "klaviyo",
        "omnisend",
    ]
    assert all(record["product_category"] for record in records)


def test_intent_facets_have_deterministic_fallback_and_enum_validation():
    facets = detect_intent_facets(
        "Reddit discussions about Klaviyo pricing and alternatives"
    )

    assert "community_intelligence" in facets
    assert "competitor_pricing" in facets
    assert "alternative_research" in facets
    assert validate_intent_facets(
        ["community_intelligence", "invented_facet"]
    ) == ["community_intelligence"]


def test_chinese_competitor_release_intent_detects_product_updates_and_trends():
    facets = detect_intent_facets("竞品最新发布的产业动态、功能升级")

    assert "competitor_monitoring" in facets
    assert "market_intelligence" in facets
    assert "trend_research" in facets
    assert "product_update_research" in facets
    assert validate_intent_facets(["product_update_research"]) == [
        "product_update_research"
    ]
