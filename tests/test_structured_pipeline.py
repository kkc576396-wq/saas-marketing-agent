"""Offline integration coverage for normalized research documents."""

import json

from workflow.normalizer import normalize_tool_result
from workflow.opportunity_classifier import opportunity_classifier_node
from workflow.output_contract import build_output_contract
from workflow.research_graph import analyzer_node, verifier_node
from workflow.scoring import scoring_node


def test_structured_reddit_document_reaches_output_contract():
    tool_result = {
        "iteration": 1,
        "source": "Agent-Reach Reddit",
        "queries": ["Klaviyo pricing complaints"],
        "content": json.dumps(
            [
                {
                    "channel": "reddit",
                    "query": "Klaviyo pricing complaints",
                    "content": """- id: post-456
  title: Klaviyo pricing complaints from Shopify merchants
  subreddit: r/Emailmarketing
  url: https://www.reddit.com/r/Emailmarketing/comments/post-456/example/
  selftext: Shopify merchants complain that Klaviyo pricing becomes expensive as their lists grow.
""",
                }
            ]
        ),
    }
    documents = normalize_tool_result(tool_result)
    state = {
        "topic": "Klaviyo pricing complaints",
        "documents": documents,
        "selected_sources": ["Agent-Reach Reddit"],
    }

    analyzed = analyzer_node(state)
    state.update(analyzed)
    verified = verifier_node(state)
    state.update(verified)
    state.update(scoring_node(state))
    state.update(opportunity_classifier_node(state))

    contract = build_output_contract(state)

    assert len(state["candidate_insights"]) == 1
    assert state["verification_results"][0]["passed"] is True
    assert contract["eligible_insights"][0]["title"] == "Klaviyo pricing complaints from Shopify merchants"
    assert contract["eligible_insights"][0]["sources"][0]["url"].endswith("/example")
    assert not contract["eligible_insights"][0]["title"].startswith("[")


def _document(
    document_id,
    title,
    summary,
    url,
    *,
    published_at="2026-07-01T00:00:00+00:00",
    source_bias="independent_source",
):
    return {
        "document_id": document_id,
        "source_type": "AnySearch",
        "title": title,
        "summary": summary,
        "url": url,
        "published_at": published_at,
        "retrieved_at": "2026-08-09T00:00:00+00:00",
        "date_status": "confirmed" if published_at else "unknown",
        "date_confidence": 1.0 if published_at else 0.0,
        "page_type": "article",
        "source_bias": source_bias,
        "raw_content": summary,
        "query": "email marketing competitor trends",
        "iteration": 1,
    }


def test_time_sensitive_verifier_rejects_stale_competitor_evidence():
    documents = [
        _document(
            "stale-1",
            "Klaviyo email marketing pricing update",
            "Klaviyo changed email marketing pricing for Shopify merchants and ecommerce brands.",
            "https://example.com/klaviyo-pricing-update",
            published_at="2023-01-01T00:00:00+00:00",
        )
    ]
    state = {
        "topic": "邮件营销竞品趋势",
        "translated_query": "Email marketing competitor trends",
        "research_date": "2026-08-09",
        "freshness_required": True,
        "documents": documents,
    }
    state.update(analyzer_node(state))

    verified = verifier_node(state)

    assert verified["insights"] == []
    assert "source_outside_freshness_window" in verified["verification_results"][0]["rejection_reasons"]


def test_verifier_hard_rejects_post_outside_explicit_30_day_window():
    documents = [
        _document(
            "reddit-old-1",
            "Email marketing workflow discussion",
            "A Shopify merchant discusses an email marketing automation workflow and asks for practical advice.",
            "https://www.reddit.com/r/Emailmarketing/comments/old/example/",
            published_at="2026-07-01T00:00:00+00:00",
            source_bias="community_source",
        )
    ]
    state = {
        "topic": "Reddit email marketing discussions",
        "translated_query": "Reddit email marketing discussions",
        "research_date": "2026-08-11",
        "freshness_required": True,
        "freshness_window_days": 30,
        "freshness_window_explicit": True,
        "documents": documents,
    }
    state.update(analyzer_node(state))

    verified = verifier_node(state)
    result = verified["verification_results"][0]

    assert verified["insights"] == []
    assert result["source_age_days"] == 41
    assert result["within_requested_freshness_window"] is False
    assert "source_outside_freshness_window" in result["rejection_reasons"]


def test_verifier_rejects_generic_topic_overlap_without_product_or_competitor_evidence():
    documents = [
        _document(
            "generic-1",
            "General software market report",
            "A general report about enterprise software spending without email platforms or ecommerce automation.",
            "https://example.com/general-software-report",
        )
    ]
    state = {
        "topic": "邮件营销竞品趋势",
        "translated_query": "Email marketing competitor trends",
        "research_date": "2026-08-09",
        "freshness_required": True,
        "documents": documents,
    }
    state.update(analyzer_node(state))

    verified = verifier_node(state)
    reasons = verified["verification_results"][0]["rejection_reasons"]

    assert verified["insights"] == []
    assert "missing_email_marketing_relevance" in reasons
    assert "missing_competitor_evidence" in reasons


def test_known_competitor_automatically_satisfies_product_relevance():
    documents = [
        _document(
            "known-competitor-1",
            "Klaviyo pricing complaints",
            (
                "A Shopify merchant reports that Klaviyo became expensive as "
                "the customer list grew and is now evaluating Omnisend instead."
            ),
            "https://www.reddit.com/r/shopify/comments/example/klaviyo_pricing",
            source_bias="community_source",
        )
    ]
    documents[0]["source_type"] = "Agent-Reach Reddit"
    state = {
        "topic": "邮件营销软件用户讨论",
        "translated_query": "Email marketing software user discussions",
        "research_date": "2026-08-09",
        "freshness_required": True,
        "documents": documents,
    }
    state.update(analyzer_node(state))

    verified = verifier_node(state)
    result = verified["verification_results"][0]

    assert result["passed"] is True
    assert result["matched_product_terms"] == []
    assert result["matched_competitors"] == ["klaviyo", "omnisend"]
    assert result["product_relevant"] is True
    assert result["product_relevance_basis"] == "known_competitor_entity"
    assert "missing_email_marketing_relevance" not in result["rejection_reasons"]


def test_ambiguous_brand_without_product_context_does_not_satisfy_relevance():
    documents = [
        _document(
            "ambiguous-drip-1",
            "Drip coffee market discussion",
            (
                "A detailed discussion of drip coffee equipment, brewing "
                "temperature, filters, beans, and cafe purchasing preferences."
            ),
            "https://example.com/drip-coffee-market",
        )
    ]
    state = {
        "topic": "邮件营销软件用户讨论",
        "translated_query": "Email marketing software user discussions",
        "research_date": "2026-08-09",
        "freshness_required": True,
        "documents": documents,
    }
    state.update(analyzer_node(state))

    verified = verifier_node(state)
    result = verified["verification_results"][0]

    assert result["matched_competitors"] == []
    assert result["product_relevant"] is False
    assert "missing_email_marketing_relevance" in result["rejection_reasons"]


def test_incidental_competitor_mention_does_not_satisfy_product_relevance():
    documents = [
        _document(
            "incidental-competitor-1",
            "How we built our customer data stack",
            (
                "The team connected records across its sales organization. "
                "Tools reviewed included Twilio, Pipedrive, Klaviyo, and "
                "several internal customer data services."
            ),
            "https://example.com/customer-data-stack",
        )
    ]
    state = {
        "topic": "邮件营销竞品趋势",
        "translated_query": "Email marketing competitor trends",
        "research_date": "2026-08-09",
        "freshness_required": True,
        "documents": documents,
    }
    state.update(analyzer_node(state))

    verified = verifier_node(state)
    result = verified["verification_results"][0]

    assert result["primary_competitors"] == []
    assert result["incidental_competitors"] == ["klaviyo"]
    assert result["product_relevant"] is False
    assert "missing_email_marketing_relevance" in result["rejection_reasons"]
    assert "missing_competitor_evidence" in result["rejection_reasons"]


def test_analyzer_merges_same_content_from_different_urls():
    summary = (
        "A Shopify merchant reports that Klaviyo pricing became expensive as "
        "the subscriber list grew and is now comparing alternative platforms."
    )
    documents = [
        _document(
            "crosspost-a",
            "Leaving Klaviyo after a price increase",
            summary,
            "https://www.reddit.com/r/shopify/comments/a/crosspost",
            source_bias="community_source",
        ),
        _document(
            "crosspost-b",
            "Klaviyo pricing migration",
            summary,
            "https://www.reddit.com/r/ecommerce/comments/b/crosspost",
            source_bias="community_source",
        ),
    ]

    analyzed = analyzer_node({"documents": documents})

    assert len(analyzed["candidate_insights"]) == 1
    candidate = analyzed["candidate_insights"][0]
    assert candidate["duplicate_count"] == 2
    assert len(candidate["duplicate_sources"]) == 1


def test_broad_vendor_claim_requires_independent_corroboration():
    vendor = _document(
        "vendor-1",
        "Omnisend email marketing industry trends report",
        "Omnisend reports major email marketing automation trends for ecommerce merchants in 2026.",
        "https://www.omnisend.com/reports/email-marketing-trends",
        source_bias="vendor_source",
    )
    state = {
        "topic": "邮件营销竞品趋势",
        "translated_query": "Email marketing competitor trends",
        "research_date": "2026-08-09",
        "freshness_required": True,
        "documents": [vendor],
    }
    state.update(analyzer_node(state))

    rejected = verifier_node(state)
    assert "insufficient_independent_sources" in rejected["verification_results"][0]["rejection_reasons"]

    independent = _document(
        "independent-1",
        "Independent Omnisend ecommerce automation analysis",
        "An independent analysis confirms Omnisend email marketing automation changes for ecommerce merchants.",
        "https://example.com/omnisend-analysis",
    )
    state["documents"] = [vendor, independent]
    state.update(analyzer_node(state))
    corroborated = verifier_node(state)

    vendor_result = next(
        item for item in corroborated["verification_results"] if item["insight"] == "vendor-1"
    )
    assert "insufficient_independent_sources" not in vendor_result["rejection_reasons"]
    assert vendor_result["independent_domain_count"] == 2


def test_community_experience_does_not_require_cross_domain_corroboration():
    community_post = _document(
        "reddit-experience-1",
        "My Klaviyo email marketing experience as the market changes",
        (
            "I run a Shopify store and found Klaviyo email marketing pricing "
            "too expensive as our list grew, so we started evaluating Omnisend."
        ),
        "https://www.reddit.com/r/shopify/comments/example/klaviyo_experience",
        source_bias="community_source",
    )
    community_post["source_type"] = "Agent-Reach Reddit"
    state = {
        "topic": "邮件营销市场动态和用户讨论",
        "translated_query": "Email marketing market dynamics and user discussions",
        "research_date": "2026-08-09",
        "freshness_required": True,
        "documents": [community_post],
    }
    state.update(analyzer_node(state))

    verified = verifier_node(state)
    result = verified["verification_results"][0]

    assert result["broad_claim"] is True
    assert result["claim_type"] == "community_observation"
    assert result["claim_scope"] == "single_user_experience"
    assert result["cross_domain_required"] is False
    assert "insufficient_independent_sources" not in result["rejection_reasons"]
    assert result["passed"] is True

    state.update(verified)
    state.update(scoring_node(state))
    state.update(opportunity_classifier_node(state))
    record = build_output_contract(state)["eligible_insights"][0]

    assert record["claim_type"] == "community_observation"
    assert record["claim_scope"] == "single_user_experience"
    assert record["verification"]["cross_domain_required"] is False
    assert record["usage_constraints"]


def test_product_update_verifier_rejects_review_and_keeps_dated_release():
    release = _document(
        "omnisend-release",
        "What’s new: Omnisend’s June 2026 updates",
        (
            "Omnisend shipped Dark Mode and new reporting capabilities for "
            "email marketing automation teams."
        ),
        "https://www.omnisend.com/blog/whats-new-june-2026",
    )
    review = _document(
        "attentive-review",
        "Attentive Review 2026",
        (
            "This email marketing review compares Attentive pricing and "
            "mentions new features for Shopify brands."
        ),
        "https://example.com/attentive-review",
    )
    state = {
        "topic": "竞品最新发布的功能升级",
        "translated_query": "Latest feature upgrades released by competitors",
        "intent_facets": [
            "product_update_research",
            "competitor_monitoring",
        ],
        "research_date": "2026-08-10",
        "freshness_required": True,
        "documents": [release, review],
    }
    state.update(analyzer_node(state))

    verified = verifier_node(state)
    by_id = {
        item["insight"]: item for item in verified["verification_results"]
    }

    assert by_id["omnisend-release"]["product_update_evidence"]["passed"] is True
    assert "missing_product_update_evidence" not in by_id[
        "omnisend-release"
    ]["rejection_reasons"]
    assert "product_update_review_or_comparison" in by_id[
        "attentive-review"
    ]["rejection_reasons"]
