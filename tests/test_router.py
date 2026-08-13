"""SmartPush source-router tests."""

from workflow.router import (
    AGENT_REACH_REDDIT,
    AGENT_REACH_RSS,
    AGENT_REACH_WEB,
    ANYSEARCH,
    classify_query,
    source_router_node,
)


def test_latest_saas_email_marketing_trends_uses_anysearch():
    decision = classify_query("Latest SaaS email marketing trends")

    assert decision["selected_sources"] == [ANYSEARCH]


def test_email_automation_complaints_use_agent_reach_reddit():
    decision = classify_query(
        "What are SaaS founders complaining about email automation?"
    )

    assert decision["selected_sources"] == [AGENT_REACH_REDDIT]


def test_klaviyo_competitors_analysis_uses_anysearch_and_agent_reach():
    decision = classify_query("Klaviyo competitors analysis")

    assert ANYSEARCH in decision["selected_sources"]
    assert AGENT_REACH_WEB in decision["selected_sources"]


def test_market_dynamics_and_user_discussions_use_multiple_sources():
    decision = classify_query(
        "Email marketing software market dynamics and user discussions"
    )

    assert decision["selected_sources"] == [
        ANYSEARCH,
        AGENT_REACH_RSS,
        AGENT_REACH_REDDIT,
    ]
    assert "market dynamics" in decision["signals"]["market_dynamic_triggers"]
    assert "discussions" in decision["signals"]["community_triggers"]


def test_router_consumes_structured_intent_facets():
    decision = source_router_node(
        {
            "translated_query": "Research a software company",
            "intent_facets": [
                "competitor_pricing",
                "community_intelligence",
            ],
        }
    )

    assert decision["selected_sources"] == [ANYSEARCH, AGENT_REACH_REDDIT]
    assert "structured intent facets" in decision["source_reasoning"]


def test_competitor_product_releases_use_search_web_and_rss_without_reddit():
    decision = classify_query(
        "latest competitor product releases and feature upgrades"
    )

    assert decision["selected_sources"] == [
        ANYSEARCH,
        AGENT_REACH_WEB,
        AGENT_REACH_RSS,
    ]
    assert decision["signals"]["product_update_triggers"]
    assert AGENT_REACH_REDDIT not in decision["selected_sources"]
