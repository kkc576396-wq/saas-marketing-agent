"""Tests for automatic translation and platform-specific query rewriting."""

import workflow.query_rewriter as query_rewriter

from workflow.query_rewriter import (
    is_valid_reddit_query,
    parse_explicit_freshness_window_days,
    query_rewriter_node,
    rewrite_query,
)


def test_explicit_freshness_window_parser_supports_chinese_and_english():
    assert parse_explicit_freshness_window_days("搜索30天内的 Reddit 帖子") == 30
    assert parse_explicit_freshness_window_days("posts from the last 2 weeks") == 14
    assert parse_explicit_freshness_window_days("within 3 months") == 90


def test_rewriter_forces_english_content_and_preserves_exact_30_day_window(monkeypatch):
    monkeypatch.setattr(
        "workflow.query_rewriter._model_rewrite",
        lambda query: {
            "research_objective": "Recent Reddit email marketing discussions",
            "translated_query": "Reddit email marketing discussions last 30 days",
            "source_queries": {
                ANYSEARCH: ["Reddit email marketing discussions"],
                AGENT_REACH_REDDIT: ["email marketing discussions"],
            },
            "content_intent": {
                "requested": True,
                "deliverable_type": "reddit_reply",
                "deliverable_description": "Replies to recent Reddit discussions",
                "request_evidence": "生成回复",
                "platform": "reddit",
                "language": "Chinese",
                "requires_post_selection": True,
                "requires_brand_rag": False,
            },
        },
    )

    result = rewrite_query(
        "搜索30天内Reddit关于邮件营销的讨论帖子，生成回复",
        research_date="2026-08-11",
    )

    assert result["freshness_window_days"] == 30
    assert result["freshness_required"] is True
    assert result["freshness_window_explicit"] is True
    assert result["content_intent"]["language"] == "English"
    assert any(
        "2026-07-12" in query for query in result["source_queries"][ANYSEARCH]
    )
from workflow.router import (
    AGENT_REACH_REDDIT,
    AGENT_REACH_RSS,
    AGENT_REACH_WEB,
    ANYSEARCH,
    source_router_node,
)


def test_chinese_query_is_translated_without_user_instruction(monkeypatch):
    monkeypatch.setattr("workflow.query_rewriter._model_rewrite", lambda query: None)

    result = rewrite_query("邮件营销企业最新动态")

    assert "email marketing" in result["translated_query"]
    assert "latest" in result["translated_query"]
    assert result["source_queries"][ANYSEARCH]
    assert result["source_queries"][AGENT_REACH_REDDIT]
    assert all(len(query) <= 160 for query in result["source_queries"][AGENT_REACH_REDDIT])


def test_query_rewriter_node_preserves_original_and_sets_english_queries(monkeypatch):
    monkeypatch.setattr("workflow.query_rewriter._model_rewrite", lambda query: None)

    result = query_rewriter_node({"topic": "邮件营销企业最新动态"})

    assert result["original_query"] == "邮件营销企业最新动态"
    assert "email marketing" in result["translated_query"]
    assert result["search_queries"] == result["source_queries"][ANYSEARCH]
    assert "intent_facets" in result
    assert "detected_entities" in result


def test_router_uses_translated_intent_before_platform_queries():
    result = source_router_node(
        {
            "topic": "中文原始查询",
            "translated_query": "Klaviyo pricing complaints",
            "search_queries": ["generic market query"],
        }
    )

    assert ANYSEARCH in result["selected_sources"]
    assert AGENT_REACH_REDDIT in result["selected_sources"]


def test_competitor_discussion_is_translated_and_routes_to_reddit(monkeypatch):
    monkeypatch.setattr("workflow.query_rewriter._model_rewrite", lambda query: None)

    rewritten = rewrite_query("关于竞品动态的讨论")
    routed = source_router_node(
        {
            "topic": "关于竞品动态的讨论",
            "translated_query": rewritten["translated_query"],
            "search_queries": rewritten["source_queries"][ANYSEARCH],
        }
    )

    assert "competitor" in rewritten["translated_query"]
    assert ANYSEARCH in routed["selected_sources"]
    assert AGENT_REACH_REDDIT in routed["selected_sources"]


def test_llm_reddit_queries_are_validated_but_not_limited_to_templates(monkeypatch):
    monkeypatch.setenv("QUERY_REWRITER_USE_LLM", "1")
    monkeypatch.setattr(
        "workflow.query_rewriter._model_rewrite",
        lambda query: {
            "translated_query": "Klaviyo deliverability complaints",
            "detected_entities": [
                {"name": "Invented Brand", "entity_type": "known_competitor"}
            ],
            "intent_facets": [
                "community_intelligence",
                "invented_facet",
            ],
            "source_queries": {
                ANYSEARCH: ["Klaviyo deliverability product updates"],
                AGENT_REACH_REDDIT: [
                    "deliverability surcharge merchants",
                    "Why are Shopify merchants leaving Klaviyo?",
                    "Klaviyo pricing complaint",
                ],
            },
            "hyde_terms": ["deliverability", "merchant complaints"],
            "reasoning": "LLM generated intent-specific queries.",
            "content_intent": {
                "requested": False,
                "deliverable_type": None,
                "deliverable_description": "",
                "request_evidence": "",
                "platform": "",
            },
        },
    )

    result = rewrite_query("Klaviyo deliverability complaints")
    reddit_queries = result["source_queries"][AGENT_REACH_REDDIT]

    assert "deliverability surcharge merchants" in reddit_queries
    assert "Why are Shopify merchants leaving Klaviyo?" not in reddit_queries
    assert all(is_valid_reddit_query(query) for query in reddit_queries)
    assert len(reddit_queries) == 5
    assert result["detected_entities"] == [
        {
            "name": "klaviyo",
            "canonical_name": "klaviyo",
            "entity_type": "known_competitor",
            "product_category": "email_marketing_automation",
        }
    ]
    assert "community_intelligence" in result["intent_facets"]
    assert "invented_facet" not in result["intent_facets"]


def test_rewriter_separates_research_objective_and_content_intent(monkeypatch):
    monkeypatch.setattr(
        "workflow.query_rewriter._model_rewrite",
        lambda query: {
            "research_objective": (
                "Recent Shopify merchant opinions about Klaviyo pricing"
            ),
            "translated_query": (
                "Recent Shopify merchant opinions about Klaviyo pricing"
            ),
            "source_queries": {
                ANYSEARCH: ["Klaviyo pricing Shopify merchant opinions"],
                AGENT_REACH_REDDIT: ["Klaviyo pricing complaints"],
            },
            "content_intent": {
                "requested": True,
                "deliverable_type": "reddit_reply",
                "deliverable_description": "Natural English Reddit replies",
                "request_evidence": "筛选 Reddit 帖子并生成自然的英文回复",
                "platform": "reddit",
                "language": "English",
                "audience": "Shopify merchants",
                "tone": ["natural", "non-promotional"],
                "constraints": ["Do not hard-sell SmartPush"],
                "requires_post_selection": True,
                "requires_brand_rag": True,
            },
        },
    )

    result = query_rewriter_node(
        {
            "topic": (
                "调查 Shopify 商户对 Klaviyo 定价的看法，筛选 Reddit 帖子并生成"
                "自然的英文回复，不要硬推销 SmartPush"
            )
        }
    )

    assert result["topic"] == (
        "Recent Shopify merchant opinions about Klaviyo pricing"
    )
    assert "生成" in result["raw_user_request"]
    assert result["content_intent"]["requested"] is True
    assert result["content_intent"]["deliverable_type"] == "reddit_reply"
    assert result["content_intent"]["constraints"] == [
        "Do not hard-sell SmartPush"
    ]
    assert result["content_intent"]["requires_content_generation"] is True


def test_rewriter_repairs_internally_conflicting_content_intent(monkeypatch):
    original = "调查竞品30天内功能升级，生成竞品报告"
    monkeypatch.setattr(
        "workflow.query_rewriter._model_rewrite",
        lambda query: {
            "research_objective": "Competitor feature upgrades in the last 30 days",
            "translated_query": "Competitor feature upgrades in the last 30 days",
            "source_queries": {},
            "content_intent": {
                "requested": False,
                "deliverable_type": None,
                "deliverable_description": "",
                "request_evidence": "",
                "platform": "report",
            },
        },
    )
    monkeypatch.setattr(
        "workflow.query_rewriter._repair_content_intent",
        lambda request, previous, violations: {
            "requested": True,
            "deliverable_type": "competitor_report",
            "deliverable_description": "An English competitor feature-update report",
            "request_evidence": "生成竞品报告",
            "platform": "report",
            "language": "English",
            "audience": "internal stakeholders",
            "tone": ["analytical"],
            "constraints": ["last 30 days only"],
            "requires_post_selection": False,
            "requires_brand_rag": True,
        },
    )

    result = rewrite_query(original, research_date="2026-08-11")

    assert result["content_intent"]["requested"] is True
    assert result["content_intent"]["deliverable_type"] == "competitor_report"
    assert result["content_intent"]["request_evidence"] == "生成竞品报告"
    assert result["content_intent"]["requires_content_generation"] is True


def test_rewriter_accepts_semantic_deliverable_without_rule_vocabulary(monkeypatch):
    original = "梳理同行最近的产品变化，形成一份供管理层决策的材料"
    monkeypatch.setattr(
        "workflow.query_rewriter._model_rewrite",
        lambda query: {
            "research_objective": "Recent product changes among peer platforms",
            "translated_query": "Recent product changes among peer platforms",
            "source_queries": {},
            "content_intent": {
                "requested": True,
                "deliverable_type": "competitor_report",
                "deliverable_description": "Decision material about peer product changes",
                "request_evidence": "形成一份供管理层决策的材料",
                "platform": "report",
                "language": "English",
                "audience": "management",
                "tone": ["analytical"],
                "constraints": [],
                "requires_post_selection": False,
                "requires_brand_rag": True,
            },
        },
    )

    result = rewrite_query(original, research_date="2026-08-11")

    assert result["content_intent"]["deliverable_type"] == "competitor_report"
    assert result["content_intent"]["request_evidence"] in original


def test_fallback_outputs_entities_and_intent_facets(monkeypatch):
    monkeypatch.setattr("workflow.query_rewriter._model_rewrite", lambda query: None)

    result = rewrite_query("Reddit 上关于 Klaviyo 定价和替代方案的讨论")

    assert "pricing" in result["translated_query"].casefold()
    assert result["detected_entities"][0]["canonical_name"] == "klaviyo"
    assert "community_intelligence" in result["intent_facets"]
    assert "competitor_pricing" in result["intent_facets"]
    assert "alternative_research" in result["intent_facets"]


def test_model_loader_passes_openai_compatible_base_url(monkeypatch):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "qwen3.7-plus")
    monkeypatch.setenv("QUERY_REWRITER_MODEL", "glm-5.1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/compatible-mode/v1")
    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChatOpenAI)

    query_rewriter._load_openai_rewriter()

    assert captured["model"] == "glm-5.1"
    assert captured["base_url"] == "https://example.test/compatible-mode/v1"
    assert captured["timeout"] == 15.0
    assert captured["max_retries"] == 0
    assert captured["extra_body"] == {"enable_thinking": False}
    assert captured["model_kwargs"] == {
        "response_format": {"type": "json_object"}
    }


def test_time_sensitive_rewrite_replaces_stale_generated_year(monkeypatch):
    monkeypatch.setattr(
        "workflow.query_rewriter._model_rewrite",
        lambda query: {
            "translated_query": "Email marketing competitor trends",
            "source_queries": {
                ANYSEARCH: ["Email marketing competitor trends 2024"],
            },
            "reasoning": "Generated market trend query.",
            "content_intent": {
                "requested": False,
                "deliverable_type": None,
                "deliverable_description": "",
                "request_evidence": "",
                "platform": "",
            },
        },
    )

    result = rewrite_query(
        "邮件营销竞品趋势",
        research_date="2026-08-09",
        freshness_window_days=365,
        freshness_required=True,
    )

    assert all("2024" not in query for query in result["source_queries"][ANYSEARCH])
    assert any("2026" in query for query in result["source_queries"][ANYSEARCH])
    assert result["freshness_required"] is True


def test_explicit_historical_year_is_preserved(monkeypatch):
    monkeypatch.setattr("workflow.query_rewriter._model_rewrite", lambda query: None)

    result = rewrite_query(
        "email marketing trends 2024",
        research_date="2026-08-09",
        freshness_required=True,
    )

    assert "2024" in result["source_queries"][ANYSEARCH][0]


def test_chinese_competitor_release_query_preserves_update_intent(monkeypatch):
    monkeypatch.setattr("workflow.query_rewriter._model_rewrite", lambda query: None)

    rewritten = rewrite_query(
        "竞品最新发布的产业动态、功能升级",
        research_date="2026-08-10",
    )
    routed = source_router_node(rewritten)

    assert "industry developments" in rewritten["translated_query"].casefold()
    assert "feature upgrades" in rewritten["translated_query"].casefold()
    assert "、" not in rewritten["translated_query"]
    assert "product_update_research" in rewritten["intent_facets"]
    assert "trend_research" in rewritten["intent_facets"]
    assert rewritten["source_queries"][ANYSEARCH][:3] == [
        "klaviyo latest product releases and feature updates 2026",
        "omnisend latest product releases and feature updates 2026",
        "mailchimp latest product releases and feature updates 2026",
    ]
    assert set(routed["selected_sources"]) == {
        ANYSEARCH,
        AGENT_REACH_WEB,
        AGENT_REACH_RSS,
    }
    assert AGENT_REACH_REDDIT not in routed["selected_sources"]
