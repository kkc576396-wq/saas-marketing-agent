"""Tests for provider response normalization and de-duplication."""

import json

from workflow.normalizer import deduplicate_documents, normalize_tool_result


def _reddit_result(content: str):
    return {
        "iteration": 1,
        "source": "Agent-Reach Reddit",
        "queries": ["Klaviyo pricing complaints"],
        "content": json.dumps(
            [
                {
                    "channel": "reddit",
                    "query": "Klaviyo pricing complaints",
                    "content": content,
                }
            ]
        ),
    }


def test_reddit_yaml_becomes_structured_document():
    result = _reddit_result(
        """- id: post-123
  title: Klaviyo alternatives for ecommerce
  subreddit: r/Emailmarketing
  url: https://www.reddit.com/r/Emailmarketing/comments/post-123/example/
  selftext: Merchants are comparing Klaviyo with lower-cost alternatives.
  created_utc: 1773161445
"""
    )

    documents = normalize_tool_result(result)

    assert len(documents) == 1
    assert documents[0]["document_id"] == "reddit-post-123"
    assert documents[0]["title"] == "Klaviyo alternatives for ecommerce"
    assert "lower-cost alternatives" in documents[0]["summary"]
    assert documents[0]["url"].endswith("/example")
    assert documents[0]["published_at"]


def test_empty_reddit_payload_produces_no_documents():
    assert normalize_tool_result(_reddit_result("[]")) == []


def test_duplicate_documents_are_kept_once():
    result = _reddit_result(
        """- id: post-123
  title: Klaviyo alternatives for ecommerce
  url: https://www.reddit.com/r/Emailmarketing/comments/post-123/example/
  selftext: Merchants are comparing Klaviyo with alternatives.
"""
    )
    documents = normalize_tool_result(result)

    assert len(deduplicate_documents(documents + documents)) == 1


def test_anysearch_raw_markdown_becomes_atomic_results():
    result = {
        "iteration": 1,
        "source": "AnySearch",
        "queries": ["Klaviyo pricing complaints"],
        "content": """## Search Results (1 results, 100ms)
### 1. Klaviyo pricing overview
- **URL**: https://example.com/klaviyo-pricing
- Pricing and plans are compared for ecommerce merchants.
""",
    }

    documents = normalize_tool_result(result)

    assert len(documents) == 1
    assert documents[0]["title"] == "Klaviyo pricing overview"
    assert documents[0]["url"] == "https://example.com/klaviyo-pricing"
    assert "ecommerce merchants" in documents[0]["summary"]


def test_normalizer_records_real_date_retrieval_time_and_source_bias():
    result = {
        "iteration": 1,
        "source": "AnySearch",
        "queries": ["Omnisend 2026 report"],
        "content": """## Search Results (1 results, 100ms)
### 1. Omnisend 2026 ecommerce report
- **URL**: https://www.omnisend.com/resources/reports/2026-ecommerce-report
- **Published**: 2026-07-01
- The report discusses current ecommerce email marketing trends and automation.
""",
    }

    document = normalize_tool_result(result)[0]

    assert document["published_at"].startswith("2026-07-01")
    assert document["retrieved_at"]
    assert document["date_status"] == "confirmed"
    assert document["date_confidence"] == 1.0
    assert document["source_bias"] == "vendor_source"


def test_anysearch_date_is_inferred_from_search_snippet():
    result = {
        "iteration": 1,
        "source": "AnySearch",
        "queries": ["email marketing trends 2026"],
        "content": """## Search Results (1 results, 100ms)
### 1. Email Marketing Trends August 2026
- **URL**: https://example.com/email-trends-2026
- Researched current sources on 2026-07-31 and summarized email marketing automation trends.
""",
    }

    document = normalize_tool_result(result)[0]

    assert document["published_at"].startswith("2026-07-31")
    assert document["date_status"] == "confirmed"
    assert document["date_confidence"] == 0.9


def test_anysearch_result_linking_to_reddit_is_a_community_source():
    result = {
        "iteration": 1,
        "source": "AnySearch",
        "queries": ["Mailchimp alternatives Reddit"],
        "content": """## Search Results (1 results, 100ms)
### 1. Mailchimp alternatives for ecommerce
- **URL**: https://www.reddit.com/r/ecommerce/comments/example
- Published on 2026-07-20. Merchants compare Mailchimp alternatives for ecommerce.
""",
    }

    document = normalize_tool_result(result)[0]

    assert document["source_bias"] == "community_source"
