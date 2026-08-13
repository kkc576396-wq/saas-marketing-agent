"""High-precision candidate deduplication tests."""

from workflow.semantic_dedup import semantic_deduplicate_documents


def _document(document_id, title, summary, url, source_type="Agent-Reach Reddit"):
    return {
        "document_id": document_id,
        "title": title,
        "summary": summary,
        "url": url,
        "source_type": source_type,
        "published_at": "2026-08-01T00:00:00+00:00",
        "date_confidence": 1.0,
    }


def test_cross_subreddit_copy_is_one_candidate_cluster():
    summary = (
        "Our Shopify store moved away from Klaviyo because pricing increased "
        "as the subscriber list grew, and segmentation became harder to manage."
    )
    documents = [
        _document("a", "Leaving Klaviyo", summary, "https://reddit.com/r/shopify/a"),
        _document("b", "Klaviyo migration", summary, "https://reddit.com/r/ecommerce/b"),
    ]

    result = semantic_deduplicate_documents(documents)

    assert len(result) == 1
    assert result[0]["duplicate_count"] == 2
    assert len(result[0]["duplicate_sources"]) == 1


def test_independently_worded_complaints_remain_separate():
    documents = [
        _document(
            "a",
            "Klaviyo pricing problem",
            "Our monthly bill doubled after importing inactive profiles, so we are reviewing alternatives.",
            "https://reddit.com/r/shopify/a",
        ),
        _document(
            "b",
            "Klaviyo pricing problem",
            "A small apparel store found the segmentation limits confusing and needed outside implementation help.",
            "https://reddit.com/r/ecommerce/b",
        ),
    ]

    assert len(semantic_deduplicate_documents(documents)) == 2


def test_direct_source_wins_over_search_snippet_for_same_content():
    summary = (
        "A merchant reported that Mailchimp automation became difficult to use "
        "after the Shopify catalog and customer segments expanded substantially."
    )
    documents = [
        _document(
            "search",
            "Mailchimp merchant feedback",
            summary,
            "https://search.example/mailchimp",
            source_type="AnySearch",
        ),
        _document(
            "reddit",
            "Mailchimp automation feedback",
            summary,
            "https://reddit.com/r/shopify/reddit",
        ),
    ]

    result = semantic_deduplicate_documents(documents)

    assert len(result) == 1
    assert result[0]["document_id"] == "reddit"
    assert result[0]["duplicate_sources"][0]["document_id"] == "search"
