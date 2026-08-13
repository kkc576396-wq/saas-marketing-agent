import sqlite3
from datetime import UTC, datetime, timedelta

from workflow.memory_manager import MemoryManager, SQLiteMemoryStore
from workflow.memory_nodes import make_memory_commit_node, make_memory_prefetch_node
from workflow.memory_tools import MemoryTool


class KeywordEmbeddings:
    def embed(self, texts):
        vectors = []
        for text in texts:
            value = text.casefold()
            vectors.append(
                [
                    float("reddit" in value),
                    float("concise" in value or "short" in value),
                    float("competitor" in value or "klaviyo" in value),
                ]
            )
        return vectors


def _manager(tmp_path, *, max_records=5_000, brand_search=None):
    return MemoryManager(
        store=SQLiteMemoryStore(tmp_path / "memory.sqlite3"),
        embedding_client=KeywordEmbeddings(),
        brand_search=brand_search,
        max_records=max_records,
    )


def test_memory_add_merges_duplicates_and_searches_scoped_records(tmp_path):
    manager = _manager(tmp_path)
    first = manager.add(
        content="Prefer concise, practical Reddit replies.",
        memory_type="user_preference",
        user_id="user-a",
        source_refs=["user-message-1"],
        importance=0.9,
        confidence=1.0,
    )
    duplicate = manager.add(
        content="Prefer concise, practical Reddit replies.",
        memory_type="user_preference",
        user_id="user-a",
        source_refs=["user-message-2"],
        importance=0.8,
        confidence=0.9,
    )
    manager.add(
        content="Use a detailed competitor report structure.",
        memory_type="task_learning",
        user_id="user-b",
        source_refs=["run-2"],
    )

    assert first["operation"] == "created"
    assert duplicate["operation"] == "merged"
    assert duplicate["memory"]["memory_id"] == first["memory"]["memory_id"]
    assert duplicate["memory"]["source_refs"] == [
        "user-message-1",
        "user-message-2",
    ]

    result = manager.search(
        query="concise Reddit reply",
        user_id="user-a",
        memory_layers=["mid_term"],
        top_k=5,
    )
    assert result["result_count"] == 1
    assert result["results"][0]["memory_id"] == first["memory"]["memory_id"]
    assert result["results"][0]["not_fact_evidence"] is True


def test_long_term_add_is_candidate_and_search_reads_approved_brand_rag(tmp_path):
    def fake_brand_search(query, *, corpora=None, usage="public_content", top_k=5):
        return {
            "results": [
                {
                    "chunk_id": "brand-asset:01-positioning",
                    "document_id": "brand-asset",
                    "corpus": "brand",
                    "content": "Approved SmartPush positioning.",
                    "authority": "brand_approved",
                    "source_url": "https://example.test/brand",
                    "score": 0.91,
                }
            ]
        }

    manager = _manager(tmp_path, brand_search=fake_brand_search)
    candidate = manager.add(
        content="Candidate SmartPush lifecycle claim.",
        memory_type="approved_claim",
        memory_layer="long_term",
        status="active",
        source_refs=["product-doc-1"],
    )
    result = manager.search(
        query="SmartPush positioning",
        memory_layers=["long_term"],
        top_k=5,
    )

    assert candidate["memory"]["status"] == "candidate"
    assert candidate["requires_approval"] is True
    assert result["results"][0]["status"] == "approved"
    assert result["results"][0]["memory_id"] == "brand-asset:01-positioning"


def test_importance_and_time_forgetting_are_soft_and_recoverable(tmp_path):
    manager = _manager(tmp_path)
    low = manager.add(
        content="Low value temporary task note.",
        memory_type="task_learning",
        importance=0.1,
        source_refs=["run-low"],
    )["memory"]
    high = manager.add(
        content="Important user-approved campaign decision.",
        memory_type="campaign_decision",
        importance=0.9,
        source_refs=["run-high"],
    )["memory"]

    forgotten = MemoryTool(manager).execute(
        "forget",
        strategy="importance_based",
        threshold=0.2,
    )
    assert forgotten["forgotten_count"] == 1
    assert forgotten["memory_ids"] == [low["memory_id"]]
    remaining = manager.search(query="campaign decision", top_k=10)
    assert [item["memory_id"] for item in remaining["results"]] == [
        high["memory_id"]
    ]

    old = manager.add(
        content="Old unused task history.",
        memory_type="task_learning",
        importance=0.5,
        source_refs=["run-old"],
    )["memory"]
    old_time = (datetime.now(UTC) - timedelta(days=45)).isoformat()
    with sqlite3.connect(manager.store.path) as connection:
        connection.execute(
            "UPDATE memories SET created_at = ?, updated_at = ? WHERE memory_id = ?",
            (old_time, old_time, old["memory_id"]),
        )
    time_result = manager.forget(strategy="time_based", max_age_days=30)
    assert time_result["forgotten_count"] == 1
    assert time_result["memory_ids"] == [old["memory_id"]]


def test_capacity_forgetting_uses_configured_limit_and_threshold(tmp_path):
    manager = _manager(tmp_path, max_records=2)
    ids = []
    for index, importance in enumerate((0.1, 0.2, 0.8, 0.9), start=1):
        ids.append(
            manager.add(
                content=f"Distinct task memory number {index}.",
                memory_type="task_learning",
                importance=importance,
                source_refs=[f"run-{index}"],
            )["memory"]["memory_id"]
        )

    result = MemoryTool(manager).execute(
        "forget",
        strategy="capacity_based",
        threshold=0.3,
    )
    assert result["forgotten_count"] == 2
    assert result["memory_ids"] == ids[:2]
    assert result["remaining_over_capacity"] == 0


def test_memory_nodes_prefetch_and_commit_without_failing_the_main_run(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("MEMORY_ENABLED", "true")
    manager = _manager(tmp_path)
    manager.add(
        content="Prefer concise Reddit replies.",
        memory_type="user_preference",
        source_refs=["user-message"],
    )
    state = {
        "topic": "Write a concise Reddit reply",
        "raw_user_request": "Write a concise Reddit reply",
        "content_intent": {"deliverable_type": "reddit_reply"},
        "eligible_insights": [
            {
                "insight_id": "reddit-1",
                "sources": [{"url": "https://reddit.com/r/test/1"}],
            }
        ],
        "final_content": {"replies": [{"reply": "Example"}]},
        "executor_summary": "Generated one test reply.",
        "output_file": str(tmp_path / "run-1.json"),
    }

    prefetched = make_memory_prefetch_node(manager)(state)
    committed = make_memory_commit_node(manager)({**state, **prefetched})

    assert prefetched["memory_prefetch_status"] == "ready"
    assert prefetched["memory_prefetch_results"][0]["not_fact_evidence"] is True
    assert committed["memory_commit_status"] == "saved"
    assert len(committed["memory_commit_ids"]) == 1
