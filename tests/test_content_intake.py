import json

from langchain_core.messages import AIMessage

from workflow.content_intake import content_intake_node, load_research_output


def _insight(insight_id, score=80, passed=True, source_type="AnySearch"):
    return {
        "insight_id": insight_id,
        "title": f"{insight_id} title",
        "summary": f"{insight_id} summary",
        "source_type": source_type,
        "sources": [{"title": "Source", "url": f"https://example.com/{insight_id}"}],
        "verification": {"passed": passed, "confidence": 0.9},
        "scoring": {"total_score": score},
        "usage_constraints": ["Use as an individual observation."] if source_type == "Reddit" else [],
    }


def test_content_intake_fallback_only_selects_verified_scored_candidates(monkeypatch):
    monkeypatch.setattr("workflow.content_intake._load_model", lambda: None)
    result = content_intake_node({
        "research_output": {
            "eligible_insights": [_insight("good", 82), _insight("low", 59), _insight("bad", 95, False)],
            "alternative_insights": [_insight("alternative", 78)],
        },
        "content_goal": "Write a practical retention guide",
        "audience": "Shopify SMB merchants",
        "channel": "blog",
        "language": "English",
    })
    assert result["selected_insight_ids"] == ["good", "alternative"]
    assert result["requires_more_research"] is False


def test_model_selection_is_sanitized_and_unknown_ids_are_dropped(monkeypatch):
    class FakeModel:
        def invoke(self, messages):
            return AIMessage(content=json.dumps({
                "selected_insight_ids": ["good", "unknown", "bad", "good"],
                "content_angle": "A focused angle",
                "evidence_map": [{"insight_id": "good", "supports": "claim", "usage_constraint": "cite source"}],
                "requires_more_research": False,
            }))

    monkeypatch.setattr("workflow.content_intake._load_model", lambda: FakeModel())
    result = content_intake_node({
        "research_output": {"eligible_insights": [_insight("good"), _insight("bad", 90, False)], "alternative_insights": []},
        "content_goal": "Guide",
    })
    assert result["selected_insight_ids"] == ["good"]
    assert result["evidence_map"] == [{
        "insight_id": "good",
        "supports": "claim",
        "source_urls": ["https://example.com/good"],
        "usage_constraints": [],
    }]
    assert result["content_angle"] == "A focused angle"


def test_load_research_output_reads_json_contract(tmp_path):
    path = tmp_path / "research.json"
    path.write_text(json.dumps({"eligible_insights": [], "alternative_insights": []}), encoding="utf-8")
    assert load_research_output(path)["eligible_insights"] == []
