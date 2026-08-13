import json

from langchain_core.messages import AIMessage

import workflow.content_planner as content_planner
from workflow.content_planner import make_content_planner_node


def _state():
    insight = {
        "insight_id": "insight-001",
        "title": "Merchant pricing concern",
        "summary": "A merchant reported a pricing concern.",
        "source_type": "Reddit",
        "sources": [{"title": "Post", "url": "https://reddit.com/r/test"}],
        "verification": {"passed": True, "confidence": 0.9},
        "scoring": {"total_score": 82},
        "opportunity_type": "product_feedback",
    }
    return {
        "raw_user_request": "Research pricing concerns and write a Reddit reply",
        "research_objective": "Klaviyo pricing concerns",
        "content_intent": {
            "type": "reddit_reply",
            "platform": "reddit",
            "language": "English",
            "requires_content_generation": True,
            "requires_brand_rag": True,
        },
        "insights": [insight],
        "candidate_insights": [insight],
        "verification_results": [
            {"insight": "insight-001", "passed": True, "confidence": 0.9}
        ],
        "insight_scores": [
            {"insight": "insight-001", "total_score": 82}
        ],
        "opportunity_types": [],
        "recommended_channels": [],
        "documents": [],
        "intent_facets": [],
    }


class FakePlannerModel:
    def invoke(self, messages):
        return AIMessage(
            content=json.dumps(
                {
                    "plan_id": "content-plan-001",
                    "final_goal": "Create a directly reusable Reddit reply",
                    "content_type": "reddit_reply",
                    "steps": [
                        {
                            "step_id": "step-001",
                            "objective": "Select relevant verified evidence",
                            "required_inputs": ["research_output"],
                            "suggested_tools": [],
                            "expected_output": "selected_evidence",
                        },
                        {
                            "step_id": "step-002",
                            "objective": "Retrieve approved product knowledge",
                            "required_inputs": ["selected_evidence"],
                            "suggested_tools": ["brand_rag_search"],
                            "expected_output": "product_context",
                        },
                        {
                            "step_id": "step-003",
                            "objective": "Generate the copy-ready reply",
                            "required_inputs": ["all_previous_results"],
                            "suggested_tools": [],
                            "expected_output": "draft",
                        },
                    ],
                    "success_criteria": ["Natural Reddit tone"],
                    "planning_reasoning": "Evidence and brand context precede drafting.",
                }
            )
        )


class ContentBlockPlannerModel(FakePlannerModel):
    def invoke(self, messages):
        response = super().invoke(messages)

        class BlockResponse:
            content = [{"type": "text", "text": response.content}]
            response_metadata = {"finish_reason": "stop"}

        return BlockResponse()


class RepairingPlannerModel(FakePlannerModel):
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="I started planning but failed to format the response."
            )
        return super().invoke(messages)


def test_content_planner_generates_structured_plan():
    result = make_content_planner_node(FakePlannerModel())(_state())

    assert result["content_planner_status"] == "planned"
    assert result["content_plan"]["content_type"] == "reddit_reply"
    assert [step["step_id"] for step in result["content_plan"]["steps"]] == [
        "step-001",
        "step-003",
    ]
    assert result["content_plan"]["steps"][0]["expected_output"] == (
        "selected_context"
    )
    assert all(
        step["suggested_tools"] == []
        for step in result["content_plan"]["steps"]
    )


def test_content_planner_parses_compatible_content_blocks():
    result = make_content_planner_node(ContentBlockPlannerModel())(_state())

    assert result["content_planner_status"] == "planned"
    assert result["content_plan"]["plan_id"] == "content-plan-001"


def test_content_planner_retries_one_invalid_format(monkeypatch):
    monkeypatch.setenv("CONTENT_PLANNER_FORMAT_RETRIES", "1")
    model = RepairingPlannerModel()

    result = make_content_planner_node(model)(_state())

    assert result["content_planner_status"] == "planned"
    assert model.calls == 2


def test_content_planner_model_reuses_research_credentials(monkeypatch):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "shared-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.delenv("CONTENT_PLANNER_MODEL", raising=False)
    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChatOpenAI)

    content_planner.load_content_planner_model()

    assert captured["model"] == "qwen3.7-flash"
    assert captured["base_url"] == "https://example.test/v1"
    assert captured["temperature"] == 0
    assert captured["extra_body"] == {"enable_thinking": False}
    assert captured["model_kwargs"] == {
        "response_format": {"type": "json_object"}
    }
