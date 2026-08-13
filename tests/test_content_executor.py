import json

from langchain_core.messages import AIMessage

import workflow.content_executor as content_executor
from workflow.content_executor import (
    make_content_executor_node,
    select_executor_model_role,
)


def _state():
    insight = {
        "insight_id": "insight-001",
        "title": "Merchant pain point",
        "summary": "A merchant described an automation pain point.",
        "source_type": "Reddit",
        "sources": [{"title": "Post", "url": "https://reddit.com/r/test"}],
        "verification": {"passed": True, "confidence": 0.9},
        "scoring": {"total_score": 82},
    }
    return {
        "raw_user_request": "Research merchant pain and write a Reddit post",
        "content_intent": {
            "type": "reddit_promotion",
            "platform": "reddit",
            "language": "English",
            "requires_content_generation": True,
        },
        "content_plan": {
            "plan_id": "content-plan-001",
            "final_goal": "Create a Reddit post",
            "steps": [
                {
                    "step_id": "step-001",
                    "objective": "Select relevant evidence",
                    "required_inputs": ["research_output"],
                    "suggested_tools": [],
                    "expected_output": "selected_evidence",
                },
                {
                    "step_id": "step-002",
                    "objective": "Generate the Reddit post",
                    "required_inputs": ["selected_evidence"],
                    "suggested_tools": [],
                    "expected_output": "draft",
                },
            ],
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


class StepwiseExecutorModel:
    def __init__(self):
        self.contexts = []

    def invoke(self, messages):
        context = json.loads(messages[-1][1])
        self.contexts.append(context)
        step = context["current_step"]
        if step["step_id"] == "step-001":
            result = {"selected_insight_ids": ["insight-001", "invented-id"]}
            evidence_ids = ["insight-001", "invented-id"]
        else:
            result = {"content": "A concise evidence-based Reddit post."}
            evidence_ids = ["insight-001"]
        return AIMessage(
            content=json.dumps(
                {
                    "step_id": step["step_id"],
                    "status": "completed",
                    "result_type": step["expected_output"],
                    "result": result,
                    "used_evidence_ids": evidence_ids,
                    "execution_summary": f"Completed {step['step_id']}",
                    "blocking_reason": "",
                    "missing_inputs": [],
                }
            )
        )


def test_executor_runs_one_step_and_passes_history_to_the_next_call():
    model = StepwiseExecutorModel()
    node = make_content_executor_node(model)
    state = _state()

    first = node(state)
    state.update(first)
    second = node(state)

    assert first["current_step_index"] == 1
    assert first["executor_status"] == "step_completed"
    assert first["execution_history"][0]["result"]["selected_insight_ids"] == [
        "insight-001"
    ]
    assert first["execution_history"][0]["used_evidence_ids"] == [
        "insight-001"
    ]
    assert second["current_step_index"] == 2
    assert second["executor_status"] == "plan_completed"
    assert len(second["execution_history"]) == 2
    assert len(model.contexts[1]["execution_history"]) == 1
    assert model.contexts[1]["current_step"]["step_id"] == "step-002"


def test_reddit_reply_final_step_generates_one_english_reply_for_each_top_five_post():
    state = _state()
    insights = []
    for index in range(5):
        insight = {
            "insight_id": f"reddit-{index + 1}",
            "title": f"Email marketing discussion {index + 1}",
            "summary": "A merchant asks for practical email marketing workflow advice.",
            "source_type": "Reddit",
            "sources": [
                {
                    "title": f"Post {index + 1}",
                    "url": f"https://www.reddit.com/r/Emailmarketing/comments/post-{index + 1}/example/",
                    "published_at": "2026-08-10T00:00:00+00:00",
                }
            ],
            "verification": {"passed": True, "confidence": 0.9},
            "scoring": {"total_score": 90 - index},
        }
        insights.append(insight)
    state.update(
        {
            "content_intent": {
                **state["content_intent"],
                "type": "reddit_reply",
            },
            "insights": insights,
            "candidate_insights": insights,
            "verification_results": [
                {"insight": item["insight_id"], "passed": True, "confidence": 0.9}
                for item in insights
            ],
            "insight_scores": [
                {"insight": item["insight_id"], "total_score": 90 - index}
                for index, item in enumerate(insights)
            ],
            "current_step_index": 1,
            "execution_history": [],
        }
    )

    class TopFiveReplyModel:
        def invoke(self, messages):
            context = json.loads(messages[-1][1])
            targets = context["reddit_reply_targets"]
            assert len(targets) == 5
            return AIMessage(
                content=json.dumps(
                    {
                        "step_id": context["current_step"]["step_id"],
                        "status": "completed",
                        "result_type": "draft",
                        "result": {
                            "replies": [
                                {
                                    "post_url": target["post_url"],
                                    "reply": f"Helpful English reply for post {index + 1}.",
                                }
                                for index, target in enumerate(targets)
                            ]
                        },
                        "used_evidence_ids": [],
                        "used_rag_chunk_ids": [],
                        "execution_summary": "Generated five replies.",
                        "blocking_reason": "",
                        "missing_inputs": [],
                    }
                )
            )

    result = make_content_executor_node(TopFiveReplyModel())(state)

    replies = result["final_content"]["replies"]
    assert result["final_content"]["reply_count"] == 5
    assert len(replies) == 5
    assert all(item["post_url"].startswith("https://www.reddit.com/") for item in replies)
    assert all("Helpful English reply" in item["reply"] for item in replies)
    assert result["execution_history"][0]["used_evidence_ids"] == [
        f"reddit-{index}" for index in range(1, 6)
    ]


def test_final_writing_retries_when_model_returns_chinese():
    state = _state()
    state["current_step_index"] = 1

    class LanguageRepairModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            content = "这是一条中文内容。" if self.calls == 1 else "English-only content."
            return AIMessage(
                content=json.dumps(
                    {
                        "step_id": "step-002",
                        "status": "completed",
                        "result_type": "draft",
                        "result": {"content": content},
                        "used_evidence_ids": [],
                        "used_rag_chunk_ids": [],
                        "execution_summary": "Drafted content.",
                        "blocking_reason": "",
                        "missing_inputs": [],
                    }
                )
            )

    model = LanguageRepairModel()
    result = make_content_executor_node(model)(state)

    assert model.calls == 2
    assert result["final_content"] == {"content": "English-only content."}


def test_final_content_is_plain_text_without_asterisks():
    state = _state()
    state["current_step_index"] = 1

    class MarkdownModel:
        def invoke(self, messages):
            return AIMessage(
                content=json.dumps(
                    {
                        "step_id": "step-002",
                        "status": "completed",
                        "result_type": "draft",
                        "result": {
                            "content": "**Headline**\n\n* First item\n*italic* text\n***"
                        },
                        "used_evidence_ids": [],
                        "used_rag_chunk_ids": [],
                        "execution_summary": "Drafted content.",
                        "blocking_reason": "",
                        "missing_inputs": [],
                    }
                )
            )

    result = make_content_executor_node(MarkdownModel())(state)

    assert result["final_content"] == {
        "content": "Headline\n\n- First item\nitalic text\n"
    }
    assert "*" not in json.dumps(result["final_content"])
    assert result["execution_history"][-1]["result"] == result["final_content"]


def test_publish_cleanup_recurses_through_reddit_replies():
    cleaned = content_executor._sanitize_publish_content(
        {
            "replies": [
                {"reply": "**Useful reply**\n* First point"},
                {"reply": "Another *directly copyable* reply"},
            ]
        }
    )

    content_executor._validate_publish_content(cleaned)
    assert cleaned == {
        "replies": [
            {"reply": "Useful reply\n- First point"},
            {"reply": "Another directly copyable reply"},
        ]
    }


def test_executor_blocks_instead_of_inventing_missing_rag():
    class BlockedModel:
        def invoke(self, messages):
            context = json.loads(messages[-1][1])
            return AIMessage(
                content=json.dumps(
                    {
                        "step_id": context["current_step"]["step_id"],
                        "status": "blocked",
                        "result_type": "product_context",
                        "result": {},
                        "used_evidence_ids": [],
                        "execution_summary": "Brand RAG is unavailable.",
                        "blocking_reason": "Required brand knowledge tool is unavailable.",
                        "missing_inputs": ["brand_rag_search"],
                    }
                )
            )

    state = _state()
    state["content_plan"]["steps"] = [
        {
            "step_id": "step-001",
            "objective": "Retrieve approved SmartPush product knowledge",
            "required_inputs": [],
            "suggested_tools": ["brand_rag_search"],
            "expected_output": "product_context",
        }
    ]

    result = make_content_executor_node(BlockedModel())(state)

    assert result["executor_status"] == "blocked"
    assert result["current_step_index"] == 0
    assert result["execution_history"][0]["missing_inputs"] == [
        "brand_rag_search"
    ]


def test_content_executor_model_reuses_research_credentials(monkeypatch):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "shared-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.delenv("CONTENT_EXECUTOR_MODEL", raising=False)
    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChatOpenAI)

    content_executor.load_content_executor_model()

    assert captured["model"] == "qwen3.6-plus"
    assert captured["base_url"] == "https://example.test/v1"
    assert captured["temperature"] == 0
    assert captured["extra_body"] == {"enable_thinking": False}


def test_executor_routes_prep_standard_and_premium_writing_models():
    state = _state()
    prep_step, writing_step = state["content_plan"]["steps"]

    assert select_executor_model_role(
        state,
        current_step=prep_step,
        current_index=0,
        step_count=2,
        executor_mode="plan",
    ) == "prep"
    assert select_executor_model_role(
        state,
        current_step=writing_step,
        current_index=1,
        step_count=2,
        executor_mode="plan",
    ) == "writing"

    state["content_intent"]["type"] = "homepage_promotion"
    assert select_executor_model_role(
        state,
        current_step=writing_step,
        current_index=1,
        step_count=2,
        executor_mode="plan",
    ) == "premium"


def test_executor_repairs_one_malformed_step_response(monkeypatch):
    class RepairingModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="Here is the draft, but it is outside the JSON object."
                )
            return AIMessage(
                content=json.dumps(
                    {
                        "step_id": "step-001",
                        "status": "completed",
                        "result_type": "selected_evidence",
                        "result": {"selected_insight_ids": ["insight-001"]},
                        "used_evidence_ids": ["insight-001"],
                        "used_rag_chunk_ids": [],
                        "execution_summary": "Selected verified evidence.",
                        "blocking_reason": "",
                        "missing_inputs": [],
                    }
                )
            )

    monkeypatch.setenv("CONTENT_EXECUTOR_FORMAT_RETRIES", "1")
    state = _state()
    state["content_plan"]["steps"] = state["content_plan"]["steps"][:1]
    model = RepairingModel()

    result = make_content_executor_node(model)(state)

    assert result["executor_status"] == "plan_completed"
    assert result["executor_iterations"] == 2
    assert result["execution_history"][0]["result"] == {
        "selected_insight_ids": ["insight-001"]
    }
    assert model.calls == 2


def test_executor_format_failure_reports_step_and_finish_reason(monkeypatch):
    class AlwaysMalformedModel:
        def invoke(self, messages):
            message = AIMessage(content="unfinished output")
            message.response_metadata = {"finish_reason": "length"}
            return message

    monkeypatch.setenv("CONTENT_EXECUTOR_FORMAT_RETRIES", "1")
    state = _state()
    state["content_plan"]["steps"] = state["content_plan"]["steps"][:1]

    try:
        make_content_executor_node(AlwaysMalformedModel())(state)
    except ValueError as exc:
        error = str(exc)
    else:
        raise AssertionError("Expected malformed Executor output to fail")

    assert "plan:step-001" in error
    assert "finish_reason=length" in error


def test_executor_rejects_late_tool_calls_after_rag_prefetch():
    class ToolAwareModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "brand_rag_search",
                        "args": {
                            "query": "SmartPush approved product capabilities",
                            "corpora": ["product"],
                            "usage": "public_content",
                            "top_k": 3,
                        },
                        "id": "rag-call-1",
                        "type": "tool_call",
                    }
                ],
            )
    state = _state()
    state["content_plan"]["steps"] = [
        {
            "step_id": "step-001",
            "objective": "Retrieve approved SmartPush product knowledge",
            "required_inputs": [],
            "suggested_tools": ["brand_rag_search"],
            "expected_output": "product_context",
        }
    ]
    model = ToolAwareModel()
    executor = make_content_executor_node(model)

    result = executor(state)

    assert result["executor_status"] == "blocked"
    assert "prefetch" in result["executor_summary"]
    assert result["content_messages"] == []
    assert model.calls == 1


def test_revision_mode_uses_only_verification_approved_evidence():
    class RevisionModel:
        def invoke(self, messages):
            context = json.loads(messages[-1][1])
            assert context["executor_mode"] == "revision"
            assert context["available_tools"] == []
            return AIMessage(
                content=json.dumps(
                    {
                        "step_id": "revision-001",
                        "status": "completed",
                        "result_type": "revised_content",
                        "result": {"content": "Scoped revised copy."},
                        "used_evidence_ids": ["insight-001", "invented"],
                        "used_rag_chunk_ids": ["approved-rag", "invented-rag"],
                        "execution_summary": "Applied the bounded revision.",
                        "blocking_reason": "",
                        "missing_inputs": [],
                    }
                )
            )

    state = _state()
    state.update(
        {
            "executor_mode": "revision",
            "final_content": {"content": "Broad original copy."},
            "revision_steps": [
                {
                    "step_id": "revision-001",
                    "target_ids": ["claim-001"],
                    "action": "qualify",
                    "instruction": "Scope the claim.",
                    "allowed_evidence_ids": ["insight-001"],
                    "allowed_rag_chunk_ids": ["approved-rag"],
                    "expected_output": "revised_content",
                }
            ],
            "current_revision_step_index": 0,
        }
    )

    result = make_content_executor_node(RevisionModel())(state)

    assert result["executor_status"] == "revision_completed"
    assert result["revision_history"][0]["used_evidence_ids"] == [
        "insight-001"
    ]
    assert result["revision_history"][0]["used_rag_chunk_ids"] == [
        "approved-rag"
    ]
    assert result["final_content"] == {"content": "Scoped revised copy."}
