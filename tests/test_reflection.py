import json

from langchain_core.messages import AIMessage

import workflow.reflection as reflection
from workflow.reflection import (
    DEFAULT_MAX_REFLECTION_ITERATIONS,
    assess_reflection_risk,
    make_reflection_question_node,
    make_reflection_verification_node,
)


def test_reflection_default_is_one_round():
    assert DEFAULT_MAX_REFLECTION_ITERATIONS == 1


def test_reflection_risk_gate_allows_only_low_risk_reddit_advice_to_stay_light():
    state = _state()
    state["content_intent"]["type"] = "reddit_reply"
    state["final_content"] = {
        "content": "I would start by mapping the welcome flow and checking each handoff."
    }

    result = assess_reflection_risk(state)

    assert result["reflection_mode"] == "light"
    assert result["reflection_risk_reasons"] == []


def test_reflection_risk_gate_escalates_numbers_competitors_capabilities_and_trends():
    cases = [
        ("This improved results by 20%.", "number_or_metric"),
        ("Klaviyo offers a cheaper plan.", "competitor_fact"),
        ("SmartPush can automate this workflow.", "product_capability"),
        ("This is a growing market trend.", "market_trend"),
    ]
    for content, reason in cases:
        state = _state()
        state["content_intent"]["type"] = "reddit_reply"
        state["final_content"] = {"content": content}

        result = assess_reflection_risk(state)

        assert result["reflection_mode"] == "full"
        assert reason in result["reflection_risk_reasons"]


def test_homepage_and_competitor_reports_always_use_full_cove():
    for content_type in ("homepage_promotion", "competitor_report"):
        state = _state()
        state["content_intent"]["type"] = content_type
        state["final_content"] = {"content": "Simple copy without facts."}

        result = assess_reflection_risk(state)

        assert result["reflection_mode"] == "full"
        assert "content_type_requires_full_cove" in result[
            "reflection_risk_reasons"
        ]


def _state():
    insight = {
        "insight_id": "insight-001",
        "title": "Supported finding",
        "summary": "A merchant reported a narrowly scoped pain point.",
        "source_type": "Reddit",
        "sources": [{"title": "Post", "url": "https://reddit.com/r/test"}],
        "verification": {"passed": True, "confidence": 0.9},
        "scoring": {"total_score": 82},
    }
    return {
        "raw_user_request": "Write an evidence-based Reddit post",
        "content_intent": {
            "type": "reddit_promotion",
            "platform": "reddit",
            "language": "English",
        },
        "content_plan": {"steps": [{"step_id": "step-001"}]},
        "final_content": {"content": "Most merchants have this problem."},
        "insights": [insight],
        "candidate_insights": [insight],
        "verification_results": [
            {"insight": "insight-001", "passed": True, "confidence": 0.9}
        ],
        "insight_scores": [{"insight": "insight-001", "total_score": 82}],
        "opportunity_types": [],
        "recommended_channels": [],
        "documents": [],
        "intent_facets": [],
    }


class QuestionModel:
    def invoke(self, messages):
        return AIMessage(
            content=json.dumps(
                {
                    "claim_checks": [
                        {
                            "claim_id": "claim-001",
                            "draft_excerpt": "Most merchants have this problem.",
                            "claim": "Most merchants have this problem.",
                            "claim_type": "market_trend",
                            "risk_level": "high",
                            "verification_question": "Does the evidence establish prevalence?",
                            "required_evidence_type": "independent market evidence",
                        }
                    ],
                    "quality_checks": [
                        {
                            "issue_id": "issue-001",
                            "category": "compliance",
                            "severity": "high",
                            "draft_excerpt": "",
                            "problem": "Brand affiliation is not disclosed.",
                            "revision_instruction": "Add a clear affiliation disclosure.",
                        }
                    ],
                    "review_summary": "One broad claim and one disclosure issue.",
                }
            )
        )


class VerificationModel:
    def invoke(self, messages):
        context = json.loads(messages[-1][1])
        assert "draft_to_review" not in context
        return AIMessage(
            content=json.dumps(
                {
                    "claim_results": [
                        {
                            "claim_id": "claim-001",
                            "verdict": "supported",
                            "answer": "The evidence is one individual observation.",
                            "evidence_ids": ["insight-001", "invented-id"],
                            "rag_chunk_ids": ["invented-chunk"],
                            "replacement_guidance": "Scope this as one merchant report.",
                        }
                    ],
                    "quality_results": [
                        {
                            "issue_id": "issue-001",
                            "verdict": "confirmed",
                            "explanation": "The requested Reddit content needs disclosure.",
                        }
                    ],
                    "revision_steps": [
                        {
                            "target_ids": ["claim-001", "issue-001"],
                            "action": "qualify",
                            "instruction": "Scope the observation and add disclosure.",
                            "allowed_evidence_ids": [
                                "insight-001",
                                "invented-id",
                            ],
                            "allowed_rag_chunk_ids": ["invented-chunk"],
                        }
                    ],
                    "verification_summary": "Revision is required.",
                }
            )
        )


def test_two_model_reflection_downgrades_uncited_support_and_builds_revisions():
    state = _state()
    question_updates = make_reflection_question_node(QuestionModel())(state)
    state.update(question_updates)
    verification_updates = make_reflection_verification_node(
        VerificationModel()
    )(state)

    claim = verification_updates["reflection_verification_results"][0]
    assert question_updates["reflection_iterations"] == 1
    assert claim["verdict"] == "supported"
    assert claim["evidence_ids"] == ["insight-001"]
    assert claim["rag_chunk_ids"] == []
    assert verification_updates["reflection_status"] == "revision_required"
    assert verification_updates["executor_mode"] == "revision"
    assert verification_updates["revision_steps"][0]["allowed_evidence_ids"] == [
        "insight-001"
    ]


def test_supported_claim_without_allowlisted_evidence_is_not_accepted():
    state = _state()
    state.update(make_reflection_question_node(QuestionModel())(state))

    class UncitedVerification:
        def invoke(self, messages):
            return AIMessage(
                content=json.dumps(
                    {
                        "claim_results": [
                            {
                                "claim_id": "claim-001",
                                "verdict": "supported",
                                "answer": "Trust me.",
                                "evidence_ids": ["invented"],
                                "rag_chunk_ids": [],
                                "replacement_guidance": "Remove the broad claim.",
                            }
                        ],
                        "quality_results": [
                            {"issue_id": "issue-001", "verdict": "dismissed"}
                        ],
                        "revision_steps": [],
                        "verification_summary": "",
                    }
                )
            )

    result = make_reflection_verification_node(UncitedVerification())(state)

    assert result["reflection_verification_results"][0]["verdict"] == (
        "insufficient_evidence"
    )
    assert result["reflection_status"] == "revision_required"
    assert result["revision_steps"][0]["target_ids"] == ["claim-001"]


def test_internal_rag_cannot_support_a_public_claim(monkeypatch):
    state = _state()
    state["rag_tool_history"] = [
        {"chunk_ids": ["internal-icp:01"], "step_index": 0}
    ]
    state.update(make_reflection_question_node(QuestionModel())(state))
    monkeypatch.setattr(
        "workflow.reflection.get_chunks_by_ids",
        lambda chunk_ids: [
            {
                "chunk_id": "internal-icp:01",
                "content": "Private audience strategy.",
                "approved_for_external_use": False,
                "visibility": "internal_only",
            }
        ],
    )

    class InternalOnlyVerification:
        def invoke(self, messages):
            return AIMessage(
                content=json.dumps(
                    {
                        "claim_results": [
                            {
                                "claim_id": "claim-001",
                                "verdict": "supported",
                                "answer": "Internal notes support it.",
                                "evidence_ids": [],
                                "rag_chunk_ids": ["internal-icp:01"],
                                "replacement_guidance": "Remove it.",
                            }
                        ],
                        "quality_results": [
                            {"issue_id": "issue-001", "verdict": "dismissed"}
                        ],
                        "revision_steps": [],
                        "verification_summary": "",
                    }
                )
            )

    result = make_reflection_verification_node(InternalOnlyVerification())(state)

    assert result["reflection_verification_results"][0]["verdict"] == (
        "insufficient_evidence"
    )
    assert result["reflection_verification_results"][0]["rag_chunk_ids"] == []


def test_reflection_models_use_requested_defaults_and_shared_endpoint(monkeypatch):
    captured = []

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "shared-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.delenv("REFLECTION_QUESTION_MODEL", raising=False)
    monkeypatch.delenv("VERIFICATION_MODEL", raising=False)
    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChatOpenAI)

    reflection.load_reflection_question_model()
    reflection.load_verification_model()

    assert captured[0]["model"] == "deepseek-v4-flash"
    assert captured[1]["model"] == "qwen3.7-plus"
    assert all(item["base_url"] == "https://example.test/v1" for item in captured)
    assert all(item["temperature"] == 0 for item in captured)
    assert all(
        item["extra_body"] == {"enable_thinking": False}
        for item in captured
    )
    assert "model_kwargs" not in captured[0]
    assert captured[1]["model_kwargs"] == {
        "response_format": {"type": "json_object"}
    }


def test_reflection_question_planner_repairs_malformed_json(monkeypatch):
    class RepairingQuestionModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="Audit notes without JSON")
            return AIMessage(
                content=json.dumps(
                    {
                        "claim_checks": [],
                        "quality_checks": [],
                        "review_summary": "No issues found.",
                    }
                )
            )

    monkeypatch.setenv("REFLECTION_FORMAT_RETRIES", "1")
    model = RepairingQuestionModel()

    result = make_reflection_question_node(model)(_state())

    assert result["reflection_question_status"] == "questions_ready"
    assert result["reflection_question_plan"]["review_summary"] == (
        "No issues found."
    )
    assert model.calls == 2


def test_reflection_question_timeout_preserves_completed_draft():
    class APITimeoutError(Exception):
        pass

    class TimeoutModel:
        def invoke(self, messages):
            raise APITimeoutError("request timed out")

    state = _state()
    original_draft = state["final_content"]
    state.update(make_reflection_question_node(TimeoutModel())(state))

    assert state["final_content"] == original_draft
    assert state["reflection_question_status"] == "timeout"
    assert state["reflection_status"] == "completed_with_review_warning"
    assert "preserved" in state["verification_summary"]


def test_reflection_verification_timeout_preserves_completed_draft():
    class APITimeoutError(Exception):
        pass

    class TimeoutModel:
        def invoke(self, messages):
            raise APITimeoutError("request timed out")

    state = _state()
    state.update(make_reflection_question_node(QuestionModel())(state))
    original_draft = state["final_content"]
    state.update(make_reflection_verification_node(TimeoutModel())(state))

    assert state["final_content"] == original_draft
    assert state["reflection_status"] == "completed_with_review_warning"
