import json
import threading

from langchain_core.messages import AIMessage

from workflow.marketing_graph import build_marketing_graph
from workflow.memory_manager import MemoryManager, SQLiteMemoryStore


class CompleteResearchModel:
    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return AIMessage(content="Research complete", tool_calls=[])


class RecordingPlannerModel:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(
            content=json.dumps(
                {
                    "plan_id": "content-plan-001",
                    "final_goal": "Write a Reddit post",
                    "steps": [
                        {
                            "step_id": "step-001",
                            "objective": "Select evidence for the post",
                            "required_inputs": ["research_output"],
                            "suggested_tools": [],
                            "expected_output": "selected_evidence",
                        },
                        {
                            "step_id": "step-002",
                            "objective": "Draft the post from prior results",
                            "required_inputs": ["selected_evidence"],
                            "suggested_tools": [],
                            "expected_output": "draft",
                        },
                    ],
                    "success_criteria": ["Use verified evidence"],
                    "planning_reasoning": "Plan before solving.",
                }
            )
        )


class RecordingExecutorModel:
    def __init__(self):
        self.calls = 0
        self.contexts = []

    def invoke(self, messages):
        self.calls += 1
        context = json.loads(messages[-1][1])
        self.contexts.append(context)
        step = context["current_step"]
        return AIMessage(
            content=json.dumps(
                {
                    "step_id": step["step_id"],
                    "status": "completed",
                    "result_type": step["expected_output"],
                    "result": {"content": "Generated step output"},
                    "used_evidence_ids": [],
                    "execution_summary": "Step completed",
                    "blocking_reason": "",
                    "missing_inputs": [],
                }
            )
        )


class PassQuestionModel:
    def invoke(self, messages):
        return AIMessage(
            content=json.dumps(
                {
                    "claim_checks": [],
                    "quality_checks": [],
                    "review_summary": "No checks required.",
                }
            )
        )


class PassVerificationModel:
    def invoke(self, messages):
        return AIMessage(
            content=json.dumps(
                {
                    "claim_results": [],
                    "quality_results": [],
                    "revision_steps": [],
                    "verification_summary": "Draft passed.",
                }
            )
        )


def fake_rag_search(query, *, corpora=None, usage="public_content", top_k=6):
    corpus = (corpora or ["brand"])[0]
    return {
        "query": query,
        "usage": usage,
        "corpora": corpora or [],
        "results": [
            {
                "chunk_id": f"{corpus}:prefetch",
                "corpus": corpus,
                "content": f"Approved {corpus} guidance.",
                "approved_for_external_use": usage == "public_content",
            }
        ],
    }


class MemoryEmbeddings:
    def embed(self, texts):
        return [
            [float("reddit" in text.casefold()), float("concise" in text.casefold())]
            for text in texts
        ]


def test_marketing_graph_runs_research_then_content_planner(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "workflow.query_rewriter._model_rewrite",
        lambda query: {
            "research_objective": "Latest email automation market trends",
            "translated_query": "Latest email automation market trends",
            "source_queries": {},
            "content_intent": {
                "type": "reddit_promotion",
                "platform": "reddit",
                "language": "English",
                "requires_brand_rag": True,
            },
        },
    )
    planner = RecordingPlannerModel()
    executor = RecordingExecutorModel()
    output_file = tmp_path / "marketing_output.json"

    result = build_marketing_graph(
        research_model=CompleteResearchModel(),
        content_planner_model=planner,
        content_executor_model=executor,
        reflection_question_model=PassQuestionModel(),
        verification_model=PassVerificationModel(),
        rag_prefetch_search=fake_rag_search,
    ).invoke(
        {
            "topic": "调查最新邮件自动化趋势并生成英文 Reddit 宣传文章",
            "max_iterations": 1,
            "output_file": str(output_file),
        }
    )

    assert planner.calls == 1
    assert executor.calls == 2
    assert result["research_objective"] == "Latest email automation market trends"
    assert result["content_plan"]["plan_id"] == "content-plan-001"
    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert payload["content_intent"]["deliverable_type"] == "reddit_promotion"
    assert payload["content_planner_status"] == "planned"
    assert payload["content_plan"]["steps"][0]["step_id"] == "step-001"
    assert payload["executor_status"] == "plan_completed"
    assert len(payload["execution_history"]) == 2
    assert executor.contexts[1]["execution_history"][0]["step_id"] == "step-001"
    assert payload["reflection_status"] == "passed"
    assert payload["reflection_iterations"] == 1
    assert payload["draft_checkpoint_status"] == "saved"
    assert payload["rag_prefetch_status"] == "ready"
    assert payload["reflection_mode"] == "full"


def test_marketing_graph_includes_parallel_rag_and_risk_gate_nodes():
    nodes = set(build_marketing_graph().get_graph().nodes)

    assert "content_executor" in nodes
    assert "rag_prefetch" in nodes
    assert "memory_prefetch" in nodes
    assert "memory_commit" in nodes
    assert "research_done" in nodes
    assert "content_tools" not in nodes
    assert "draft_checkpoint" in nodes
    assert "reflection_risk_gate" in nodes
    assert "reflection_question_planner" in nodes
    assert "reflection_verification" in nodes


def test_marketing_graph_skips_content_planner_for_research_only(monkeypatch, tmp_path):
    monkeypatch.setattr("workflow.query_rewriter._model_rewrite", lambda query: None)
    planner = RecordingPlannerModel()
    executor = RecordingExecutorModel()

    result = build_marketing_graph(
        research_model=CompleteResearchModel(),
        content_planner_model=planner,
        content_executor_model=executor,
        reflection_question_model=PassQuestionModel(),
        verification_model=PassVerificationModel(),
    ).invoke(
        {
            "topic": "Research email automation trends",
            "max_iterations": 1,
            "output_file": str(tmp_path / "research-only.json"),
        }
    )

    assert planner.calls == 0
    assert executor.calls == 0
    assert result["requires_content_generation"] is False
    assert result.get("content_plan", {}) == {}


def test_marketing_graph_runs_revision_then_rechecks(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "workflow.query_rewriter._model_rewrite",
        lambda query: {
            "research_objective": "Email automation trend",
            "translated_query": "Email automation trend",
            "source_queries": {},
            "content_intent": {
                "type": "reddit_promotion",
                "platform": "reddit",
                "language": "English",
                "requires_brand_rag": False,
            },
        },
    )

    class OneStepPlanner:
        def invoke(self, messages):
            return AIMessage(
                content=json.dumps(
                    {
                        "plan_id": "content-plan-001",
                        "final_goal": "Write a post",
                        "steps": [
                            {
                                "step_id": "step-001",
                                "objective": "Write the post",
                                "required_inputs": [],
                                "suggested_tools": [],
                                "expected_output": "draft",
                            }
                        ],
                        "success_criteria": [],
                        "planning_reasoning": "One drafting step.",
                    }
                )
            )

    class RevisionAwareExecutor:
        def __init__(self):
            self.modes = []

        def invoke(self, messages):
            context = json.loads(messages[-1][1])
            self.modes.append(context["executor_mode"])
            content = (
                "Revised post with disclosure."
                if context["executor_mode"] == "revision"
                else "Initial post."
            )
            return AIMessage(
                content=json.dumps(
                    {
                        "step_id": context["current_step"]["step_id"],
                        "status": "completed",
                        "result_type": context["current_step"]["expected_output"],
                        "result": {"content": content},
                        "used_evidence_ids": [],
                        "used_rag_chunk_ids": [],
                        "execution_summary": "Step completed.",
                        "blocking_reason": "",
                        "missing_inputs": [],
                    }
                )
            )

    class TwoRoundQuestions:
        def invoke(self, messages):
            context = json.loads(messages[-1][1])
            issues = []
            if context["reflection_round"] == 1:
                issues = [
                    {
                        "issue_id": "issue-001",
                        "category": "compliance",
                        "severity": "high",
                        "draft_excerpt": "Initial post.",
                        "problem": "Missing affiliation disclosure.",
                        "revision_instruction": "Add an affiliation disclosure.",
                    }
                ]
            return AIMessage(
                content=json.dumps(
                    {
                        "claim_checks": [],
                        "quality_checks": issues,
                        "review_summary": "Round review.",
                    }
                )
            )

    class IssueVerifier:
        def invoke(self, messages):
            context = json.loads(messages[-1][1])
            issues = context["reflection_question_plan"]["quality_checks"]
            return AIMessage(
                content=json.dumps(
                    {
                        "claim_results": [],
                        "quality_results": [
                            {
                                "issue_id": item["issue_id"],
                                "verdict": "confirmed",
                                "explanation": "Disclosure is required.",
                            }
                            for item in issues
                        ],
                        "revision_steps": [
                            {
                                "target_ids": ["issue-001"],
                                "action": "compliance_fix",
                                "instruction": "Add a clear affiliation disclosure.",
                                "allowed_evidence_ids": [],
                                "allowed_rag_chunk_ids": [],
                            }
                        ]
                        if issues
                        else [],
                        "verification_summary": "Checked.",
                    }
                )
            )

    executor = RevisionAwareExecutor()
    result = build_marketing_graph(
        research_model=CompleteResearchModel(),
        content_planner_model=OneStepPlanner(),
        content_executor_model=executor,
        reflection_question_model=TwoRoundQuestions(),
        verification_model=IssueVerifier(),
        rag_prefetch_search=fake_rag_search,
    ).invoke(
        {
            "topic": "调查并生成 Reddit 文章",
            "max_iterations": 1,
            "max_reflection_iterations": 2,
            "output_file": str(tmp_path / "revision.json"),
        }
    )

    assert executor.modes == ["plan", "plan", "revision"]
    assert len(result["revision_history"]) == 1
    assert result["reflection_iterations"] == 2
    assert result["reflection_status"] == "passed"
    assert [item["status"] for item in result["reflection_history"]] == [
        "revision_required",
        "passed",
    ]
    assert result["final_content"] == {
        "content": "Revised post with disclosure."
    }


def test_marketing_graph_caps_reddit_research_at_two_rounds(monkeypatch, tmp_path):
    monkeypatch.setattr("workflow.query_rewriter._model_rewrite", lambda query: None)

    result = build_marketing_graph(
        research_model=CompleteResearchModel(),
        content_planner_model=RecordingPlannerModel(),
        content_executor_model=RecordingExecutorModel(),
        reflection_question_model=PassQuestionModel(),
        verification_model=PassVerificationModel(),
    ).invoke(
        {
            "topic": "Search Reddit discussions about email marketing",
            "max_iterations": 5,
            "output_file": str(tmp_path / "reddit-cap.json"),
        }
    )

    assert result["max_iterations"] == 2


def test_marketing_graph_saves_draft_when_reflection_times_out(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "workflow.query_rewriter._model_rewrite",
        lambda query: {
            "research_objective": "Reddit email marketing discussions",
            "translated_query": "Reddit email marketing discussions",
            "source_queries": {},
            "content_intent": {
                "type": "reddit_promotion",
                "platform": "reddit",
                "language": "English",
                "requires_brand_rag": False,
            },
        },
    )

    class APITimeoutError(Exception):
        pass

    class TimeoutReflection:
        def invoke(self, messages):
            raise APITimeoutError("request timed out")

    output_file = tmp_path / "reflection-timeout.json"
    result = build_marketing_graph(
        research_model=CompleteResearchModel(),
        content_planner_model=RecordingPlannerModel(),
        content_executor_model=RecordingExecutorModel(),
        reflection_question_model=TimeoutReflection(),
        verification_model=PassVerificationModel(),
        rag_prefetch_search=fake_rag_search,
    ).invoke(
        {
            "topic": "调查 Reddit 并生成宣传文章",
            "max_iterations": 5,
            "output_file": str(output_file),
        }
    )

    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert result["draft_checkpoint_status"] == "saved"
    assert result["reflection_status"] == "completed_with_review_warning"
    assert result["final_content"] == {"content": "Generated step output"}
    assert payload["final_content"] == result["final_content"]


def test_low_risk_reddit_reply_uses_one_light_review(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "workflow.query_rewriter._model_rewrite",
        lambda query: {
            "research_objective": "Email workflow advice",
            "translated_query": "Email workflow advice",
            "source_queries": {},
            "content_intent": {
                "type": "reddit_reply",
                "platform": "reddit",
                "language": "English",
                "requires_brand_rag": False,
            },
        },
    )

    class VerificationMustNotRun:
        def invoke(self, messages):
            raise AssertionError("Low-risk Reddit review must skip Verification")

    result = build_marketing_graph(
        research_model=CompleteResearchModel(),
        content_planner_model=RecordingPlannerModel(),
        content_executor_model=RecordingExecutorModel(),
        reflection_question_model=PassQuestionModel(),
        verification_model=VerificationMustNotRun(),
        rag_prefetch_search=fake_rag_search,
    ).invoke(
        {
            "topic": "给 Reddit 用户写一条建议型回复",
            "max_iterations": 1,
            "output_file": str(tmp_path / "light-review.json"),
        }
    )

    assert result["reflection_mode"] == "light"
    assert result["reflection_risk_reasons"] == []
    assert result["reflection_question_status"] == "light_review_completed"
    assert result["reflection_status"] == "passed"
    assert result["reflection_iterations"] == 1


def test_risky_reddit_reply_uses_full_two_model_cove(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "workflow.query_rewriter._model_rewrite",
        lambda query: {
            "research_objective": "Email workflow advice",
            "translated_query": "Email workflow advice",
            "source_queries": {},
            "content_intent": {
                "type": "reddit_reply",
                "platform": "reddit",
                "language": "English",
                "requires_brand_rag": False,
            },
        },
    )

    class RiskyExecutor(RecordingExecutorModel):
        def invoke(self, messages):
            response = super().invoke(messages)
            context = self.contexts[-1]
            if context["current_step"] == context["full_plan"]["steps"][-1]:
                response.content = json.dumps(
                    {
                        "step_id": context["current_step"]["step_id"],
                        "status": "completed",
                        "result_type": context["current_step"]["expected_output"],
                        "result": {"content": "The plan costs $29 per month."},
                        "used_evidence_ids": [],
                        "used_rag_chunk_ids": [],
                        "execution_summary": "Step completed",
                        "blocking_reason": "",
                        "missing_inputs": [],
                    }
                )
            return response

    class CountingVerification(PassVerificationModel):
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            return super().invoke(messages)

    verification = CountingVerification()
    result = build_marketing_graph(
        research_model=CompleteResearchModel(),
        content_planner_model=RecordingPlannerModel(),
        content_executor_model=RiskyExecutor(),
        reflection_question_model=PassQuestionModel(),
        verification_model=verification,
        rag_prefetch_search=fake_rag_search,
    ).invoke(
        {
            "topic": "给 Reddit 用户写一条建议型回复",
            "max_iterations": 1,
            "output_file": str(tmp_path / "full-cove.json"),
        }
    )

    assert result["reflection_mode"] == "full"
    assert "number_or_metric" in result["reflection_risk_reasons"]
    assert "pricing_or_plan" in result["reflection_risk_reasons"]
    assert verification.calls == 1


def test_rag_prefetch_runs_in_parallel_and_joins_before_planner(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_ENABLED", "true")
    monkeypatch.setattr(
        "workflow.query_rewriter._model_rewrite",
        lambda query: {
            "research_objective": "Email workflow advice",
            "translated_query": "Email workflow advice",
            "source_queries": {},
            "content_intent": {
                "type": "reddit_reply",
                "platform": "reddit",
                "language": "English",
                "requires_brand_rag": True,
            },
        },
    )
    prefetch_started = threading.Event()

    class WaitingResearch(CompleteResearchModel):
        def __init__(self):
            self.observed_parallel_prefetch = False

        def invoke(self, messages):
            self.observed_parallel_prefetch = prefetch_started.wait(timeout=1)
            return super().invoke(messages)

    def signalling_rag_search(query, *, corpora=None, usage="public_content", top_k=6):
        prefetch_started.set()
        return fake_rag_search(
            query, corpora=corpora, usage=usage, top_k=top_k
        )

    research = WaitingResearch()
    executor = RecordingExecutorModel()
    memory_manager = MemoryManager(
        store=SQLiteMemoryStore(tmp_path / "memory.sqlite3"),
        embedding_client=MemoryEmbeddings(),
    )
    memory_manager.add(
        content="Prefer concise Reddit replies.",
        memory_type="user_preference",
        source_refs=["user-message-1"],
    )
    result = build_marketing_graph(
        research_model=research,
        content_planner_model=RecordingPlannerModel(),
        content_executor_model=executor,
        reflection_question_model=PassQuestionModel(),
        verification_model=PassVerificationModel(),
        rag_prefetch_search=signalling_rag_search,
        memory_manager=memory_manager,
    ).invoke(
        {
            "topic": "给 Reddit 用户写一条建议型回复",
            "max_iterations": 1,
            "output_file": str(tmp_path / "parallel-prefetch.json"),
        }
    )

    assert research.observed_parallel_prefetch is True
    assert result["research_completed"] is True
    assert result["rag_prefetch_status"] == "ready"
    assert len(result["rag_prefetch_results"]) == 3
    assert len(executor.contexts) == 2
    assert executor.contexts[0]["rag_prefetch"]["status"] == "ready"
    assert executor.contexts[0]["medium_term_memory"]["status"] == "ready"
    assert result["memory_commit_status"] == "saved"
