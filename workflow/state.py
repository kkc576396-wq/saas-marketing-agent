"""State contract for the LangGraph research workflow."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class ResearchState(TypedDict, total=False):
    """Shared state passed between research workflow nodes.

    The workflow initializes omitted optional fields in ``planner_node`` so a
    caller can start a run with only ``topic`` and ``max_iterations``.
    """

    topic: str
    research_date: str
    freshness_window_days: int
    freshness_required: bool
    freshness_window_explicit: bool
    original_query: str
    translated_query: str
    search_queries: list[str]
    source_queries: dict[str, list[str]]
    detected_entities: list[dict[str, Any]]
    intent_facets: list[str]
    query_reasoning: str
    hyde_terms: list[str]
    query_history: list[dict[str, Any]]
    # ReAct research history. ``add_messages`` lets the model and tool
    # executor append observations without replacing earlier turns.
    messages: Annotated[list[AnyMessage], add_messages]
    research_agent_status: str
    research_agent_reasoning: str
    research_tool_history: list[dict[str, Any]]
    documents: list[dict[str, Any]]
    search_iterations: int
    max_iterations: int
    recommended_sources: list[str]
    source_plan_reasoning: str
    selected_sources: list[str]
    source_reasoning: str
    tool_results: list[dict[str, Any]]
    # Candidates are structured normalized insight records. ``Any`` keeps
    # backwards compatibility with legacy string-based unit fixtures.
    insights: list[Any]
    candidate_insights: list[Any]
    verification_results: list[dict[str, Any]]
    insight_scores: list[dict[str, Any]]
    opportunity_types: list[dict[str, Any]]
    recommended_channels: list[dict[str, Any]]
    eligible_insights: list[dict[str, Any]]
    alternative_insights: list[dict[str, Any]]
    rejected_insights: list[dict[str, Any]]
    output_file: str


class MarketingState(ResearchState, total=False):
    """Superset state for the end-to-end Research → Content workflow."""

    # Immutable user request. Research rewriting may derive a cleaner search
    # objective, but downstream content planning always receives the original.
    raw_user_request: str
    research_objective: str
    content_intent: dict[str, Any]
    requires_content_generation: bool
    content_plan: dict[str, Any]
    content_planner_status: str
    content_planner_reasoning: str
    current_step_index: int
    current_step_attempt: int
    execution_history: list[dict[str, Any]]
    execution_artifacts: dict[str, Any]
    executor_iterations: int
    max_executor_iterations: int
    executor_status: str
    executor_summary: str
    # Tool-calling conversation for the current solving step only. This list
    # is replaced (not message-reduced) so it can be cleared between steps.
    content_messages: list[AnyMessage]
    content_active_step_id: str
    rag_tool_history: list[dict[str, Any]]
    rag_prefetch_status: str
    rag_prefetch_results: list[dict[str, Any]]
    rag_prefetch_errors: list[str]
    # Medium-term episodic memory. Approved long-term assets stay in Brand RAG.
    memory_brand_id: str
    memory_user_id: str
    memory_namespace: str
    memory_prefetch_status: str
    memory_prefetch_results: list[dict[str, Any]]
    memory_prefetch_errors: list[str]
    memory_commit_status: str
    memory_commit_ids: list[str]
    memory_commit_errors: list[str]
    research_completed: bool
    # Reflection/CoVe state. Question generation and verification are separate
    # LLM nodes; revisions are executed by the existing Content Executor.
    final_content: Any
    draft_checkpoint_status: str
    reflection_mode: str
    reflection_risk_level: str
    reflection_risk_reasons: list[str]
    reflection_question_plan: dict[str, Any]
    reflection_question_status: str
    reflection_verification_results: list[dict[str, Any]]
    reflection_history: list[dict[str, Any]]
    verification_summary: str
    reflection_status: str
    reflection_iterations: int
    max_reflection_iterations: int
    revision_steps: list[dict[str, Any]]
    current_revision_step_index: int
    revision_history: list[dict[str, Any]]
    executor_mode: str
    save_after_revision: bool
