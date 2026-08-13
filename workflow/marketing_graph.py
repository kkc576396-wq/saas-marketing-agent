"""End-to-end Marketing graph through the Content planning phase."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from .content_executor import (
    make_content_executor_node,
    route_after_content_executor,
)
from .content_intent import content_requested
from .content_planner import make_content_planner_node
from .content_tools import make_rag_prefetch_node
from .memory_manager import MemoryManager
from .memory_nodes import make_memory_commit_node, make_memory_prefetch_node
from .research_agent import (
    make_research_agent_node,
    research_tools_node,
    route_after_research_agent,
    route_after_research_tools,
)
from .research_graph import evaluation_node, planning_node, save_output_node
from .reflection import (
    assess_reflection_risk,
    make_reflection_question_node,
    make_reflection_verification_node,
    route_after_reflection_questions,
    route_after_reflection_verification,
)
from .state import MarketingState


def route_after_evaluation(state: MarketingState) -> str:
    """Plan content only when the Rewriter detected a generation request."""

    return (
        "content_planner"
        if content_requested(state.get("content_intent", {}))
        else "save"
    )


def build_marketing_graph(
    research_model: Any | None = None,
    content_planner_model: Any | None = None,
    content_executor_model: Any | None = None,
    reflection_question_model: Any | None = None,
    verification_model: Any | None = None,
    rag_prefetch_search: Any | None = None,
    memory_manager: MemoryManager | None = None,
):
    """Build Research, memory, Plan-and-Solve Content, and Reflection."""

    base_research_agent_node = make_research_agent_node(research_model)

    # LangGraph derives a node's input schema from its type annotation. The
    # shared Research functions are annotated with ResearchState, so wrap the
    # two nodes that must also read/write MarketingState-only fields.
    def marketing_planning_node(state: MarketingState) -> dict[str, Any]:
        working = dict(state)
        raw_request = str(
            state.get("raw_user_request")
            or state.get("original_query")
            or state.get("topic", "")
        ).strip()
        if raw_request:
            working["original_query"] = raw_request
        return planning_node(working)

    def marketing_research_agent_node(
        state: MarketingState,
    ) -> dict[str, Any]:
        return base_research_agent_node(state)

    def marketing_save_node(state: MarketingState) -> dict[str, Any]:
        return save_output_node(state)

    def research_done_node(state: MarketingState) -> dict[str, Any]:
        return {"research_completed": True}

    def draft_checkpoint_node(state: MarketingState) -> dict[str, Any]:
        """Persist a usable draft before optional Reflection begins."""

        working = {**state, "draft_checkpoint_status": "saved"}
        updates = save_output_node(working)
        return {**updates, "draft_checkpoint_status": "saved"}

    graph = StateGraph(MarketingState)
    graph.add_node("planning", marketing_planning_node)
    graph.add_node("research_agent", marketing_research_agent_node)
    graph.add_node("tools", research_tools_node)
    graph.add_node("research_done", research_done_node)
    graph.add_node("rag_prefetch", make_rag_prefetch_node(rag_prefetch_search))
    graph.add_node("memory_prefetch", make_memory_prefetch_node(memory_manager))
    graph.add_node("evaluation", evaluation_node)
    graph.add_node(
        "content_planner", make_content_planner_node(content_planner_model)
    )
    graph.add_node(
        "content_executor", make_content_executor_node(content_executor_model)
    )
    graph.add_node("draft_checkpoint", draft_checkpoint_node)
    graph.add_node("reflection_risk_gate", assess_reflection_risk)
    graph.add_node(
        "reflection_question_planner",
        make_reflection_question_node(reflection_question_model),
    )
    graph.add_node(
        "reflection_verification",
        make_reflection_verification_node(verification_model),
    )
    graph.add_node("memory_commit", make_memory_commit_node(memory_manager))
    graph.add_node("save", marketing_save_node)

    graph.add_edge(START, "planning")
    graph.add_edge("planning", "research_agent")
    graph.add_edge("planning", "rag_prefetch")
    graph.add_edge("planning", "memory_prefetch")
    graph.add_conditional_edges(
        "research_agent",
        route_after_research_agent,
        {"tools": "tools", "evaluation": "research_done"},
    )
    graph.add_conditional_edges(
        "tools",
        route_after_research_tools,
        {"research_agent": "research_agent", "evaluation": "research_done"},
    )
    graph.add_edge(
        ["research_done", "rag_prefetch", "memory_prefetch"], "evaluation"
    )
    graph.add_conditional_edges(
        "evaluation",
        route_after_evaluation,
        {"content_planner": "content_planner", "save": "memory_commit"},
    )
    graph.add_edge("content_planner", "content_executor")
    graph.add_conditional_edges(
        "content_executor",
        route_after_content_executor,
        {
            "content_executor": "content_executor",
            "draft_checkpoint": "draft_checkpoint",
            "reflection_question_planner": "reflection_question_planner",
            "save": "memory_commit",
        },
    )
    graph.add_edge("draft_checkpoint", "reflection_risk_gate")
    graph.add_edge("reflection_risk_gate", "reflection_question_planner")
    graph.add_conditional_edges(
        "reflection_question_planner",
        route_after_reflection_questions,
        {
            "reflection_verification": "reflection_verification",
            "content_executor": "content_executor",
            "save": "memory_commit",
        },
    )
    graph.add_conditional_edges(
        "reflection_verification",
        route_after_reflection_verification,
        {"content_executor": "content_executor", "save": "memory_commit"},
    )
    graph.add_edge("memory_commit", "save")
    graph.add_edge("save", END)
    return graph.compile()


marketing_graph = build_marketing_graph()
