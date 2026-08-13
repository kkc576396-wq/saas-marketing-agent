"""Offline tests for the AnySearch research graph."""

import json
import importlib
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage

from workflow.research_agent import research_tools_node
from workflow.research_graph import build_research_graph
from workflow.router import AGENT_REACH_REDDIT


class ScriptedResearchModel:
    """Emit one AnySearch call per model turn until the graph budget stops it."""

    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "anysearch_search",
                    "args": {
                        "query": f"email marketing research round {self.calls}",
                        "max_results": 5,
                    },
                    "id": f"call-{self.calls}",
                    "type": "tool_call",
                }
            ],
        )


def test_compiled_graph_uses_only_current_bounded_react_nodes():
    """Guard against accidentally reconnecting retired workflow nodes."""

    nodes = set(build_research_graph().get_graph().nodes)

    assert {"planning", "research_agent", "tools", "evaluation", "save"}.issubset(nodes)
    assert "retrieval" not in nodes
    assert "reflection" not in nodes
    assert "query_rewriter" not in nodes
    assert "source_router" not in nodes
    assert "verifier" not in nodes


def test_research_graph_respects_max_iterations(monkeypatch, tmp_path: Path):
    calls = []

    def fake_search(tool_name, arguments, api_key=None):
        calls.append((tool_name, arguments))
        return """## Search Results
### 1. Useful market finding
- **URL**: https://example.com/research
- **Published Time**: 2026-08-01
Email marketing automation market evidence for Shopify merchants.
"""

    monkeypatch.setattr("tools.anysearch_tool._call_anysearch", fake_search)

    output_file = tmp_path / "research_output.json"
    result = build_research_graph(model=ScriptedResearchModel()).invoke(
        {
            "topic": "AI SaaS marketing",
            "max_iterations": 2,
            "output_file": str(output_file),
        }
    )

    assert len(calls) == 2
    assert result["search_iterations"] == 2
    assert "verification_results" in result
    assert "insight_scores" in result
    assert output_file.exists()
    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert "documents" not in payload
    assert "tool_results" not in payload
    assert "alternative_insights" in payload
    assert "detected_entities" in payload
    assert "intent_facets" in payload


def test_research_graph_executes_exactly_five_iterations(monkeypatch, tmp_path: Path):
    calls = []

    def fake_search(tool_name, arguments, api_key=None):
        calls.append((tool_name, arguments))
        return """## Search Results
### 1. AI SaaS marketing result
- **URL**: https://example.com/ai-saas
- **Published Time**: 2026-08-01
AI SaaS marketing finding for ecommerce teams.
"""

    monkeypatch.setattr("tools.anysearch_tool._call_anysearch", fake_search)

    result = build_research_graph(model=ScriptedResearchModel()).invoke(
        {
            "topic": "AI SaaS marketing",
            "max_iterations": 5,
            "output_file": str(tmp_path / "research_output.json"),
        }
    )

    assert len(calls) == 5
    assert result["search_iterations"] == 5


def test_reddit_research_is_capped_at_two_iterations(monkeypatch, tmp_path: Path):
    calls = []

    def fake_search(tool_name, arguments, api_key=None):
        calls.append((tool_name, arguments))
        return """## Search Results
### 1. Reddit email marketing discussion
- **URL**: https://reddit.com/r/Emailmarketing/comments/example
- **Published Time**: 2026-08-01
A merchant discussed an email marketing workflow.
"""

    monkeypatch.setattr("tools.anysearch_tool._call_anysearch", fake_search)

    result = build_research_graph(model=ScriptedResearchModel()).invoke(
        {
            "topic": "Search Reddit discussions about email marketing",
            "max_iterations": 5,
            "output_file": str(tmp_path / "reddit_output.json"),
        }
    )

    assert len(calls) == 2
    assert result["search_iterations"] == 2
    assert result["max_iterations"] == 2


def test_llm_reddit_tool_calls_run_in_parallel_with_bounded_concurrency(monkeypatch):
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_invoke(arguments):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return json.dumps(
            [
                {
                    "channel": "reddit",
                    "query": arguments["queries"][0],
                    "content": "[]",
                }
            ]
        )

    research_agent_module = importlib.import_module("workflow.research_agent")
    monkeypatch.setitem(
        research_agent_module.TOOL_BY_NAME,
        "agent_reach_search",
        SimpleNamespace(invoke=fake_invoke),
    )
    tool_calls = [
        {
            "name": "agent_reach_search",
            "args": {
                "channel": "reddit",
                "queries": [query],
                "max_results": 5,
            },
            "id": f"reddit-{index}",
            "type": "tool_call",
        }
        for index, query in enumerate(
            [
                "Klaviyo complaints",
                "Klaviyo alternatives",
                "Klaviyo pricing issues",
                "switching from Klaviyo",
            ]
        )
    ]
    result = research_tools_node(
        {
            "topic": "email marketing discussions",
            "messages": [AIMessage(content="", tool_calls=tool_calls)],
            "selected_sources": [AGENT_REACH_REDDIT],
            "search_iterations": 0,
            "max_iterations": 5,
            "documents": [],
            "tool_results": [],
        }
    )

    assert max_active == 2
    assert result["search_iterations"] == 1
    assert len(result["tool_results"]) == 4
    assert [item["queries"][0] for item in result["tool_results"]] == [
        "Klaviyo complaints",
        "Klaviyo alternatives",
        "Klaviyo pricing issues",
        "switching from Klaviyo",
    ]
