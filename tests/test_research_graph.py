"""Offline tests for the AnySearch research graph."""

from pathlib import Path

from workflow.research_graph import build_research_graph


def test_research_graph_respects_max_iterations(monkeypatch, tmp_path: Path):
    calls = []

    def fake_search(tool_name, arguments, api_key=None):
        calls.append((tool_name, arguments))
        return "# Result\nA useful market finding."

    monkeypatch.setattr("tools.anysearch_tool._call_anysearch", fake_search)

    output_file = tmp_path / "research_output.json"
    result = build_research_graph().invoke(
        {
            "topic": "AI SaaS marketing",
            "max_iterations": 2,
            "output_file": str(output_file),
        }
    )

    assert len(calls) == 2
    assert result["search_iterations"] == 2
    assert "verification_results" in result
    assert output_file.exists()


def test_research_graph_executes_exactly_five_iterations(monkeypatch, tmp_path: Path):
    calls = []

    def fake_search(tool_name, arguments, api_key=None):
        calls.append((tool_name, arguments))
        return "# AI SaaS marketing result\nAI SaaS marketing finding."

    monkeypatch.setattr("tools.anysearch_tool._call_anysearch", fake_search)

    result = build_research_graph().invoke(
        {
            "topic": "AI SaaS marketing",
            "max_iterations": 5,
            "output_file": str(tmp_path / "research_output.json"),
        }
    )

    assert len(calls) == 5
    assert result["search_iterations"] == 5
