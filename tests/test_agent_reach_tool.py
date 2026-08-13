"""Mocked tests for the Agent-Reach LangGraph tools."""

import json

import pytest

from tools import agent_reach_tool


def test_agent_reach_search_supports_at_most_five_queries(monkeypatch):
    calls = []

    def fake_run_channel(channel, query, max_results):
        calls.append((channel, query, max_results))
        return f"mock result for {query}"

    monkeypatch.setattr(agent_reach_tool, "_run_channel", fake_run_channel)

    result = agent_reach_tool.agent_reach_search.invoke(
        {
            "channel": "web",
            "queries": [
                "https://example.com/1",
                "https://example.com/2",
                "https://example.com/3",
                "https://example.com/4",
                "https://example.com/5",
            ],
        }
    )

    assert len(calls) == 5
    assert len(json.loads(result)) == 5


def test_agent_reach_search_rejects_more_than_five_queries():
    with pytest.raises(Exception, match="between 1 and 5"):
        agent_reach_tool.agent_reach_search.invoke(
            {
                "channel": "web",
                "queries": [f"https://example.com/{index}" for index in range(6)],
            }
        )


def test_agent_reach_doctor_uses_mocked_health_response(monkeypatch):
    class FakeAgentReach:
        def doctor(self):
            return {"web": {"status": "ok"}, "reddit": {"status": "off"}}

    monkeypatch.setattr(agent_reach_tool, "AgentReach", FakeAgentReach, raising=False)

    # Patch the import path used inside agent_reach_doctor.
    import agent_reach

    monkeypatch.setattr(agent_reach, "AgentReach", FakeAgentReach)
    result = json.loads(agent_reach_tool.agent_reach_doctor.invoke({}))
    assert result["web"]["status"] == "ok"


def test_rss_channel_uses_bounded_http_timeout(monkeypatch):
    captured = {}

    class FakeResponse:
        content = b"""<?xml version="1.0"?><rss version="2.0"><channel>
        <item><title>Update</title><link>https://example.test/update</link></item>
        </channel></rss>"""

        def raise_for_status(self):
            return None

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(agent_reach_tool.requests, "get", fake_get)

    result = json.loads(
        agent_reach_tool._run_channel(
            "rss", "https://example.test/feed.xml", max_results=5
        )
    )

    assert captured["timeout"] == 15.0
    assert result[0]["title"] == "Update"
