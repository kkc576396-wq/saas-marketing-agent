import workflow.research_agent as research_agent


def test_research_model_routes_reddit_to_flash_and_competitors_to_plus(monkeypatch):
    monkeypatch.delenv("RESEARCH_FAST_MODEL", raising=False)
    monkeypatch.delenv("RESEARCH_BROAD_MODEL", raising=False)

    assert research_agent.select_research_model_name(
        {"content_intent": {"type": "reddit_reply", "platform": "reddit"}}
    ) == "qwen3.7-flash"
    assert research_agent.select_research_model_name(
        {"content_intent": {"requested": True, "deliverable_type": "competitor_report"}}
    ) == "qwen3.7-plus"
    assert research_agent.select_research_model_name(
        {"intent_facets": ["market_intelligence"]}
    ) == "qwen3.7-plus"


def test_research_model_disables_thinking(monkeypatch):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "shared-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChatOpenAI)

    research_agent.load_research_model("qwen3.7-flash")

    assert captured["model"] == "qwen3.7-flash"
    assert captured["extra_body"] == {"enable_thinking": False}
