import json
import subprocess

import pytest

from web.feishu import (
    FeishuPublisher,
    FeishuPublishError,
    feishu_title,
    final_content_text,
)


def test_extracts_final_content_and_title():
    result = {
        "topic": "Fallback topic",
        "content_plan": {"final_goal": "Competitor report"},
        "final_content": {"title": "Market update", "content": "Report body"},
    }

    assert final_content_text(result["final_content"]) == "Report body"
    assert feishu_title(result) == "Market update"


def test_publisher_imports_star_free_markdown_through_cli(tmp_path):
    cli = tmp_path / "lark-cli"
    cli.write_text("fake executable", encoding="utf-8")
    calls = []

    def fake_runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "identity": "bot",
                    "data": {
                        "document": {
                            "document_id": "docx-test-id",
                            "url": "https://example.feishu.cn/docx/docx-test-id",
                        }
                    },
                }
            ),
            stderr="",
        )

    publisher = FeishuPublisher(cli_path=str(cli), identity="bot", runner=fake_runner)
    result = publisher.publish(
        title="Competitor *Report*",
        content="**Summary**\n\n- First finding\nSecond paragraph",
    )

    command, kwargs = calls[0]
    assert command[1:3] == ["docs", "+create"]
    assert command[command.index("--doc-format") + 1] == "markdown"
    assert command[command.index("--as") + 1] == "bot"
    assert "*" not in kwargs["input"]
    assert result["document_id"] == "docx-test-id"
    assert result["document_url"].endswith("/docx-test-id")


def test_publisher_surfaces_cli_json_error(tmp_path):
    cli = tmp_path / "lark-cli"
    cli.write_text("fake executable", encoding="utf-8")

    def fake_runner(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr=json.dumps(
                {
                    "ok": False,
                    "error": {
                        "message": "No authenticated identity",
                        "hint": "Run lark-cli auth login",
                    },
                }
            ),
        )

    publisher = FeishuPublisher(cli_path=str(cli), runner=fake_runner)
    with pytest.raises(FeishuPublishError, match="auth login"):
        publisher.publish(title="Report", content="Body")
