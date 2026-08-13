"""Tests for the local Research Agent web layer."""

from __future__ import annotations

import http.client
import json
import threading
import time

import pytest

from web.app import ResearchService, _validate_topic, create_server


def test_validate_topic_rejects_empty_input():
    with pytest.raises(ValueError, match="请输入调研主题"):
        _validate_topic("  ")


def test_research_service_returns_completed_output():
    def fake_runner(topic, progress):
        progress("research_agent", {"selected_sources": ["AnySearch"]})
        progress("tools", {"search_iterations": 5})
        return {
            "topic": topic,
            "search_iterations": 5,
            "eligible_insights": [{"title": "Test insight"}],
            "alternative_insights": [],
        }

    service = ResearchService(fake_runner)
    try:
        submitted = service.submit("邮件营销趋势")
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            job = service.get(submitted["job_id"])
            if job["status"] == "succeeded":
                break
            time.sleep(0.01)
        assert job["stage"] == "complete"
        assert job["search_iterations"] == 5
        assert job["result"]["topic"] == "邮件营销趋势"
    finally:
        service.close()


def test_service_tracks_content_rag_and_reflection_progress():
    def fake_runner(topic, progress):
        progress(
            "planning",
            {
                "max_iterations": 2,
                "requires_content_generation": True,
                "content_intent": {"type": "reddit_reply", "platform": "reddit"},
            },
        )
        progress(
            "content_planner",
            {"content_plan": {"steps": [{"step_id": "step-001"}]}},
        )
        progress(
            "content_executor",
            {"current_step_index": 1, "executor_iterations": 2},
        )
        progress(
            "rag_prefetch",
            {
                "rag_tool_history": [
                    {"tool_name": "brand_rag_search", "chunk_ids": ["chunk-1"]}
                ],
                "rag_prefetch_status": "ready",
            },
        )
        progress(
            "draft_checkpoint",
            {"draft_checkpoint_status": "saved"},
        )
        progress(
            "reflection_verification",
            {
                "reflection_iterations": 1,
                "reflection_status": "revision_required",
                "revision_steps": [{"step_id": "revision-001"}],
                "executor_mode": "revision",
            },
        )
        return {
            "topic": topic,
            "search_iterations": 1,
            "eligible_insights": [],
            "alternative_insights": [],
            "final_content": {"content": "Final reply"},
        }

    service = ResearchService(fake_runner)
    try:
        submitted = service.submit("调查并回复 Reddit 帖子")
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            job = service.get(submitted["job_id"])
            if job["status"] == "succeeded":
                break
            time.sleep(0.01)

        assert job["content_requested"] is True
        assert job["max_iterations"] == 2
        assert job["content_intent"]["type"] == "reddit_reply"
        assert job["content_plan_steps"] == 1
        assert job["executor_iterations"] == 2
        assert job["rag_call_count"] == 1
        assert job["rag_prefetch_status"] == "ready"
        assert job["reflection_iterations"] == 1
        assert job["revision_step_count"] == 1
        assert job["executor_mode"] == "revision"
        assert job["draft_checkpoint_status"] == "saved"
    finally:
        service.close()


def test_http_api_submits_and_returns_research_job():
    def fake_runner(topic, progress):
        progress("tools", {"search_iterations": 5})
        return {
            "topic": topic,
            "search_iterations": 5,
            "eligible_insights": [],
            "alternative_insights": [],
        }

    service = ResearchService(fake_runner)
    server = create_server("127.0.0.1", 0, service=service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
        connection.request("GET", "/")
        page_response = connection.getresponse()
        page = page_response.read().decode("utf-8")
        assert page_response.status == 200
        assert "SmartPush Marketing Agent Lab" in page
        assert "Reflection 审查结果" in page
        assert "发布到飞书" in page

        body = json.dumps({"topic": "Klaviyo pricing"})
        connection.request(
            "POST",
            "/api/research",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        submitted = json.loads(response.read())
        assert response.status == 202

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            connection.request("GET", f"/api/research/{submitted['job_id']}")
            status_response = connection.getresponse()
            job = json.loads(status_response.read())
            if job["status"] == "succeeded":
                break
            time.sleep(0.01)

        assert job["result"]["topic"] == "Klaviyo pricing"
        assert job["max_iterations"] == 5
    finally:
        server.shutdown()
        server.server_close()
        service.close()
        thread.join(timeout=2)


def test_service_publishes_non_reddit_result_to_feishu_once():
    class FakePublisher:
        configured = True

        def __init__(self):
            self.calls = []

        def publish(self, *, title, content):
            self.calls.append({"title": title, "content": content})
            return {
                "status": "published",
                "document_id": "docx-123",
                "document_url": "https://feishu.cn/docx/docx-123",
                "title": title,
            }

    def fake_runner(topic, progress):
        return {
            "topic": topic,
            "content_intent": {
                "type": "competitor_report",
                "platform": "feishu",
            },
            "content_plan": {"final_goal": "Competitor update report"},
            "final_content": {"content": "Final report body."},
        }

    publisher = FakePublisher()
    service = ResearchService(fake_runner, feishu_publisher=publisher)
    try:
        submitted = service.submit("Competitor updates")
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            job = service.get(submitted["job_id"])
            if job["status"] == "succeeded":
                break
            time.sleep(0.01)

        first = service.publish_to_feishu(submitted["job_id"])
        second = service.publish_to_feishu(submitted["job_id"])

        assert first == second
        assert len(publisher.calls) == 1
        assert publisher.calls[0]["content"] == "Final report body."
    finally:
        service.close()


def test_service_rejects_reddit_publication_to_feishu():
    class FakePublisher:
        configured = True

    def fake_runner(topic, progress):
        return {
            "topic": topic,
            "content_intent": {"type": "reddit_reply", "platform": "reddit"},
            "final_content": {"content": "Manual Reddit reply"},
        }

    service = ResearchService(fake_runner, feishu_publisher=FakePublisher())
    try:
        submitted = service.submit("Reddit reply")
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            job = service.get(submitted["job_id"])
            if job["status"] == "succeeded":
                break
            time.sleep(0.01)
        with pytest.raises(ValueError, match="仅支持复制"):
            service.publish_to_feishu(submitted["job_id"])
    finally:
        service.close()


def test_http_api_publishes_completed_result_to_feishu():
    class FakePublisher:
        configured = True

        def publish(self, *, title, content):
            return {
                "status": "published",
                "document_id": "docx-http",
                "document_url": "https://feishu.cn/docx/docx-http",
                "title": title,
            }

    def fake_runner(topic, progress):
        return {
            "topic": topic,
            "content_intent": {
                "type": "homepage_promotion",
                "platform": "homepage",
            },
            "final_content": {"content": "Homepage copy."},
        }

    service = ResearchService(fake_runner, feishu_publisher=FakePublisher())
    server = create_server("127.0.0.1", 0, service=service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
        connection.request(
            "POST",
            "/api/research",
            body=json.dumps({"topic": "Homepage campaign"}),
            headers={"Content-Type": "application/json"},
        )
        submitted_response = connection.getresponse()
        submitted = json.loads(submitted_response.read())
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            connection.request("GET", f"/api/research/{submitted['job_id']}")
            status_response = connection.getresponse()
            job = json.loads(status_response.read())
            if job["status"] == "succeeded":
                break
            time.sleep(0.01)

        connection.request(
            "POST",
            f"/api/research/{submitted['job_id']}/publish/feishu",
            body="{}",
            headers={"Content-Type": "application/json"},
        )
        publish_response = connection.getresponse()
        publication = json.loads(publish_response.read())

        assert publish_response.status == 200
        assert publication["document_id"] == "docx-http"
    finally:
        server.shutdown()
        server.server_close()
        service.close()
        thread.join(timeout=2)
