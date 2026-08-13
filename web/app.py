"""Dependency-free local console for the end-to-end Marketing Agent."""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlsplit

from workflow.marketing_graph import marketing_graph
from workflow.research_graph import DEFAULT_OUTPUT_FILE
from .feishu import (
    FeishuPublisher,
    FeishuPublishError,
    feishu_title,
    final_content_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_TOPIC_LENGTH = 2_000

STAGE_LABELS = {
    "queued": "等待开始",
    "planning": "识别研究与内容意图",
    "research_agent": "Research Agent 判断证据缺口",
    "tools": "检索并提取外部证据",
    "rag_prefetch": "并行预取品牌与平台 RAG",
    "memory_prefetch": "并行读取中期任务记忆",
    "research_done": "Research 分支等待 RAG 汇合",
    "evaluation": "验证、评分并筛选研究洞察",
    "content_planner": "Content Planner 生成完整计划",
    "content_executor": "Content Executor 执行当前步骤",
    "draft_checkpoint": "保存可恢复的内容草稿",
    "reflection_risk_gate": "评估事实风险并选择审查模式",
    "reflection_question_planner": "Reflection 拆分审查问题",
    "reflection_verification": "Verification 独立核验证据",
    "memory_commit": "保存可复用的中期任务记忆",
    "save": "保存最终内容与完整状态",
    "complete": "完整营销链路已完成",
    "failed": "营销任务执行失败",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _validate_topic(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("请输入调研主题。")
    topic = " ".join(value.split())
    if not topic:
        raise ValueError("请输入调研主题。")
    if len(topic) > MAX_TOPIC_LENGTH:
        raise ValueError(f"调研主题不能超过 {MAX_TOPIC_LENGTH} 个字符。")
    return topic


ProgressCallback = Callable[[str, dict[str, Any]], None]
ResearchRunner = Callable[[str, ProgressCallback], dict[str, Any]]


def run_research(topic: str, progress: ProgressCallback) -> dict[str, Any]:
    """Stream one real LangGraph run and return its persisted output contract."""

    initial_state = {
        "topic": topic,
        "max_iterations": 5,
        "output_file": str(DEFAULT_OUTPUT_FILE),
    }
    progress("planning", {})
    content_requested = False
    for update in marketing_graph.stream(
        initial_state,
        config={"recursion_limit": 64},
        stream_mode="updates",
    ):
        if not isinstance(update, dict):
            continue
        for node, node_update in update.items():
            normalized_update = node_update if isinstance(node_update, dict) else {}
            if node == "planning":
                content_requested = bool(
                    normalized_update.get("requires_content_generation", False)
                )
            progress(node, normalized_update)
            if node == "planning":
                progress("research_agent", {})
            elif node == "research_agent":
                next_stage = (
                    "tools"
                    if normalized_update.get("research_agent_status")
                    == "tool_calls_requested"
                    else "research_done"
                )
                progress(next_stage, {})
            elif node == "tools":
                reached_limit = int(
                    normalized_update.get("search_iterations", 0)
                ) >= int(normalized_update.get("max_iterations", 5))
                progress("research_done" if reached_limit else "research_agent", {})
            elif node in {"rag_prefetch", "memory_prefetch"}:
                progress("research_agent", {})
            elif node == "research_done":
                progress("evaluation", {})
            elif node == "evaluation":
                next_stage = (
                    "content_planner"
                    if content_requested
                    else "memory_commit"
                )
                progress(next_stage, {})
            elif node == "content_planner":
                progress("content_executor", {})
            elif node == "content_executor":
                executor_status = normalized_update.get("executor_status")
                if executor_status in {
                    "step_completed",
                    "revision_step_completed",
                }:
                    next_stage = "content_executor"
                elif executor_status == "plan_completed":
                    next_stage = "draft_checkpoint"
                elif executor_status == "revision_completed":
                    next_stage = (
                        "memory_commit"
                        if normalized_update.get("reflection_status")
                        == "revision_applied_at_limit"
                        else "draft_checkpoint"
                    )
                else:
                    next_stage = "memory_commit"
                progress(next_stage, {})
            elif node == "draft_checkpoint":
                progress("reflection_risk_gate", {})
            elif node == "reflection_risk_gate":
                progress("reflection_question_planner", {})
            elif node == "reflection_question_planner":
                next_stage = (
                    "reflection_verification"
                    if normalized_update.get("reflection_question_status")
                    == "questions_ready"
                    else "content_executor"
                    if normalized_update.get("reflection_status")
                    == "revision_required"
                    else "memory_commit"
                )
                progress(next_stage, {})
            elif node == "reflection_verification":
                next_stage = (
                    "content_executor"
                    if normalized_update.get("reflection_status")
                    == "revision_required"
                    else "memory_commit"
                )
                progress(next_stage, {})
            elif node == "memory_commit":
                progress("save", {})

    return json.loads(DEFAULT_OUTPUT_FILE.read_text(encoding="utf-8"))


class ResearchService:
    """Run research jobs serially and expose thread-safe status snapshots."""

    def __init__(
        self,
        runner: ResearchRunner = run_research,
        *,
        max_workers: int = 1,
        feishu_publisher: Any | None = None,
    ) -> None:
        self._runner = runner
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.feishu_publisher = feishu_publisher or FeishuPublisher.from_environment()
        # OpenCLI uses a shared browser bridge, so real jobs intentionally run
        # one at a time even though the HTTP server remains responsive.
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="research-web",
        )

    def submit(self, topic: str) -> dict[str, Any]:
        topic = _validate_topic(topic)
        job_id = uuid.uuid4().hex
        created_at = _now()
        job = {
            "job_id": job_id,
            "topic": topic,
            "status": "queued",
            "stage": "queued",
            "stage_label": STAGE_LABELS["queued"],
            "search_iterations": 0,
            "max_iterations": 5,
            "selected_sources": [],
            "content_requested": False,
            "content_intent": {},
            "content_plan_steps": 0,
            "current_step_index": 0,
            "executor_iterations": 0,
            "executor_mode": "plan",
            "rag_call_count": 0,
            "reflection_iterations": 0,
            "max_reflection_iterations": 1,
            "reflection_status": "not_started",
            "reflection_mode": "not_assessed",
            "reflection_risk_level": "not_assessed",
            "reflection_risk_reasons": [],
            "rag_prefetch_status": "not_started",
            "memory_prefetch_status": "not_started",
            "memory_commit_status": "not_started",
            "draft_checkpoint_status": "not_saved",
            "revision_step_count": 0,
            "current_revision_step_index": 0,
            "created_at": created_at,
            "updated_at": created_at,
            "elapsed_seconds": 0.0,
            "result": None,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job
        self._executor.submit(self._execute, job_id)
        return self.get(job_id) or job

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return json.loads(json.dumps(job, ensure_ascii=False))

    def publish_to_feishu(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            existing = job.get("feishu_publication")
            if isinstance(existing, dict) and existing.get("status") == "published":
                return json.loads(json.dumps(existing, ensure_ascii=False))
            if isinstance(existing, dict) and existing.get("status") == "publishing":
                raise ValueError("该任务正在发布到飞书，请勿重复提交。")
            if job.get("status") != "succeeded":
                raise ValueError("任务完成后才能发布到飞书。")
            result = job.get("result")
            if not isinstance(result, dict):
                raise ValueError("任务没有可发布的最终结果。")
            intent = result.get("content_intent", {})
            content_type = str(
                intent.get("deliverable_type") or intent.get("type") or ""
            ).casefold() if isinstance(intent, dict) else ""
            platform = str(intent.get("platform", "")).casefold() if isinstance(intent, dict) else ""
            if platform == "reddit" or content_type.startswith("reddit_"):
                raise ValueError("Reddit 内容在 MVP 中仅支持复制后手动发布。")
            content = final_content_text(result.get("final_content"))
            if not content:
                raise ValueError("任务没有可发布到飞书的最终正文。")
            job["feishu_publication"] = {"status": "publishing"}
            job["updated_at"] = _now()
            title = feishu_title(result)
        try:
            publication = self.feishu_publisher.publish(
                title=title,
                content=content,
            )
        except Exception:
            with self._lock:
                self._jobs[job_id]["feishu_publication"] = {"status": "failed"}
                self._jobs[job_id]["updated_at"] = _now()
            raise
        with self._lock:
            self._jobs[job_id]["feishu_publication"] = publication
            self._jobs[job_id]["updated_at"] = _now()
        return json.loads(json.dumps(publication, ensure_ascii=False))

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.update(changes)
            job["updated_at"] = _now()

    def _execute(self, job_id: str) -> None:
        started = time.monotonic()
        self._update(job_id, status="running")

        def progress(stage: str, update: dict[str, Any]) -> None:
            changes: dict[str, Any] = {
                "stage": stage,
                "stage_label": STAGE_LABELS.get(stage, stage),
                "elapsed_seconds": round(time.monotonic() - started, 1),
            }
            if "search_iterations" in update:
                changes["search_iterations"] = int(update["search_iterations"])
            if "max_iterations" in update:
                changes["max_iterations"] = int(update["max_iterations"])
            if "selected_sources" in update:
                changes["selected_sources"] = list(update["selected_sources"])
            if "source_reasoning" in update:
                changes["source_reasoning"] = str(update["source_reasoning"])
            if "requires_content_generation" in update:
                changes["content_requested"] = bool(
                    update["requires_content_generation"]
                )
            if "content_intent" in update and isinstance(
                update["content_intent"], dict
            ):
                changes["content_intent"] = dict(update["content_intent"])
            if "content_plan" in update and isinstance(update["content_plan"], dict):
                changes["content_plan_steps"] = len(
                    update["content_plan"].get("steps", [])
                )
            for key in (
                "current_step_index",
                "executor_iterations",
                "reflection_iterations",
                "max_reflection_iterations",
                "current_revision_step_index",
            ):
                if key in update:
                    changes[key] = int(update[key])
            if "executor_mode" in update:
                changes["executor_mode"] = str(update["executor_mode"])
            if "reflection_status" in update:
                changes["reflection_status"] = str(update["reflection_status"])
            if "reflection_mode" in update:
                changes["reflection_mode"] = str(update["reflection_mode"])
            if "reflection_risk_level" in update:
                changes["reflection_risk_level"] = str(
                    update["reflection_risk_level"]
                )
            if "reflection_risk_reasons" in update:
                changes["reflection_risk_reasons"] = list(
                    update["reflection_risk_reasons"]
                )
            if "rag_prefetch_status" in update:
                changes["rag_prefetch_status"] = str(update["rag_prefetch_status"])
            if "memory_prefetch_status" in update:
                changes["memory_prefetch_status"] = str(
                    update["memory_prefetch_status"]
                )
            if "memory_commit_status" in update:
                changes["memory_commit_status"] = str(
                    update["memory_commit_status"]
                )
            if "draft_checkpoint_status" in update:
                changes["draft_checkpoint_status"] = str(
                    update["draft_checkpoint_status"]
                )
            if "revision_steps" in update and isinstance(update["revision_steps"], list):
                changes["revision_step_count"] = len(update["revision_steps"])
            if "rag_tool_history" in update and isinstance(
                update["rag_tool_history"], list
            ):
                changes["rag_call_count"] = len(update["rag_tool_history"])
            self._update(job_id, **changes)

        try:
            topic = str(self.get(job_id)["topic"])
            result = self._runner(topic, progress)
            self._update(
                job_id,
                status="succeeded",
                stage="complete",
                stage_label=STAGE_LABELS["complete"],
                search_iterations=int(result.get("search_iterations", 5)),
                elapsed_seconds=round(time.monotonic() - started, 1),
                result=result,
            )
        except Exception as exc:  # The exact tool/workflow error is shown in UI.
            self._update(
                job_id,
                status="failed",
                stage="failed",
                stage_label=STAGE_LABELS["failed"],
                elapsed_seconds=round(time.monotonic() - started, 1),
                error=f"{type(exc).__name__}: {exc}",
            )

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)


class ResearchRequestHandler(BaseHTTPRequestHandler):
    """Serve the local interface and its small JSON API."""

    server_version = "SmartPushMarketingAgent/2.0"

    @property
    def research_service(self) -> ResearchService:
        return self.server.research_service  # type: ignore[attr-defined]

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path: Path) -> None:
        if not path.is_file() or STATIC_DIR not in path.resolve().parents:
            self._json(HTTPStatus.NOT_FOUND, {"error": "页面不存在。"})
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self'; script-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; base-uri 'none'; "
            "frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        path = unquote(urlsplit(self.path).path)
        if path == "/api/health":
            self._json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "max_iterations": 5,
                    "reddit_max_iterations": 2,
                    "executor_plan_steps": 2,
                    "max_reflection_iterations": 1,
                    "feishu_configured": self.research_service.feishu_publisher.configured,
                },
            )
            return
        if path.startswith("/api/research/"):
            job_id = path.removeprefix("/api/research/").strip("/")
            job = self.research_service.get(job_id)
            if job is None:
                self._json(HTTPStatus.NOT_FOUND, {"error": "未找到该调研任务。"})
            else:
                self._json(HTTPStatus.OK, job)
            return
        if path in {"", "/"}:
            self._serve_file(STATIC_DIR / "index.html")
            return
        safe_name = Path(path.lstrip("/")).name
        if safe_name in {"app.js", "styles.css"}:
            self._serve_file(STATIC_DIR / safe_name)
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "页面不存在。"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlsplit(self.path).path
        if path.startswith("/api/research/") and path.endswith("/publish/feishu"):
            job_id = path.removeprefix("/api/research/").removesuffix(
                "/publish/feishu"
            ).strip("/")
            try:
                publication = self.research_service.publish_to_feishu(job_id)
            except KeyError:
                self._json(HTTPStatus.NOT_FOUND, {"error": "未找到该调研任务。"})
                return
            except ValueError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            except FeishuPublishError as exc:
                self._json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
                return
            except Exception as exc:
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"飞书发布失败：{type(exc).__name__}: {exc}"},
                )
                return
            self._json(HTTPStatus.OK, publication)
            return
        if path != "/api/research":
            self._json(HTTPStatus.NOT_FOUND, {"error": "接口不存在。"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 65_536:
                raise ValueError("请求内容为空或过大。")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("请求格式错误。")
            job = self.research_service.submit(payload.get("topic"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        self._json(HTTPStatus.ACCEPTED, job)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")


class ResearchHTTPServer(ThreadingHTTPServer):
    research_service: ResearchService


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    service: ResearchService | None = None,
) -> ResearchHTTPServer:
    server = ResearchHTTPServer((host, port), ResearchRequestHandler)
    server.research_service = service or ResearchService()
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="SmartPush Marketing Agent web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    server = create_server(args.host, args.port)
    print(f"SmartPush Marketing Agent Lab: http://{args.host}:{server.server_port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        server.research_service.close()


if __name__ == "__main__":
    main()
