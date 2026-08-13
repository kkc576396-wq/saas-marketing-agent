"""Publish completed marketing deliverables through the local Feishu CLI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_CLI = (
    PROJECT_ROOT
    / "tools"
    / "feishu-cli"
    / "node_modules"
    / "@larksuite"
    / "cli"
    / "bin"
    / "lark-cli"
)


class FeishuPublishError(RuntimeError):
    """Raised when the local Feishu CLI cannot create a document."""


def _load_environment() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except Exception:
        pass


def final_content_text(value: Any) -> str:
    """Extract the existing final deliverable for Markdown import."""

    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, dict):
        return ""
    for key in ("content", "report", "draft", "article", "reply"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def feishu_title(result: dict[str, Any]) -> str:
    """Choose a stable document title from the completed task state."""

    final_content = result.get("final_content")
    content_title = final_content.get("title") if isinstance(final_content, dict) else None
    plan = result.get("content_plan")
    plan_goal = plan.get("final_goal") if isinstance(plan, dict) else None
    title = next(
        (
            str(value).strip()
            for value in (content_title, plan_goal, result.get("topic"))
            if isinstance(value, str) and value.strip()
        ),
        "Marketing deliverable",
    )
    return " ".join(title.replace("*", "").split())[:800]


def _discover_cli_path() -> str:
    configured = os.getenv("FEISHU_CLI_PATH", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        return str(candidate)
    if PROJECT_CLI.is_file():
        return str(PROJECT_CLI)
    return shutil.which("lark-cli") or ""


def _json_payload(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    for line in reversed(raw.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass
class FeishuPublisher:
    """Import Markdown into Feishu Docs using an already authenticated CLI."""

    cli_path: str
    identity: str = ""
    timeout_seconds: float = 60.0
    runner: CommandRunner = field(default=subprocess.run, repr=False)

    @classmethod
    def from_environment(cls) -> "FeishuPublisher":
        _load_environment()
        return cls(
            cli_path=_discover_cli_path(),
            identity=os.getenv("FEISHU_CLI_IDENTITY", "").strip().casefold(),
            timeout_seconds=float(
                os.getenv("FEISHU_CLI_TIMEOUT_SECONDS", "60")
            ),
        )

    @property
    def configured(self) -> bool:
        return bool(self.cli_path and Path(self.cli_path).is_file())

    def publish(self, *, title: str, content: str) -> dict[str, Any]:
        clean_content = content.replace("*", "").strip()
        clean_title = title.replace("*", "").strip()
        if not clean_content:
            raise FeishuPublishError("没有可发布到飞书的最终正文。")
        if not self.configured:
            raise FeishuPublishError(
                "飞书 CLI 未安装。请先安装项目内的 @larksuite/cli，或设置 "
                "FEISHU_CLI_PATH。"
            )
        if self.identity not in {"", "bot", "user"}:
            raise FeishuPublishError("FEISHU_CLI_IDENTITY 只能是 bot、user 或留空。")

        command = [
            self.cli_path,
            "docs",
            "+create",
            "--doc-format",
            "markdown",
            "--title",
            clean_title,
            "--content",
            "-",
            "--json",
        ]
        if self.identity:
            command.extend(["--as", self.identity])

        environment = os.environ.copy()
        environment["LARKSUITE_CLI_NO_UPDATE_NOTIFIER"] = "1"
        environment["LARKSUITE_CLI_NO_SKILLS_NOTIFIER"] = "1"
        try:
            completed = self.runner(
                command,
                input=clean_content,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
                env=environment,
                cwd=PROJECT_ROOT,
            )
        except subprocess.TimeoutExpired as exc:
            raise FeishuPublishError(
                f"飞书 CLI 在 {self.timeout_seconds:g} 秒内未完成。"
            ) from exc
        except OSError as exc:
            raise FeishuPublishError(f"无法启动飞书 CLI：{exc}") from exc

        payload = _json_payload(completed.stdout) or _json_payload(completed.stderr)
        if completed.returncode != 0 or not payload or payload.get("ok") is not True:
            error = payload.get("error", {}) if isinstance(payload, dict) else {}
            message = str(error.get("message", "")).strip()
            hint = str(error.get("hint", "")).strip()
            detail = message or completed.stderr.strip() or completed.stdout.strip()
            if hint:
                detail = f"{detail}（{hint}）"
            raise FeishuPublishError(detail or "飞书 CLI 创建文档失败。")

        data = payload.get("data", {})
        document = data.get("document", {}) if isinstance(data, dict) else {}
        document_id = str(document.get("document_id", "")).strip()
        document_url = str(document.get("url", "")).strip()
        if not document_id or not document_url:
            raise FeishuPublishError("飞书 CLI 成功响应中缺少文档 ID 或 URL。")

        return {
            "status": "published",
            "document_id": document_id,
            "document_url": document_url,
            "title": clean_title,
            "identity": str(payload.get("identity", self.identity or "auto")),
            "permission_grant": data.get("permission_grant"),
        }
