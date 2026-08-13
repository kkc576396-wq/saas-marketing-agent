"""LangGraph tools for Agent-Reach routed research.

Agent-Reach is an installer/doctor/router. The actual content calls are made
through the upstream tools selected by Agent-Reach, as recommended by its
official documentation. This adapter keeps those calls behind a small,
allowlisted LangGraph tool interface.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import feedparser
import requests
from langchain_core.tools import tool


MAX_QUERIES = 5
DEFAULT_MAX_RESULTS = 5
DEFAULT_COMMAND_TIMEOUT_SECONDS = 45
DEFAULT_RSS_TIMEOUT_SECONDS = 15
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_BIN = PROJECT_ROOT / "venv" / "bin"
Channel = Literal[
    "web",
    "rss",
    "github",
    "twitter",
    "reddit",
    "bilibili",
    "xiaohongshu",
    "facebook",
    "instagram",
    "v2ex",
]


class AgentReachError(RuntimeError):
    """Raised when an Agent-Reach route cannot be executed."""


def _find_command(command: str) -> str | None:
    """Find a command in the project venv first, then on PATH."""

    local_command = VENV_BIN / command
    if local_command.is_file() and os.access(local_command, os.X_OK):
        return str(local_command)
    return shutil.which(command)


def _run_command(command: list[str], timeout: int | None = None) -> str:
    """Run an allowlisted upstream command without invoking a shell."""

    effective_timeout = timeout or int(
        os.getenv(
            "AGENT_REACH_COMMAND_TIMEOUT_SECONDS",
            str(DEFAULT_COMMAND_TIMEOUT_SECONDS),
        )
    )
    env = os.environ.copy()
    env["PATH"] = f"{VENV_BIN}{os.pathsep}{env.get('PATH', '')}"
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=effective_timeout,
            check=False,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AgentReachError(f"Agent-Reach upstream command failed: {exc}") from exc

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise AgentReachError(detail or f"Command exited with {completed.returncode}")
    return completed.stdout.strip()


def _run_channel(channel: Channel, query: str, max_results: int) -> str:
    """Route one query using the corresponding Agent-Reach upstream path."""

    if channel == "web":
        if not query.startswith(("http://", "https://")):
            raise ValueError("web queries must be http(s) URLs")
        encoded_url = quote(query, safe=":/?=&%#@,+;$-_.!~*'()")
        curl = _find_command("curl")
        if not curl:
            raise AgentReachError("curl is required for Agent-Reach web reading")
        return _run_command([curl, "-sS", "--fail", f"https://r.jina.ai/{encoded_url}"])

    if channel == "rss":
        try:
            response = requests.get(
                query,
                headers={"User-Agent": "agent-reach/1.0"},
                timeout=float(
                    os.getenv(
                        "AGENT_REACH_RSS_TIMEOUT_SECONDS",
                        str(DEFAULT_RSS_TIMEOUT_SECONDS),
                    )
                ),
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise AgentReachError(f"Agent-Reach RSS request failed: {exc}") from exc
        parsed = feedparser.parse(response.content)
        entries = [
            {
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "summary": entry.get("summary", ""),
                "published": entry.get("published", entry.get("updated", "")),
            }
            for entry in parsed.entries[:max_results]
        ]
        return json.dumps(entries, ensure_ascii=False)

    if channel == "v2ex":
        response = requests.get(
            "https://www.v2ex.com/api/topics/hot.json",
            headers={"User-Agent": "agent-reach/1.0"},
            timeout=30,
        )
        response.raise_for_status()
        topics = response.json()
        if query.strip():
            terms = query.lower().split()
            topics = [
                topic
                for topic in topics
                if all(term in str(topic).lower() for term in terms)
            ]
        return json.dumps(topics[:max_results], ensure_ascii=False)

    command_name = {
        "github": "gh",
        "twitter": "twitter",
        "bilibili": "bili",
    }.get(channel)
    if command_name:
        executable = _find_command(command_name)
        if not executable:
            raise AgentReachError(f"Agent-Reach channel '{channel}' requires {command_name}")
        if channel == "github":
            return _run_command(
                [executable, "search", "repos", query, "--sort", "stars", "--limit", str(max_results)]
            )
        if channel == "twitter":
            return _run_command([executable, "search", query, "-n", str(max_results)])
        return _run_command(
            [executable, "search", query, "--type", "video", "-n", str(max_results)]
        )

    if channel == "reddit":
        executable = _find_command("rdt")
        if executable:
            return _run_command([executable, "search", query, "--limit", str(max_results)])
        executable = _find_command("opencli")
        if executable:
            return _run_command([executable, "reddit", "search", query, "-f", "yaml"])
        raise AgentReachError("Agent-Reach Reddit requires rdt-cli or OpenCLI")

    if channel in {"xiaohongshu", "facebook", "instagram"}:
        executable = _find_command("opencli")
        if not executable:
            raise AgentReachError(f"Agent-Reach channel '{channel}' requires OpenCLI")
        return _run_command([executable, channel, "search", query, "-f", "yaml"])

    raise ValueError(f"Unsupported Agent-Reach channel: {channel}")


@tool
def agent_reach_search(
    channel: Channel,
    queries: list[str],
    max_results: int = DEFAULT_MAX_RESULTS,
) -> str:
    """Run up to five research queries through an Agent-Reach channel.

    Agent-Reach chooses the upstream backend; this tool only exposes a safe
    allowlist of channel routes. Results are returned as a JSON array so a
    ResearchState node can append them to its document collection.
    """

    if not 1 <= len(queries) <= MAX_QUERIES:
        raise ValueError(f"queries must contain between 1 and {MAX_QUERIES} items")
    if not 1 <= max_results <= 10:
        raise ValueError("max_results must be between 1 and 10")

    results: list[dict[str, Any]] = []
    for query in queries:
        if not query.strip():
            raise ValueError("queries must not contain empty strings")
        results.append(
            {
                "channel": channel,
                "query": query,
                "content": _run_channel(channel, query, max_results),
            }
        )
    return json.dumps(results, ensure_ascii=False)


@tool
def agent_reach_doctor() -> str:
    """Return machine-readable Agent-Reach channel availability."""

    try:
        from agent_reach import AgentReach

        return json.dumps(AgentReach().doctor(), ensure_ascii=False)
    except Exception as exc:  # pragma: no cover - depends on local install
        raise AgentReachError(f"Agent-Reach doctor failed: {exc}") from exc


AGENT_REACH_TOOLS = [agent_reach_search, agent_reach_doctor]
