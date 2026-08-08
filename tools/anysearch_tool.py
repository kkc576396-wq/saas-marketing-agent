"""LangChain/LangGraph tools backed by the local AnySearch skill.

This module intentionally contains only the provider adapter. Research-agent
planning, synthesis, and citation logic belong in the workflow layer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests
from langchain_core.tools import tool


ANYSEARCH_ENDPOINT = "https://api.anysearch.com/mcp"
ANYSEARCH_CLIENT = "skill/3.0.1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AnySearchError(RuntimeError):
    """Raised when AnySearch cannot complete a tool call."""


def _load_project_env() -> None:
    """Load non-empty values from the project .env without overwriting env vars."""

    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("\"'")
        if key.strip() and value and key.strip() not in os.environ:
            os.environ[key.strip()] = value


def _api_key(explicit_key: str | None = None) -> str:
    """Resolve an explicit key first, then the project's environment."""

    _load_project_env()
    return explicit_key or os.getenv("ANYSEARCH_API_KEY", "")


def _call_anysearch(
    tool_name: str,
    arguments: dict[str, Any],
    api_key: str | None = None,
) -> str:
    """Call AnySearch's JSON-RPC endpoint and return its text content."""

    headers = {
        "Content-Type": "application/json",
        "X-Anysearch-Client": ANYSEARCH_CLIENT,
    }
    key = _api_key(api_key)
    if key:
        headers["Authorization"] = f"Bearer {key}"

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }

    try:
        response = requests.post(
            ANYSEARCH_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise AnySearchError(f"AnySearch request failed: {exc}") from exc
    except ValueError as exc:
        raise AnySearchError("AnySearch returned invalid JSON") from exc

    if data.get("error"):
        error = data["error"]
        message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
        raise AnySearchError(f"AnySearch API error: {message}")

    result = data.get("result", {})
    for item in result.get("content", []):
        if item.get("type") == "text":
            return item.get("text", "")

    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def anysearch_search(
    query: str,
    domain: str | None = None,
    sub_domain: str | None = None,
    sub_domain_params: dict[str, Any] | None = None,
    max_results: int = 5,
) -> str:
    """Search the web or a supported vertical domain with AnySearch.

    For vertical searches, call ``anysearch_get_sub_domains`` first to discover
    the valid ``sub_domain`` and required parameters.
    """

    if not query.strip():
        raise ValueError("query must not be empty")
    if not 1 <= max_results <= 10:
        raise ValueError("max_results must be between 1 and 10")

    arguments: dict[str, Any] = {"query": query, "max_results": max_results}
    if domain:
        arguments["domain"] = domain
    if sub_domain:
        arguments["sub_domain"] = sub_domain
    if sub_domain_params:
        arguments["sub_domain_params"] = sub_domain_params
    return _call_anysearch("search", arguments)


@tool
def anysearch_batch_search(queries: list[dict[str, Any]]) -> str:
    """Run up to five independent searches in one AnySearch request."""

    if not 1 <= len(queries) <= 5:
        raise ValueError("queries must contain between 1 and 5 items")
    if any(not item.get("query") for item in queries):
        raise ValueError("each query item must contain a non-empty query")
    return _call_anysearch("batch_search", {"queries": queries})


@tool
def anysearch_get_sub_domains(
    domain: str | None = None,
    domains: list[str] | None = None,
) -> str:
    """Discover vertical-search sub-domains and their parameter schemas."""

    if bool(domain) == bool(domains):
        raise ValueError("provide exactly one of domain or domains")
    arguments: dict[str, Any] = {"domain": domain} if domain else {"domains": domains}
    return _call_anysearch("get_sub_domains", arguments)


@tool
def anysearch_extract(url: str) -> str:
    """Extract the full content of an HTML URL as Markdown."""

    if not url.startswith(("http://", "https://")):
        raise ValueError("url must start with http:// or https://")
    return _call_anysearch("extract", {"url": url})


ANYSEARCH_TOOLS = [
    anysearch_search,
    anysearch_batch_search,
    anysearch_get_sub_domains,
    anysearch_extract,
]

