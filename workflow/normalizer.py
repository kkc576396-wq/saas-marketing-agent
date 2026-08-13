"""Normalize provider responses into atomic, deduplicated research documents."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml


BLOCKED_CONTENT_MARKERS = (
    "captcha",
    "security verification",
    "performing security verification",
)
VENDOR_DOMAIN_MARKERS = (
    "klaviyo.com",
    "omnisend.com",
    "mailchimp.com",
    "hubspot.com",
    "drip.com",
    "attentive.com",
    "yotpo.com",
)
CURRENT_PAGE_MARKERS = (
    "pricing",
    "features",
    "release-notes",
    "changelog",
)


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def canonical_url(value: Any) -> str | None:
    """Return a stable URL suitable for document de-duplication."""

    url = _clean_text(value).rstrip(".,;)]}")
    if not url.startswith(("http://", "https://")):
        return None
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), parsed.query, ""))


def _stable_id(prefix: str, *values: Any) -> str:
    material = "|".join(_clean_text(value) for value in values if _clean_text(value))
    digest = hashlib.sha1(material.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _published_at(value: Any) -> str | None:
    if value in (None, ""):
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OverflowError):
        text = _clean_text(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(text)
        except (TypeError, ValueError, OverflowError):
            date_match = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text)
            if not date_match:
                return None
            try:
                parsed = datetime(
                    int(date_match.group(1)),
                    int(date_match.group(2)),
                    int(date_match.group(3)),
                    tzinfo=timezone.utc,
                )
            except ValueError:
                return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _date_from_text(value: str) -> tuple[str | None, float]:
    """Extract a publication date from search snippets with confidence."""

    text = _clean_text(value)
    exact_match = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text)
    if exact_match:
        parsed = _published_at(exact_match.group(0))
        return parsed, 0.9 if parsed else 0.0

    month_pattern = (
        r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
        r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
        r"Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2}),?\s+(20\d{2})\b"
    )
    month_match = re.search(month_pattern, text, re.IGNORECASE)
    if month_match:
        for pattern in ("%B %d %Y", "%b %d %Y"):
            try:
                parsed = datetime.strptime(
                    f"{month_match.group(1)} {month_match.group(2)} {month_match.group(3)}",
                    pattern,
                ).replace(tzinfo=timezone.utc)
                return parsed.isoformat(), 0.8
            except ValueError:
                continue

    month_year_match = re.search(
        month_pattern.replace(r"\s+(\d{1,2}),?", ""),
        text,
        re.IGNORECASE,
    )
    if month_year_match:
        for pattern in ("%B %Y", "%b %Y"):
            try:
                parsed = datetime.strptime(
                    f"{month_year_match.group(1)} {month_year_match.group(2)}",
                    pattern,
                ).replace(tzinfo=timezone.utc)
                return parsed.isoformat(), 0.6
            except ValueError:
                continue

    year_match = re.search(r"\b(20\d{2})\b", text)
    if year_match:
        parsed = datetime(int(year_match.group(1)), 1, 1, tzinfo=timezone.utc)
        return parsed.isoformat(), 0.4
    return None, 0.0


def _page_type(url: str, title: str) -> str:
    material = f"{url} {title}".casefold()
    if "pricing" in material:
        return "pricing_page"
    if any(marker in material for marker in CURRENT_PAGE_MARKERS):
        return "current_product_page"
    return "article"


def _source_bias(source_type: str, url: str) -> str:
    hostname = urlsplit(url).netloc.casefold()
    if "Reddit" in source_type or hostname == "reddit.com" or hostname.endswith(".reddit.com"):
        return "community_source"
    if any(hostname == marker or hostname.endswith(f".{marker}") for marker in VENDOR_DOMAIN_MARKERS):
        return "vendor_source"
    return "independent_source"


def _is_blocked(text: str) -> bool:
    lowered = text.casefold()
    return any(marker in lowered for marker in BLOCKED_CONTENT_MARKERS)


def _document(
    *,
    document_id: str,
    source_type: str,
    title: str,
    summary: str,
    url: str | None,
    published_at: str | None,
    published_date_confidence: float | None = None,
    raw_content: str,
    query: str,
    iteration: int,
) -> dict[str, Any] | None:
    title = _clean_text(title)
    summary = _clean_text(summary)
    if not title or not summary or not url or _is_blocked(raw_content):
        return None
    normalized_published_at = _published_at(published_at)
    page_type = _page_type(url, title)
    if normalized_published_at:
        date_status = "confirmed"
        date_confidence = (
            float(published_date_confidence)
            if published_date_confidence is not None
            else 1.0
        )
    elif page_type in {"pricing_page", "current_product_page"}:
        date_status = "current_page"
        date_confidence = 0.6
    else:
        date_status = "unknown"
        date_confidence = 0.0
    return {
        "document_id": document_id,
        "source_type": source_type,
        "title": title[:300],
        "summary": summary[:2000],
        "url": url,
        "published_at": normalized_published_at,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "date_status": date_status,
        "date_confidence": date_confidence,
        "page_type": page_type,
        "source_bias": _source_bias(source_type, url),
        "raw_content": raw_content,
        "query": query,
        "iteration": iteration,
    }


def _parse_reddit_content(tool_result: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        payloads = json.loads(str(tool_result.get("content", "")))
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(payloads, list):
        return []

    documents: list[dict[str, Any]] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        query = _clean_text(payload.get("query"))
        raw_content = str(payload.get("content") or "")
        if not raw_content.strip() or raw_content.strip() in {"[]", "{}"}:
            continue
        try:
            posts = yaml.safe_load(raw_content)
        except yaml.YAMLError:
            continue
        if not isinstance(posts, list):
            continue
        for post in posts:
            if not isinstance(post, dict):
                continue
            post_id = _clean_text(post.get("id"))
            title = _clean_text(post.get("title"))
            body = _clean_text(post.get("selftext") or post.get("text") or title)
            url = canonical_url(post.get("url") or post.get("permalink"))
            if not post_id:
                post_id = _stable_id("post", url or title, body[:200])
            document = _document(
                document_id=f"reddit-{post_id}",
                source_type="Agent-Reach Reddit",
                title=title,
                summary=body,
                url=url,
                published_at=_published_at(post.get("created_utc")),
                raw_content=raw_content,
                query=query,
                iteration=int(tool_result.get("iteration", 0)),
            )
            if document:
                documents.append(document)
    return documents


def _parse_anysearch_content(tool_result: dict[str, Any]) -> list[dict[str, Any]]:
    content = str(tool_result.get("content") or "")
    lines = content.splitlines()
    documents: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    blocks: list[str] = []

    def flush() -> None:
        if not current:
            return
        title = str(current.get("title", ""))
        url = canonical_url(current.get("url"))
        summary = _clean_text(" ".join(blocks))
        if title and url and summary:
            published_at = current.get("published_at")
            date_confidence = 1.0 if published_at else None
            if not published_at:
                published_at, date_confidence = _date_from_text(
                    f"{title} {summary[:1000]}"
                )
            documents.append(
                _document(
                    document_id=_stable_id("anysearch", url),
                    source_type="AnySearch",
                    title=title,
                    summary=summary,
                    url=url,
                    published_at=published_at,
                    published_date_confidence=date_confidence,
                    raw_content="\n".join(blocks),
                    query=str((tool_result.get("queries") or [""])[0]),
                    iteration=int(tool_result.get("iteration", 0)),
                )
            )

    for line in lines:
        if line.startswith("### "):
            flush()
            current = {"title": re.sub(r"^###\s+\d+\.\s*", "", line)}
            blocks = []
            continue
        if current is None:
            continue
        if line.strip().startswith("- **URL**:"):
            current["url"] = line.split(":", 1)[1].strip()
            continue
        if re.match(r"\s*- \*\*(?:Published(?: Time)?|Date)\*\*:", line, re.IGNORECASE):
            current["published_at"] = line.split(":", 1)[1].strip()
            continue
        if line.startswith("## Search Results") or line.strip() == "---":
            continue
        if line.strip():
            blocks.append(line.strip())
    flush()
    return [document for document in documents if document]


def _parse_web_or_rss_content(tool_result: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        payloads = json.loads(str(tool_result.get("content", "")))
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(payloads, list):
        return []

    source = str(tool_result.get("source", ""))
    source_type = "Agent-Reach RSS" if "RSS" in source else "Agent-Reach Web"
    documents: list[dict[str, Any]] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        raw_content = str(payload.get("content") or payload.get("summary") or "")
        url = canonical_url(payload.get("query") or payload.get("link"))
        source_url = re.search(r"URL Source:\s*(https?://\S+)", raw_content)
        url = canonical_url(source_url.group(1)) if source_url else url
        title_match = re.search(r"Title:\s*(.+)", raw_content)
        title = _clean_text(title_match.group(1) if title_match else payload.get("title") or url)
        published_match = re.search(r"Published Time:\s*(.+)", raw_content)
        published_at = (
            _clean_text(published_match.group(1))
            if published_match
            else payload.get("published_at") or payload.get("published")
        )
        summary = raw_content.split("Markdown Content:", 1)[-1]
        if source_type == "Agent-Reach RSS":
            summary = payload.get("summary") or summary
        document = _document(
            document_id=_stable_id("web" if source_type.endswith("Web") else "rss", url or title),
            source_type=source_type,
            title=title,
            summary=str(summary),
            url=url,
            published_at=_clean_text(published_at) or None,
            raw_content=raw_content,
            query=str(payload.get("query") or payload.get("link") or ""),
            iteration=int(tool_result.get("iteration", 0)),
        )
        if document:
            documents.append(document)
    return documents


def normalize_tool_result(tool_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse one aggregate provider response into atomic research documents."""

    source = str(tool_result.get("source", ""))
    if source == "Agent-Reach Reddit":
        return _parse_reddit_content(tool_result)
    if source == "AnySearch":
        return _parse_anysearch_content(tool_result)
    if source in {"Agent-Reach Web", "Agent-Reach RSS"}:
        return _parse_web_or_rss_content(tool_result)
    return []


def deduplicate_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first occurrence of each source document across iterations."""

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for document in documents:
        identity = str(document.get("document_id") or canonical_url(document.get("url")) or "")
        if not identity or identity in seen:
            continue
        seen.add(identity)
        unique.append(document)
    return unique
