"""Robust JSON extraction for OpenAI-compatible chat model responses."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


def _text_fragments(value: Any) -> list[str]:
    """Extract text without stringifying structured content blocks."""

    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        fragments: list[str] = []
        for key in ("text", "output_text", "content", "value"):
            if key in value:
                fragments.extend(_text_fragments(value[key]))
        return fragments
    if isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray)
    ):
        fragments: list[str] = []
        for item in value:
            fragments.extend(_text_fragments(item))
        return fragments

    for attribute in ("text", "output_text", "content"):
        nested = getattr(value, attribute, None)
        if nested is not None and nested is not value:
            fragments = _text_fragments(nested)
            if fragments:
                return fragments
    return []


def response_text(response: Any) -> str:
    """Return the visible assistant text from strings or content blocks."""

    content = getattr(response, "content", response)
    return "\n".join(
        fragment.strip()
        for fragment in _text_fragments(content)
        if fragment.strip()
    ).strip()


def _json_object_from_text(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE
    ).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        # Compatible providers sometimes prepend prose or a reasoning section.
        # raw_decode lets us select the first complete object instead of assuming
        # everything between the first and last brace belongs to one object.
        for match in re.finditer(r"\{", cleaned):
            try:
                payload, _ = decoder.raw_decode(cleaned[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        return None
    return payload if isinstance(payload, dict) else None


def parse_json_response(response: Any) -> dict[str, Any] | None:
    """Parse one JSON object across common compatible-provider formats."""

    content = getattr(response, "content", response)
    if isinstance(content, dict):
        return content

    primary_text = response_text(response)
    if primary_text:
        parsed = _json_object_from_text(primary_text)
        if parsed is not None:
            return parsed

    # Some reasoning-model gateways return an empty content field and place the
    # generated text in additional_kwargs. This is a fallback, not the default,
    # so visible final output always wins when both are present.
    additional = getattr(response, "additional_kwargs", {})
    if isinstance(additional, Mapping):
        for key in ("output_text", "content", "reasoning_content"):
            for fragment in _text_fragments(additional.get(key)):
                parsed = _json_object_from_text(fragment)
                if parsed is not None:
                    return parsed
    return None


def response_diagnostic(response: Any) -> str:
    """Create a short, non-secret diagnostic suitable for an exception."""

    metadata = getattr(response, "response_metadata", {})
    finish_reason = "unknown"
    if isinstance(metadata, Mapping):
        finish_reason = str(metadata.get("finish_reason", "unknown"))
    content = getattr(response, "content", response)
    preview = response_text(response).replace("\n", " ")[:180]
    return (
        f"content_type={type(content).__name__}, "
        f"finish_reason={finish_reason}, preview={preview!r}"
    )
