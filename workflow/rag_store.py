"""Small persistent vector store for curated SmartPush brand knowledge."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Protocol, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_ROOT = PROJECT_ROOT / "knowledge"
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "rag" / "brand_index.json"
DEFAULT_EMBEDDING_MODEL = "qwen3.7-text-embedding"
INDEX_VERSION = 1
MAX_TOP_K = 10


class EmbeddingClient(Protocol):
    """Minimal interface used by the index builder and search tests."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class OpenAICompatibleEmbeddingClient:
    """Embedding adapter sharing the project's OpenAI-compatible endpoint."""

    def __init__(self, model: str | None = None, timeout: float | None = None):
        try:
            from dotenv import load_dotenv

            load_dotenv(PROJECT_ROOT / ".env", override=False)
        except Exception:
            pass

        api_key = (
            os.getenv("RAG_EMBEDDING_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        ).strip()
        if not api_key:
            raise RuntimeError(
                "RAG_EMBEDDING_API_KEY or OPENAI_API_KEY is required to build/search the RAG index"
            )

        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": api_key}
        base_url = (
            os.getenv("RAG_EMBEDDING_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or ""
        ).strip()
        if base_url:
            kwargs["base_url"] = base_url
        if timeout is not None:
            kwargs["timeout"] = max(1.0, float(timeout))
        self.client = OpenAI(**kwargs)
        self.model = model or os.getenv(
            "RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )
        self.batch_size = max(
            1, min(20, int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "20")))
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        normalized = [str(text) for text in texts]
        for start in range(0, len(normalized), self.batch_size):
            response = self.client.embeddings.create(
                model=self.model,
                input=normalized[start : start + self.batch_size],
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend([list(item.embedding) for item in ordered])
        return vectors


def _frontmatter_value(value: str) -> Any:
    cleaned = value.strip()
    if cleaned.casefold() in {"true", "false"}:
        return cleaned.casefold() == "true"
    return cleaned


def _parse_document(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"Knowledge document is missing front matter: {path}")
    try:
        header, body = text[4:].split("\n---\n", 1)
    except ValueError as exc:
        raise ValueError(f"Knowledge document has invalid front matter: {path}") from exc

    metadata: dict[str, Any] = {}
    for line in header.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition(":")
        if not separator:
            raise ValueError(f"Invalid front matter line in {path}: {line}")
        metadata[key.strip()] = _frontmatter_value(value)
    required = {
        "document_id",
        "corpus",
        "authority",
        "visibility",
        "approved_for_external_use",
        "source_title",
        "source_url",
        "usage_constraints",
    }
    missing = sorted(required.difference(metadata))
    if missing:
        raise ValueError(f"Knowledge document {path} is missing: {', '.join(missing)}")
    try:
        metadata["knowledge_path"] = str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        # Custom roots are useful for isolated tests and one-off imports.
        metadata["knowledge_path"] = str(path)
    return metadata, body.strip()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _section_chunks(body: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"(?m)^##\s+(.+?)\s*$", body))
    if not matches:
        title = re.sub(r"(?m)^#\s+", "", body.splitlines()[0]).strip()
        return [(title or "Knowledge", body)]

    chunks: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        section = body[match.start() : end].strip()
        if section:
            chunks.append((match.group(1).strip(), section))
    return chunks


def load_knowledge_chunks(root: Path = KNOWLEDGE_ROOT) -> list[dict[str, Any]]:
    """Load curated Markdown sections as stable vector-store chunks."""

    chunks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for path in sorted(root.rglob("*.md")):
        metadata, body = _parse_document(path)
        document_id = str(metadata["document_id"])
        for index, (section_title, content) in enumerate(
            _section_chunks(body), start=1
        ):
            chunk_id = f"{document_id}:{index:02d}-{_slug(section_title)}"
            if chunk_id in seen_ids:
                raise ValueError(f"Duplicate knowledge chunk ID: {chunk_id}")
            seen_ids.add(chunk_id)
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "section_title": section_title,
                    "content": content,
                    **metadata,
                }
            )
    if not chunks:
        raise ValueError(f"No knowledge Markdown files found in {root}")
    return chunks


def _corpus_fingerprint(chunks: Sequence[dict[str, Any]], model: str) -> str:
    payload = [
        {
            key: chunk.get(key)
            for key in sorted(chunk)
            if key != "embedding"
        }
        for chunk in chunks
    ]
    raw = json.dumps(
        {"model": model, "chunks": payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_index(
    *,
    root: Path = KNOWLEDGE_ROOT,
    index_path: Path = DEFAULT_INDEX_PATH,
    embedding_client: EmbeddingClient | None = None,
    model: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Embed curated chunks and persist a dependency-free JSON index."""

    active_model = model or os.getenv(
        "RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
    )
    chunks = load_knowledge_chunks(root)
    fingerprint = _corpus_fingerprint(chunks, active_model)
    if index_path.exists() and not force:
        existing = json.loads(index_path.read_text(encoding="utf-8"))
        if (
            existing.get("fingerprint") == fingerprint
            and existing.get("embedding_model") == active_model
        ):
            return existing

    client = embedding_client or OpenAICompatibleEmbeddingClient(active_model)
    vectors = client.embed([chunk["content"] for chunk in chunks])
    if len(vectors) != len(chunks):
        raise ValueError("Embedding endpoint returned an unexpected vector count")
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1 or not dimensions or next(iter(dimensions)) < 1:
        raise ValueError("Embedding endpoint returned invalid vector dimensions")

    indexed_chunks = [
        {**chunk, "embedding": [float(value) for value in vector]}
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    payload = {
        "index_version": INDEX_VERSION,
        "embedding_model": active_model,
        "embedding_dimensions": next(iter(dimensions)),
        "fingerprint": fingerprint,
        "chunk_count": len(indexed_chunks),
        "chunks": indexed_chunks,
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return payload


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Query and index embedding dimensions do not match")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def search_index(
    query: str,
    *,
    corpora: Sequence[str] | None = None,
    usage: str = "public_content",
    top_k: int = 6,
    index_path: Path = DEFAULT_INDEX_PATH,
    embedding_client: EmbeddingClient | None = None,
) -> dict[str, Any]:
    """Search the persistent index while enforcing external-use boundaries."""

    cleaned_query = str(query or "").strip()
    if not cleaned_query:
        raise ValueError("RAG query must not be empty")
    if usage not in {"public_content", "internal_strategy"}:
        raise ValueError("usage must be public_content or internal_strategy")
    if not index_path.exists():
        raise FileNotFoundError(
            f"RAG index not found at {index_path}. Run scripts/build_brand_rag_index.py first."
        )

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    model = str(payload.get("embedding_model") or DEFAULT_EMBEDDING_MODEL)
    client = embedding_client or OpenAICompatibleEmbeddingClient(model)
    query_vectors = client.embed([cleaned_query])
    if len(query_vectors) != 1:
        raise ValueError("Embedding endpoint did not return one query vector")

    allowed_corpora = {
        str(item).strip().casefold() for item in (corpora or []) if str(item).strip()
    }
    candidates: list[dict[str, Any]] = []
    for chunk in payload.get("chunks", []):
        if not isinstance(chunk, dict):
            continue
        if allowed_corpora and str(chunk.get("corpus", "")).casefold() not in allowed_corpora:
            continue
        if usage == "public_content" and not bool(
            chunk.get("approved_for_external_use", False)
        ):
            continue
        score = _cosine_similarity(query_vectors[0], chunk.get("embedding", []))
        candidates.append({**chunk, "score": round(score, 6)})

    candidates.sort(key=lambda item: item["score"], reverse=True)
    limit = max(1, min(MAX_TOP_K, int(top_k)))
    results = []
    for chunk in candidates[:limit]:
        results.append(
            {
                key: chunk.get(key)
                for key in (
                    "chunk_id",
                    "document_id",
                    "section_title",
                    "content",
                    "corpus",
                    "authority",
                    "visibility",
                    "approved_for_external_use",
                    "source_title",
                    "source_url",
                    "secondary_source_url",
                    "retrieved_at",
                    "usage_constraints",
                    "knowledge_path",
                    "score",
                )
                if chunk.get(key) not in (None, "")
            }
        )
    return {
        "query": cleaned_query,
        "usage": usage,
        "corpora": sorted(allowed_corpora),
        "embedding_model": model,
        "result_count": len(results),
        "results": results,
    }


def get_chunks_by_ids(
    chunk_ids: Sequence[str],
    *,
    index_path: Path = DEFAULT_INDEX_PATH,
) -> list[dict[str, Any]]:
    """Read exact previously retrieved chunks without another embedding call."""

    requested = [
        str(chunk_id).strip() for chunk_id in chunk_ids if str(chunk_id).strip()
    ]
    if not requested or not index_path.exists():
        return []
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    chunks_by_id = {
        str(chunk.get("chunk_id", "")): chunk
        for chunk in payload.get("chunks", [])
        if isinstance(chunk, dict) and chunk.get("chunk_id")
    }
    fields = (
        "chunk_id",
        "document_id",
        "section_title",
        "content",
        "corpus",
        "authority",
        "visibility",
        "approved_for_external_use",
        "source_title",
        "source_url",
        "secondary_source_url",
        "retrieved_at",
        "usage_constraints",
        "knowledge_path",
    )
    return [
        {
            key: chunks_by_id[chunk_id].get(key)
            for key in fields
            if chunks_by_id[chunk_id].get(key) not in (None, "")
        }
        for chunk_id in requested
        if chunk_id in chunks_by_id
    ]
