"""Controlled medium-term memory and long-term brand-asset access.

Medium-term memories and long-term asset candidates live in SQLite. Approved
long-term assets remain in the curated Brand RAG and are searched through the
existing RAG adapter. Automated forgetting is deliberately limited to SQLite
records; approved Brand RAG assets require the separate curation workflow.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, Sequence

from .rag_store import OpenAICompatibleEmbeddingClient, search_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEMORY_DB_PATH = PROJECT_ROOT / "data" / "memory" / "memory.sqlite3"
DEFAULT_BRAND_ID = "smartpush"
DEFAULT_NAMESPACE = "marketing"
DEFAULT_MAX_RECORDS = 5_000
VALID_LAYERS = {"mid_term", "long_term"}
VALID_STATUSES = {
    "candidate",
    "active",
    "superseded",
    "forgotten",
    "expired",
}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


class EmbeddingClient(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime | None = None) -> str:
    return (value or _now()).astimezone(UTC).isoformat()


def _parse_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid ISO datetime: {text}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_load(value: Any, fallback: Any) -> Any:
    try:
        return json.loads(str(value)) if value not in (None, "") else fallback
    except (TypeError, ValueError):
        return fallback


def _bounded_score(name: str, value: Any) -> float:
    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return score


def _normalized_layer(value: Any) -> str:
    aliases = {
        "episodic": "mid_term",
        "medium_term": "mid_term",
        "semantic": "long_term",
    }
    normalized = str(value or "mid_term").strip().casefold()
    layer = aliases.get(normalized, normalized)
    if layer not in VALID_LAYERS:
        raise ValueError("memory_layer must be mid_term or long_term")
    return layer


def _content_hash(content: str) -> str:
    normalized = " ".join(content.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _lexical_similarity(query: str, content: str) -> float:
    query_tokens = set(TOKEN_PATTERN.findall(query.casefold()))
    content_tokens = set(TOKEN_PATTERN.findall(content.casefold()))
    if not query_tokens or not content_tokens:
        return 0.0
    return len(query_tokens.intersection(content_tokens)) / len(query_tokens)


class SQLiteMemoryStore:
    """Small transactional store for episodic memories and asset candidates."""

    def __init__(self, path: Path | str = DEFAULT_MEMORY_DB_PATH):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 10000")
        self._initialize(connection)
        return connection

    @staticmethod
    def _initialize(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                brand_id TEXT NOT NULL,
                user_id TEXT,
                namespace TEXT NOT NULL,
                memory_layer TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                importance REAL NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                source_run_id TEXT,
                source_refs_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                embedding_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_accessed_at TEXT,
                access_count INTEGER NOT NULL DEFAULT 0,
                expires_at TEXT,
                supersedes_id TEXT,
                forgotten_at TEXT,
                forget_reason TEXT,
                pinned INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(supersedes_id) REFERENCES memories(memory_id)
            );
            CREATE INDEX IF NOT EXISTS idx_memories_scope
                ON memories(brand_id, user_id, namespace, memory_layer, status);
            CREATE INDEX IF NOT EXISTS idx_memories_type
                ON memories(brand_id, memory_type, status);
            CREATE INDEX IF NOT EXISTS idx_memories_retention
                ON memories(brand_id, status, pinned, importance, last_accessed_at);
            CREATE INDEX IF NOT EXISTS idx_memories_hash
                ON memories(brand_id, user_id, namespace, memory_type, content_hash);

            CREATE TABLE IF NOT EXISTS memory_events (
                event_id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                action TEXT NOT NULL,
                actor TEXT,
                reason TEXT,
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(memory_id) REFERENCES memories(memory_id)
            );
            CREATE INDEX IF NOT EXISTS idx_memory_events_memory
                ON memory_events(memory_id, created_at);
            """
        )
        connection.commit()

    @staticmethod
    def _record(row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        record["source_refs"] = _json_load(record.pop("source_refs_json", "[]"), [])
        record["metadata"] = _json_load(record.pop("metadata_json", "{}"), {})
        record["embedding"] = _json_load(record.pop("embedding_json", None), None)
        record["pinned"] = bool(record.get("pinned", 0))
        return record

    def add(self, record: dict[str, Any]) -> dict[str, Any]:
        """Insert or merge an exact duplicate inside one memory scope."""

        now = _iso()
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT * FROM memories
                WHERE brand_id = ?
                  AND user_id IS ?
                  AND namespace = ?
                  AND memory_layer = ?
                  AND memory_type = ?
                  AND content_hash = ?
                  AND status IN ('candidate', 'active')
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (
                    record["brand_id"],
                    record.get("user_id"),
                    record["namespace"],
                    record["memory_layer"],
                    record["memory_type"],
                    record["content_hash"],
                ),
            ).fetchone()
            if existing is not None:
                prior = self._record(existing)
                refs = list(
                    dict.fromkeys([*prior.get("source_refs", []), *record.get("source_refs", [])])
                )
                metadata = {**prior.get("metadata", {}), **record.get("metadata", {})}
                merged_embedding = prior.get("embedding") or record.get("embedding")
                connection.execute(
                    """
                    UPDATE memories
                    SET importance = ?, confidence = ?, source_refs_json = ?,
                        metadata_json = ?, source_run_id = COALESCE(?, source_run_id),
                        expires_at = COALESCE(?, expires_at),
                        embedding_json = COALESCE(embedding_json, ?), updated_at = ?
                    WHERE memory_id = ?
                    """,
                    (
                        max(float(prior["importance"]), float(record["importance"])),
                        max(float(prior["confidence"]), float(record["confidence"])),
                        _json_dump(refs),
                        _json_dump(metadata),
                        record.get("source_run_id"),
                        record.get("expires_at"),
                        _json_dump(merged_embedding) if merged_embedding else None,
                        now,
                        prior["memory_id"],
                    ),
                )
                self._event(
                    connection,
                    prior["memory_id"],
                    "merge",
                    record.get("actor"),
                    "Exact scoped duplicate merged",
                    {"source_refs": refs},
                )
                merged = connection.execute(
                    "SELECT * FROM memories WHERE memory_id = ?",
                    (prior["memory_id"],),
                ).fetchone()
                return {"operation": "merged", "memory": self._record(merged)}

            columns = (
                "memory_id", "brand_id", "user_id", "namespace", "memory_layer",
                "memory_type", "content", "content_hash", "importance", "confidence",
                "status", "source_run_id", "source_refs_json", "metadata_json",
                "embedding_json", "created_at", "updated_at", "last_accessed_at",
                "access_count", "expires_at", "supersedes_id", "forgotten_at",
                "forget_reason", "pinned",
            )
            values = (
                record["memory_id"], record["brand_id"], record.get("user_id"),
                record["namespace"], record["memory_layer"], record["memory_type"],
                record["content"], record["content_hash"], record["importance"],
                record["confidence"], record["status"], record.get("source_run_id"),
                _json_dump(record.get("source_refs", [])),
                _json_dump(record.get("metadata", {})),
                _json_dump(record["embedding"]) if record.get("embedding") else None,
                now, now, None, 0, record.get("expires_at"),
                record.get("supersedes_id"), None, None, int(bool(record.get("pinned"))),
            )
            placeholders = ",".join("?" for _ in columns)
            connection.execute(
                f"INSERT INTO memories ({','.join(columns)}) VALUES ({placeholders})",
                values,
            )
            if record.get("supersedes_id"):
                connection.execute(
                    """
                    UPDATE memories SET status = 'superseded', updated_at = ?
                    WHERE memory_id = ? AND brand_id = ?
                    """,
                    (now, record["supersedes_id"], record["brand_id"]),
                )
            self._event(
                connection,
                record["memory_id"],
                "add",
                record.get("actor"),
                "Memory added",
                {"memory_layer": record["memory_layer"], "status": record["status"]},
            )
            row = connection.execute(
                "SELECT * FROM memories WHERE memory_id = ?", (record["memory_id"],)
            ).fetchone()
            return {"operation": "created", "memory": self._record(row)}

    @staticmethod
    def _event(
        connection: sqlite3.Connection,
        memory_id: str,
        action: str,
        actor: str | None,
        reason: str,
        payload: dict[str, Any],
    ) -> None:
        connection.execute(
            """
            INSERT INTO memory_events
                (event_id, memory_id, action, actor, reason, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"evt_{uuid.uuid4().hex}", memory_id, action, actor, reason,
                _json_dump(payload), _iso(),
            ),
        )

    def candidates(
        self,
        *,
        brand_id: str,
        user_id: str | None,
        layers: Sequence[str],
        namespaces: Sequence[str] | None = None,
        memory_types: Sequence[str] | None = None,
        include_candidates: bool = False,
        include_expired: bool = False,
    ) -> list[dict[str, Any]]:
        statuses = ["active"]
        if include_candidates:
            statuses.append("candidate")
        where = ["brand_id = ?"]
        params: list[Any] = [brand_id]
        if user_id:
            where.append("(user_id IS NULL OR user_id = ?)")
            params.append(user_id)
        else:
            where.append("user_id IS NULL")
        where.append(f"memory_layer IN ({','.join('?' for _ in layers)})")
        params.extend(layers)
        where.append(f"status IN ({','.join('?' for _ in statuses)})")
        params.extend(statuses)
        if namespaces:
            where.append(f"namespace IN ({','.join('?' for _ in namespaces)})")
            params.extend(namespaces)
        if memory_types:
            where.append(f"memory_type IN ({','.join('?' for _ in memory_types)})")
            params.extend(memory_types)
        if not include_expired:
            where.append("(expires_at IS NULL OR expires_at > ?)")
            params.append(_iso())
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM memories WHERE {' AND '.join(where)}",
                params,
            ).fetchall()
        return [self._record(row) for row in rows]

    def touch(self, memory_ids: Sequence[str]) -> None:
        if not memory_ids:
            return
        now = _iso()
        with self._connect() as connection:
            connection.executemany(
                """
                UPDATE memories
                SET last_accessed_at = ?, access_count = access_count + 1,
                    updated_at = ?
                WHERE memory_id = ?
                """,
                [(now, now, memory_id) for memory_id in memory_ids],
            )

    def forget(
        self,
        *,
        brand_id: str,
        user_id: str | None,
        namespace: str | None,
        memory_layer: str,
        strategy: str,
        threshold: float | None,
        max_age_days: int | None,
        max_records: int,
        memory_id: str | None,
        dry_run: bool,
        actor: str | None,
    ) -> dict[str, Any]:
        """Select and soft-forget records inside a mandatory brand scope."""

        where = ["brand_id = ?", "memory_layer = ?", "pinned = 0"]
        params: list[Any] = [brand_id, memory_layer]
        if user_id:
            where.append("user_id = ?")
            params.append(user_id)
        else:
            where.append("user_id IS NULL")
        if namespace:
            where.append("namespace = ?")
            params.append(namespace)
        if memory_layer == "long_term":
            where.append("status = 'candidate'")
        else:
            where.append("status IN ('active', 'candidate')")
        if memory_id:
            where.append("memory_id = ?")
            params.append(memory_id)

        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM memories WHERE {' AND '.join(where)}",
                params,
            ).fetchall()
            records = [self._record(row) for row in rows]
            selected: list[dict[str, Any]]
            if strategy == "importance_based":
                if threshold is None:
                    raise ValueError("threshold is required for importance_based forgetting")
                selected = [item for item in records if float(item["importance"]) < threshold]
            elif strategy == "time_based":
                if max_age_days is None or int(max_age_days) < 1:
                    raise ValueError("max_age_days must be at least 1")
                cutoff = _now() - timedelta(days=int(max_age_days))
                selected = []
                for item in records:
                    reference = _parse_time(item.get("last_accessed_at")) or _parse_time(item["created_at"])
                    expires = _parse_time(item.get("expires_at"))
                    if (reference and reference < cutoff) or (expires and expires <= _now()):
                        selected.append(item)
            elif strategy == "capacity_based":
                if max_records < 1:
                    raise ValueError("max_records must be at least 1")
                excess = max(0, len(records) - max_records)
                eligible = [
                    item for item in records
                    if threshold is None or float(item["importance"]) <= threshold
                ]
                eligible.sort(
                    key=lambda item: (
                        float(item["importance"]),
                        str(item.get("last_accessed_at") or item["created_at"]),
                    )
                )
                selected = eligible[:excess]
            elif strategy == "targeted":
                if not memory_id:
                    raise ValueError("memory_id is required for targeted forgetting")
                selected = records
            else:
                raise ValueError(
                    "strategy must be importance_based, time_based, capacity_based, or targeted"
                )

            ids = [str(item["memory_id"]) for item in selected]
            if not dry_run and ids:
                now = _iso()
                reason = f"{strategy} memory policy"
                connection.executemany(
                    """
                    UPDATE memories
                    SET status = 'forgotten', forgotten_at = ?, forget_reason = ?,
                        updated_at = ?
                    WHERE memory_id = ?
                    """,
                    [(now, reason, now, memory_id_value) for memory_id_value in ids],
                )
                for memory_id_value in ids:
                    self._event(
                        connection,
                        memory_id_value,
                        "forget",
                        actor,
                        reason,
                        {"strategy": strategy},
                    )

        remaining_over_capacity = (
            max(0, len(records) - len(selected) - max_records)
            if strategy == "capacity_based"
            else 0
        )
        return {
            "strategy": strategy,
            "dry_run": bool(dry_run),
            "matched_count": len(selected),
            "forgotten_count": 0 if dry_run else len(selected),
            "memory_ids": ids,
            "recoverable": True,
            "remaining_over_capacity": remaining_over_capacity,
        }


class MemoryManager:
    """Policy layer shared by graph nodes and LangChain memory tools."""

    def __init__(
        self,
        *,
        store: SQLiteMemoryStore | None = None,
        embedding_client: EmbeddingClient | None = None,
        brand_search: Any | None = None,
        default_brand_id: str | None = None,
        default_namespace: str | None = None,
        max_records: int | None = None,
    ):
        configured_path = Path(
            os.getenv("MEMORY_DB_PATH", str(DEFAULT_MEMORY_DB_PATH))
        )
        if not configured_path.is_absolute():
            configured_path = PROJECT_ROOT / configured_path
        self.store = store or SQLiteMemoryStore(configured_path)
        self._embedding_client = embedding_client
        self._embedding_load_attempted = embedding_client is not None
        self.brand_search = brand_search or search_index
        self.default_brand_id = default_brand_id or os.getenv(
            "MEMORY_DEFAULT_BRAND_ID", DEFAULT_BRAND_ID
        )
        self.default_namespace = default_namespace or os.getenv(
            "MEMORY_DEFAULT_NAMESPACE", DEFAULT_NAMESPACE
        )
        self.max_records = max_records or max(
            1, int(os.getenv("MEMORY_MAX_ACTIVE_PER_BRAND", str(DEFAULT_MAX_RECORDS)))
        )

    def _embeddings(self) -> EmbeddingClient | None:
        if self._embedding_load_attempted:
            return self._embedding_client
        self._embedding_load_attempted = True
        if os.getenv("MEMORY_EMBEDDING_ENABLED", "true").casefold() not in {
            "1", "true", "yes", "on"
        }:
            return None
        try:
            self._embedding_client = OpenAICompatibleEmbeddingClient(
                os.getenv("MEMORY_EMBEDDING_MODEL") or None,
                timeout=float(os.getenv("MEMORY_EMBEDDING_TIMEOUT_SECONDS", "15")),
            )
        except Exception:
            self._embedding_client = None
        return self._embedding_client

    def _embed_one(self, text: str) -> tuple[list[float] | None, str]:
        client = self._embeddings()
        if client is None:
            return None, "embedding_unavailable"
        try:
            values = client.embed([text])
            if len(values) != 1 or not values[0]:
                return None, "embedding_invalid"
            return [float(value) for value in values[0]], "ready"
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def add(
        self,
        *,
        content: str,
        memory_type: str,
        brand_id: str | None = None,
        user_id: str | None = None,
        namespace: str | None = None,
        memory_layer: str = "mid_term",
        importance: float = 0.5,
        confidence: float = 0.7,
        status: str = "active",
        source_run_id: str | None = None,
        source_refs: Sequence[str] | None = None,
        metadata: dict[str, Any] | None = None,
        expires_at: str | None = None,
        supersedes_id: str | None = None,
        pinned: bool = False,
        actor: str | None = "memory_manager",
    ) -> dict[str, Any]:
        cleaned = " ".join(str(content or "").split()).strip()
        if not cleaned:
            raise ValueError("content must not be empty")
        cleaned_type = str(memory_type or "").strip()
        if not cleaned_type:
            raise ValueError("memory_type must not be empty")
        layer = _normalized_layer(memory_layer)
        normalized_status = str(status or "active").strip().casefold()
        if normalized_status not in VALID_STATUSES:
            raise ValueError(f"Unsupported memory status: {normalized_status}")
        if layer == "long_term":
            normalized_status = "candidate"
        expiry = _parse_time(expires_at)
        embedding, embedding_status = self._embed_one(cleaned)
        refs = [str(item) for item in (source_refs or []) if str(item).strip()]
        if layer == "long_term" and not refs:
            raise ValueError("long-term asset candidates require source_refs")
        record = {
            "memory_id": f"mem_{uuid.uuid4().hex}",
            "brand_id": str(brand_id or self.default_brand_id).strip(),
            "user_id": str(user_id).strip() if user_id else None,
            "namespace": str(namespace or self.default_namespace).strip(),
            "memory_layer": layer,
            "memory_type": cleaned_type,
            "content": cleaned,
            "content_hash": _content_hash(cleaned),
            "importance": _bounded_score("importance", importance),
            "confidence": _bounded_score("confidence", confidence),
            "status": normalized_status,
            "source_run_id": str(source_run_id).strip() if source_run_id else None,
            "source_refs": refs,
            "metadata": {
                **(metadata or {}),
                "embedding_status": embedding_status,
                "approved_for_external_use": False if layer == "long_term" else None,
            },
            "embedding": embedding,
            "expires_at": _iso(expiry) if expiry else None,
            "supersedes_id": supersedes_id,
            "pinned": bool(pinned),
            "actor": actor,
        }
        if not record["brand_id"] or not record["namespace"]:
            raise ValueError("brand_id and namespace must not be empty")
        result = self.store.add(record)
        result["memory_layer"] = layer
        result["requires_approval"] = layer == "long_term"
        return result

    def search(
        self,
        *,
        query: str,
        brand_id: str | None = None,
        user_id: str | None = None,
        namespaces: Sequence[str] | None = None,
        memory_types: Sequence[str] | None = None,
        memory_layers: Sequence[str] | None = None,
        purpose: str = "content_generation",
        top_k: int = 5,
        include_candidates: bool = False,
        include_expired: bool = False,
        corpora: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        cleaned_query = " ".join(str(query or "").split()).strip()
        if not cleaned_query:
            raise ValueError("query must not be empty")
        layers = [_normalized_layer(item) for item in (memory_layers or ["mid_term"])]
        layers = list(dict.fromkeys(layers))
        limit = max(1, min(20, int(top_k)))
        resolved_brand = str(brand_id or self.default_brand_id).strip()
        results: list[dict[str, Any]] = []

        if "mid_term" in layers or ("long_term" in layers and include_candidates):
            sqlite_layers = [layer for layer in layers if layer == "mid_term"]
            if "long_term" in layers and include_candidates:
                sqlite_layers.append("long_term")
            candidates = self.store.candidates(
                brand_id=resolved_brand,
                user_id=user_id,
                layers=sqlite_layers,
                namespaces=namespaces,
                memory_types=memory_types,
                include_candidates=include_candidates,
                include_expired=include_expired,
            )
            query_embedding = None
            if candidates and any(item.get("embedding") for item in candidates):
                query_embedding, _ = self._embed_one(cleaned_query)
            now = _now()
            for item in candidates:
                semantic = _cosine(query_embedding or [], item.get("embedding") or [])
                lexical = _lexical_similarity(cleaned_query, str(item.get("content", "")))
                relevance = max(semantic, lexical)
                created = _parse_time(item.get("created_at")) or now
                age_days = max(0.0, (now - created).total_seconds() / 86_400)
                freshness = 1.0 / (1.0 + age_days / 30.0)
                access_signal = min(1.0, math.log1p(int(item.get("access_count", 0))) / 4.0)
                score = (
                    0.45 * relevance
                    + 0.25 * float(item["importance"])
                    + 0.15 * float(item["confidence"])
                    + 0.10 * freshness
                    + 0.05 * access_signal
                )
                results.append(
                    {
                        **{key: value for key, value in item.items() if key != "embedding"},
                        "retrieval_score": round(score, 6),
                        "semantic_score": round(semantic, 6),
                        "lexical_score": round(lexical, 6),
                        "not_fact_evidence": item["memory_layer"] == "mid_term",
                    }
                )

        long_term_error = ""
        if "long_term" in layers:
            usage = "internal_strategy" if purpose == "internal_strategy" else "public_content"
            try:
                payload = self.brand_search(
                    cleaned_query,
                    corpora=corpora,
                    usage=usage,
                    top_k=limit,
                )
                for item in payload.get("results", []):
                    if not isinstance(item, dict):
                        continue
                    results.append(
                        {
                            "memory_id": str(item.get("chunk_id", "")),
                            "brand_id": resolved_brand,
                            "memory_layer": "long_term",
                            "memory_type": str(item.get("corpus", "brand_asset")),
                            "content": str(item.get("content", "")),
                            "status": "approved",
                            "importance": 1.0,
                            "confidence": 1.0,
                            "source_refs": [
                                str(value)
                                for value in (
                                    item.get("source_url"),
                                    item.get("secondary_source_url"),
                                )
                                if value
                            ],
                            "metadata": {
                                key: item.get(key)
                                for key in (
                                    "document_id", "chunk_id", "section_title",
                                    "authority", "usage_constraints", "knowledge_path",
                                )
                                if item.get(key) not in (None, "")
                            },
                            "retrieval_score": float(item.get("score", 0.0)),
                            "not_fact_evidence": False,
                        }
                    )
            except Exception as exc:
                long_term_error = f"{type(exc).__name__}: {exc}"

        results.sort(key=lambda item: float(item.get("retrieval_score", 0.0)), reverse=True)
        selected = results[:limit]
        self.store.touch(
            [
                str(item["memory_id"])
                for item in selected
                if item.get("memory_layer") == "mid_term"
                or item.get("status") == "candidate"
            ]
        )
        return {
            "query": cleaned_query,
            "brand_id": resolved_brand,
            "memory_layers": layers,
            "result_count": len(selected),
            "results": selected,
            "long_term_error": long_term_error,
        }

    def forget(
        self,
        *,
        strategy: str,
        brand_id: str | None = None,
        user_id: str | None = None,
        namespace: str | None = None,
        memory_layer: str = "mid_term",
        threshold: float | None = None,
        max_age_days: int | None = None,
        max_records: int | None = None,
        memory_id: str | None = None,
        dry_run: bool = False,
        actor: str | None = "memory_manager",
    ) -> dict[str, Any]:
        layer = _normalized_layer(memory_layer)
        normalized_strategy = str(strategy or "").strip().casefold()
        bounded_threshold = (
            _bounded_score("threshold", threshold) if threshold is not None else None
        )
        result = self.store.forget(
            brand_id=str(brand_id or self.default_brand_id).strip(),
            user_id=str(user_id).strip() if user_id else None,
            namespace=str(namespace).strip() if namespace else None,
            memory_layer=layer,
            strategy=normalized_strategy,
            threshold=bounded_threshold,
            max_age_days=max_age_days,
            max_records=max_records or self.max_records,
            memory_id=memory_id,
            dry_run=bool(dry_run),
            actor=actor,
        )
        result["memory_layer"] = layer
        result["approved_long_term_assets_protected"] = layer == "long_term"
        return result


_default_manager: MemoryManager | None = None


def get_default_memory_manager() -> MemoryManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = MemoryManager()
    return _default_manager
