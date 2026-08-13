import json
from pathlib import Path

from workflow.rag_store import build_index, load_knowledge_chunks, search_index


class KeywordEmbeddings:
    def embed(self, texts):
        vectors = []
        for text in texts:
            value = text.casefold()
            vectors.append(
                [
                    float("product" in value or "automation" in value),
                    float("reddit" in value or "community" in value),
                    float("private" in value or "internal" in value),
                ]
            )
        return vectors


def _write_doc(path: Path, *, internal: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """---
document_id: {document_id}
corpus: {corpus}
authority: {authority}
visibility: {visibility}
approved_for_external_use: {approved}
source_title: Test source
source_url: https://example.test/source
usage_constraints: Test boundary.
---

# Knowledge

## {heading}

{content}
""".format(
            document_id="private-audience" if internal else "public-product",
            corpus="audience" if internal else "product",
            authority="confidential_internal" if internal else "official_public",
            visibility="internal_only" if internal else "public",
            approved="false" if internal else "true",
            heading="Internal audience" if internal else "Product automation",
            content=(
                "Private internal strategy for a narrow audience."
                if internal
                else "Product automation supports lifecycle campaigns."
            ),
        ),
        encoding="utf-8",
    )


def test_curated_repository_knowledge_has_stable_metadata():
    chunks = load_knowledge_chunks()

    assert len(chunks) >= 20
    assert all(chunk["chunk_id"] for chunk in chunks)
    assert all("usage_constraints" in chunk for chunk in chunks)
    assert any(chunk["visibility"] == "internal_only" for chunk in chunks)
    assert any(chunk["authority"] == "official_public" for chunk in chunks)


def test_vector_search_enforces_public_and_internal_boundaries(tmp_path):
    root = tmp_path / "knowledge"
    _write_doc(root / "product.md", internal=False)
    _write_doc(root / "audience.md", internal=True)
    index_path = tmp_path / "index.json"

    payload = build_index(
        root=root,
        index_path=index_path,
        embedding_client=KeywordEmbeddings(),
        model="qwen3.7-text-embedding",
    )
    public = search_index(
        "private internal audience",
        usage="public_content",
        top_k=5,
        index_path=index_path,
        embedding_client=KeywordEmbeddings(),
    )
    internal = search_index(
        "private internal audience",
        usage="internal_strategy",
        top_k=5,
        index_path=index_path,
        embedding_client=KeywordEmbeddings(),
    )

    assert payload["embedding_model"] == "qwen3.7-text-embedding"
    assert payload["chunk_count"] == 2
    assert {item["document_id"] for item in public["results"]} == {
        "public-product"
    }
    assert {item["document_id"] for item in internal["results"]} == {
        "private-audience",
        "public-product",
    }
    persisted = json.loads(index_path.read_text(encoding="utf-8"))
    assert persisted["embedding_dimensions"] == 3
