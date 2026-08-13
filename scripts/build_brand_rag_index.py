"""Build or refresh the local SmartPush brand RAG index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workflow.rag_store import DEFAULT_INDEX_PATH, build_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    args = parser.parse_args()
    payload = build_index(index_path=args.index_path, force=args.force)
    print(
        json.dumps(
            {
                "index_path": str(args.index_path),
                "embedding_model": payload["embedding_model"],
                "embedding_dimensions": payload["embedding_dimensions"],
                "chunk_count": payload["chunk_count"],
                "fingerprint": payload["fingerprint"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
