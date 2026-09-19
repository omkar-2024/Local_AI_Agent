"""Upload one or more trusted local documents to the backend knowledge base.

Examples (Windows PowerShell):
    python scripts/ingest_knowledge.py data/knowledge_documents/safety_sop.pdf

    python scripts/ingest_knowledge.py data/knowledge_documents/*.pdf

The backend must be running on http://localhost:8000.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument(
        "--url",
        default="http://localhost:8000/api/v1/knowledge/ingest-file",
    )
    args = parser.parse_args()

    for path in args.files:
        if not path.is_file():
            print(f"SKIP: {path} does not exist")
            continue

        with path.open("rb") as handle:
            response = httpx.post(
                args.url,
                files={
                    "file": (
                        path.name,
                        handle,
                        "application/octet-stream",
                    )
                },
                timeout=300.0,
            )

        response.raise_for_status()
        print(response.json())


if __name__ == "__main__":
    main()
