"""
Author: Luigi Saetta
Date last modified: 2026-02-16
Python Version: 3.11
License: MIT

Description:
    Delete all chunks for one or more documents from an existing
    Oracle Vector Search collection, using METADATA.source.
"""

from __future__ import annotations

import argparse
import re
from typing import Dict, List

from multimodal_extraction.db.db_utils import get_db_connection
from multimodal_extraction.db.oraclevs_admin import OracleVSAdmin
from multimodal_extraction.utils import get_console_logger

logger = get_console_logger()

_IDENT_RE = re.compile(r"^[A-Z][A-Z0-9_$#]*$")


def _safe_ident(name: str) -> str:
    normalized = name.strip().upper()
    if not _IDENT_RE.fullmatch(normalized):
        raise ValueError(f"Unsafe or invalid Oracle identifier: {name!r}")
    return normalized


def _count_chunks_by_source(conn, collection_name: str, sources: List[str]) -> Dict[str, int]:
    """
    Return per-source chunk counts for the provided sources.
    """
    out: Dict[str, int] = {}
    for src in sources:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT COUNT(*)
                FROM {collection_name}
                WHERE json_value(METADATA, '$.source') = :doc
                """,
                [src],
            )
            row = cur.fetchone()
            out[src] = int(row[0]) if row and row[0] is not None else 0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete all chunks for one or more documents from a collection."
    )
    parser.add_argument(
        "collection_name",
        type=str,
        help="Existing collection table name (Oracle identifier).",
    )
    parser.add_argument(
        "doc_names",
        type=str,
        nargs="+",
        help="One or more document source names (METADATA.source) to delete.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview counts only; do not delete.",
    )
    args = parser.parse_args()

    collection_name = _safe_ident(args.collection_name)
    doc_names = [d.strip() for d in args.doc_names if d and d.strip()]
    if not doc_names:
        raise SystemExit("No valid document names provided.")

    with get_db_connection() as conn:
        existing = OracleVSAdmin.list_collections(conn)
        if collection_name not in existing:
            raise SystemExit(
                f"Collection not found: {collection_name}. Existing collections: {', '.join(existing[:20])}"
            )

        counts_before = _count_chunks_by_source(conn, collection_name, doc_names)

        logger.info("")
        logger.info("Collection: %s", collection_name)
        logger.info("Documents requested: %d", len(doc_names))
        total_to_delete = 0
        for src in doc_names:
            n = counts_before.get(src, 0)
            total_to_delete += n
            logger.info("  - %s -> %d chunk(s)", src, n)
        logger.info("Total chunk(s) matched: %d", total_to_delete)

        if args.dry_run:
            logger.info("Dry-run mode: no rows deleted.")
            logger.info("")
            return

        OracleVSAdmin.delete_documents(conn, collection_name, doc_names)

        counts_after = _count_chunks_by_source(conn, collection_name, doc_names)
        remaining = sum(counts_after.values())
        logger.info("Delete completed.")
        logger.info("Remaining chunk(s) for requested docs: %d", remaining)
        logger.info("")


if __name__ == "__main__":
    main()

