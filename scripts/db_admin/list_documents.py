"""
Author: Luigi Saetta
Date last modified: 2026-02-25
Python Version: 3.11
License: MIT

Description:
    Utility to list document sources (METADATA.source) in an existing
    Oracle Vector Search collection.
"""

import argparse

from multimodal_extraction.db.db_utils import get_db_connection
from multimodal_extraction.db.oraclevs_admin import OracleVSAdmin
from multimodal_extraction.utils import get_console_logger


def main() -> None:
    logger = get_console_logger()

    parser = argparse.ArgumentParser(
        description="List document names (METADATA.source) in a collection."
    )
    parser.add_argument(
        "collection_name",
        type=str,
        help="Existing collection table name (Oracle identifier).",
    )
    args = parser.parse_args()

    with get_db_connection() as conn:
        existing = OracleVSAdmin.list_collections(conn)
        collection_name = args.collection_name.strip().upper()

        if collection_name not in existing:
            raise SystemExit(
                f"Collection not found: {collection_name}. Existing collections: {', '.join(existing[:20])}"
            )

        documents = OracleVSAdmin.list_documents_in_collection(conn, collection_name)

    logger.info("")
    logger.info("Collection: %s", collection_name)
    if not documents:
        logger.info("No documents found.")
        logger.info("")
        return

    logger.info("Documents found (%d):", len(documents))
    for doc in documents:
        logger.info("  - %s", doc)
    logger.info("")


if __name__ == "__main__":
    main()
