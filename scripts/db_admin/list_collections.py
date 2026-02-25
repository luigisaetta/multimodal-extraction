"""
Author: Luigi Saetta
Date last modified: 2026-02-23
Python Version: 3.11
License: MIT

Description:
    Utility to list existing Oracle Vector Search collections.
"""

from multimodal_extraction.db.db_utils import get_db_connection
from multimodal_extraction.db.oraclevs_admin import OracleVSAdmin
from multimodal_extraction.utils import get_console_logger


def main() -> None:
    logger = get_console_logger()

    with get_db_connection() as conn:
        collections = OracleVSAdmin.list_collections(conn)

    logger.info("")
    if not collections:
        logger.info("No collections found.")
        logger.info("")
        return

    logger.info("Available collections (%d):", len(collections))
    for name in collections:
        logger.info("  - %s", name)
    logger.info("")


if __name__ == "__main__":
    main()
