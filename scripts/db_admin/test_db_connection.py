"""
Author: Luigi Saetta
Date last modified: 2026-01-14
Python Version: 3.11
License: MIT

Description:
    Script to test DB connection to Vector Store
"""

from multimodal_extraction.db.db_utils import get_db_connection
from multimodal_extraction.utils import get_console_logger


def main() -> None:
    logger = get_console_logger()

    try:
        with get_db_connection() as conn:
            logger.info("")
            logger.info("Connection OK")
            logger.info("")
    except Exception as e:
        logger.error("Error testing connection...")
        logger.error(e)


if __name__ == "__main__":
    main()
