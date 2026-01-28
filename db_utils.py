"""
Author: Luigi Saetta
Date last modified: 2026-01-14
Python Version: 3.11
License: MIT

Description:
    Utility functions for DB operations
"""

import oracledb

from utils import get_console_logger

from config_private import CONNECT_ARGS

logger = get_console_logger()


def get_db_connection():
    """
    get a connection to db
    """

    try:
        logger.info("")
        logger.info(
            "Connecting as USER: %s to DSN: %s",
            CONNECT_ARGS["user"],
            CONNECT_ARGS["dsn"],
        )

        return oracledb.connect(**CONNECT_ARGS)
    except oracledb.Error as e:
        logger.error("Database connection failed: %s", str(e))
        raise


def get_table_comment(conn, table_name: str) -> str:
    """
    Return the table-level comment for an Oracle table as a single line of text.

    Args:
        conn: An open oracledb connection.
        table_name: Table name (case-insensitive, unquoted identifier).

    Returns:
        The table comment, or an empty string if not present.
    """
    table_name = table_name.strip().upper()

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT comments
            FROM user_tab_comments
            WHERE table_name = :t
            """,
            {"t": table_name},
        )
        row = cur.fetchone()

    return row[0] if row and row[0] else ""


def print_table_comment(conn, table_name: str) -> None:
    """
    Print the table-level comment for an Oracle table.

    Args:
        conn: An open oracledb connection.
        table_name: Table name (case-insensitive, unquoted identifier).
    """
    table_name = table_name.strip().upper()
    comment = get_table_comment(conn, table_name)

    print(f"Table comment for {table_name}:")
    print(comment if comment else "(no table comment)")
