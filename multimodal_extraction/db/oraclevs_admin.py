"""
Author: Luigi Saetta
Date last modified: 2026-02-01
Python Version: 3.11
License: MIT

Description:
    Extension of OracleVS class to add utility methods for administration and
    management specific to Oracle Vector Search collections.

Important:
    This class assumes that METADATA is a JSON column containing a '$.source'
    field identifying the document each chunk belongs to.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Sequence

from oracledb import Connection, DB_TYPE_VECTOR
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import DistanceStrategy

# moved to LangChain 1.x compatibility
from langchain_oracledb import OracleVS

from multimodal_extraction.config import DEBUG, EMBED_MODEL_ID
from multimodal_extraction.utils import get_console_logger

logger = get_console_logger()

# to avoid SQL injection on identifiers (table names etc.)
_VALID_IDENT = re.compile(r"^[A-Z][A-Z0-9_$#]*$")


def _safe_ident(name: str) -> str:
    """
    Validate and return a safe Oracle identifier (uppercased).

    Raises:
        ValueError: if name contains invalid characters.
    """
    ident = name.strip().upper()
    if not _VALID_IDENT.fullmatch(ident):
        raise ValueError(f"Invalid Oracle identifier: {name!r}")
    return ident


def _escape_sql_string_literal(value: str) -> str:
    """Escape a Python string for use as a single-quoted Oracle SQL literal."""
    return value.replace("'", "''")


class OracleVSAdmin(OracleVS):
    """
    Admin utilities for Oracle Vector Store collections.
    """

    @classmethod
    def list_collections(cls, connection: Connection) -> List[str]:
        """
        Return a list of all collections (tables) that contain VECTOR columns
        in the current schema.
        """
        query = """
            SELECT DISTINCT table_name
            FROM user_tab_columns
            WHERE data_type = 'VECTOR'
            ORDER BY table_name ASC
        """

        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

        return [row[0] for row in rows]

    @classmethod
    def list_documents_in_collection(
        cls, connection: Connection, collection_name: str
    ) -> List[str]:
        """
        Return the list of distinct document sources (METADATA.source) found
        in the specified collection.
        """
        safe_name = _safe_ident(collection_name)

        query = f"""
            SELECT DISTINCT json_value(METADATA, '$.source') AS source
            FROM {safe_name}
            WHERE json_value(METADATA, '$.source') IS NOT NULL
            ORDER BY source ASC
        """

        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

        return [row[0] for row in rows if row and row[0] is not None]

    @classmethod
    def list_documents_with_chunk_counts(
        cls, connection: Connection, collection_name: str
    ) -> List[Dict[str, Any]]:
        """
        Return list of {"document": <source>, "n_chunks": <count>} for the given collection.

        This is the "single GROUP BY" query you used in the Streamlit app, moved here
        to avoid duplication.

        Notes:
          - expects a METADATA JSON column with '$.source'
        """
        safe_name = _safe_ident(collection_name)

        sql = f"""
            SELECT
                json_value(METADATA, '$.source') AS document,
                COUNT(*) AS n_chunks
            FROM {safe_name}
            WHERE json_value(METADATA, '$.source') IS NOT NULL
            GROUP BY json_value(METADATA, '$.source')
            ORDER BY document ASC
        """

        with connection.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

        out: List[Dict[str, Any]] = []
        for row in rows:
            # row[0] is document/source, row[1] is count
            out.append({"document": row[0], "n_chunks": int(row[1])})
        return out

    @classmethod
    def analyze_collection(cls, connection: Connection, collection_name: str) -> str:
        """
        Analyze a collection and return a short report including:
          - total rows (chunks)
          - vector dimensions distribution
          - vector formats distribution
        """
        safe_name = _safe_ident(collection_name)
        sql = f"SELECT * FROM {safe_name}"

        with connection.cursor() as cur:
            cur.execute(sql)
            descs = cur.description  # column metadata

            records = 0
            dim_counter: Counter[int] = Counter()
            format_counter: Counter[Any] = Counter()

            for row in cur:
                records += 1
                for idx, _ in enumerate(row):
                    info = descs[idx]
                    if info.type_code == DB_TYPE_VECTOR:
                        dims = info.vector_dimensions
                        fmt = info.vector_format
                        dim_counter[dims] += 1
                        format_counter[fmt] += 1

        report = f"Analyzed collection: {safe_name}\n"
        report += f"    Total chunks fetched: {records}\n"
        report += f"    Vector dimensions seen (count): {dict(dim_counter)}\n"
        report += f"    Vector formats seen (count): {dict(format_counter)}\n"
        return report

    @classmethod
    def delete_documents(
        cls, connection: Connection, collection_name: str, doc_names: Sequence[str]
    ) -> None:
        """
        Delete all chunks whose METADATA.source is in doc_names.

        Args:
            connection: open Oracle connection
            collection_name: table name
            doc_names: iterable of source names to delete
        """
        safe_name = _safe_ident(collection_name)

        sql = f"""
            DELETE FROM {safe_name}
            WHERE json_value(METADATA, '$.source') = :doc
        """

        with connection.cursor() as cur:
            for doc_name in doc_names:
                if DEBUG:
                    logger.info("Dropping document source: %s", doc_name)
                cur.execute(sql, [doc_name])

        connection.commit()

    @classmethod
    def drop_collection(cls, connection: Connection, collection_name: str) -> None:
        """
        Drop a collection table.
        """
        safe_name = _safe_ident(collection_name)
        sql = f"DROP TABLE {safe_name}"

        with connection.cursor() as cur:
            cur.execute(sql)

        connection.commit()

    @classmethod
    def annotate_collection_table(
        cls,
        connection: Connection,
        table_name: str,
        embedding_model: str = EMBED_MODEL_ID,
    ) -> None:
        """
        Annotate an Oracle table with collection metadata.
        """
        safe_table = _safe_ident(table_name)
        comment = f"RAG collection metadata: embedding_model={embedding_model}"
        comment_sql = _escape_sql_string_literal(comment)
        sql = f"COMMENT ON TABLE {safe_table} IS '{comment_sql}'"

        with connection.cursor() as cur:
            cur.execute(sql)
        connection.commit()

    @classmethod
    def create_empty_collection(
        cls,
        connection: Connection,
        collection_name: str,
        embedding: Any,
        embedding_model: str = EMBED_MODEL_ID,
    ) -> str:
        """
        Create a new empty collection table using a temporary bootstrap row.

        Notes:
          - Uses OracleVS schema creation path via from_documents(...)
          - Deletes the bootstrap row immediately after table creation
        """
        safe_name = _safe_ident(collection_name)
        existing = cls.list_collections(connection)
        if safe_name in existing:
            raise ValueError(f"Collection already exists: {safe_name}")

        bootstrap_source = "__bootstrap__"
        bootstrap_doc = Document(
            page_content="bootstrap row used for collection creation",
            metadata={"source": bootstrap_source, "page": 0},
        )

        cls.from_documents(
            client=connection,
            documents=[bootstrap_doc],
            embedding=embedding,
            table_name=safe_name,
            distance_strategy=DistanceStrategy.COSINE,
        )
        cls.delete_documents(connection, safe_name, [bootstrap_source])
        cls.annotate_collection_table(connection, safe_name, embedding_model)

        return safe_name
