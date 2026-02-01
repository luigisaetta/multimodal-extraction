"""
Author: Luigi Saetta
Date last modified: 2026-02-01
Python Version: 3.11
License: MIT

Batch loading (new collection only)

Creates a new collection and loads a directory of PDFs by:
1) Extracting text using the unified pipeline (Docling for TEXT_PDF, VLM OCR for SCANNED_PDF)
2) Chunking as: one page = one chunk
3) Embedding and loading into Oracle Vector Search
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from glob import glob
from pathlib import Path
from typing import List

from langchain_community.vectorstores.utils import DistanceStrategy

from classify_pdf import ClassifyConfig, classify_pdf
from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL_ID
from db_utils import get_db_connection
from oci_models import get_embedding_model
from ocr_output_chunking_utils import (
    chunks_to_langchain_documents,
    ocr_output_text_to_chunks,
)
from oraclevs_admin import OracleVSAdmin
from text_from_pdf_scanner import OcrConfig, run_ocr_pipeline
from utils import compute_stats, get_console_logger

logger = get_console_logger()


_IDENTIFIER_RE = re.compile(r"^[A-Z][A-Z0-9_$#]*$")


def _assert_safe_identifier(name: str) -> str:
    """
    Validate an Oracle identifier (simple, unquoted).
    Prevents SQL injection since identifiers cannot be bound.
    """
    normalized = name.strip().upper()
    if not _IDENTIFIER_RE.fullmatch(normalized):
        raise ValueError(f"Unsafe or invalid Oracle identifier: {name!r}")
    return normalized


def _escape_sql_string_literal(value: str) -> str:
    """Escape a Python string for use as a single-quoted Oracle SQL literal."""
    return value.replace("'", "''")


def annotate_collection_table(conn, table_name: str, embedding_model: str) -> None:
    """
    Annotate an Oracle table with the embedding model used for the collection.
    Stored as a table COMMENT.
    """
    safe_table = _assert_safe_identifier(table_name)
    comment = f"RAG collection metadata: embedding_model={embedding_model}"
    comment_sql = _escape_sql_string_literal(comment)

    sql = f"COMMENT ON TABLE {safe_table} IS '{comment_sql}'"
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def get_list_collections() -> List[str]:
    """Get the list of existing collections."""
    with get_db_connection() as conn:
        return OracleVSAdmin.list_collections(conn)


def _classify_pdf_quick(pdf_path: Path) -> str:
    """
    Classify PDF with same heuristics used elsewhere.
    Returns: TEXT_PDF / SCANNED_PDF / MIXED_OR_UNKNOWN
    """
    classify_cfg = ClassifyConfig(
        sample_pages=10,
        min_text_chars_doc=200,
        min_text_chars_page=50,
        scanned_if_image_pages_ratio_ge=0.6,
        strong_text_chars=5000,
    )
    detected_label, _ = classify_pdf(pdf_path, classify_cfg)
    return detected_label


def extract_and_chunk_pdf(
    pdf_path: Path,
    ocr_model_id: str,
    describe_figures: bool,
    max_pages: int | None,
) -> list:
    """
    Extract text (Docling for TEXT_PDF, VLM OCR for scanned) and chunk as 1-page=1-chunk.
    Returns: List[langchain_core.documents.Document]
    """
    detected_pdf_type = _classify_pdf_quick(pdf_path)
    logger.info("PDF type: %s | %s", detected_pdf_type, pdf_path.name)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / f"{pdf_path.stem}_output.txt"

        ocr_cfg = OcrConfig(
            model_id=ocr_model_id,
            out_path=out_path,
            dpi=200,
            max_pages=max_pages,
            extra_prompt="",
            save_images=False,
            images_dir=None,
            describe_figures=bool(describe_figures),
            # key fields for the "auto" branch:
            text_extraction_mode="auto",
            input_pdf_type=detected_pdf_type,
            min_text_chars_page=50,
        )

        extracted_text = run_ocr_pipeline(pdf_path, ocr_cfg)

    # NOTE: current chunker keeps the same API but now produces 1 chunk per page
    chunks = ocr_output_text_to_chunks(
        full_text=extracted_text,
        source_name=pdf_path.name,
        max_chunk_size=CHUNK_SIZE,  # kept for API compatibility (ignored now)
        overlap=CHUNK_OVERLAP,  # kept for API compatibility (ignored now)
        add_header=True,
    )
    return chunks_to_langchain_documents(chunks)


def main() -> None:
    """Main."""
    parser = argparse.ArgumentParser(
        description="Batch load PDFs into a NEW collection."
    )
    parser.add_argument("new_collection_name", type=str, help="New collection name.")
    parser.add_argument(
        "documents_dir", type=str, help="Directory containing PDFs to load."
    )
    parser.add_argument(
        "--ocr-model-id",
        type=str,
        default="openai.gpt-5.2",
        help="Multimodal model id used by the extraction pipeline (for scanned PDFs).",
    )
    parser.add_argument(
        "--no-describe-figures",
        action="store_false",
        dest="describe_figures",
        help="Disable figure/diagram description (multimodal call).",
    )

    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Max pages to process per PDF (0 = all).",
    )
    parser.set_defaults(describe_figures=True)

    args = parser.parse_args()

    new_collection_name = _assert_safe_identifier(args.new_collection_name)
    documents_dir = Path(args.documents_dir).expanduser().resolve()

    if not documents_dir.exists() or not documents_dir.is_dir():
        raise SystemExit(f"Not a directory: {documents_dir}")

    logger.info("")
    logger.info("Batch loading PDFs into collection %s ...", new_collection_name)
    logger.info("")

    # Check collection doesn't exist
    existing = get_list_collections()
    if new_collection_name in existing:
        logger.error(
            "Error: collection %s already exists. Exiting.", new_collection_name
        )
        sys.exit(1)

    # List PDFs
    pdf_files = sorted(glob(str(documents_dir / "*.pdf")))
    if not pdf_files:
        logger.info("No PDF files found in: %s", documents_dir)
        return

    logger.info("These PDFs will be loaded:")
    for p in pdf_files:
        logger.info(p)
    logger.info("")

    max_pages = None if int(args.max_pages) == 0 else int(args.max_pages)

    # Extract + chunk
    all_docs = []
    for pdf_file in pdf_files:
        pdf_path = Path(pdf_file).resolve()
        logger.info("Extracting + chunking: %s", pdf_path.name)
        docs = extract_and_chunk_pdf(
            pdf_path=pdf_path,
            ocr_model_id=args.ocr_model_id,
            describe_figures=bool(args.describe_figures),
            max_pages=max_pages,
        )
        all_docs.extend(docs)

    if not all_docs:
        logger.info("No chunks produced. Nothing to load.")
        return

    # Embed + load
    embed_model = get_embedding_model()

    logger.info("")
    logger.info(
        "Embedding and loading %s chunks into %s ...",
        len(all_docs),
        new_collection_name,
    )

    with get_db_connection() as conn:
        OracleVSAdmin.from_documents(
            client=conn,
            documents=all_docs,
            embedding=embed_model,
            table_name=new_collection_name,
            distance_strategy=DistanceStrategy.COSINE,
        )
        annotate_collection_table(conn, new_collection_name, EMBED_MODEL_ID)

    logger.info("Loading completed.")
    logger.info("")

    mean, stdev, perc_75 = compute_stats(all_docs)
    logger.info("Statistics on chunks length distribution:")
    logger.info("Total num. of chunks loaded: %s", len(all_docs))
    logger.info("Avg. length: %s (chars)", mean)
    logger.info("Std dev: %s (chars)", stdev)
    logger.info("75-perc: %s (chars)", perc_75)
    logger.info("")


if __name__ == "__main__":
    main()
