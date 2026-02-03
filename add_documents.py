"""
Author: Luigi Saetta
Date last modified: 2026-02-02
Python Version: 3.11
License: MIT

Add documents to an EXISTING Oracle Vector Search collection.

Behavior:
- Read PDFs from a directory
- Check if each PDF was already loaded (METADATA.source == filename)
- Skip already loaded PDFs
- Extract text using unified pipeline:
    - TEXT_PDF -> Docling (if enabled) or pypdf
    - SCANNED_PDF -> VLM OCR
    - MIXED -> per-page fallback
- Optionally (default ON) describe figures via a second multimodal call
- Chunking: 1 page = 1 chunk (via your existing chunker API)
- Load chunks into the existing collection using OracleVSAdmin.add_documents(...)
"""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

from tqdm import tqdm

from classify_pdf import ClassifyConfig, classify_pdf
from config import CHUNK_OVERLAP, CHUNK_SIZE
from db_utils import get_db_connection
from oci_models import get_embedding_model
from ocr_output_chunking_utils import (
    chunks_to_langchain_documents,
    ocr_output_text_to_chunks,
)
from oraclevs_admin import OracleVSAdmin
from text_from_pdf_scanner import OcrConfig, run_ocr_pipeline
from utils import get_console_logger

logger = get_console_logger()

_IDENT_RE = re.compile(r"^[A-Z][A-Z0-9_$#]*$")


def _safe_ident(name: str) -> str:
    """Validate a simple unquoted Oracle identifier."""
    normalized = name.strip().upper()
    if not _IDENT_RE.fullmatch(normalized):
        raise ValueError(f"Unsafe or invalid Oracle identifier: {name!r}")
    return normalized


def _list_pdfs(directory: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(directory.glob(pattern))


def _classify_pdf_quick(pdf_path: Path) -> str:
    """
    Classify PDF with the same heuristics used elsewhere.
    Returns: TEXT_PDF / SCANNED_PDF / MIXED_OR_UNKNOWN
    """
    cfg = ClassifyConfig(
        sample_pages=10,
        min_text_chars_doc=200,
        min_text_chars_page=50,
        scanned_if_image_pages_ratio_ge=0.6,
        strong_text_chars=5000,
    )
    detected_label, _ = classify_pdf(pdf_path, cfg)
    return detected_label


def _get_loaded_sources(conn, collection_name: str) -> Set[str]:
    """
    Read distinct sources already loaded in the collection.
    Uses METADATA.source convention.
    """
    sources = OracleVSAdmin.list_documents_in_collection(conn, collection_name)
    # normalize to allow reliable comparison
    return {str(s).strip() for s in sources if s is not None}


def _extract_and_chunk_pdf(
    pdf_path: Path,
    ocr_model_id: str,
    describe_figures: bool,
    max_pages: Optional[int],
) -> list:
    """
    Extract text and chunk as 1-page=1-chunk (via your chunker API).
    Returns: List[langchain_core.documents.Document]
    """
    detected_type = _classify_pdf_quick(pdf_path)
    logger.info("PDF type: %s | %s", detected_type, pdf_path.name)

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
            # auto mode branch
            text_extraction_mode="auto",
            input_pdf_type=detected_type,
            min_text_chars_page=50,
        )

        extracted_text = run_ocr_pipeline(pdf_path, ocr_cfg)

    # API compatibility: CHUNK_SIZE / CHUNK_OVERLAP are kept but should be ignored
    # if you've switched to "one page = one chunk" internally.
    chunks = ocr_output_text_to_chunks(
        full_text=extracted_text,
        source_name=pdf_path.name,
        max_chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
        add_header=True,
    )
    return chunks_to_langchain_documents(chunks)


def _partition_pdfs(
    pdfs: Iterable[Path], loaded_sources: Set[str]
) -> Tuple[List[Path], List[Path]]:
    """
    Split PDFs into (to_process, skipped) comparing by filename against METADATA.source.
    """
    to_process: List[Path] = []
    skipped: List[Path] = []

    for p in pdfs:
        name = p.name.strip()
        if name in loaded_sources:
            skipped.append(p)
        else:
            to_process.append(p)

    return to_process, skipped


def main() -> None:
    """
    Main
    """
    parser = argparse.ArgumentParser(
        description="Add PDFs to an EXISTING Oracle Vector Search collection."
    )
    parser.add_argument(
        "collection_name",
        type=str,
        help="Existing collection table name (Oracle identifier).",
    )
    parser.add_argument(
        "documents_dir",
        type=str,
        help="Directory containing PDFs to load.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan PDFs recursively in subdirectories.",
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
        help="Disable figure/diagram description (second multimodal call).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Max pages to process per PDF (0 = all).",
    )
    parser.set_defaults(describe_figures=True)

    args = parser.parse_args()

    collection_name = _safe_ident(args.collection_name)
    docs_dir = Path(args.documents_dir).expanduser().resolve()
    if not docs_dir.exists() or not docs_dir.is_dir():
        raise SystemExit(f"Not a directory: {docs_dir}")

    max_pages = None if int(args.max_pages) == 0 else int(args.max_pages)

    logger.info("")
    logger.info("Add documents into existing collection: %s", collection_name)
    logger.info("Directory: %s", docs_dir)
    logger.info("Describe figures: %s", bool(args.describe_figures))
    logger.info("")

    pdfs = _list_pdfs(docs_dir, recursive=bool(args.recursive))
    if not pdfs:
        logger.info("No PDF files found in: %s", docs_dir)
        return

    # 1) Read loaded sources once
    with get_db_connection() as conn:
        # Optional: validate collection exists
        existing = OracleVSAdmin.list_collections(conn)
        if collection_name not in existing:
            raise SystemExit(
                f"Collection not found: {collection_name}. "
                f"Existing collections: {', '.join(existing[:20])}"
            )

        loaded_sources = _get_loaded_sources(conn, collection_name)

    # 2) Split to_process vs skipped
    to_process, skipped = _partition_pdfs(pdfs, loaded_sources)

    logger.info("PDFs found: %d", len(pdfs))
    logger.info("Already loaded (skip): %d", len(skipped))
    logger.info("To process: %d", len(to_process))
    logger.info("")

    if not to_process:
        logger.info("Nothing to do: all PDFs are already loaded.")
        return

    # 3) Extract+chunk, collecting docs
    all_docs = []
    failed = []

    for pdf_path in tqdm(to_process, desc="Extract+chunk PDFs"):
        try:
            logger.info("Extracting + chunking: %s", pdf_path.name)
            docs = _extract_and_chunk_pdf(
                pdf_path=pdf_path,
                ocr_model_id=args.ocr_model_id,
                describe_figures=bool(args.describe_figures),
                max_pages=max_pages,
            )
            all_docs.extend(docs)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            failed.append((pdf_path, exc))
            logger.error("************************************")
            logger.error("Error processing %s: %s", pdf_path.name, exc)
            logger.error("Skipping this file.")
            logger.error("************************************")

    if not all_docs:
        logger.info("No chunks produced. Nothing to load.")
        if failed:
            logger.info("Failed files: %d", len(failed))
        return

    # 4) Load into existing collection
    logger.info("")
    logger.info("Loading %d chunks into %s ...", len(all_docs), collection_name)

    embed_model = get_embedding_model()

    with get_db_connection() as conn:
        oracle_vs = OracleVSAdmin(
            client=conn,
            table_name=collection_name,
            embedding_function=embed_model,
        )
        oracle_vs.add_documents(all_docs)

    logger.info("Loading completed.")
    logger.info("")

    # 5) Summary
    logger.info("Summary")
    logger.info("  PDFs scanned: %d", len(pdfs))
    logger.info("  PDFs skipped (already loaded): %d", len(skipped))
    logger.info("  PDFs attempted: %d", len(to_process))
    logger.info("  PDFs failed: %d", len(failed))
    logger.info("  Total chunks loaded: %d", len(all_docs))

    if failed:
        logger.info("")
        logger.info("Failed files list:")
        for p, exc in failed:
            logger.info("  - %s: %s", p.name, type(exc).__name__)


if __name__ == "__main__":
    main()
