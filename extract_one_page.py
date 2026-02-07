"""
Author: Luigi Saetta
Date last modified: 2026-02-03
Python Version: 3.11
License: MIT

Extract text + describe figures for a SINGLE PDF page.

- Reads a PDF path and a 1-based page number
- Extracts text using:
    - pypdf/Docling (if requested) OR
    - multimodal OCR (VLM) OR
    - auto fallback: try pypdf/Docling, else VLM
- Always renders the requested page to an image if figures are enabled
- Optionally appends a [FIGURES] section

It reuses the same helpers/prompts as text_from_pdf_scanner.py.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from PIL import Image
import pypdfium2 as pdfium
from pypdf import PdfReader

from docling.document_converter import DocumentConverter

from oci_models import get_llm
from text_from_pdf_scanner import (
    call_multimodal_llm_text_only,
    call_multimodal_llm_figures_only,
    append_figures_block,
    is_blank_page,
    page_has_enough_text,
)
from docling_post_processing import cleanup_docling_text_keep_captions
from utils import get_console_logger
from config import DOCKLING_ENABLED, ENABLE_CLEANUP

logger = get_console_logger()

PdfTypeLabel = Literal["TEXT_PDF", "SCANNED_PDF", "MIXED_OR_UNKNOWN"]
TextExtractionMode = Literal["auto", "pypdf", "vlm"]


# ----------------------------
# Config
# ----------------------------
@dataclass
class SinglePageConfig:
    """
    Configs for single-page extraction.
    """

    model_id: str
    page: int  # 1-based
    text_mode: TextExtractionMode = "auto"
    input_pdf_type: Optional[PdfTypeLabel] = None
    min_text_chars_page: int = 50

    dpi: int = 200
    max_side: int = 1600
    jpeg_quality: int = 90

    blank_white_threshold: int = 245
    blank_min_nonwhite_ratio: float = 0.01
    blank_use_center_crop: bool = True
    blank_placeholder: str = "[BLANK PAGE SKIPPED]"

    extra_prompt: str = ""

    # Figures
    describe_figures: bool = True


# ----------------------------
# Image helpers (same logic)
# ----------------------------
def render_single_page(pdf_path: Path, page_1based: int, dpi: int) -> Image.Image:
    """
    render a single page to a PIL image using pdfium (similar as in text_from_pdf_scanner.py)
    """
    pdf_doc = pdfium.PdfDocument(str(pdf_path))
    total_pages = len(pdf_doc)
    if page_1based < 1 or page_1based > total_pages:
        raise ValueError(f"Invalid page {page_1based}. PDF has {total_pages} pages.")

    page_idx0 = page_1based - 1
    page = pdf_doc[page_idx0]
    scale = dpi / 72.0
    bitmap = page.render(scale=scale)
    return bitmap.to_pil()


# ----------------------------
# Text extraction: pypdf/docling (single page)
# ----------------------------
def extract_text_single_page_pypdf(pdf_path: Path, page_1based: int) -> str:
    """
    Extract page text using pypdf.

    Returns:
        page text
        Empty string for pages with no extractable text.
    """
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    if page_1based < 1 or page_1based > total_pages:
        raise ValueError(f"Invalid page {page_1based}. PDF has {total_pages} pages.")

    page = reader.pages[page_1based - 1]
    return (page.extract_text() or "").strip()


def extract_text_single_page_docling(pdf_path: Path, page_1based: int) -> str:
    """
    Extract per-page Markdown from a TEXT_PDF using Docling.
    (Uses the same export_to_markdown pagination placeholder technique,
     then picks the requested page.)
    """
    pdf_path = Path(pdf_path).expanduser().resolve()

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    doc = result.document

    page_break = "\n\n<<<DOCLING_PAGE_BREAK>>>\n\n"
    md = doc.export_to_markdown(
        enable_chart_tables=True,
        compact_tables=True,
        page_break_placeholder=page_break,
        include_annotations=False,
        escape_html=True,
        escape_underscores=True,
        image_placeholder="",
    )

    pages = [p.strip() for p in md.split(page_break)]
    if page_1based < 1 or page_1based > len(pages):
        raise ValueError(
            f"Invalid page {page_1based}. Docling produced {len(pages)} pages."
        )

    out = pages[page_1based - 1]
    if ENABLE_CLEANUP:
        out = cleanup_docling_text_keep_captions(out)
    return out


def resolve_text_mode(cfg: SinglePageConfig) -> TextExtractionMode:
    """
    Resolve effective text extraction mode based on config.
     - If text_mode is explicitly set to "pypdf" or "vlm", use that.
     - If text_mode is "auto", decide based on input_pdf_type hint:
        - TEXT_PDF -> pypdf/docling
        - SCANNED_PDF -> vlm (OCR)
        - MIXED_OR_UNKNOWN -> start with pypdf/docling, fallback to vlm
          if extracted text is too short
    """
    if cfg.text_mode != "auto":
        return cfg.text_mode

    if cfg.input_pdf_type == "TEXT_PDF":
        return "pypdf"
    if cfg.input_pdf_type == "SCANNED_PDF":
        return "vlm"
    return "auto"


# ----------------------------
# Single-page pipeline
# ----------------------------
def extract_single_page(pdf_path: Path, cfg: SinglePageConfig) -> str:
    """
    Extract text + optionally describe figures for a single PDF page based
    on the provided config.
    """
    pdf_path = Path(pdf_path).expanduser().resolve()
    effective_mode = resolve_text_mode(cfg)

    logger.info(
        "Page=%s | text_mode=%s (input_pdf_type=%s) | describe_figures=%s",
        cfg.page,
        effective_mode,
        cfg.input_pdf_type,
        cfg.describe_figures,
    )

    # We need an image if:
    # - vlm text OR
    # - describe_figures OR
    # - auto fallback might require OCR
    need_image = cfg.describe_figures or (effective_mode in ("vlm", "auto"))
    page_img = None
    if need_image:
        page_img = render_single_page(pdf_path, cfg.page, dpi=cfg.dpi)

    # If we might call the model, do blank detection
    if page_img is not None and (
        effective_mode in ("vlm", "auto") or cfg.describe_figures
    ):
        if is_blank_page(
            page_img,
            white_threshold=cfg.blank_white_threshold,
            min_nonwhite_ratio=cfg.blank_min_nonwhite_ratio,
            use_center_crop=cfg.blank_use_center_crop,
        ):
            return cfg.blank_placeholder

    # Try local text extraction if mode allows
    page_text = ""
    if effective_mode in ("pypdf", "auto"):
        if DOCKLING_ENABLED:
            logger.info("Extracting page text via Docling...")
            page_text = extract_text_single_page_docling(pdf_path, cfg.page)
        else:
            logger.info("Extracting page text via pypdf...")
            page_text = extract_text_single_page_pypdf(pdf_path, cfg.page)

    # Decide if we need the model
    need_llm_for_text = (effective_mode == "vlm") or (
        effective_mode == "auto"
        and not page_has_enough_text(page_text, cfg.min_text_chars_page)
    )
    need_llm = need_llm_for_text or cfg.describe_figures

    llm = None
    if need_llm:
        logger.info("Loading LLM: %s", cfg.model_id)
        llm = get_llm(model_id=cfg.model_id)

    # Text extraction with model if required
    if need_llm_for_text:
        if llm is None or page_img is None:
            raise RuntimeError("Need LLM text extraction but LLM/image not available.")
        logger.info("Calling LLM for TEXT OCR on this page...")
        page_text = call_multimodal_llm_text_only(
            llm,
            page_img,
            extra_prompt=cfg.extra_prompt,
            max_side=cfg.max_side,
            jpeg_quality=cfg.jpeg_quality,
        )

    # Figures (optional)
    if cfg.describe_figures:
        if llm is None or page_img is None:
            raise RuntimeError(
                "describe_figures=True requires both llm and page image."
            )
        logger.info("Calling LLM for FIGURES description on this page...")
        figs = call_multimodal_llm_figures_only(
            llm,
            page_img,
            max_side=cfg.max_side,
            jpeg_quality=cfg.jpeg_quality,
        )
        page_text = append_figures_block(page_text, figs)

    return page_text.strip()


# ----------------------------
# CLI
# ----------------------------
def main() -> None:
    """
    Main
    """
    parser = argparse.ArgumentParser(
        description="Extract text + describe figures for a SINGLE PDF page."
    )
    parser.add_argument("pdf", type=str, help="Path to the PDF file")
    parser.add_argument(
        "page",
        type=int,
        help="Page number (1-based).",
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default="openai.gpt-5.2",
        help="Multimodal model id (used for OCR and/or figures).",
    )
    parser.add_argument(
        "--text-mode",
        type=str,
        default="auto",
        choices=["auto", "pypdf", "vlm"],
        help="Text extraction mode: auto (fallback), pypdf/docling, or vlm (OCR).",
    )
    parser.add_argument(
        "--input-pdf-type",
        type=str,
        default=None,
        choices=["TEXT_PDF", "SCANNED_PDF", "MIXED_OR_UNKNOWN"],
        help="Optional hint (only used when --text-mode=auto).",
    )
    parser.add_argument(
        "--min-text-chars-page",
        type=int,
        default=50,
        help="AUTO: if local text has fewer chars, fallback to VLM OCR.",
    )

    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--jpeg-quality", type=int, default=85)

    parser.add_argument(
        "--no-describe-figures",
        action="store_false",
        dest="describe_figures",
        help="Disable [FIGURES] description for this page.",
    )
    parser.set_defaults(describe_figures=True)

    parser.add_argument("--extra-prompt", type=str, default="")
    parser.add_argument(
        "--no-center-crop",
        action="store_true",
        help="Disable center crop for blank detection.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    cfg = SinglePageConfig(
        model_id=args.model_id,
        page=int(args.page),
        text_mode=args.text_mode,
        input_pdf_type=args.input_pdf_type,
        min_text_chars_page=int(args.min_text_chars_page),
        dpi=int(args.dpi),
        max_side=int(args.max_side),
        jpeg_quality=int(args.jpeg_quality),
        extra_prompt=args.extra_prompt,
        describe_figures=bool(args.describe_figures),
        blank_use_center_crop=not bool(args.no_center_crop),
    )

    out = extract_single_page(Path(args.pdf), cfg)

    # Print to stdout (easy piping / redirect)
    print("")
    print(f"===== {Path(args.pdf).name} | PAGE {args.page} =====")
    print(out)
    print("")
    logger.info("Done")


if __name__ == "__main__":
    main()
