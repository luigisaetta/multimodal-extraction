# text_from_pdf_scanner.py
"""
Author: Luigi Saetta
Python: 3.11+
License: MIT

Scanned / Text PDF -> (optional) page images -> multimodal LLM -> text (+ optional figures)

Features:
- If PDF is TEXT_PDF: extract text with pypdf (no multimodal OCR), per-page.
- If PDF is SCANNED_PDF: render pages to images and OCR via multimodal LLM, per-page.
- If PDF is MIXED_OR_UNKNOWN or mode=auto: per-page fallback:
    - try pypdf text
    - if too little text on the page, fallback to multimodal OCR for that page
- Blank-page detection (skip VLM calls and emit placeholder)
- OPTIONAL: describe figures/diagrams (append at end of each page)
- Single output text file with per-page footer
- Prompts for multimodal LLM in prompts.py
"""

from __future__ import annotations

from pathlib import Path
import argparse
import base64
import io
import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Literal, cast

from docling.document_converter import DocumentConverter
from PIL import Image
import pypdfium2 as pdfium
from pypdf import PdfReader
from langchain_core.messages import HumanMessage

from multimodal_extraction.models.oci_models import get_llm
from multimodal_extraction.prompts.prompts import build_ocr_text_prompt, build_figures_prompt
from multimodal_extraction.ocr.docling_post_processing import cleanup_docling_text_keep_captions
from multimodal_extraction.utils import get_console_logger
from multimodal_extraction.config import DOCKLING_ENABLED, ENABLE_CLEANUP

logger = get_console_logger()

PdfTypeLabel = Literal["TEXT_PDF", "SCANNED_PDF", "MIXED_OR_UNKNOWN"]
TextExtractionMode = Literal["auto", "pypdf", "vlm"]
ImageFormat = Literal["jpeg", "png"]


# ----------------------------
# Config
# ----------------------------
@dataclass
class OcrConfig:
    """
    Configuration for OCR pipeline.
    """

    model_id: str
    out_path: Path

    # Rendering / pages
    dpi: int = 200
    max_pages: Optional[int] = None
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    save_images: bool = False
    images_dir: Optional[Path] = None

    # Text extraction strategy
    text_extraction_mode: TextExtractionMode = "auto"
    input_pdf_type: Optional[PdfTypeLabel] = None

    # pypdf -> per-page fallback threshold (auto/mixed)
    min_text_chars_page: int = 50

    # Prompt
    extra_prompt: str = ""

    # blank detection (used for VLM calls)
    blank_white_threshold: int = 245
    blank_min_nonwhite_ratio: float = 0.01
    blank_use_center_crop: bool = True

    # image encoding for LLM
    max_side: int = 1600
    image_format: ImageFormat = "jpeg"
    jpeg_quality: int = 85

    # placeholder
    blank_placeholder: str = "[BLANK PAGE SKIPPED]"

    # figures
    describe_figures: bool = False

    # resilience
    page_max_retries: int = 2
    page_retry_backoff_sec: float = 2.0
    continue_on_page_error: bool = True
    page_error_placeholder: str = "[PAGE PROCESSING ERROR]"

    # progress checkpoint
    save_progress_checkpoint: bool = True
    checkpoint_every_pages: int = 5
    checkpoint_path: Optional[Path] = None


# ----------------------------
# Image helpers
# ----------------------------
def pil_to_data_url(
    img: Image.Image,
    max_side: int = 1600,
    quality: int = 85,
    image_format: ImageFormat = "jpeg",
) -> str:
    """Convert PIL image to data URL (JPEG default, PNG optional)."""
    if image_format == "jpeg" and img.mode != "RGB":
        img = img.convert("RGB")

    width, height = img.size
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        img = img.resize((int(width * scale), int(height * scale)))

    buf = io.BytesIO()
    if image_format == "jpeg":
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        mime = "image/jpeg"
    elif image_format == "png":
        # Uncompressed PNG path (explicitly requested option).
        img.save(buf, format="PNG", compress_level=0, optimize=False)
        mime = "image/png"
    else:
        raise ValueError(f"Unsupported image_format={image_format!r}.")

    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def render_pdf_pages(
    pdf_path: Path,
    dpi: int = 200,
    page_numbers: Optional[List[int]] = None,
) -> List[Image.Image]:
    """Render each PDF page to a PIL image using pdfium."""
    pdf_doc = pdfium.PdfDocument(str(pdf_path))
    total_pages = len(pdf_doc)
    selected_pages = page_numbers or list(range(1, total_pages + 1))
    images: List[Image.Image] = []
    scale = dpi / 72.0
    for page_1based in selected_pages:
        if page_1based < 1 or page_1based > total_pages:
            raise ValueError(f"Invalid page {page_1based}. PDF has {total_pages} pages.")
        page = pdf_doc[page_1based - 1]
        bitmap = page.render(scale=scale)
        images.append(bitmap.to_pil())
    return images


def render_single_page(pdf_path: Path, page_1based: int, dpi: int = 200) -> Image.Image:
    """Render a single 1-based page to a PIL image using pdfium."""
    pdf_doc = pdfium.PdfDocument(str(pdf_path))
    total_pages = len(pdf_doc)
    if page_1based < 1 or page_1based > total_pages:
        raise ValueError(f"Invalid page {page_1based}. PDF has {total_pages} pages.")

    page_idx0 = page_1based - 1
    page = pdf_doc[page_idx0]
    scale = dpi / 72.0
    bitmap = page.render(scale=scale)
    return bitmap.to_pil()


def is_blank_page(
    img: Image.Image,
    white_threshold: int = 245,
    min_nonwhite_ratio: float = 0.01,
    use_center_crop: bool = True,
) -> bool:
    """
    Detect if a page is essentially blank.

    - convert to grayscale
    - optionally crop center (ignore margins/header/footer)
    - compute fraction of pixels darker than white_threshold
    """
    gray = img.convert("L")
    width, height = gray.size

    if use_center_crop:
        gray = gray.crop(
            (int(width * 0.1), int(height * 0.1), int(width * 0.9), int(height * 0.9))
        )
        width, height = gray.size

    pixels = gray.load()
    total = width * height
    non_white = 0

    for x in range(width):
        for y in range(height):
            if pixels[x, y] < white_threshold:
                non_white += 1

    ratio = non_white / total
    return ratio < min_nonwhite_ratio


def format_page_block(page_idx: int, text: str) -> str:
    """Format a page block with footer."""
    footer = f"\n\n--- PAGE {page_idx} ---\n\n"
    return text.rstrip() + footer


def resolve_selected_pages(cfg: OcrConfig, total_pages: int) -> List[int]:
    """
    Resolve the exact list of 1-based pages to process.
    """
    start_page = 1 if cfg.start_page is None else int(cfg.start_page)
    end_page = total_pages if cfg.end_page is None else int(cfg.end_page)

    if start_page < 1:
        raise ValueError(f"start_page must be >= 1 (got {start_page}).")
    if end_page < 1:
        raise ValueError(f"end_page must be >= 1 (got {end_page}).")
    if start_page > total_pages:
        raise ValueError(
            f"start_page={start_page} is out of bounds. PDF has {total_pages} pages."
        )
    if end_page > total_pages:
        raise ValueError(
            f"end_page={end_page} is out of bounds. PDF has {total_pages} pages."
        )
    if start_page > end_page:
        raise ValueError(
            f"Invalid page window: start_page ({start_page}) > end_page ({end_page})."
        )

    pages = list(range(start_page, end_page + 1))
    if cfg.max_pages is not None:
        pages = pages[: int(cfg.max_pages)]
    return pages


# ----------------------------
# Text extraction (pypdf)
# ----------------------------
def extract_text_pages_pypdf(
    pdf_path: Path,
    page_numbers: Optional[List[int]] = None,
    max_pages: Optional[int] = None,
) -> List[str]:
    """
    Extract per-page text using pypdf.

    Returns:
        List of page texts (len == number of pages considered).
        Empty string for pages with no extractable text.
    """
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    selected_pages = page_numbers or list(range(1, total_pages + 1))
    if max_pages is not None:
        selected_pages = selected_pages[:max_pages]

    page_texts: List[str] = []
    for page_1based in selected_pages:
        if page_1based < 1 or page_1based > total_pages:
            raise ValueError(f"Invalid page {page_1based}. PDF has {total_pages} pages.")
        page = reader.pages[page_1based - 1]
        text = page.extract_text() or ""
        page_texts.append(text.strip())
    return page_texts


def extract_text_single_page_pypdf(pdf_path: Path, page_1based: int) -> str:
    """
    Extract one page text using pypdf.
    Returns empty string when no extractable text is present.
    """
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    if page_1based < 1 or page_1based > total_pages:
        raise ValueError(f"Invalid page {page_1based}. PDF has {total_pages} pages.")

    page = reader.pages[page_1based - 1]
    return (page.extract_text() or "").strip()


def extract_text_pages_docling(
    pdf_path: Path,
    page_numbers: Optional[List[int]] = None,
    max_pages: Optional[int] = None,
) -> List[str]:
    """
    Extract per-page Markdown from a TEXT_PDF using Docling.
    Tables are rendered as Markdown. Images are suppressed (no placeholders).

    Returns:
        List[str]: per-page markdown (len == number of pages considered).
    """
    pdf_path = Path(pdf_path).expanduser().resolve()

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    doc = result.document

    # Token unlikely to appear in normal content
    page_break = "\n\n<<<DOCLING_PAGE_BREAK>>>\n\n"

    md = doc.export_to_markdown(
        # Core: keep tables and paginate
        enable_chart_tables=True,
        compact_tables=True,
        page_break_placeholder=page_break,
        # “Light markdown”: reduce noise
        include_annotations=False,
        escape_html=True,
        escape_underscores=True,
        # Suppress image placeholders without needing ImageRefMode
        image_placeholder="",
    )

    pages = [p.strip() for p in md.split(page_break)]
    total_pages = len(pages)

    selected_pages = page_numbers or list(range(1, total_pages + 1))
    if max_pages is not None:
        selected_pages = selected_pages[:max_pages]

    page_slice: List[str] = []
    for page_1based in selected_pages:
        if page_1based < 1 or page_1based > total_pages:
            raise ValueError(
                f"Invalid page {page_1based}. Docling produced {total_pages} pages."
            )
        page_slice.append(pages[page_1based - 1])

    if ENABLE_CLEANUP:
        # ✅ CLEANUP: remove short label noise but keep captions (Figure X / Table Y)
        page_slice = [cleanup_docling_text_keep_captions(p) for p in page_slice]
    else:
        logger.info("Cleanup disabled...")

    return page_slice


def extract_text_single_page_docling(pdf_path: Path, page_1based: int) -> str:
    """
    Extract one page markdown using Docling.
    Reuses the same page-break placeholder strategy used in multi-page export.
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


def page_has_enough_text(page_text: str, min_chars: int) -> bool:
    """Heuristic: decide whether pypdf extraction is 'good enough' for a page."""
    if not page_text:
        return False
    # count non-whitespace chars
    non_ws = sum(1 for c in page_text if not c.isspace())
    return non_ws >= min_chars


def process_page_with_strategy(
    *,
    effective_mode: TextExtractionMode,
    page_img: Optional[Image.Image],
    candidate_text: str,
    llm,
    cfg: OcrConfig,
    need_llm: bool,
) -> str:
    """
    Shared per-page logic used by both multi-page and single-page flows.
    Returns the final page text (optionally with [FIGURES] appended).
    """
    # If we might call VLM (text or figures), do blank detection using the image
    if page_img is not None and need_llm:
        if is_blank_page(
            page_img,
            white_threshold=cfg.blank_white_threshold,
            min_nonwhite_ratio=cfg.blank_min_nonwhite_ratio,
            use_center_crop=cfg.blank_use_center_crop,
        ):
            return cfg.blank_placeholder

    # ---- TEXT (choose strategy) ----
    page_text = ""

    if effective_mode == "pypdf":
        page_text = candidate_text

    elif effective_mode == "vlm":
        if llm is None or page_img is None:
            raise RuntimeError("VLM mode requires both llm and page image.")
        logger.info("  - calling LLM for text extraction (vlm)...")
        page_text = call_multimodal_llm_text_only(
            llm,
            page_img,
            extra_prompt=cfg.extra_prompt,
            max_side=cfg.max_side,
            jpeg_quality=cfg.jpeg_quality,
            image_format=cfg.image_format,
        )
        if page_text is None:
            raise RuntimeError("Text extraction call returned no content.")

    else:
        # auto: try Docling/pypdf, fallback to VLM if page looks empty/weak
        if page_has_enough_text(candidate_text, cfg.min_text_chars_page):
            page_text = candidate_text
        else:
            if llm is None or page_img is None:
                raise RuntimeError(
                    "AUTO mode fallback requires both llm and page image."
                )
            logger.info("  - pypdf weak/empty; fallback to LLM OCR for text...")
            page_text = call_multimodal_llm_text_only(
                llm,
                page_img,
                extra_prompt=cfg.extra_prompt,
                max_side=cfg.max_side,
                jpeg_quality=cfg.jpeg_quality,
                image_format=cfg.image_format,
            )
            if page_text is None:
                raise RuntimeError("Fallback OCR call returned no content.")

    # ---- FIGURES (optional) ----
    if cfg.describe_figures:
        if llm is None or page_img is None:
            raise RuntimeError(
                "describe_figures=True requires both llm and page image."
            )
        logger.info("  - calling LLM for figures description...")
        figs_text = call_multimodal_llm_figures_only(
            llm,
            page_img,
            max_side=cfg.max_side,
            jpeg_quality=cfg.jpeg_quality,
            image_format=cfg.image_format,
        )
        if figs_text is None:
            raise RuntimeError("Figures description call returned no content.")
        page_text = append_figures_block(page_text, figs_text)

    return (page_text or "").strip()


# ----------------------------
# Multimodal LLM calls
# ----------------------------
def call_multimodal_llm_text_only(
    llm,
    page_img: Image.Image,
    extra_prompt: str,
    max_side: int,
    jpeg_quality: int,
    image_format: ImageFormat = "jpeg",
) -> Optional[str]:
    """
    Ask the model for ONLY transcribed text (no JSON).
    This is far more stable across providers (Gemini included).
    """
    data_url = pil_to_data_url(
        page_img,
        max_side=max_side,
        quality=jpeg_quality,
        image_format=image_format,
    )
    prompt_text = build_ocr_text_prompt(extra_prompt=extra_prompt)

    msg = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    )
    try:
        res = llm.invoke([msg])
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Error extracting text: %s", exc)
        logger.error("Skipping this image.")
        return None

    return str(getattr(res, "content", res)).strip()


def call_multimodal_llm_figures_only(
    llm,
    page_img: Image.Image,
    max_side: int,
    jpeg_quality: int,
    image_format: ImageFormat = "jpeg",
) -> Optional[str]:
    """
    Describe ONLY figures/diagrams/technical drawings in the page.
    Ignore tables. If none, return exactly: NONE
    """
    data_url = pil_to_data_url(
        page_img,
        max_side=max_side,
        quality=jpeg_quality,
        image_format=image_format,
    )
    prompt_text = build_figures_prompt()

    msg = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    )
    try:
        res = llm.invoke([msg])
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Error describing figures: %s", exc)
        logger.error("Skipping this image.")
        return None

    return str(getattr(res, "content", res)).strip()


def append_figures_block(page_text: str, figures_text: str) -> str:
    """
    Append [FIGURES] block to page text if figures_text is valid.
    If figures_text is empty or "NONE", return page_text unchanged.
    """
    cleaned = (figures_text or "").strip()
    if not cleaned or cleaned.upper() == "NONE":
        return page_text
    return page_text.rstrip() + "\n\n[FIGURES]\n" + cleaned + "\n"


def process_page_with_retries(
    *,
    source_page: int,
    effective_mode: TextExtractionMode,
    page_img: Optional[Image.Image],
    candidate_text: str,
    llm,
    cfg: OcrConfig,
    need_llm: bool,
) -> str:
    """
    Process one page with retry/backoff.
    """
    attempts = max(1, int(cfg.page_max_retries) + 1)
    last_exc: Optional[Exception] = None

    for attempt_idx in range(attempts):
        try:
            return process_page_with_strategy(
                effective_mode=effective_mode,
                page_img=page_img,
                candidate_text=candidate_text,
                llm=llm,
                cfg=cfg,
                need_llm=need_llm,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            last_exc = exc
            current_try = attempt_idx + 1
            if current_try < attempts:
                delay_sec = max(0.0, float(cfg.page_retry_backoff_sec)) * current_try
                logger.warning(
                    "Page %d failed (attempt %d/%d): %s: %s. Retrying in %.1fs",
                    source_page,
                    current_try,
                    attempts,
                    type(exc).__name__,
                    exc,
                    delay_sec,
                )
                if delay_sec > 0:
                    time.sleep(delay_sec)
                continue

            logger.error(
                "Page %d failed after %d attempt(s): %s: %s",
                source_page,
                attempts,
                type(exc).__name__,
                exc,
            )
            if cfg.continue_on_page_error:
                return cfg.page_error_placeholder
            raise

    if cfg.continue_on_page_error:
        return cfg.page_error_placeholder

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Page {source_page} failed for unknown reasons.")


def write_progress_checkpoint(
    *,
    checkpoint_path: Path,
    parts: List[str],
    processed_pages: int,
    total_pages: int,
    last_page_number: int,
) -> None:
    """
    Persist a lightweight in-progress checkpoint for long OCR jobs.
    """
    status = (
        "\n================== CHECKPOINT ==================\n"
        f"STATUS: IN_PROGRESS\n"
        f"PROCESSED_PAGES: {processed_pages}/{total_pages}\n"
        f"LAST_PAGE: {last_page_number}\n"
    )
    checkpoint_text = "".join(parts) + status
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(checkpoint_text, encoding="utf-8")


# ----------------------------
# Strategy resolution
# ----------------------------
def resolve_text_mode(cfg: OcrConfig) -> TextExtractionMode:
    """
    Resolve the effective text extraction mode based on cfg.text_extraction_mode and cfg.input_pdf_type.
    """
    if cfg.text_extraction_mode != "auto":
        return cfg.text_extraction_mode

    if cfg.input_pdf_type == "TEXT_PDF":
        return "pypdf"
    if cfg.input_pdf_type == "SCANNED_PDF":
        return "vlm"
    # MIXED_OR_UNKNOWN or None -> auto per-page fallback
    return "auto"


# ----------------------------
# Pipeline
# ----------------------------
def run_ocr_pipeline(pdf_path: Path, cfg: OcrConfig) -> str:
    """
    Run OCR pipeline and write a single output file.
    Returns the full output text (same content written to disk).
    """
    pdf_path = Path(pdf_path).expanduser().resolve()
    cfg.out_path = Path(cfg.out_path).expanduser().resolve()
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_image_format = str(cfg.image_format).lower()
    if normalized_image_format not in {"jpeg", "png"}:
        raise ValueError(
            f"Invalid image_format={cfg.image_format!r}. Allowed: jpeg, png."
        )
    cfg.image_format = cast(ImageFormat, normalized_image_format)

    if cfg.save_images:
        if cfg.images_dir is None:
            cfg.images_dir = cfg.out_path.parent / "images"
        cfg.images_dir.mkdir(parents=True, exist_ok=True)

    effective_mode = resolve_text_mode(cfg)
    total_pages = len(PdfReader(str(pdf_path)).pages)
    selected_pages = resolve_selected_pages(cfg, total_pages)
    if not selected_pages:
        raise RuntimeError("No pages selected for processing.")

    logger.info(
        "Effective text extraction mode: %s (input_pdf_type=%s)",
        effective_mode,
        cfg.input_pdf_type,
    )
    logger.info(
        "Selected pages: %d page(s), from %d to %d",
        len(selected_pages),
        selected_pages[0],
        selected_pages[-1],
    )

    # 1) Decide what we need:
    # - pypdf mode: pypdf text; render images ONLY if describe_figures=True
    # - vlm mode: render images; call VLM OCR; (figures uses same images)
    # - auto mode: try pypdf; render images (needed for fallback OCR and/or figures)
    need_images = cfg.describe_figures or (effective_mode in ("vlm", "auto"))

    # 2) Extract pypdf text pages if needed
    pypdf_page_texts: Optional[List[str]] = None
    if effective_mode in ("pypdf", "auto"):
        # TODO see if there is a better way to plug docling here
        if DOCKLING_ENABLED:
            logger.info("Extracting text via Docling...")
            pypdf_page_texts = extract_text_pages_docling(
                pdf_path,
                page_numbers=selected_pages,
            )
            logger.info("Docling extracted %d pages.", len(pypdf_page_texts))
        else:
            # use pypdf
            logger.info("Extracting text via pypdf...")
            pypdf_page_texts = extract_text_pages_pypdf(
                pdf_path,
                page_numbers=selected_pages,
            )
            logger.info("pypdf extracted %d pages.", len(pypdf_page_texts))

    # 3) Render images if needed
    page_images: Optional[List[Image.Image]] = None
    if need_images:
        logger.info("Rendering pages to images...")
        page_images = render_pdf_pages(
            pdf_path,
            dpi=cfg.dpi,
            page_numbers=selected_pages,
        )
        logger.info("Rendered %d pages.", len(page_images))

        # optional: persist images for debugging
        if cfg.save_images and cfg.images_dir:
            for idx, img in enumerate(page_images):
                source_page = selected_pages[idx]
                img_path = cfg.images_dir / f"page_{source_page:04d}.png"
                img.save(img_path)

    # Determine number of pages to process
    candidates = []
    if pypdf_page_texts is not None:
        candidates.append(len(pypdf_page_texts))
    if page_images is not None:
        candidates.append(len(page_images))
    if not candidates:
        raise RuntimeError(
            "No pages to process (neither pypdf nor image rendering produced pages)."
        )
    num_pages = min(candidates)

    # 4) Load LLM only if we might call it
    need_llm = effective_mode in ("vlm", "auto") or cfg.describe_figures
    llm = None
    if need_llm:
        logger.info("Loading LLM: %s", cfg.model_id)
        llm = get_llm(model_id=cfg.model_id)

    # 5) Assemble output
    parts: List[str] = []
    filename = os.path.basename(str(pdf_path))
    parts.append(f"SOURCE PDF: {filename}\n")
    parts.append(f"DPI: {cfg.dpi}\n")
    parts.append(f"MODEL_ID: {cfg.model_id}\n")
    parts.append(f"TEXT_MODE: {effective_mode}\n")
    parts.append(f"IMAGE_FORMAT: {cfg.image_format}\n")
    parts.append(f"INPUT_PDF_TYPE: {cfg.input_pdf_type}\n")
    parts.append(f"DESCRIBE_FIGURES: {cfg.describe_figures}\n")
    parts.append(f"PAGE_MAX_RETRIES: {cfg.page_max_retries}\n")
    parts.append(f"CONTINUE_ON_PAGE_ERROR: {cfg.continue_on_page_error}\n")
    parts.append("\n==================== BEGIN TEXT ====================\n\n")

    checkpoint_path: Optional[Path] = None
    if cfg.save_progress_checkpoint:
        checkpoint_path = (
            Path(cfg.checkpoint_path).expanduser().resolve()
            if cfg.checkpoint_path
            else cfg.out_path.with_suffix(cfg.out_path.suffix + ".checkpoint")
        )

    for idx in range(num_pages):
        source_page = selected_pages[idx]
        logger.info("Processing page %d (%d/%d) ...", source_page, idx + 1, num_pages)

        # Get image if available
        page_img = page_images[idx] if page_images is not None else None
        candidate_text = ""
        if pypdf_page_texts is not None:
            candidate_text = pypdf_page_texts[idx]

        page_text = process_page_with_retries(
            source_page=source_page,
            effective_mode=effective_mode,
            page_img=page_img,
            candidate_text=candidate_text,
            llm=llm,
            cfg=cfg,
            need_llm=need_llm,
        )
        parts.append(format_page_block(source_page, page_text))

        if checkpoint_path is not None:
            every_n = max(1, int(cfg.checkpoint_every_pages))
            processed_pages = idx + 1
            if processed_pages % every_n == 0 or processed_pages == num_pages:
                write_progress_checkpoint(
                    checkpoint_path=checkpoint_path,
                    parts=parts,
                    processed_pages=processed_pages,
                    total_pages=num_pages,
                    last_page_number=source_page,
                )

    parts.append("\n===================== END TEXT =====================\n")
    parts.append(f"TOTAL PAGES: {num_pages}\n")

    full_text = "".join(parts)
    cfg.out_path.write_text(full_text, encoding="utf-8")
    logging.info("Wrote output to %s", cfg.out_path)
    return full_text


# ----------------------------
# CLI
# ----------------------------
def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="PDF → Text (+ optional figures)")
    parser.add_argument("pdf", type=str, help="Path to the PDF file")

    parser.add_argument(
        "--model-id",
        type=str,
        default="meta.llama-4-maverick-17b-128e-instruct-fp8",
        help="LLM model id",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="./out_ocr/output.txt",
        help="Output text file path",
    )

    # Strategy inputs
    parser.add_argument(
        "--text-mode",
        type=str,
        default="auto",
        choices=["auto", "pypdf", "vlm"],
        help="Text extraction mode: auto (per-page fallback), pypdf (text only), vlm (multimodal OCR).",
    )
    parser.add_argument(
        "--input-pdf-type",
        type=str,
        default=None,
        choices=["TEXT_PDF", "SCANNED_PDF", "MIXED_OR_UNKNOWN"],
        help="Optional PDF type hint (from your classifier). Used only when --text-mode=auto.",
    )
    parser.add_argument(
        "--min-text-chars-page",
        type=int,
        default=50,
        help="AUTO mode: if pypdf page text has fewer non-whitespace chars, fallback to VLM.",
    )

    # Rendering / OCR
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--start-page", type=int, default=None)
    parser.add_argument("--end-page", type=int, default=None)
    parser.add_argument("--page-max-retries", type=int, default=2)
    parser.add_argument("--page-retry-backoff-sec", type=float, default=2.0)
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop entire job on first page failure (default is continue with placeholder).",
    )
    parser.add_argument(
        "--no-progress-checkpoint",
        action="store_true",
        help="Disable writing periodic progress checkpoints.",
    )
    parser.add_argument("--checkpoint-every-pages", type=int, default=5)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--extra-prompt", type=str, default="")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--images-dir", type=str, default=None)

    # blank detection
    parser.add_argument("--blank-white-threshold", type=int, default=245)
    parser.add_argument("--blank-min-nonwhite-ratio", type=float, default=0.01)
    parser.add_argument(
        "--no-center-crop",
        action="store_true",
        help="Disable center crop for blank detection",
    )

    # image payload
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument(
        "--image-format",
        type=str,
        default="jpeg",
        choices=["jpeg", "png"],
        help="Image format sent to multimodal LLM (default: jpeg).",
    )
    parser.add_argument("--jpeg-quality", type=int, default=85)

    # figures
    parser.add_argument(
        "--describe-figures",
        action="store_true",
        help="Append a [FIGURES] section per page (figures/diagrams only).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    cfg = OcrConfig(
        model_id=args.model_id,
        out_path=Path(args.out_path),
        dpi=args.dpi,
        max_pages=args.max_pages,
        start_page=args.start_page,
        end_page=args.end_page,
        page_max_retries=int(args.page_max_retries),
        page_retry_backoff_sec=float(args.page_retry_backoff_sec),
        continue_on_page_error=not bool(args.fail_fast),
        save_progress_checkpoint=not bool(args.no_progress_checkpoint),
        checkpoint_every_pages=int(args.checkpoint_every_pages),
        checkpoint_path=Path(args.checkpoint_path) if args.checkpoint_path else None,
        extra_prompt=args.extra_prompt,
        save_images=bool(args.save_images),
        images_dir=Path(args.images_dir) if args.images_dir else None,
        blank_white_threshold=args.blank_white_threshold,
        blank_min_nonwhite_ratio=args.blank_min_nonwhite_ratio,
        blank_use_center_crop=not bool(args.no_center_crop),
        max_side=args.max_side,
        image_format=args.image_format,
        jpeg_quality=args.jpeg_quality,
        describe_figures=bool(args.describe_figures),
        text_extraction_mode=args.text_mode,
        input_pdf_type=args.input_pdf_type,
        min_text_chars_page=int(args.min_text_chars_page),
    )

    # from command lie you can run the entire pipeline
    run_ocr_pipeline(Path(args.pdf), cfg)


if __name__ == "__main__":
    main()
