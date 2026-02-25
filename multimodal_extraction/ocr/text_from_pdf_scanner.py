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
import hashlib
import io
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, cast

from docling.document_converter import DocumentConverter
from PIL import Image
import pypdfium2 as pdfium
from pypdf import PdfReader, PdfWriter
from langchain_core.messages import HumanMessage

from multimodal_extraction.models.oci_models import get_llm
from multimodal_extraction.prompts.prompts import (
    build_ocr_text_prompt,
    build_figures_prompt,
    build_complex_table_detection_prompt,
)
from multimodal_extraction.ocr.docling_post_processing import cleanup_docling_text_keep_captions
from multimodal_extraction.utils import get_console_logger
from multimodal_extraction.config import (
    DOCLING_ENABLED,
    DOCLING_TIMEOUT_SEC,
    DOCLING_FALLBACK_TO_PYPDF,
    DOCLING_MAX_CHUNK_PAGES,
    ENABLE_CLEANUP,
    ENABLE_MODEL_COMPARISON,
    REFERENCE_MODEL_ID,
    MODEL_COMPARISON_CACHE_DIR,
    ENABLE_COMPLEX_TABLE_DETECTION,
)

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

    # optional model comparison (WER against reference model)
    enable_model_comparison: bool = ENABLE_MODEL_COMPARISON
    reference_model_id: str = REFERENCE_MODEL_ID
    comparison_cache_dir: Path = Path(MODEL_COMPARISON_CACHE_DIR)
    comparison_result: Optional[Dict[str, Any]] = None

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


def _tokenize_for_wer(text: str) -> List[str]:
    """Tokenize text into normalized word units for WER."""
    lowered = (text or "").lower()
    tokens = re.findall(r"\w+", lowered, flags=re.UNICODE)
    return tokens


def _edit_distance_words(ref_tokens: List[str], hyp_tokens: List[str]) -> int:
    """Levenshtein distance on token lists."""
    n = len(ref_tokens)
    m = len(hyp_tokens)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            sub_cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + sub_cost,  # substitution
            )
        prev = curr
    return prev[m]


def compute_wer(reference_text: str, hypothesis_text: str) -> float:
    """Compute word error rate (WER)."""
    ref_tokens = _tokenize_for_wer(reference_text)
    hyp_tokens = _tokenize_for_wer(hypothesis_text)
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return _edit_distance_words(ref_tokens, hyp_tokens) / len(ref_tokens)


def _sanitize_model_id(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id.strip())


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _reference_cache_path(
    *,
    cache_dir: Path,
    pdf_digest: str,
    reference_model_id: str,
    page_number: int,
    prompt_fingerprint: str,
    image_format: str,
    max_side: int,
    jpeg_quality: int,
) -> Path:
    model_key = _sanitize_model_id(reference_model_id)
    return (
        cache_dir
        / model_key
        / pdf_digest
        / f"p{page_number:05d}_{prompt_fingerprint}_{image_format}_{max_side}_{jpeg_quality}.txt"
    )


def _read_cached_reference_text(cache_path: Path) -> Optional[str]:
    if not cache_path.exists():
        return None
    return cache_path.read_text(encoding="utf-8")


def _write_cached_reference_text(cache_path: Path, text: str) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")


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
    timeout_sec: Optional[float] = None,
) -> List[str]:
    """
    Extract per-page Markdown from a TEXT_PDF using Docling.
    Tables are rendered as Markdown. Images are suppressed (no placeholders).

    Returns:
        List[str]: per-page markdown (len == number of pages considered).
    """
    pdf_path = Path(pdf_path).expanduser().resolve()

    pages = _extract_all_pages_docling_guarded(pdf_path, timeout_sec=timeout_sec)
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


def _extract_all_pages_docling(pdf_path: Path) -> List[str]:
    """
    Convert a PDF with Docling and return all pages as Markdown slices.
    """
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
    return [p.strip() for p in md.split(page_break)]


def _extract_all_pages_docling_guarded(
    pdf_path: Path,
    timeout_sec: Optional[float],
) -> List[str]:
    """
    Run Docling extraction, optionally guarded by a hard timeout.
    """
    if timeout_sec is None or float(timeout_sec) <= 0:
        return _extract_all_pages_docling(pdf_path)

    timeout_val = float(timeout_sec)
    helper_code = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "from multimodal_extraction.ocr.text_from_pdf_scanner import _extract_all_pages_docling\n"
        "pages = _extract_all_pages_docling(Path(sys.argv[1]))\n"
        "Path(sys.argv[2]).write_text(json.dumps(pages), encoding='utf-8')\n"
    )

    tmp_out_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
            tmp_out_path = Path(tmp_file.name)

        # Use a fresh interpreter process (exec-based), not fork, to avoid
        # macOS CoreFoundation/Objective-C fork-safety crashes.
        proc = subprocess.run(
            [sys.executable, "-c", helper_code, str(pdf_path), str(tmp_out_path)],
            capture_output=True,
            text=True,
            timeout=timeout_val,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"Docling timed out after {timeout_val:.1f}s while processing {pdf_path.name}."
        ) from exc

    try:
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            details = stderr or stdout or f"exit code {proc.returncode}"
            raise RuntimeError(f"Docling subprocess failed: {details}")

        if tmp_out_path is None or not tmp_out_path.exists():
            raise RuntimeError(
                f"Docling subprocess exited without returning content for {pdf_path.name}."
            )

        payload = json.loads(tmp_out_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise RuntimeError(
                f"Docling subprocess returned unexpected payload type: {type(payload).__name__}."
            )
        return cast(List[str], payload)
    finally:
        if tmp_out_path is not None and tmp_out_path.exists():
            tmp_out_path.unlink(missing_ok=True)


def _write_pdf_subset(
    source_pdf_path: Path,
    page_numbers: List[int],
    target_pdf_path: Path,
) -> None:
    """
    Write a temporary PDF containing only the selected 1-based pages.
    """
    reader = PdfReader(str(source_pdf_path))
    writer = PdfWriter()
    total_pages = len(reader.pages)
    for page_1based in page_numbers:
        if page_1based < 1 or page_1based > total_pages:
            raise ValueError(
                f"Invalid page {page_1based}. PDF has {total_pages} pages."
            )
        writer.add_page(reader.pages[page_1based - 1])
    with target_pdf_path.open("wb") as handle:
        writer.write(handle)


def _extract_docling_for_page_numbers(
    pdf_path: Path,
    page_numbers: List[int],
    timeout_sec: Optional[float],
) -> List[str]:
    """
    Run Docling on a temporary subset PDF and return one text per requested page.
    """
    if not page_numbers:
        return []

    tmp_pdf_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_pdf_path = Path(tmp_file.name)

        _write_pdf_subset(pdf_path, page_numbers, tmp_pdf_path)
        page_texts = extract_text_pages_docling(
            tmp_pdf_path,
            timeout_sec=timeout_sec,
        )
        if len(page_texts) != len(page_numbers):
            raise RuntimeError(
                "Docling subset extraction returned an unexpected page count "
                f"(expected {len(page_numbers)}, got {len(page_texts)})."
            )
        return page_texts
    finally:
        if tmp_pdf_path is not None and tmp_pdf_path.exists():
            tmp_pdf_path.unlink(missing_ok=True)


def _extract_candidate_text_pages_docling_resilient(
    pdf_path: Path,
    *,
    selected_pages: List[int],
    docling_timeout_sec: Optional[float],
    docling_fallback_to_pypdf: bool,
    docling_max_chunk_pages: int,
) -> List[str]:
    """
    Resilient Docling extraction:
    - process in chunks
    - on chunk failure, recursively split
    - if a single page still fails, fallback to pypdf only for that page
    """
    if not selected_pages:
        return []

    page_to_text: Dict[int, str] = {}

    def process_segment(segment_pages: List[int]) -> None:
        if not segment_pages:
            return
        seg_start = segment_pages[0]
        seg_end = segment_pages[-1]
        logger.info(
            "Docling chunk start: pages %d-%d (%d page(s))",
            seg_start,
            seg_end,
            len(segment_pages),
        )
        try:
            segment_texts = _extract_docling_for_page_numbers(
                pdf_path=pdf_path,
                page_numbers=segment_pages,
                timeout_sec=docling_timeout_sec,
            )
            for page_num, text in zip(segment_pages, segment_texts):
                page_to_text[page_num] = text
            logger.info(
                "Docling chunk completed: pages %d-%d (%d page(s))",
                seg_start,
                seg_end,
                len(segment_pages),
            )
            return
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if len(segment_pages) == 1:
                page_num = segment_pages[0]
                if not docling_fallback_to_pypdf:
                    raise RuntimeError(
                        f"Docling failed on page {page_num} and fallback is disabled."
                    ) from exc
                logger.warning(
                    "Docling failed on page %d (%s: %s). Falling back to pypdf for this page.",
                    page_num,
                    type(exc).__name__,
                    exc,
                )
                page_to_text[page_num] = extract_text_single_page_pypdf(pdf_path, page_num)
                return

            mid = len(segment_pages) // 2
            left = segment_pages[:mid]
            right = segment_pages[mid:]
            logger.warning(
                "Docling failed on pages %d-%d (%s: %s). Splitting chunk into [%d-%d] and [%d-%d].",
                segment_pages[0],
                segment_pages[-1],
                type(exc).__name__,
                exc,
                left[0],
                left[-1],
                right[0],
                right[-1],
            )
            process_segment(left)
            process_segment(right)

    chunk_size = max(1, int(docling_max_chunk_pages))
    for start_idx in range(0, len(selected_pages), chunk_size):
        process_segment(selected_pages[start_idx : start_idx + chunk_size])

    return [page_to_text[p] for p in selected_pages]


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


def call_multimodal_llm_complex_table_only(
    llm,
    page_img: Image.Image,
    max_side: int,
    jpeg_quality: int,
    image_format: ImageFormat = "jpeg",
) -> Optional[bool]:
    """
    Return True if the page contains a complex table, False otherwise.
    Returns None only on call failure.
    """
    data_url = pil_to_data_url(
        page_img,
        max_side=max_side,
        quality=jpeg_quality,
        image_format=image_format,
    )
    prompt_text = build_complex_table_detection_prompt()

    msg = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    )

    started = time.perf_counter()
    try:
        res = llm.invoke([msg])
    except Exception as exc:  # pylint: disable=broad-exception-caught
        elapsed = time.perf_counter() - started
        logger.error("Error checking complex table presence: %s", exc)
        logger.error("Complex table check failed after %.2fs", elapsed)
        return None

    elapsed = time.perf_counter() - started
    raw = str(getattr(res, "content", res)).strip()
    normalized = raw.upper()
    if normalized.startswith("YES"):
        decision = True
    elif normalized.startswith("NO"):
        decision = False
    else:
        logger.warning(
            "Unexpected complex table classifier output: %r. Falling back to NO.",
            raw,
        )
        decision = False

    logger.info(
        "Complex table check completed in %.2fs | result=%s",
        elapsed,
        "YES" if decision else "NO",
    )
    return decision


def page_contains_complex_table(
    pdf_path: Path,
    *,
    page_1based: int,
    model_id: str,
    dpi: int = 200,
    max_side: int = 1600,
    jpeg_quality: int = 85,
    image_format: ImageFormat = "jpeg",
    enabled: bool = ENABLE_COMPLEX_TABLE_DETECTION,
) -> bool:
    """
    Check if a single PDF page contains a complex table using the selected VLM.
    """
    if not enabled:
        logger.info("Complex table detection disabled by config/flag.")
        return False

    page_img = render_single_page(pdf_path, page_1based=page_1based, dpi=dpi)

    llm = get_llm(model_id=model_id)
    result = call_multimodal_llm_complex_table_only(
        llm,
        page_img,
        max_side=max_side,
        jpeg_quality=jpeg_quality,
        image_format=image_format,
    )
    if result is None:
        raise RuntimeError("Complex table detection call returned no content.")
    return result


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


def extract_candidate_text_pages(
    pdf_path: Path,
    *,
    effective_mode: TextExtractionMode,
    selected_pages: List[int],
    docling_enabled: bool = DOCLING_ENABLED,
    docling_timeout_sec: Optional[float] = DOCLING_TIMEOUT_SEC,
    docling_fallback_to_pypdf: bool = DOCLING_FALLBACK_TO_PYPDF,
    docling_max_chunk_pages: int = DOCLING_MAX_CHUNK_PAGES,
) -> Optional[List[str]]:
    """
    Extract text candidate pages for pypdf/auto modes.
    When Docling is enabled, it is tried first and can fallback to pypdf on failure/timeout.
    """
    if effective_mode not in ("pypdf", "auto"):
        return None

    if docling_enabled:
        logger.info(
            "Extracting text via Docling (timeout=%ss, chunk=%s pages)...",
            "off" if not docling_timeout_sec else docling_timeout_sec,
            docling_max_chunk_pages,
        )
        try:
            page_texts = _extract_candidate_text_pages_docling_resilient(
                pdf_path,
                selected_pages=selected_pages,
                docling_timeout_sec=docling_timeout_sec,
                docling_fallback_to_pypdf=docling_fallback_to_pypdf,
                docling_max_chunk_pages=docling_max_chunk_pages,
            )
            logger.info("Docling extracted %d pages.", len(page_texts))
            return page_texts
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if not docling_fallback_to_pypdf:
                raise
            logger.warning(
                "Docling failed (%s: %s). Falling back to pypdf.",
                type(exc).__name__,
                exc,
            )

    logger.info("Extracting text via pypdf...")
    page_texts = extract_text_pages_pypdf(
        pdf_path,
        page_numbers=selected_pages,
    )
    logger.info("pypdf extracted %d pages.", len(page_texts))
    return page_texts


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

    cfg.comparison_cache_dir = Path(cfg.comparison_cache_dir).expanduser().resolve()

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
    comparison_enabled = bool(cfg.enable_model_comparison)
    need_images = (
        cfg.describe_figures
        or (effective_mode in ("vlm", "auto"))
        or comparison_enabled
    )

    # 2) Extract pypdf text pages if needed
    pypdf_page_texts = extract_candidate_text_pages(
        pdf_path,
        effective_mode=effective_mode,
        selected_pages=selected_pages,
    )

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

    reference_llm = None
    reference_model_id = (cfg.reference_model_id or "").strip()
    prompt_fingerprint = hashlib.sha1(
        build_ocr_text_prompt(extra_prompt=cfg.extra_prompt).encode("utf-8")
    ).hexdigest()[:12]
    pdf_digest = _sha256_file(pdf_path)

    comparison_pages_evaluated = 0
    comparison_cache_hits = 0
    comparison_cache_misses = 0
    comparison_errors = 0
    comparison_page_metrics: List[Dict[str, Any]] = []

    if comparison_enabled:
        if not reference_model_id:
            logger.warning(
                "Model comparison enabled but no reference_model_id provided. Comparison disabled for this run."
            )
            comparison_enabled = False
        else:
            if llm is not None and reference_model_id == cfg.model_id:
                reference_llm = llm
            else:
                logger.info("Loading reference model for comparison: %s", reference_model_id)
                reference_llm = get_llm(model_id=reference_model_id)

    # 5) Assemble output
    parts: List[str] = []
    filename = os.path.basename(str(pdf_path))
    parts.append(f"SOURCE PDF: {filename}\n")
    parts.append(f"DPI: {cfg.dpi}\n")
    parts.append(f"MODEL_ID: {cfg.model_id}\n")
    parts.append(f"TEXT_MODE: {effective_mode}\n")
    parts.append(f"IMAGE_FORMAT: {cfg.image_format}\n")
    parts.append(f"MODEL_COMPARISON_ENABLED: {comparison_enabled}\n")
    if comparison_enabled:
        parts.append(f"REFERENCE_MODEL_ID: {reference_model_id}\n")
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

        if comparison_enabled:
            try:
                if reference_llm is None:
                    raise RuntimeError("Reference model is not loaded.")

                reference_img = page_img
                if reference_img is None:
                    reference_img = render_single_page(
                        pdf_path,
                        page_1based=source_page,
                        dpi=cfg.dpi,
                    )

                cache_path = _reference_cache_path(
                    cache_dir=cfg.comparison_cache_dir,
                    pdf_digest=pdf_digest,
                    reference_model_id=reference_model_id,
                    page_number=source_page,
                    prompt_fingerprint=prompt_fingerprint,
                    image_format=cfg.image_format,
                    max_side=cfg.max_side,
                    jpeg_quality=cfg.jpeg_quality,
                )
                reference_text = _read_cached_reference_text(cache_path)
                cache_hit = reference_text is not None

                if not cache_hit:
                    comparison_cache_misses += 1
                    reference_text = call_multimodal_llm_text_only(
                        reference_llm,
                        reference_img,
                        extra_prompt=cfg.extra_prompt,
                        max_side=cfg.max_side,
                        jpeg_quality=cfg.jpeg_quality,
                        image_format=cfg.image_format,
                    )
                    if reference_text is None:
                        raise RuntimeError("Reference OCR call returned no content.")
                    _write_cached_reference_text(cache_path, reference_text)
                else:
                    comparison_cache_hits += 1

                page_wer = compute_wer(reference_text, page_text)
                comparison_pages_evaluated += 1
                comparison_page_metrics.append(
                    {
                        "page": source_page,
                        "wer": page_wer,
                        "cache_hit": cache_hit,
                    }
                )
                logger.info(
                    "Model comparison | page=%d | ref=%s | wer=%.4f",
                    source_page,
                    reference_model_id,
                    page_wer,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                comparison_errors += 1
                logger.warning(
                    "Model comparison skipped on page %d: %s: %s",
                    source_page,
                    type(exc).__name__,
                    exc,
                )

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

    mean_wer = None
    if comparison_page_metrics:
        mean_wer = sum(p["wer"] for p in comparison_page_metrics) / len(
            comparison_page_metrics
        )
        logger.info(
            "Model comparison summary | ref=%s | pages=%d | mean_wer=%.4f | cache_hits=%d | cache_misses=%d | errors=%d",
            reference_model_id,
            comparison_pages_evaluated,
            mean_wer,
            comparison_cache_hits,
            comparison_cache_misses,
            comparison_errors,
        )
    elif comparison_enabled:
        logger.info(
            "Model comparison summary | ref=%s | pages=0 | cache_hits=%d | cache_misses=%d | errors=%d",
            reference_model_id,
            comparison_cache_hits,
            comparison_cache_misses,
            comparison_errors,
        )

    cfg.comparison_result = {
        "enabled": comparison_enabled,
        "reference_model_id": reference_model_id if comparison_enabled else None,
        "pages_evaluated": comparison_pages_evaluated,
        "mean_wer": mean_wer,
        "cache_hits": comparison_cache_hits,
        "cache_misses": comparison_cache_misses,
        "errors": comparison_errors,
    }

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
    parser.add_argument(
        "--enable-model-comparison",
        action="store_true",
        help="Compute WER against a reference model output (cached on local filesystem).",
    )
    parser.add_argument(
        "--reference-model-id",
        type=str,
        default=REFERENCE_MODEL_ID,
        help="Reference model used for WER comparison.",
    )
    parser.add_argument(
        "--comparison-cache-dir",
        type=str,
        default=MODEL_COMPARISON_CACHE_DIR,
        help="Cache directory for per-page reference OCR outputs.",
    )

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
        enable_model_comparison=bool(args.enable_model_comparison),
        reference_model_id=args.reference_model_id,
        comparison_cache_dir=Path(args.comparison_cache_dir),
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
