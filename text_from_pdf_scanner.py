# text_from_pdf_scanner.py
"""
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

Author: Luigi Saetta
Python: 3.11+
License: MIT
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Literal

from PIL import Image
import pypdfium2 as pdfium
from pypdf import PdfReader
from langchain_core.messages import HumanMessage

from oci_models import get_llm
from prompts import build_ocr_text_prompt, build_figures_prompt
from utils import get_console_logger

logger = get_console_logger()

PdfTypeLabel = Literal["TEXT_PDF", "SCANNED_PDF", "MIXED_OR_UNKNOWN"]
TextExtractionMode = Literal["auto", "pypdf", "vlm"]


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
    jpeg_quality: int = 85

    # placeholder
    blank_placeholder: str = "[BLANK PAGE SKIPPED]"

    # figures
    describe_figures: bool = False


# ----------------------------
# Image helpers
# ----------------------------
def pil_to_data_url(img: Image.Image, max_side: int = 1600, quality: int = 85) -> str:
    """Convert PIL image to JPEG data URL (base64), resizing to keep payload manageable."""
    if img.mode != "RGB":
        img = img.convert("RGB")

    width, height = img.size
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        img = img.resize((int(width * scale), int(height * scale)))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def render_pdf_pages(
    pdf_path: Path, dpi: int = 200, max_pages: Optional[int] = None
) -> List[Image.Image]:
    """Render each PDF page to a PIL image using pdfium."""
    pdf_doc = pdfium.PdfDocument(str(pdf_path))
    total_pages = len(pdf_doc)
    if max_pages is not None:
        total_pages = min(total_pages, max_pages)

    images: List[Image.Image] = []
    scale = dpi / 72.0
    for page_idx in range(total_pages):
        page = pdf_doc[page_idx]
        bitmap = page.render(scale=scale)
        images.append(bitmap.to_pil())
    return images


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


# ----------------------------
# Text extraction (pypdf)
# ----------------------------
def extract_text_pages_pypdf(
    pdf_path: Path, max_pages: Optional[int] = None
) -> List[str]:
    """
    Extract per-page text using pypdf.

    Returns:
        List of page texts (len == number of pages considered).
        Empty string for pages with no extractable text.
    """
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    if max_pages is not None:
        total_pages = min(total_pages, max_pages)

    page_texts: List[str] = []
    for page_idx in range(total_pages):
        page = reader.pages[page_idx]
        text = page.extract_text() or ""
        page_texts.append(text.strip())
    return page_texts


def page_has_enough_text(page_text: str, min_chars: int) -> bool:
    """Heuristic: decide whether pypdf extraction is 'good enough' for a page."""
    if not page_text:
        return False
    # count non-whitespace chars
    non_ws = sum(1 for c in page_text if not c.isspace())
    return non_ws >= min_chars


# ----------------------------
# Multimodal LLM calls
# ----------------------------
def call_multimodal_llm_text_only(
    llm,
    page_img: Image.Image,
    extra_prompt: str,
    max_side: int,
    jpeg_quality: int,
) -> str:
    """
    Ask the model for ONLY transcribed text (no JSON).
    This is far more stable across providers (Gemini included).
    """
    data_url = pil_to_data_url(page_img, max_side=max_side, quality=jpeg_quality)
    prompt_text = build_ocr_text_prompt(extra_prompt=extra_prompt)

    msg = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    )
    res = llm.invoke([msg])
    if hasattr(res, "content"):
        return str(res.content).strip()
    return str(res).strip()


def call_multimodal_llm_figures_only(
    llm,
    page_img: Image.Image,
    max_side: int,
    jpeg_quality: int,
) -> str:
    """
    Describe ONLY figures/diagrams/technical drawings in the page.
    Ignore tables. If none, return exactly: NONE
    """
    data_url = pil_to_data_url(page_img, max_side=max_side, quality=jpeg_quality)
    prompt_text = build_figures_prompt()

    msg = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    )
    res = llm.invoke([msg])
    if hasattr(res, "content"):
        return str(res.content).strip()
    return str(res).strip()


def append_figures_block(page_text: str, figures_text: str) -> str:
    """
    Append [FIGURES] block to page text if figures_text is valid.
    If figures_text is empty or "NONE", return page_text unchanged.
    """
    cleaned = (figures_text or "").strip()
    if not cleaned:
        return page_text
    if cleaned.upper() == "NONE":
        return page_text
    return page_text.rstrip() + "\n\n[FIGURES]\n" + cleaned + "\n"


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

    if cfg.save_images:
        if cfg.images_dir is None:
            cfg.images_dir = cfg.out_path.parent / "images"
        cfg.images_dir.mkdir(parents=True, exist_ok=True)

    effective_mode = resolve_text_mode(cfg)
    logger.info(
        "Effective text extraction mode: %s (input_pdf_type=%s)",
        effective_mode,
        cfg.input_pdf_type,
    )

    # 1) Decide what we need:
    # - pypdf mode: pypdf text; render images ONLY if describe_figures=True
    # - vlm mode: render images; call VLM OCR; (figures uses same images)
    # - auto mode: try pypdf; render images (needed for fallback OCR and/or figures)
    need_images = cfg.describe_figures or (effective_mode in ("vlm", "auto"))

    # 2) Extract pypdf text pages if needed
    pypdf_page_texts: Optional[List[str]] = None
    if effective_mode in ("pypdf", "auto"):
        logger.info("Extracting text via pypdf...")
        pypdf_page_texts = extract_text_pages_pypdf(pdf_path, max_pages=cfg.max_pages)
        logger.info("pypdf extracted %d pages.", len(pypdf_page_texts))

    # 3) Render images if needed
    page_images: Optional[List[Image.Image]] = None
    if need_images:
        logger.info("Rendering pages to images...")
        page_images = render_pdf_pages(pdf_path, dpi=cfg.dpi, max_pages=cfg.max_pages)
        logger.info("Rendered %d pages.", len(page_images))

        # optional: persist images for debugging
        if cfg.save_images and cfg.images_dir:
            for idx, img in enumerate(page_images, start=1):
                img_path = cfg.images_dir / f"page_{idx:04d}.png"
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
    parts.append(f"INPUT_PDF_TYPE: {cfg.input_pdf_type}\n")
    parts.append(f"DESCRIBE_FIGURES: {cfg.describe_figures}\n")
    parts.append("\n==================== BEGIN TEXT ====================\n\n")

    for idx in range(1, num_pages + 1):
        logger.info("Processing page %d/%d ...", idx, num_pages)

        # Get image if available
        page_img = page_images[idx - 1] if page_images is not None else None

        # If we might call VLM (text or figures), do blank detection using the image
        if page_img is not None and need_llm:
            if is_blank_page(
                page_img,
                white_threshold=cfg.blank_white_threshold,
                min_nonwhite_ratio=cfg.blank_min_nonwhite_ratio,
                use_center_crop=cfg.blank_use_center_crop,
            ):
                parts.append(format_page_block(idx, cfg.blank_placeholder))
                continue

        # ---- TEXT (choose strategy) ----
        page_text = ""

        if effective_mode == "pypdf":
            # Always use pypdf for text
            assert pypdf_page_texts is not None
            page_text = pypdf_page_texts[idx - 1]

        elif effective_mode == "vlm":
            # Always use multimodal OCR for text
            if llm is None or page_img is None:
                raise RuntimeError("VLM mode requires both llm and page image.")
            logger.info("  - calling LLM for text extraction (vlm)...")
            page_text = call_multimodal_llm_text_only(
                llm,
                page_img,
                extra_prompt=cfg.extra_prompt,
                max_side=cfg.max_side,
                jpeg_quality=cfg.jpeg_quality,
            )

        else:
            # auto: try pypdf, fallback to VLM if page looks empty/weak
            assert pypdf_page_texts is not None
            candidate_text = pypdf_page_texts[idx - 1]
            if page_has_enough_text(candidate_text, cfg.min_text_chars_page):
                page_text = candidate_text
            else:
                if llm is None or page_img is None:
                    # This can happen if someone sets describe_figures=False AND
                    # disables images somehow; but in auto we force images anyway.
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
                )

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
            )
            page_text = append_figures_block(page_text, figs_text)

        parts.append(format_page_block(idx, page_text))

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
        extra_prompt=args.extra_prompt,
        save_images=bool(args.save_images),
        images_dir=Path(args.images_dir) if args.images_dir else None,
        blank_white_threshold=args.blank_white_threshold,
        blank_min_nonwhite_ratio=args.blank_min_nonwhite_ratio,
        blank_use_center_crop=not bool(args.no_center_crop),
        max_side=args.max_side,
        jpeg_quality=args.jpeg_quality,
        describe_figures=bool(args.describe_figures),
        text_extraction_mode=args.text_mode,
        input_pdf_type=args.input_pdf_type,
        min_text_chars_page=int(args.min_text_chars_page),
    )

    run_ocr_pipeline(Path(args.pdf), cfg)


if __name__ == "__main__":
    main()
