# text_from_pdf_scanner.py
"""
Scanned PDF -> page images -> multimodal LLM -> text

Features:
- Render PDF pages to images (pdfium)
- Blank-page detection (skip and emit placeholder)
- Multimodal LLM OCR per-page (LangChain model from get_llm)
- OPTIONAL: describe figures/diagrams (append at end of each page)
- Single output text file with per-page footer
- prompts for multimodal LLM in prompts.py

Author: Luigi Saetta
Python: 3.11+
"""

import os
import argparse
import base64
import logging
import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PIL import Image
import pypdfium2 as pdfium
from langchain_core.messages import HumanMessage

from oci_models import get_llm
from prompts import build_ocr_text_prompt, build_figures_prompt
from utils import get_console_logger


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
    dpi: int = 200
    max_pages: Optional[int] = None
    extra_prompt: str = ""
    save_images: bool = False
    images_dir: Optional[Path] = None

    # blank detection
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


logger = get_console_logger()


# ----------------------------
# Image helpers
# ----------------------------
def pil_to_data_url(img: Image.Image, max_side: int = 1600, quality: int = 85) -> str:
    """Convert PIL image to JPEG data URL (base64), resizing to keep payload manageable."""
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def render_pdf_pages(
    pdf_path: Path, dpi: int = 200, max_pages: Optional[int] = None
) -> List[Image.Image]:
    """Render each PDF page to a PIL image using pdfium."""
    pdf = pdfium.PdfDocument(str(pdf_path))
    n_pages = len(pdf)
    if max_pages is not None:
        n_pages = min(n_pages, max_pages)

    images: List[Image.Image] = []
    scale = dpi / 72.0
    for i in range(n_pages):
        page = pdf[i]
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
    w, h = gray.size

    if use_center_crop:
        gray = gray.crop((int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)))
        w, h = gray.size

    pixels = gray.load()
    total = w * h
    non_white = 0

    # Simple loop (fast enough for typical PDFs; optimize later if needed)
    for x in range(w):
        for y in range(h):
            if pixels[x, y] < white_threshold:
                non_white += 1

    ratio = non_white / total
    return ratio < min_nonwhite_ratio


def format_page_block(page_idx: int, text: str) -> str:
    """
    Format a page block with footer.
    """
    footer = f"\n\n--- PAGE {page_idx} ---\n\n"
    return text.rstrip() + footer


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

    # changed to produce the correct language and to produce
    # a more detailed description
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
    figures_text = (figures_text or "").strip()
    if not figures_text:
        return page_text
    if figures_text.upper() == "NONE":
        return page_text
    return page_text.rstrip() + "\n\n[FIGURES]\n" + figures_text + "\n"


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

    logger.info("Rendering pages...")
    pages = render_pdf_pages(pdf_path, dpi=cfg.dpi, max_pages=cfg.max_pages)
    logger.info("Rendered %d pages.", len(pages))

    logger.info("Loading LLM: %s", cfg.model_id)
    llm = get_llm(model_id=cfg.model_id, temperature=0.0, max_tokens=4000)

    parts: List[str] = []
    filename = os.path.basename(pdf_path)
    parts.append(f"SOURCE PDF: {filename}\n")
    parts.append(f"DPI: {cfg.dpi}\n")
    parts.append(f"MODEL_ID: {cfg.model_id}\n")
    parts.append(f"DESCRIBE_FIGURES: {cfg.describe_figures}\n")

    parts.append("\n==================== BEGIN TEXT ====================\n\n")

    for idx, img in enumerate(pages, start=1):
        logger.info("Processing page %d/%d ...", idx, len(pages))

        if cfg.save_images and cfg.images_dir:
            img_path = cfg.images_dir / f"page_{idx:04d}.png"
            img.save(img_path)

        if is_blank_page(
            img,
            white_threshold=cfg.blank_white_threshold,
            min_nonwhite_ratio=cfg.blank_min_nonwhite_ratio,
            use_center_crop=cfg.blank_use_center_crop,
        ):
            parts.append(format_page_block(idx, cfg.blank_placeholder))
            continue

        logger.info("  - calling LLM for text extraction...")
        text = call_multimodal_llm_text_only(
            llm,
            img,
            extra_prompt=cfg.extra_prompt,
            max_side=cfg.max_side,
            jpeg_quality=cfg.jpeg_quality,
        )

        if cfg.describe_figures:
            logger.info("  - calling LLM for figures description...")
            figs = call_multimodal_llm_figures_only(
                llm,
                img,
                max_side=cfg.max_side,
                jpeg_quality=cfg.jpeg_quality,
            )
            text = append_figures_block(text, figs)

        parts.append(format_page_block(idx, text))

    parts.append("\n===================== END TEXT =====================\n")
    parts.append(f"TOTAL PAGES: {len(pages)}\n")

    full_text = "".join(parts)
    cfg.out_path.write_text(full_text, encoding="utf-8")
    logging.info("Wrote output to %s", cfg.out_path)
    return full_text


# ----------------------------
# CLI
# ----------------------------
def main():
    """
    Main
    """
    parser = argparse.ArgumentParser(
        description="Scanned PDF → OCR text (blank pages skipped)"
    )
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
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--extra-prompt", type=str, default="")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--images-dir", type=str, default=None)
    parser.add_argument("--blank-white-threshold", type=int, default=245)
    parser.add_argument("--blank-min-nonwhite-ratio", type=float, default=0.01)
    parser.add_argument(
        "--no-center-crop",
        action="store_true",
        help="Disable center crop for blank detection",
    )
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--jpeg-quality", type=int, default=85)
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
        save_images=args.save_images,
        images_dir=Path(args.images_dir) if args.images_dir else None,
        blank_white_threshold=args.blank_white_threshold,
        blank_min_nonwhite_ratio=args.blank_min_nonwhite_ratio,
        blank_use_center_crop=not args.no_center_crop,
        max_side=args.max_side,
        jpeg_quality=args.jpeg_quality,
        describe_figures=bool(args.describe_figures),
    )

    run_ocr_pipeline(Path(args.pdf), cfg)


if __name__ == "__main__":
    main()
