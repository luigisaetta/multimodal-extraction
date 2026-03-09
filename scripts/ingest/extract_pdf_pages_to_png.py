"""
Author: Luigi Saetta
Date last modified: 2026-02-19
Python Version: 3.11+
License: MIT

Extract all pages from all PDFs in an input directory and save them as PNG files.

Output layout:
- <base_dir>/<pdf_stem>/page0001.png
- <base_dir>/<pdf_stem>/page0002.png
- ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

from multimodal_extraction.utils import get_console_logger

logger = get_console_logger()


def render_pdf_pages_reuse_or_fallback(pdf_path: Path, dpi: int):
    """
    Reuse project renderer when available; fallback to local pdfium renderer
    if optional OCR dependencies (e.g. docling) are missing.
    """
    try:
        from multimodal_extraction.ocr.text_from_pdf_scanner import render_pdf_pages

        return render_pdf_pages(pdf_path=pdf_path, dpi=dpi)
    except ModuleNotFoundError as exc:
        logger.warning(
            "Could not import shared renderer (%s). Falling back to local pdfium rendering.",
            exc,
        )

    try:
        import pypdfium2 as pdfium
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PDF rendering dependency not found. Install either project OCR deps "
            "(to reuse text_from_pdf_scanner renderer) or pypdfium2."
        ) from exc

    pdf_doc = pdfium.PdfDocument(str(pdf_path))
    scale = dpi / 72.0
    images = []
    for page_idx0 in range(len(pdf_doc)):
        page = pdf_doc[page_idx0]
        bitmap = page.render(scale=scale)
        images.append(bitmap.to_pil())
    return images


def list_pdf_files(input_dir: Path) -> list[Path]:
    """Return non-recursive PDF files sorted by name."""
    return sorted(
        [
            p
            for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".pdf"
        ],
        key=lambda p: p.name.lower(),
    )


def extract_pages_for_pdf(pdf_path: Path, out_root: Path, dpi: int) -> int:
    """Render all pages for one PDF and save page images."""
    out_dir = out_root / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=False)

    images = render_pdf_pages_reuse_or_fallback(pdf_path=pdf_path, dpi=dpi)
    for idx, image in enumerate(images, start=1):
        out_path = out_dir / f"page{idx:04d}.png"
        image.save(out_path, format="PNG")

    return len(images)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract all pages from all PDFs in a directory as PNG files. "
            "Output: <base-dir>/<pdf-name>/page0001.png"
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=str,
        help="Directory containing PDF files.",
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        type=str,
        help="Root output directory.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Rendering DPI (default: 200).",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    base_dir = Path(args.base_dir).expanduser().resolve()
    dpi = int(args.dpi)

    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Invalid --input-dir: {input_dir}")
    if dpi <= 0:
        raise ValueError("--dpi must be a positive integer.")

    base_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list_pdf_files(input_dir)
    if not pdf_files:
        logger.warning("No PDF files found in: %s", input_dir)
        return

    logger.info("Found %d PDF file(s) in %s", len(pdf_files), input_dir)

    total_pages = 0
    skipped_pdfs = 0
    for i, pdf_path in enumerate(pdf_files, start=1):
        out_dir = base_dir / pdf_path.stem
        if out_dir.exists():
            skipped_pdfs += 1
            logger.info(
                "[%d/%d] Skipping %s (destination already exists: %s)",
                i,
                len(pdf_files),
                pdf_path.name,
                out_dir,
            )
            continue

        logger.info("[%d/%d] Processing %s", i, len(pdf_files), pdf_path.name)
        pages = extract_pages_for_pdf(pdf_path=pdf_path, out_root=base_dir, dpi=dpi)
        total_pages += pages
        logger.info("Saved %d page(s) under %s", pages, base_dir / pdf_path.stem)

    logger.info(
        "Done. PDFs=%d, skipped=%d, processed=%d, pages=%d, output=%s",
        len(pdf_files),
        skipped_pdfs,
        len(pdf_files) - skipped_pdfs,
        total_pages,
        base_dir,
    )


if __name__ == "__main__":
    main()
