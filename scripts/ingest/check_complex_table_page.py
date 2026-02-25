"""
Author: Luigi Saetta
Date last modified: 2026-02-18
Python Version: 3.11
License: MIT

Check if a SINGLE PDF page contains a complex table using a selected VLM.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from multimodal_extraction.config import ENABLE_COMPLEX_TABLE_DETECTION
from multimodal_extraction.ocr.text_from_pdf_scanner import page_contains_complex_table
from multimodal_extraction.utils import get_console_logger

logger = get_console_logger()


def main() -> None:
    """
    CLI entrypoint.
    """
    parser = argparse.ArgumentParser(
        description="Check whether one PDF page contains a complex table."
    )
    parser.add_argument("pdf", type=str, help="Path to PDF file")
    parser.add_argument("page", type=int, help="Page number (1-based)")
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Multimodal model id used for the check.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument(
        "--image-format",
        type=str,
        default="jpeg",
        choices=["jpeg", "png"],
        help="Image encoding format sent to the model.",
    )
    parser.add_argument(
        "--enable-check",
        action="store_true",
        help="Force-enable the check even if disabled in config.",
    )
    parser.add_argument(
        "--disable-check",
        action="store_true",
        help="Force-disable the check for this run.",
    )
    args = parser.parse_args()

    if args.enable_check and args.disable_check:
        raise ValueError("Use only one between --enable-check and --disable-check.")

    enabled = ENABLE_COMPLEX_TABLE_DETECTION
    if args.enable_check:
        enabled = True
    if args.disable_check:
        enabled = False

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    start = time.perf_counter()
    contains = page_contains_complex_table(
        pdf_path,
        page_1based=int(args.page),
        model_id=args.model_id,
        dpi=int(args.dpi),
        max_side=int(args.max_side),
        jpeg_quality=int(args.jpeg_quality),
        image_format=args.image_format,
        enabled=enabled,
    )
    elapsed = time.perf_counter() - start

    logger.info("Complex table check total elapsed time: %.2fs", elapsed)
    print(f"COMPLEX_TABLE={'YES' if contains else 'NO'}")


if __name__ == "__main__":
    main()
