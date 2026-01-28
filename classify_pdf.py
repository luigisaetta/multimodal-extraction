"""
Classify PDFs in a directory as:
- TEXT_PDF (extractable text)
- SCANNED_PDF (image-only / scanned)
- MIXED_OR_UNKNOWN (hybrid or ambiguous)

Decision policy (robust heuristic):
- Strong text signal wins: if extracted text chars >= STRONG_TEXT_CHARS, classify as TEXT_PDF
  even if every page contains decorative images (logos/watermarks) -> avoids false MIXED.
- Otherwise combine:
  1) extracted text length (pypdf/PyPDF2)
  2) presence of image XObjects in page resources
- Minimal, non-noisy output: one line per PDF (unless you keep the "mixed -> scanned fallback",
  which intentionally prints MIXED/UNKNOWN then SCANNED_PDF for safety).

Python 3.11+
Dependencies: pypdf (recommended) or PyPDF2
  pip install pypdf
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from pypdf import PdfReader


# -----------------------------
# Placeholder processors
# -----------------------------
def process_text_pdf(pdf_path: Path) -> None:
    """
    For now empty processor that just logs the text PDF.
    """
    logging.info("TEXT_PDF     | %s", pdf_path)


def process_scanned_pdf(pdf_path: Path) -> None:
    """
    For now empty processor that just logs the scanned PDF.
    """
    logging.info("SCANNED_PDF  | %s", pdf_path)


# -----------------------------
# Classification
# -----------------------------
@dataclass
class ClassifyConfig:
    """PDF classification configuration."""

    # Scan at most N pages per PDF for speed/robustness trade-off
    sample_pages: int = 10

    # Minimum extracted text characters to consider "text present"
    min_text_chars_doc: int = 200
    min_text_chars_page: int = 50

    # If images are found on many sampled pages and text is low -> scanned
    scanned_if_image_pages_ratio_ge: float = 0.6

    # Strong text threshold: if extracted chars >= this, classify as TEXT_PDF
    # This prevents false MIXED for PDFs that include logos/watermarks on every page.
    strong_text_chars: int = 5000


def _safe_extract_text(reader: PdfReader, page_index: int) -> str:
    try:
        page = reader.pages[page_index]
        txt = page.extract_text() or ""
        # normalize whitespace a bit (keeps signal stable)
        return " ".join(txt.split())
    except Exception:
        return ""


def _page_has_images(reader: PdfReader, page_index: int) -> bool:
    """
    Check if a PDF page references image XObjects.
    NOTE: many "TEXT" PDFs include decorative images (logos/watermarks) on every page.
    This signal is useful but should not override a strong text signal.
    """
    try:
        page = reader.pages[page_index]
        resources = page.get("/Resources") or {}
        xobj = resources.get("/XObject") if hasattr(resources, "get") else None
        if not xobj:
            return False

        xobj = xobj.get_object() if hasattr(xobj, "get_object") else xobj
        if not hasattr(xobj, "items"):
            return False

        for _, obj in xobj.items():
            try:
                obj = obj.get_object() if hasattr(obj, "get_object") else obj
                subtype = obj.get("/Subtype")
                if subtype == "/Image":
                    return True
            except Exception:
                continue
        return False
    except Exception:
        return False


def classify_pdf(pdf_path: Path, cfg: ClassifyConfig) -> Tuple[str, Optional[str]]:
    """
    Returns: (label, reason)
      label in {"TEXT_PDF", "SCANNED_PDF", "MIXED_OR_UNKNOWN"}
    """
    try:
        reader = PdfReader(str(pdf_path))
        n_pages = len(reader.pages)
        if n_pages == 0:
            return "MIXED_OR_UNKNOWN", "zero_pages"

        sample_n = min(cfg.sample_pages, n_pages)

        # sample uniformly to reduce bias
        stride = max(1, n_pages // sample_n)
        sampled_indices = list(range(0, n_pages, stride))[:sample_n]

        total_text_chars = 0
        text_pages = 0
        image_pages = 0

        for i in sampled_indices:
            txt = _safe_extract_text(reader, i)
            tlen = len(txt)
            total_text_chars += tlen
            if tlen >= cfg.min_text_chars_page:
                text_pages += 1

            if _page_has_images(reader, i):
                image_pages += 1

        image_ratio = image_pages / max(1, len(sampled_indices))

        # --- KEY FIX: strong text signal wins ---
        # Avoid misclassifying "text PDFs with logos/watermarks on every page" as MIXED.
        if total_text_chars >= cfg.strong_text_chars:
            return (
                "TEXT_PDF",
                f"text_chars={total_text_chars}, text_pages={text_pages}, image_ratio={image_ratio:.2f}",
            )

        has_text = total_text_chars >= cfg.min_text_chars_doc or text_pages > 0
        has_images = image_pages > 0

        if has_text and (
            not has_images or image_ratio < cfg.scanned_if_image_pages_ratio_ge
        ):
            return (
                "TEXT_PDF",
                f"text_chars={total_text_chars}, text_pages={text_pages}, image_ratio={image_ratio:.2f}",
            )

        if (
            (not has_text)
            and has_images
            and image_ratio >= cfg.scanned_if_image_pages_ratio_ge
        ):
            return (
                "SCANNED_PDF",
                f"text_chars={total_text_chars}, text_pages={text_pages}, image_ratio={image_ratio:.2f}",
            )

        return (
            "MIXED_OR_UNKNOWN",
            f"text_chars={total_text_chars}, text_pages={text_pages}, image_ratio={image_ratio:.2f}",
        )

    except Exception as e:
        return "MIXED_OR_UNKNOWN", f"read_error={type(e).__name__}"


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    """Main"""
    ap = argparse.ArgumentParser(
        description="Classify PDFs in a directory: text vs scanned"
    )
    ap.add_argument("directory", type=str, help="Directory containing PDF files")
    ap.add_argument("--recursive", action="store_true", help="Scan subdirectories")
    ap.add_argument(
        "--sample-pages", type=int, default=10, help="Max pages to sample per PDF"
    )
    ap.add_argument(
        "--min-text-doc",
        type=int,
        default=200,
        help="Min extracted chars to classify as TEXT_PDF",
    )
    ap.add_argument(
        "--min-text-page",
        type=int,
        default=50,
        help="Min extracted chars on a page to count as text page",
    )
    ap.add_argument(
        "--scanned-image-ratio",
        type=float,
        default=0.6,
        help="If >= this ratio of sampled pages have images and text is low -> SCANNED_PDF",
    )
    ap.add_argument(
        "--strong-text-chars",
        type=int,
        default=5000,
        help="If extracted text chars >= this -> TEXT_PDF (even if image_ratio is high)",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Include reason in logs (still one line per file)",
    )
    ap.add_argument(
        "--mixed-as-scanned",
        action="store_true",
        help="Treat MIXED/UNKNOWN as SCANNED_PDF (safer, but may OCR true text PDFs).",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = Path(args.directory).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    cfg = ClassifyConfig(
        sample_pages=args.sample_pages,
        min_text_chars_doc=args.min_text_doc,
        min_text_chars_page=args.min_text_page,
        scanned_if_image_pages_ratio_ge=args.scanned_image_ratio,
        strong_text_chars=args.strong_text_chars,
    )

    pattern = "**/*.pdf" if args.recursive else "*.pdf"
    pdf_files = sorted(root.glob(pattern))

    if not pdf_files:
        logging.info("No PDF files found in: %s", root)
        return

    for pdf_path in pdf_files:
        label, reason = classify_pdf(pdf_path, cfg)

        if args.verbose and reason:
            logging.info("%-12s| %s | %s", label, pdf_path, reason)
        else:
            # minimal one-line output via processor functions
            if label == "TEXT_PDF":
                process_text_pdf(pdf_path)
            elif label == "SCANNED_PDF":
                process_scanned_pdf(pdf_path)
            else:
                logging.info("MIXED/UNKNOWN| %s", pdf_path)
                if args.mixed_as_scanned:
                    process_scanned_pdf(pdf_path)


if __name__ == "__main__":
    main()
