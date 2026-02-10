"""
pdf_object_inspector.py

Inspect a PDF and report, per page:
- extracted text stats (chars, non-whitespace chars)
- presence/count of XObjects: /Image, /Form, /PS (rare), /Other
- fonts used
- basic resource keys

Goal: help diagnose "TEXT_PDF but formulas missing" cases: often formulas are embedded
as /Form XObjects or vector drawings with minimal text extraction.

Author: Luigi Saetta (script generated with ChatGPT help)
Python: 3.11+
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pypdf import PdfReader


@dataclass
class PageReport:
    page_index_1based: int
    text_chars: int
    text_non_ws_chars: int
    text_preview: str
    xobj_images: int
    xobj_forms: int
    xobj_ps: int
    xobj_other: int
    fonts_count: int
    fonts_preview: List[str]
    resource_keys: List[str]


def _safe_get_dict(obj: Any) -> Dict[str, Any]:
    """Return a plain dict from a pypdf object or {}."""
    if obj is None:
        return {}
    try:
        if hasattr(obj, "get_object"):
            obj = obj.get_object()
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _extract_text(page) -> str:
    try:
        txt = page.extract_text() or ""
        return txt
    except Exception:
        return ""


def _count_non_ws(s: str) -> int:
    return sum(1 for c in s if not c.isspace())


def _inspect_xobjects(page) -> Tuple[int, int, int, int]:
    """
    Inspect /Resources /XObject and count subtypes:
      /Image, /Form, /PS, Other
    """
    resources = _safe_get_dict(page.get("/Resources"))
    xobj = resources.get("/XObject")
    xobj_dict = _safe_get_dict(xobj)

    images = forms = ps = other = 0

    for _, obj in xobj_dict.items():
        obj_dict = _safe_get_dict(obj)
        subtype = obj_dict.get("/Subtype")
        if subtype == "/Image":
            images += 1
        elif subtype == "/Form":
            forms += 1
        elif subtype == "/PS":
            ps += 1
        elif subtype:
            other += 1
        else:
            other += 1

    return images, forms, ps, other


def _inspect_fonts(page) -> List[str]:
    """
    Return a list of font resource names present on the page (best effort).
    """
    resources = _safe_get_dict(page.get("/Resources"))
    fonts = _safe_get_dict(resources.get("/Font"))
    names: List[str] = []
    for k, v in fonts.items():
        # k is typically a NameObject like '/F1'
        font_key = str(k)
        vdict = _safe_get_dict(v)
        base = vdict.get("/BaseFont")
        subtype = vdict.get("/Subtype")
        label = font_key
        if base:
            label += f":{base}"
        if subtype:
            label += f"({subtype})"
        names.append(label)
    return names


def inspect_pdf(pdf_path: Path, max_pages: Optional[int] = None) -> List[PageReport]:
    reader = PdfReader(str(pdf_path))
    n_pages = len(reader.pages)
    if max_pages is not None:
        n_pages = min(n_pages, max_pages)

    reports: List[PageReport] = []
    for i in range(n_pages):
        page = reader.pages[i]

        txt = _extract_text(page)
        txt_stripped = txt.strip()
        preview = txt_stripped[:160].replace("\n", " ")
        if len(txt_stripped) > 160:
            preview += "…"

        images, forms, ps, other = _inspect_xobjects(page)
        fonts = _inspect_fonts(page)

        resources = _safe_get_dict(page.get("/Resources"))
        resource_keys = sorted(str(k) for k in resources.keys())

        rep = PageReport(
            page_index_1based=i + 1,
            text_chars=len(txt_stripped),
            text_non_ws_chars=_count_non_ws(txt_stripped),
            text_preview=preview,
            xobj_images=images,
            xobj_forms=forms,
            xobj_ps=ps,
            xobj_other=other,
            fonts_count=len(fonts),
            fonts_preview=fonts[:10],
            resource_keys=resource_keys,
        )
        reports.append(rep)

    return reports


def summarize(reports: List[PageReport]) -> Dict[str, Any]:
    pages = len(reports)
    total_text_chars = sum(r.text_chars for r in reports)
    pages_with_text = sum(1 for r in reports if r.text_chars > 0)
    pages_with_images = sum(1 for r in reports if r.xobj_images > 0)
    pages_with_forms = sum(1 for r in reports if r.xobj_forms > 0)

    return {
        "pages": pages,
        "total_text_chars": total_text_chars,
        "pages_with_text": pages_with_text,
        "pages_with_images": pages_with_images,
        "pages_with_forms": pages_with_forms,
        "ratio_pages_with_text": pages_with_text / max(1, pages),
        "ratio_pages_with_images": pages_with_images / max(1, pages),
        "ratio_pages_with_forms": pages_with_forms / max(1, pages),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inspect PDF objects per page (text, XObjects, fonts)."
    )
    ap.add_argument("pdf", type=str, help="Path to PDF")
    ap.add_argument("--max-pages", type=int, default=0, help="0 = all pages")
    ap.add_argument(
        "--json",
        action="store_true",
        help="Print full report as JSON (useful for tooling).",
    )
    ap.add_argument(
        "--show-pages",
        type=str,
        default="",
        help="Comma-separated list of page numbers to print (e.g. '1,2,10').",
    )

    args = ap.parse_args()
    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        raise SystemExit(f"File not found: {pdf_path}")

    max_pages = None if int(args.max_pages) == 0 else int(args.max_pages)
    reports = inspect_pdf(pdf_path, max_pages=max_pages)
    summ = summarize(reports)

    if args.json:
        payload = {"summary": summ, "pages": [asdict(r) for r in reports]}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print("=== PDF OBJECT INSPECTION ===")
    print(f"File: {pdf_path.name}")
    print(
        "Summary: "
        f"pages={summ['pages']} | "
        f"text_pages={summ['pages_with_text']} ({summ['ratio_pages_with_text']:.2f}) | "
        f"image_pages={summ['pages_with_images']} ({summ['ratio_pages_with_images']:.2f}) | "
        f"form_pages={summ['pages_with_forms']} ({summ['ratio_pages_with_forms']:.2f})"
    )
    print("")

    only_pages: Optional[set[int]] = None
    if args.show_pages.strip():
        only_pages = {int(x.strip()) for x in args.show_pages.split(",") if x.strip()}

    for r in reports:
        if only_pages is not None and r.page_index_1based not in only_pages:
            continue

        print(f"--- Page {r.page_index_1based} ---")
        print(
            f"text_chars={r.text_chars} | non_ws={r.text_non_ws_chars} | "
            f"xobj_image={r.xobj_images} | xobj_form={r.xobj_forms} | xobj_other={r.xobj_other}"
        )
        if r.text_preview:
            print(f"text_preview: {r.text_preview}")
        else:
            print("text_preview: <EMPTY>")
        print(
            f"fonts({r.fonts_count}): {', '.join(r.fonts_preview) if r.fonts_preview else '-'}"
        )
        print(f"resources: {', '.join(r.resource_keys) if r.resource_keys else '-'}")
        print("")

    print("Hint:")
    print(
        "- Many /Form XObjects with low extracted text often means formulas/diagrams are vector drawings."
    )
    print(
        "- If text is present but formulas are missing, check pages where xobj_forms > 0."
    )


if __name__ == "__main__":
    main()
