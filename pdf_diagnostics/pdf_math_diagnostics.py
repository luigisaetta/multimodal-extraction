#!/usr/bin/env python3
"""
pdf_math_diagnostics_v3.py

Diagnose why math/formulas may be missing from text extraction:
- inspects fonts (/ToUnicode etc.)
- inspects page content operators (text vs graphics)
- recursively inspects XObjects (/Form) and annotation appearance streams

Usage:
  python pdf_math_diagnostics_v3.py /path/to/file.pdf
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pypdf import PdfReader
from pypdf.generic import ContentStream, DictionaryObject

# --- Operators sets (rough classification) ---
TEXT_OPS = {
    b"BT",
    b"ET",
    b"Tj",
    b"TJ",
    b"'",
    b'"',
    b"Td",
    b"TD",
    b"Tm",
    b"Ts",
    b"Tf",
    b"Tr",
    b"Tw",
    b"Tz",
    b"TL",
}

GRAPHICS_OPS = {
    # path construction / painting
    b"m",
    b"l",
    b"c",
    b"v",
    b"y",
    b"h",
    b"re",
    b"S",
    b"s",
    b"f",
    b"F",
    b"f*",
    b"B",
    b"B*",
    b"b",
    b"b*",
    b"n",
    # graphics state / transforms
    b"q",
    b"Q",
    b"cm",
    b"w",
    b"J",
    b"j",
    b"M",
    b"d",
    b"ri",
    b"i",
    b"gs",
    b"g",
    b"G",
    b"rg",
    b"RG",
    b"k",
    b"K",
}


@dataclass
class OpsCount:
    ops_total: int = 0
    ops_text: int = 0
    ops_gfx: int = 0


def _get_obj(x: Any) -> Any:
    return x.get_object() if hasattr(x, "get_object") else x


def _get_resources(page) -> DictionaryObject:
    res = page.get("/Resources") or {}
    res = _get_obj(res)
    return res


def _content_stream_from_any(raw: Any, reader: PdfReader) -> Optional[ContentStream]:
    """
    Build a ContentStream from:
    - a StreamObject
    - an ArrayObject of streams
    - already-parsed objects

    Returns None if it cannot be parsed.
    """
    if raw is None:
        return None
    try:
        return ContentStream(raw, reader)
    except Exception:
        return None


def _count_ops_from_content(cs: Optional[ContentStream]) -> OpsCount:
    out = OpsCount()
    if cs is None:
        return out

    try:
        ops = getattr(cs, "operations", None)
        if not ops:
            return out

        out.ops_total = len(ops)
        for operands, operator in ops:
            if operator in TEXT_OPS:
                out.ops_text += 1
            if operator in GRAPHICS_OPS:
                out.ops_gfx += 1
        return out
    except Exception:
        return out


def _xobject_counts(resources: DictionaryObject) -> Tuple[int, int, int]:
    xobj = resources.get("/XObject") if hasattr(resources, "get") else None
    if not xobj:
        return 0, 0, 0

    xobj = _get_obj(xobj)
    if not hasattr(xobj, "items"):
        return 0, 0, 0

    img = form = other = 0
    for _, obj in xobj.items():
        try:
            obj = _get_obj(obj)
            subtype = obj.get("/Subtype")
            if subtype == "/Image":
                img += 1
            elif subtype == "/Form":
                form += 1
            else:
                other += 1
        except Exception:
            other += 1
    return img, form, other


def _font_report(resources: DictionaryObject) -> Dict[str, Dict[str, Any]]:
    fonts = resources.get("/Font") if hasattr(resources, "get") else None
    if not fonts:
        return {}

    fonts = _get_obj(fonts)
    if not hasattr(fonts, "items"):
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for name, f in fonts.items():
        try:
            f = _get_obj(f)
            subtype = f.get("/Subtype")
            basefont = f.get("/BaseFont")
            has_tounicode = "/ToUnicode" in f

            cid_info = None
            if subtype == "/Type0":
                desc = f.get("/DescendantFonts")
                if desc and hasattr(desc, "__len__") and len(desc) > 0:
                    d0 = _get_obj(desc[0])
                    cid_info = {
                        "cid_subtype": d0.get("/Subtype"),
                        "cid_basefont": d0.get("/BaseFont"),
                    }

            out[str(name)] = {
                "subtype": subtype,
                "basefont": str(basefont) if basefont else None,
                "has_tounicode": bool(has_tounicode),
                "cid": cid_info,
            }
        except Exception as exc:
            out[str(name)] = {"error": type(exc).__name__}
    return out


def _iter_xobject_forms(resources: DictionaryObject) -> List[Any]:
    """
    Return list of XObject Form streams found at this resources level (shallow).
    """
    forms: List[Any] = []
    xobj = resources.get("/XObject") if hasattr(resources, "get") else None
    if not xobj:
        return forms

    xobj = _get_obj(xobj)
    if not hasattr(xobj, "items"):
        return forms

    for _, obj in xobj.items():
        try:
            obj = _get_obj(obj)
            if obj.get("/Subtype") == "/Form":
                forms.append(obj)
        except Exception:
            continue
    return forms


def _count_forms_recursive(
    resources: DictionaryObject,
    reader: PdfReader,
    depth: int = 0,
    max_depth: int = 6,
) -> Tuple[int, OpsCount]:
    """
    Recursively visit XObject /Form streams, sum operators.
    Returns: (n_forms, OpsCount)
    """
    if depth > max_depth:
        return 0, OpsCount()

    total_forms = 0
    total_ops = OpsCount()

    for form in _iter_xobject_forms(resources):
        total_forms += 1

        # form content
        form_cs = _content_stream_from_any(form.get("/Contents"), reader)
        oc = _count_ops_from_content(form_cs)
        total_ops.ops_total += oc.ops_total
        total_ops.ops_text += oc.ops_text
        total_ops.ops_gfx += oc.ops_gfx

        # nested resources
        nested_res = form.get("/Resources") or {}
        nested_res = _get_obj(nested_res)
        if isinstance(nested_res, DictionaryObject):
            nf, no = _count_forms_recursive(nested_res, reader, depth + 1, max_depth)
            total_forms += nf
            total_ops.ops_total += no.ops_total
            total_ops.ops_text += no.ops_text
            total_ops.ops_gfx += no.ops_gfx

    return total_forms, total_ops


def _count_annot_appearances(page, reader: PdfReader) -> Tuple[int, int, OpsCount]:
    """
    Check annotation appearance streams (if any).
    Returns: (n_annots, n_annots_with_ap, ops_count_in_ap)
    """
    annots = page.get("/Annots")
    if not annots:
        return 0, 0, OpsCount()

    annots = _get_obj(annots)
    if not hasattr(annots, "__iter__"):
        return 0, 0, OpsCount()

    n_annots = 0
    n_with_ap = 0
    ap_ops = OpsCount()

    for a in annots:
        n_annots += 1
        try:
            a = _get_obj(a)
            ap = a.get("/AP")
            if not ap:
                continue
            ap = _get_obj(ap)
            n_with_ap += 1

            # /N is the normal appearance, can be stream or dict of states
            n = ap.get("/N")
            if not n:
                continue
            n = _get_obj(n)

            # If dict of states, just inspect all streams inside
            if isinstance(n, DictionaryObject) and hasattr(n, "items"):
                for _, st in n.items():
                    st = _get_obj(st)
                    cs = _content_stream_from_any(st.get("/Contents"), reader)
                    oc = _count_ops_from_content(cs)
                    ap_ops.ops_total += oc.ops_total
                    ap_ops.ops_text += oc.ops_text
                    ap_ops.ops_gfx += oc.ops_gfx
            else:
                # stream-like
                cs = _content_stream_from_any(n.get("/Contents"), reader)
                oc = _count_ops_from_content(cs)
                ap_ops.ops_total += oc.ops_total
                ap_ops.ops_text += oc.ops_text
                ap_ops.ops_gfx += oc.ops_gfx

        except Exception:
            continue

    return n_annots, n_with_ap, ap_ops


def _safe_preview(text: str, limit: int = 160) -> str:
    s = " ".join((text or "").split())
    if len(s) <= limit:
        return s
    return s[:limit] + "…"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python pdf_math_diagnostics_v3.py /path/to/file.pdf")

    pdf_path = Path(sys.argv[1]).expanduser().resolve()
    reader = PdfReader(str(pdf_path))

    print("=== PDF MATH DIAGNOSTICS (v3) ===")
    print(f"File: {pdf_path.name}")
    print(f"Pages: {len(reader.pages)}")
    print("")

    for i, page in enumerate(reader.pages, start=1):
        resources = _get_resources(page)

        # Text extracted (what users typically see)
        extracted = page.extract_text() or ""
        non_ws = sum(1 for c in extracted if not c.isspace())

        # --- Page content operators (FIX: don't use truthiness) ---
        content = page.get_contents()
        if content is None:
            # Fallback: raw /Contents
            content = page.get("/Contents")

        page_cs = _content_stream_from_any(content, reader)
        page_ops = _count_ops_from_content(page_cs)

        # --- XObjects ---
        x_img, x_form, x_other = _xobject_counts(resources)
        n_forms_rec, form_ops = _count_forms_recursive(resources, reader)

        # --- Annotation appearances ---
        n_ann, n_ann_ap, ap_ops = _count_annot_appearances(page, reader)

        # --- Fonts ---
        fonts = _font_report(resources)
        fonts_total = len(fonts)
        fonts_no_tounicode = sum(
            1
            for v in fonts.values()
            if isinstance(v, dict) and v.get("has_tounicode") is False
        )

        print(f"--- Page {i} ---")
        print(
            f"text_chars={len(extracted)} | non_ws={non_ws} | "
            f"page_ops={page_ops.ops_total} | page_text_ops={page_ops.ops_text} | "
            f"page_gfx_ops={page_ops.ops_gfx}"
        )
        print(
            f"xobj(shallow): image={x_img} form={x_form} other={x_other} | "
            f"xobj(rec): forms={n_forms_rec} | form_ops={form_ops.ops_total} | "
            f"form_text_ops={form_ops.ops_text} | form_gfx_ops={form_ops.ops_gfx}"
        )
        print(
            f"annots={n_ann} | annots_with_ap={n_ann_ap} | "
            f"ap_ops={ap_ops.ops_total} | ap_text_ops={ap_ops.ops_text} | "
            f"ap_gfx_ops={ap_ops.ops_gfx}"
        )
        print(f"fonts={fonts_total} | fonts_missing_ToUnicode={fonts_no_tounicode}")

        if fonts:
            for k, v in list(fonts.items())[:12]:
                if "error" in v:
                    print(f"  font {k}: ERROR={v['error']}")
                    continue
                bf = v.get("basefont")
                st = v.get("subtype")
                tu = "ToUnicode=Y" if v.get("has_tounicode") else "ToUnicode=N"
                print(f"  font {k}: {st} {bf} {tu}")

        print(f"text_preview: {_safe_preview(extracted)}")
        print("")


if __name__ == "__main__":
    main()
