from __future__ import annotations

from multimodal_extraction.chunking.ocr_output_chunking_utils import (
    chunk_text,
    extract_ocr_body,
    normalize_spaces_keep_newlines,
    parse_pages_from_ocr_output,
)


def test_normalize_spaces_keep_newlines():
    text = "a   b\t\tc\nline   2"
    out = normalize_spaces_keep_newlines(text)
    assert out == "a b c\nline 2"


def test_extract_ocr_body_without_markers_returns_original():
    text = "No markers here\nA\nB"
    assert extract_ocr_body(text) == text


def test_extract_ocr_body_with_markers_returns_only_body():
    text = (
        "HEADER\n"
        "===== BEGIN TEXT =====\n"
        "line 1\n"
        "line 2\n"
        "===== END TEXT =====\n"
        "FOOTER\n"
    )
    assert extract_ocr_body(text) == "line 1\nline 2"


def test_parse_pages_from_ocr_output_handles_crlf():
    text = (
        "===== BEGIN TEXT =====\r\n\r\n"
        "P1\r\n"
        "--- PAGE 1 ---\r\n\r\n"
        "P2\r\n"
        "--- PAGE 2 ---\r\n\r\n"
        "===== END TEXT =====\r\n"
    )
    pages = parse_pages_from_ocr_output(text)
    assert pages == [(1, "P1"), (2, "P2")]


def test_chunk_text_splits_long_text():
    text = "alpha beta gamma delta epsilon zeta eta theta"
    chunks = chunk_text(text, max_chunk_size=12, overlap=2)
    assert len(chunks) >= 2
    assert all(isinstance(c, str) and c for c in chunks)
