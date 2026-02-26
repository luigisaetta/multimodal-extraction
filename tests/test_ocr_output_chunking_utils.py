from __future__ import annotations

from pathlib import Path

from multimodal_extraction.chunking.ocr_output_chunking_utils import (
    _infer_source_name_from_ocr_header,
    chunks_to_langchain_documents,
    ocr_output_file_to_chunks,
    ocr_output_text_to_chunks,
    parse_pages_from_ocr_output,
)


def _sample_ocr_output() -> str:
    return (
        "SOURCE PDF: /tmp/docs/sample.pdf\n"
        "==================== BEGIN TEXT ====================\n\n"
        "First page text.\n"
        "--- PAGE 1 ---\n\n"
        "Second page text.\n"
        "--- PAGE 2 ---\n\n"
        "===================== END TEXT =====================\n"
        "TOTAL PAGES: 2\n"
    )


def test_parse_pages_from_ocr_output():
    pages = parse_pages_from_ocr_output(_sample_ocr_output())
    assert pages == [(1, "First page text."), (2, "Second page text.")]


def test_ocr_output_text_to_chunks_with_header():
    text = (
        "==================== BEGIN TEXT ====================\n\n"
        "Page one.\n"
        "--- PAGE 1 ---\n\n"
        "\n"
        "--- PAGE 2 ---\n\n"  # empty page should be skipped
        "Page three.\n"
        "--- PAGE 3 ---\n\n"
        "===================== END TEXT =====================\n"
    )
    chunks = ocr_output_text_to_chunks(text, source_name="a.pdf", add_header=True)

    assert len(chunks) == 2
    assert chunks[0].page_label == "1"
    assert chunks[1].page_label == "3"
    assert chunks[0].metadata["source"] == "a.pdf"
    assert chunks[0].text.startswith("---\nsource_file: a.pdf\n---\n\n")

    docs = chunks_to_langchain_documents(chunks)
    assert len(docs) == 2
    assert docs[0].metadata["page_label"] == "1"


def test_infer_source_name_from_header():
    inferred = _infer_source_name_from_ocr_header(_sample_ocr_output())
    assert inferred == "sample.pdf"


def test_ocr_output_file_to_chunks_fallback_to_output_filename(tmp_path: Path):
    p = tmp_path / "output.txt"
    p.write_text(
        "==================== BEGIN TEXT ====================\n\n"
        "Only page.\n"
        "--- PAGE 1 ---\n\n"
        "===================== END TEXT =====================\n",
        encoding="utf-8",
    )
    chunks = ocr_output_file_to_chunks(p, source_name=None, add_header=False)
    assert len(chunks) == 1
    assert chunks[0].source_name == "output.txt"


def test_ocr_output_text_to_chunks_size_based_mode_splits_page_text():
    text = (
        "==================== BEGIN TEXT ====================\n\n"
        "alpha beta gamma delta epsilon zeta eta theta iota kappa\n"
        "--- PAGE 1 ---\n\n"
        "===================== END TEXT =====================\n"
    )
    chunks = ocr_output_text_to_chunks(
        text,
        source_name="b.pdf",
        max_chunk_size=18,
        overlap=4,
        chunk_by_page=False,
        add_header=False,
    )

    assert len(chunks) >= 2
    assert all(ch.page_label == "1" for ch in chunks)
    assert all(ch.metadata["source"] == "b.pdf" for ch in chunks)
