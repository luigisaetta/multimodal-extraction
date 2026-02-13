from __future__ import annotations

from multimodal_extraction.ocr.docling_post_processing import cleanup_docling_text_keep_captions


def test_cleanup_preserves_fenced_code_blocks():
    text = (
        "Intro paragraph.\n"
        "AXIS\n"
        "NODE\n"
        "After noise.\n"
        "```python\n"
        "AXIS\n"
        "NODE\n"
        "print('keep this block unchanged')\n"
        "```\n"
        "Tail paragraph.\n"
    )

    cleaned = cleanup_docling_text_keep_captions(
        text,
        collapse_runs=True,
        run_min_lines=2,
    )

    assert "[FIGURE LABELS REMOVED]" in cleaned
    assert "```python" in cleaned
    assert "print('keep this block unchanged')" in cleaned
    assert "AXIS\nNODE" in cleaned  # kept inside fenced code


def test_cleanup_keeps_caption_lines():
    text = (
        "Figure 1: Pressure drop over length.\n"
        "Tabella 3\n"
        "Valid sentence with content.\n"
    )
    cleaned = cleanup_docling_text_keep_captions(text)

    assert "Figure 1: Pressure drop over length." in cleaned
    assert "Tabella 3" in cleaned
    assert "Valid sentence with content." in cleaned
