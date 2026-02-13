from __future__ import annotations

from multimodal_extraction.ocr.docling_post_processing import (
    _is_noise_label_line,
    _split_by_fenced_code_blocks,
    cleanup_docling_text_keep_captions,
)


def test_split_by_fenced_code_blocks_with_two_styles():
    text = (
        "intro\n"
        "```txt\n"
        "code1\n"
        "```\n"
        "middle\n"
        "~~~yaml\n"
        "k: v\n"
        "~~~\n"
        "tail\n"
    )
    segments = _split_by_fenced_code_blocks(text)
    assert len(segments) == 5
    assert segments[0][0] is False
    assert segments[1][0] is True
    assert segments[2][0] is False
    assert segments[3][0] is True
    assert segments[4][0] is False


def test_is_noise_label_line_basic_cases():
    assert _is_noise_label_line("AXIS")
    assert _is_noise_label_line("P1")
    assert not _is_noise_label_line("Figure 1: Pressure drop")
    assert not _is_noise_label_line("## Section")
    assert not _is_noise_label_line("| c1 | c2 |")


def test_cleanup_without_collapse_removes_noise_run_without_marker():
    text = "Start line.\nAXIS\nNODE\nVALUE\nEnd line.\n"
    out = cleanup_docling_text_keep_captions(
        text,
        collapse_runs=False,
        run_min_lines=2,
    )
    assert "[FIGURE LABELS REMOVED]" not in out
    assert "Start line." in out
    assert "End line." in out
    assert "AXIS" not in out
    assert "NODE" not in out
    assert "VALUE" not in out
