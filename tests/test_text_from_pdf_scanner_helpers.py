from __future__ import annotations

from pathlib import Path

import pytest

scanner = pytest.importorskip("multimodal_extraction.ocr.text_from_pdf_scanner")


def _cfg(**kwargs):
    base = scanner.OcrConfig(model_id="m", out_path=Path("/tmp/out.txt"))
    for k, v in kwargs.items():
        setattr(base, k, v)
    return base


def test_resolve_text_mode_auto_from_pdf_type():
    assert scanner.resolve_text_mode(_cfg(text_extraction_mode="auto", input_pdf_type="TEXT_PDF")) == "pypdf"
    assert scanner.resolve_text_mode(_cfg(text_extraction_mode="auto", input_pdf_type="SCANNED_PDF")) == "vlm"
    assert scanner.resolve_text_mode(_cfg(text_extraction_mode="auto", input_pdf_type="MIXED_OR_UNKNOWN")) == "auto"
    assert scanner.resolve_text_mode(_cfg(text_extraction_mode="auto", input_pdf_type=None)) == "auto"


def test_resolve_text_mode_explicit_override():
    assert scanner.resolve_text_mode(_cfg(text_extraction_mode="pypdf")) == "pypdf"
    assert scanner.resolve_text_mode(_cfg(text_extraction_mode="vlm")) == "vlm"


def test_page_has_enough_text_counts_non_whitespace():
    assert scanner.page_has_enough_text("abc  def", min_chars=6)
    assert not scanner.page_has_enough_text("a b", min_chars=3)
    assert not scanner.page_has_enough_text("", min_chars=1)


def test_append_figures_block_behavior():
    assert scanner.append_figures_block("base", "NONE") == "base"
    assert scanner.append_figures_block("base", "") == "base"
    out = scanner.append_figures_block("base", "- fig text")
    assert out.startswith("base")
    assert "[FIGURES]" in out
    assert "- fig text" in out


def test_format_page_block_footer():
    out = scanner.format_page_block(3, "hello")
    assert out.startswith("hello")
    assert "--- PAGE 3 ---" in out


def test_process_page_with_strategy_pypdf_no_llm():
    cfg = _cfg(describe_figures=False)
    out = scanner.process_page_with_strategy(
        effective_mode="pypdf",
        page_img=None,
        candidate_text="from pypdf",
        llm=None,
        cfg=cfg,
        need_llm=False,
    )
    assert out == "from pypdf"


def test_process_page_with_strategy_auto_fallback_to_llm(monkeypatch):
    cfg = _cfg(describe_figures=False, min_text_chars_page=100)
    monkeypatch.setattr(scanner, "is_blank_page", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        scanner,
        "call_multimodal_llm_text_only",
        lambda *args, **kwargs: "OCR RESULT",
    )

    out = scanner.process_page_with_strategy(
        effective_mode="auto",
        page_img=object(),
        candidate_text="tiny",
        llm=object(),
        cfg=cfg,
        need_llm=True,
    )
    assert out == "OCR RESULT"


def test_process_page_with_strategy_blank_page_short_circuit(monkeypatch):
    cfg = _cfg(describe_figures=False, blank_placeholder="[SKIP]")
    monkeypatch.setattr(scanner, "is_blank_page", lambda *args, **kwargs: True)

    out = scanner.process_page_with_strategy(
        effective_mode="auto",
        page_img=object(),
        candidate_text="whatever",
        llm=object(),
        cfg=cfg,
        need_llm=True,
    )
    assert out == "[SKIP]"


def test_process_page_with_strategy_adds_figures_block(monkeypatch):
    cfg = _cfg(describe_figures=True)
    monkeypatch.setattr(scanner, "is_blank_page", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        scanner,
        "call_multimodal_llm_figures_only",
        lambda *args, **kwargs: "- fig details",
    )

    out = scanner.process_page_with_strategy(
        effective_mode="pypdf",
        page_img=object(),
        candidate_text="text body",
        llm=object(),
        cfg=cfg,
        need_llm=True,
    )
    assert "text body" in out
    assert "[FIGURES]" in out
    assert "- fig details" in out


def test_process_page_with_strategy_vlm_requires_llm_and_image():
    cfg = _cfg(describe_figures=False)
    with pytest.raises(RuntimeError):
        scanner.process_page_with_strategy(
            effective_mode="vlm",
            page_img=None,
            candidate_text="",
            llm=None,
            cfg=cfg,
            need_llm=True,
        )


def test_resolve_selected_pages_default_full_document():
    cfg = _cfg()
    assert scanner.resolve_selected_pages(cfg, total_pages=4) == [1, 2, 3, 4]


def test_resolve_selected_pages_single_page():
    cfg = _cfg(start_page=3, end_page=3)
    assert scanner.resolve_selected_pages(cfg, total_pages=8) == [3]


def test_resolve_selected_pages_range_and_max_pages():
    cfg = _cfg(start_page=2, end_page=7, max_pages=3)
    assert scanner.resolve_selected_pages(cfg, total_pages=10) == [2, 3, 4]


def test_resolve_selected_pages_invalid_window():
    cfg = _cfg(start_page=5, end_page=2)
    with pytest.raises(ValueError):
        scanner.resolve_selected_pages(cfg, total_pages=10)


def test_process_page_with_retries_succeeds_after_retry(monkeypatch):
    cfg = _cfg(page_max_retries=2, page_retry_backoff_sec=0.0, continue_on_page_error=True)
    state = {"n": 0}

    def flaky(*args, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("temporary")
        return "ok"

    monkeypatch.setattr(scanner, "process_page_with_strategy", flaky)
    out = scanner.process_page_with_retries(
        source_page=7,
        effective_mode="pypdf",
        page_img=None,
        candidate_text="x",
        llm=None,
        cfg=cfg,
        need_llm=False,
    )
    assert out == "ok"
    assert state["n"] == 2


def test_process_page_with_retries_continue_on_error(monkeypatch):
    cfg = _cfg(page_max_retries=1, page_retry_backoff_sec=0.0, continue_on_page_error=True)
    monkeypatch.setattr(
        scanner, "process_page_with_strategy", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    out = scanner.process_page_with_retries(
        source_page=2,
        effective_mode="pypdf",
        page_img=None,
        candidate_text="x",
        llm=None,
        cfg=cfg,
        need_llm=False,
    )
    assert out == cfg.page_error_placeholder


def test_process_page_with_retries_fail_fast(monkeypatch):
    cfg = _cfg(page_max_retries=1, page_retry_backoff_sec=0.0, continue_on_page_error=False)
    monkeypatch.setattr(
        scanner, "process_page_with_strategy", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    with pytest.raises(RuntimeError):
        scanner.process_page_with_retries(
            source_page=3,
            effective_mode="pypdf",
            page_img=None,
            candidate_text="x",
            llm=None,
            cfg=cfg,
            need_llm=False,
        )
