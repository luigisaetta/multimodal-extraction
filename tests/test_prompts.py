from __future__ import annotations

from multimodal_extraction.prompts.prompts import build_figures_prompt, build_ocr_text_prompt


def test_build_ocr_text_prompt_appends_extra_instructions():
    p = build_ocr_text_prompt(extra_prompt="Use strict OCR for formulas.")
    assert "Return ONLY the transcribed text." in p
    assert "Do not return JSON." in p
    assert "Additional instructions:" in p
    assert "Use strict OCR for formulas." in p


def test_build_figures_prompt_has_none_contract():
    p = build_figures_prompt()
    assert "return exactly:\nNONE" in p
    assert "Generate always the description in Italian language." in p
