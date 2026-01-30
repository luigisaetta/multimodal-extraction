"""
Author: Luigi Saetta
Date last modified: 2026-01-14
Python Version: 3.11
License: MIT

    Prompt builders for the scanned-PDF OCR pipeline.

    Keep prompts centralized to improve visibility, versioning and maintainability.
"""

from __future__ import annotations

PROMPT_VERSION = "2026-01-28"


def build_ocr_text_prompt(extra_prompt: str = "") -> str:
    """
    Build the prompt for OCR text extraction from scanned documents.
    """
    base = (
        "You are performing OCR on a scanned technical document.\n"
        "Return ONLY the transcribed text.\n"
        "Rules:\n"
        "- Do not return JSON.\n"
        "- Do not wrap the output in Markdown fences.\n"
        "- Do not add page numbers.\n"
        "- Do not summarize.\n"
        "- Do not translate.\n"
        "- Preserve paragraphs and numbering using newlines.\n"
        "- Keep units and symbols exactly.\n"
        "- If unreadable, write [ILLEGIBLE].\n"
    )

    extra_prompt = (extra_prompt or "").strip()
    if extra_prompt:
        base += "\nAdditional instructions:\n" + extra_prompt + "\n"
    return base


def build_figures_prompt() -> str:
    """
    Build the prompt for figure/diagram description from scanned documents.
    """
    return (
        "You are analyzing a scanned technical document page.\n"
        "Your task is to identify and describe ONLY figures, diagrams, or technical drawings.\n"
        "IGNORE tables (tabular data), paragraphs, headers/footers, logos, watermarks,"
        " and decorative elements.\n"
        "\n"
        "IMPORTANT LANGUAGE RULE:\n"
        "- Write the description in the SAME LANGUAGE used in the visible text of the page.\n"
        "- If the page text is in Italian, write in Italian.\n"
        "- If the page text is in English, write in English.\n"
        "- If there is no visible text, or you are unsure, default to Italian.\n"
        "- Do NOT translate text from one language to another.\n"
        "\n"
        "DESCRIPTION GUIDELINES:\n"
        "- Describe WHAT the figure represents and its PURPOSE.\n"
        "- Mention key components, symbols, labels, or sections that are visible.\n"
        "- Describe relationships between elements (e.g. connections, flow, hierarchy) if present.\n"
        "- If the figure is a process or flow diagram, describe the main steps or directions.\n"
        "- If the figure is a schematic or technical drawing, describe the main elements"
        " and how they are arranged.\n"
        "- Do NOT invent details that are not clearly visible.\n"
        "- Do NOT repeat surrounding page text verbatim.\n"
        "\n"
        "If there are NO figures/diagrams/drawings on the page, return exactly:\n"
        "NONE\n"
        "\n"
        "OUTPUT FORMAT (STRICT):\n"
        "- (pos: top|middle|bottom) <concise but complete description (2–4 sentences)>\n"
        "- ... (one bullet per figure, if multiple figures are present)\n"
        "\n"
        "STYLE CONSTRAINTS:\n"
        "- Use precise technical language appropriate for an engineering or regulatory document.\n"
        "- Keep each description compact but informative.\n"
        "- Do NOT add any introductory or concluding text.\n"
    )
