# `prompts.py` — Central Prompt Builders for OCR + Figure Understanding

## What this module provides

This module centralizes **all prompt text** used by the scanned-PDF extraction pipeline, so the rest of the code can stay “prompt-agnostic”.

### Public API used by other modules
- **`PROMPT_VERSION`**
  - A simple string version tag (useful for logging, regression tracking, and reproducibility).

- **`build_ocr_text_prompt(extra_prompt: str = "") -> str`**
  - Returns the prompt used for **multimodal OCR text extraction**.
  - Supports an optional `extra_prompt` appended as “Additional instructions”.

- **`build_figures_prompt() -> str`**
  - Returns the prompt used for **figure/diagram description** (second pass per page).

---

## Key features

### 1) Deterministic OCR output constraints
`build_ocr_text_prompt()` enforces cross-provider stability by requiring:
- **no JSON**
- **no markdown fences**
- **no page numbers**
- **no summarization**
- **no translation**
- preserve paragraph structure and technical symbols
- explicit `[ILLEGIBLE]` for unreadable spans

This reduces “helpful” behavior (summaries/reformatting) that breaks evaluation and chunking.

### 2) Controlled extensibility via `extra_prompt`
The optional `extra_prompt` lets the UI add constraints without forking code:
- you can inject domain-specific rules (e.g., “keep headings”, “preserve hyphenation”)
- the base policy remains consistent and versionable

### 3) Figure prompt is intentionally scoped and language-aware
`build_figures_prompt()` explicitly:
- targets **ONLY** figures/diagrams/technical drawings
- **ignores tables** and decorative elements
- enforces **same-language output** (Italian/English), defaulting to Italian if uncertain
- forbids hallucination (“do not invent details”)
- uses a strict output style: one bullet per figure with an approximate position tag

---

## Key design decisions

### 1) Keep prompts in one place (versionable)
Moving prompts out of the pipeline code makes it easier to:
- iterate quickly
- compare prompt changes across model evaluations
- keep reproducibility (prompt version + model id)

### 2) Separate “text” and “figures” prompts (two-pass stability)
Keeping OCR and figure understanding separate avoids:
- OCR text being contaminated by long figure descriptions
- models mixing tasks (a common source of truncation or summarization)

### 3) Prefer “simple text” outputs over structured JSON
The prompts avoid JSON to maximize compatibility across providers and SDKs
(and to prevent schema drift / formatting failures during long documents).

---

## Practical notes
- If you later add **table extraction**, it should be a third prompt builder here (e.g., `build_tables_prompt()`), so the pipeline remains modular.
- If you run evaluations, log `PROMPT_VERSION` alongside model + parameters: it’s the fastest way to explain regressions.
