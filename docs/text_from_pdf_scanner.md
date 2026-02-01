# `text_from_pdf_scanner.py` — Unified PDF → Text (+ Figures) Extraction Pipeline

## What this module provides

This module implements a **single entry-point pipeline** that produces one consolidated text output (`output.txt`) from a PDF, with optional **figure/diagram descriptions** appended per page.

It supports multiple input scenarios:

- **TEXT_PDF**: extract text per page (no OCR).  
  - Uses **Docling** (preferred) to export **light Markdown**, including **tables as Markdown**.
  - Falls back to **pypdf** (plain text) when needed.
- **SCANNED_PDF**: render pages to images and do **multimodal OCR** per page.
- **MIXED_OR_UNKNOWN / auto**: per-page fallback:
  - try text extraction for that page
  - if too little text, fallback to multimodal OCR for that page

### Public API used by other modules
- **`OcrConfig` (dataclass)**  
  Central config for the whole pipeline: rendering, strategy, thresholds, prompts, blank detection, and figure extraction.
- **`run_ocr_pipeline(pdf_path: Path, cfg: OcrConfig) -> str`**  
  Main pipeline: returns the full output text and writes it to `cfg.out_path`.
- Text extraction helpers:
  - **`extract_text_pages_docling(pdf_path, max_pages) -> List[str]`** (TEXT_PDF → Markdown with tables)
  - **`extract_text_pages_pypdf(pdf_path, max_pages) -> List[str]`** (TEXT_PDF → plain text)
- Multimodal helpers:
  - **`call_multimodal_llm_text_only(...) -> str`**
  - **`call_multimodal_llm_figures_only(...) -> str`**
  - **`append_figures_block(page_text, figures_text) -> str`**

---

## Key features (what other modules rely on)

- **Per-page output structure** with a consistent footer marker:  
  `--- PAGE N ---` via `format_page_block()`
- **Blank page detection** before expensive VLM calls: `is_blank_page()`
- **Optional figures pass** (second multimodal call per page), appended under `[FIGURES]`
- **Single output file** with metadata header and final totals

---

## Key design decisions

### 1) One pipeline, multiple strategies (controlled by config)
The module avoids separate codepaths spread across the app by concentrating decisions in:

- `text_extraction_mode`: `"auto" | "pypdf" | "vlm"`
- `input_pdf_type`: `"TEXT_PDF" | "SCANNED_PDF" | "MIXED_OR_UNKNOWN" | None`

This keeps the Streamlit app thin: it passes classification + mode, and the pipeline decides.

### 2) “Auto” is per-page, not per-document
For mixed PDFs, classifying the whole file as scanned can be wasteful.  
Instead, `auto` can do **page-level fallback** using `min_text_chars_page`:
- if extracted text on a page is “enough” → keep it
- otherwise → OCR that page via VLM

This reduces cost/time on hybrid documents.

### 3) Docling for TEXT_PDF to preserve tables as Markdown
When enabled, Docling exports **light Markdown**:
- tables are kept (`enable_chart_tables=True`, `compact_tables=True`)
- pagination uses a synthetic placeholder (`page_break_placeholder`)
- output is sanitized (`escape_html`, `escape_underscores`)
- image placeholders are suppressed (`image_placeholder=""`)

Optional cleanup (`cleanup_docling_text_keep_captions`) removes “label noise” while keeping meaningful captions.

### 4) Multimodal calls are “text-only” to reduce provider variance
The VLM prompt path intentionally requests **ONLY transcribed text** (no JSON),
because some providers can truncate or “helpfully” reformat outputs unless constrained.

Figures are handled as a separate call and appended, so text extraction remains stable.

---

## Output format guarantees
- Header includes source filename and main parameters (dpi, model, figures enabled)
- Each page ends with `--- PAGE N ---`
- Figures appear only if non-empty and not `"NONE"`, under `[FIGURES]`
- The pipeline returns the exact string written to disk

---

## Practical notes / limitations
- Docling may still extract “stray” text from figures depending on layout; cleanup is provided but intentionally conservative.
- VLM quality and truncation behavior depends on the selected model; this module focuses on making the calls as deterministic as possible.
