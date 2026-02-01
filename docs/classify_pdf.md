# `classify_pdf.py` — PDF Type Classifier (TEXT vs SCANNED)

## What this module provides

This module classifies PDFs into three categories:

- **`TEXT_PDF`**: text is extractable (suitable for text-based extraction pipelines like Docling / PyPDF).
- **`SCANNED_PDF`**: pages are primarily images (requires OCR / multimodal pipeline).
- **`MIXED_OR_UNKNOWN`**: ambiguous or hybrid cases.

### Public API used by other modules
- **`ClassifyConfig` (dataclass)**  
  Configuration for the heuristic classifier (sampling, thresholds, scanned-image ratio, strong text override).
- **`classify_pdf(pdf_path: Path, cfg: ClassifyConfig) -> tuple[str, Optional[str]]`**  
  Core classifier returning `(label, reason)` where `reason` is a compact diagnostic string.
- **CLI (`main()`)**  
  Batch classification of PDFs in a directory with optional recursion and verbose logging.

### Optional/placeholder hooks
- **`process_text_pdf(pdf_path: Path)`**, **`process_scanned_pdf(pdf_path: Path)`**  
  Currently just log the detected type; intended extension points for downstream processing.

---

## Key design decisions

### 1) “Strong text wins” to avoid false MIXED
Many “true text PDFs” contain a logo/watermark on every page, which would otherwise look “image-heavy”.  
To prevent misclassification, the classifier enforces:

- If `total_text_chars >= strong_text_chars` ⇒ classify as **`TEXT_PDF`**, even if images are present.

This is the most important rule to keep the classifier stable on real-world enterprise PDFs.

### 2) Sampling instead of scanning all pages
For speed and robustness, only up to `sample_pages` pages are analyzed.  
Pages are sampled uniformly with a `stride` so the sample is not biased toward the beginning of a document.

This keeps classification cheap even for large PDFs.

### 3) Two independent signals: extracted text + image XObjects
The classifier uses two signals:
- **Text signal** from `pypdf` page `.extract_text()` (normalized whitespace).
- **Image signal** by scanning page `/Resources` → `/XObject` and detecting `Subtype == /Image`.

It then combines them via thresholds:
- document-level minimum text (`min_text_chars_doc`)
- per-page minimum text (`min_text_chars_page`)
- scanned decision if `image_pages_ratio >= scanned_if_image_pages_ratio_ge` and text is low

### 4) Minimal output by default, explainable output when needed
By default, the CLI prints one line per file (low-noise).
With `--verbose`, it also prints a `reason` string containing:
- `text_chars`
- `text_pages`
- `image_ratio`

This supports debugging/tuning thresholds without flooding logs.

---

## Known limitations / edge cases (practical)
- If a PDF is encrypted, malformed, or `pypdf` fails to parse, the result becomes **`MIXED_OR_UNKNOWN`**.
- Some PDFs have embedded text but extraction yields little (fonts/encoding issues); these may become `SCANNED_PDF` or `MIXED_OR_UNKNOWN`.
- “Image presence” is a helpful signal but not perfect: decorative images and watermarks are common (hence the strong-text override).

---

## Typical usage
- **Library usage**: call `classify_pdf(path, ClassifyConfig(...))` and route to either a text extractor (Docling) or OCR pipeline (VLM).
- **CLI usage**: run on a folder to build an inventory of scanned vs text PDFs before choosing the extraction strategy.
