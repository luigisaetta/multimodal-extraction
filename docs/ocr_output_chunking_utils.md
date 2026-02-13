# `ocr_output_chunking_utils.py` — Page-Level Chunking for OCR Output

## What this module provides

This module converts the **single text output** produced by the OCR pipeline into
**page-level chunks** suitable for embedding and loading into Oracle Vector Search.

Each chunk corresponds to **exactly one PDF page**, preserving a strict mapping
between the original document structure and the vector store.

---

## Public API

Used by the Streamlit app and loading pipeline:

- **`ocr_output_text_to_chunks(full_text, source_name, max_chunk_size, overlap, add_header)`**
  - Parses OCR output and returns one chunk per page.
  - Function signature is kept stable for backward compatibility.
  - `max_chunk_size` and `overlap` are currently ignored.

- **`ocr_output_file_to_chunks(ocr_output_path, source_name, ...)`**
  - Reads an OCR output file and delegates to `ocr_output_text_to_chunks`.

- **`chunks_to_langchain_documents(chunks)`**
  - Converts chunks into LangChain `Document` objects with attached metadata.

---

## Key features

- **One chunk = one page**
  - No intra-page splitting.
  - Tables, figures, and surrounding text always stay together.

- **Page-accurate provenance**
  - Each chunk includes:
    - source document name
    - page number
    - extraction type (`ocr`)
  - Enables precise citations and safe document deletion.

- **Optional chunk header**
  - When enabled, prepends a small, stable header with the source filename.
  - Makes chunks self-describing outside the application context.

- **Provider-agnostic**
  - Operates only on final extracted text.
  - Independent of OCR engine, Docling, or LLM provider.

---

## Key design decisions

- **Page-level chunking over size-based chunking**
  - Improves correctness for technical PDFs.
  - Avoids breaking tables and diagrams across chunks.
  - Simplifies retrieval and explainability.

- **Stable interface, flexible internals**
  - Public function signatures remain unchanged.
  - Allows future hybrid strategies (e.g. split only very large pages).

---

## Notes

- Very large pages may exceed embedding limits in rare cases.
- The module relies on OCR page footer markers (`--- PAGE N ---`);
  if they are missing, chunking cannot reconstruct pages.

---

**In short:** this module acts as a reliable bridge between OCR output and vector
storage, prioritizing traceability and document fidelity over aggressive chunk
optimization.
