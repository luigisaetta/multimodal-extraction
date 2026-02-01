# `ocr_output_chunking_utils.py` — Chunking OCR/Text Output for Vector Loading

## What this module provides

This module converts the **single-file OCR output** produced by your pipeline into:
1) structured **chunks** (with optional headers and metadata), and  
2) **LangChain `Document` objects** ready to be embedded and loaded into Oracle Vector Store.

It is designed around the output format used by the pipeline, especially the page delimiters.

### Public API used by other modules
- **`ocr_output_text_to_chunks(full_text: str, source_name: str, max_chunk_size: int, overlap: int, add_header: bool) -> list[dict]`**
  - Main function: splits `full_text` into chunks.
  - Preserves a link to the original document (`source_name`) and page context.
  - Adds overlap to reduce boundary loss for retrieval.

- **`chunks_to_langchain_documents(chunks: list[dict]) -> list[Document]`**
  - Converts chunks into `langchain_core.documents.Document`.
  - Stores text in `page_content` and metadata (source/page/chunk id etc.) in `metadata`.

---

## Key features

- **Page-aware segmentation**
  - The chunker is aware of per-page markers (e.g. `--- PAGE N ---`).
  - This allows metadata such as:
    - `page_start`, `page_end` (or similar)
    - source file name
  - The result is more traceable: retrieval can cite the PDF and relevant page(s).

- **Configurable chunk size + overlap**
  - `max_chunk_size` controls the maximum chunk length (in characters).
  - `overlap` repeats trailing content into the next chunk to improve recall for queries that hit boundaries.

- **Optional chunk headers**
  - When `add_header=True`, the chunk text is prefixed with a small header containing provenance:
    - source name
    - page range
  - This makes retrieved chunks self-describing even outside your application context.

- **Stable structure for DB loading**
  - The chunk dictionary structure is intentionally simple and consistent so:
    - downstream loaders don’t need to parse OCR format again
    - it’s easy to test and debug

---

## Key design decisions

### 1) Chunking is done on the **final pipeline output**, not raw PDF
The module assumes the extraction pipeline already performed:
- OCR / text extraction
- optional figure description injection
- page delimiting

This keeps the chunker provider-agnostic (it doesn’t care if the text came from Docling, VLM OCR, etc.).

### 2) Prefer *character-based* chunking for predictability
Token-based chunking can vary by model and tokenizer. This module uses chars to be:
- deterministic across environments
- easy to reason about and debug

(You can later add token-aware chunking if you need tighter embedding limits.)

### 3) Preserve provenance as metadata, not “out-of-band”
Instead of relying on external mappings, provenance is attached directly to each chunk:
- enables traceability in the vector store
- supports UI display (“this answer came from PDF X, page Y”)
- enables selective deletion by `METADATA.source` downstream

---

## Practical notes / limitations
- If the OCR output does not contain the expected page delimiters, page-level metadata may degrade (chunks may fall back to “unknown page”).
- Overlap increases storage and embedding cost; keep it small and measure retrieval impact.
- If you use Markdown rendering in the output (Docling), chunking still works, but extremely large tables may produce long contiguous sections; consider table-aware splitting if that becomes a bottleneck.
