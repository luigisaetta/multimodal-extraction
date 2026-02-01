# Multimodal PDF Processing Pipeline (Scanned + Text PDFs)

This repository contains a **modular, production-oriented pipeline** for classifying, extracting, post-processing, chunking, and loading technical PDFs (both scanned and text-based) into **Oracle Vector Search**.

The system is designed to:
- handle **mixed real-world PDFs** (logos, watermarks, diagrams, tables)
- combine **multimodal LLM OCR** with **Docling-based text extraction**
- remain **observable, debuggable, and evaluable** across models and prompt versions

---

## Core modules and documentation

Each major module is documented with a **1-page design-focused Markdown file** describing:
- exposed features
- how it is used by other modules
- key design decisions

### 📄 PDF classification
- **Code:** `classify_pdf.py`  
- **Docs:** [`classify_pdf.md`](./classify_pdf.md)

Classifies PDFs using robust heuristics that combine:
- extractable text signal
- image presence
- “strong text wins” rule to avoid false mixed classifications.

---

### 📄 OCR & text extraction pipeline
- **Code:** `text_from_pdf_scanner.py`  
- **Docs:** [`text_from_pdf_scanner.md`](./text_from_pdf_scanner.md)

Unified extraction pipeline that:
- automatically routes PDFs to Docling or multimodal OCR
- supports figure description as a second pass
- produces a single, page-delimited output file.

---

### 📄 Docling post-processing
- **Code:** `docling_post_processing.py`  
- **Docs:** [`docling_post_processing.md`](./docling_post_processing.md)

Normalizes Docling output by:
- filtering low-value text extracted from figures
- cleaning markdown tables
- stabilizing text structure for chunking and retrieval.

---

### 📄 Chunking utilities
- **Code:** `ocr_output_chunking_utils.py`  
- **Docs:** [`ocr_output_chunking_utils.md`](./ocr_output_chunking_utils.md)

Transforms OCR output into:
- page-aware, overlapping chunks
- LangChain `Document` objects
ready for embedding and vector storage.

---

### 📄 Prompt management
- **Code:** `prompts.py`  
- **Docs:** [`prompts.md`](./prompts.md)

Centralized prompt builders for:
- deterministic OCR text extraction
- scoped figure/diagram understanding

Includes prompt versioning for evaluation and regression tracking.

---

### 📄 Oracle Vector Store administration
- **Code:** `oraclevs_admin.py`  
- **Docs:** [`oraclevs_admin.md`](./oraclevs_admin.md)

Administrative utilities for Oracle Vector Search:
- list collections
- list documents + chunk counts
- analyze vector dimensions/formats
- delete documents or drop collections

Used by the Streamlit inspector and loaders.

---

### 📄 Database utilities
- **Code:** `db_utils.py`  
- **Docs:** [`db_utils.md`](./db_utils.md)

Centralizes:
- DB connection handling
- health checks
- configuration masking (UI-safe)

Keeps DB logic out of UI and pipeline code.

---

## Design principles (summary)

- **Separation of concerns**
  - classification ≠ extraction ≠ chunking ≠ loading
- **Provider-agnostic**
  - OCR pipeline works across LLM vendors
- **Traceability**
  - every chunk retains source + page provenance
- **Safety by default**
  - strict SQL identifier validation
  - explicit destructive operations
- **Evaluation-friendly**
  - deterministic prompts
  - page-stable output format

---

## Intended usage

This project is well-suited for:
- RAG on technical / regulatory PDFs
- mixed scanned + digital document corpora
- enterprise vector search on Oracle Database
- multimodal LLM evaluation and benchmarking

---
