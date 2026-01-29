# Multimodal PDF Ingestion for Oracle Vector Search

A Streamlit-based application and supporting utilities to ingest **technical PDFs** into **Oracle Vector Search (Oracle DB 23ai / 26ai)**, handling both **text-based** and **fully scanned** documents.

The project is designed for **RAG pipelines on complex technical documentation** (regulations, standards, engineering docs), where **recall, provenance, and multimodal understanding** matter more than generic OCR.

---

## What problem this solves (briefly)

Many real-world technical PDFs are:
- partially or fully scanned
- rich in **figures, diagrams, maps**
- written in **Italian or English**
- unsuitable for classic OCR-only pipelines

This project provides a **single ingestion pipeline** that:
- classifies PDFs reliably
- extracts text with page fidelity
- leverages **multimodal LLMs** to understand images
- chunks content with provenance
- loads everything into **Oracle Vector Search** for downstream RAG agents

---

## Main features

- **PDF classification**
  - Automatically detects `TEXT_PDF`, `SCANNED_PDF`, or mixed cases
  - Robust against logos, watermarks, and low-text pages

- **Multimodal OCR for scanned PDFs**
  - Page-by-page image rendering
  - Multimodal LLM-based text extraction
  - No forced summarization or formatting
  - Preserves original wording as much as possible

- **Technical figure & diagram understanding**
  - Optional second multimodal pass per page
  - Extracts *useful descriptions* of figures, diagrams, drawings, and maps
  - Explicitly avoids hallucinated interpretations and ignores tables
  - Appends structured figure descriptions to page content

- **Page-level provenance**
  - Every extracted block is traceable to its original PDF page
  - Page boundaries are preserved throughout the pipeline

- **Chunking optimized for RAG**
  - Per-page chunking strategy
  - Stable metadata (`source`, `page_label`, `extraction_type`)
  - Designed to maximize recall on legal / technical questions

- **Oracle Vector Search ingestion**
  - Native integration with Oracle Database Vector Search
  - Safe loading, listing, analysis, and cleanup utilities
  - Designed for production-scale document collections

- **Streamlit UI**
  - Upload and preview PDFs
  - Inspect classification results
  - Run OCR and image understanding
  - Chunk and load data into the vector store
  - Inspect intermediate outputs for debugging and validation

---

## Design philosophy

- **Accuracy over speed**
- **Recall over elegance**
- **No silent summarization**
- **Explicit provenance**
- **Multimodal by design, not as an afterthought**

This is intentionally *not* a generic OCR demo, but a foundation for building **enterprise-grade RAG systems on difficult PDFs**.

---

## Setup

* Create a conda environment with Python 3.11
* install required libraries, following [setup_libraries](./setup_libraries.txt)
* configure NGINX following [nginx config](./ngingx_streamlit_config)

## License

This code is released under **MIT** License

see [LICENSE](./LICENSE)

