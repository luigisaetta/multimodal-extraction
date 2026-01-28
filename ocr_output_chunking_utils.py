"""
Utilities for chunking OCR output produced by the scanned-PDF pipeline.

This module is intentionally NOT based on Docling:
- Input is already-extracted text (e.g., out_dir/output.txt) with page footer markers.
- Page provenance is reconstructed from the OCR footer marker: '--- PAGE N ---'.

It provides:
- Parsing OCR output into per-page blocks
- Chunking each page with LangChain's RecursiveCharacterTextSplitter
- Producing LangChain Document objects with clean metadata
- Optionally prepending a stable header to each chunk:
    ---
    source_file: <pdf file name>
    ---

Expected OCR output structure (simplified):
    ... header ...
    ==================== BEGIN TEXT ====================

    <page 1 text ...>

    --- PAGE 1 ---

    <page 2 text ...>

    --- PAGE 2 ---

    ===================== END TEXT =====================
    TOTAL PAGES: <n>

Author: Luigi Saetta
Python: 3.11+
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Matches the footer emitted by your OCR pipeline
# Example: "\n\n--- PAGE 12 ---\n\n"
_PAGE_FOOTER_RE = re.compile(r"(?im)^\s*---\s*PAGE\s+(\d+)\s*---\s*$")

# Optional boundaries (not strictly required, but helpful if you want to trim headers/footers)
# NOTE: your OCR file uses BEGIN TEXT / END TEXT)
_BEGIN_RE = re.compile(r"(?im)^[=\s]*BEGIN\s+TEXT\s*[=\s]*$")
_END_RE = re.compile(r"(?im)^[=\s]*END\s+TEXT\s*[=\s]*$")


@dataclass(frozen=True)
class OcrChunk:
    """
    A chunk generated from OCR output text.

    Attributes:
        text: The chunk content (possibly with a header prepended).
        source_name: The source document name (usually the PDF filename).
        page_label: Page number as a string, e.g. "12".
        chunk_index: Global chunk index within the returned list.
        metadata: Metadata dictionary attached to this chunk.
    """

    text: str
    source_name: str
    page_label: str
    chunk_index: int
    metadata: Dict[str, str]


def normalize_spaces_keep_newlines(text: str) -> str:
    """
    Normalize whitespace without removing newlines.

    - Collapses runs of spaces/tabs to a single space
    - Keeps newlines intact (important for OCR paragraph structure)

    Args:
        text: Input text.

    Returns:
        Normalized text.
    """
    return re.sub(r"[ \t]{2,}", " ", text)


def chunk_text(text: str, max_chunk_size: int, overlap: int = 0) -> List[str]:
    """
    Chunk text into approximate character-sized parts.

    Uses LangChain's RecursiveCharacterTextSplitter to generate chunks.

    Args:
        text: Input text to chunk.
        max_chunk_size: Target maximum chunk size in characters.
        overlap: Overlap between chunks in characters.

    Returns:
        List of chunk strings.
    """
    normalized = normalize_spaces_keep_newlines(text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=overlap,
    )
    return splitter.split_text(normalized)


def extract_ocr_body(full_text: str) -> str:
    """
    Extract the main body of OCR output, removing headers/footers.
    """
    s = full_text

    m_begin = _BEGIN_RE.search(s)
    if not m_begin:
        return s

    m_end = _END_RE.search(s, pos=m_begin.end())
    if not m_end:
        return s

    return s[m_begin.end() : m_end.start()].strip()


def parse_pages_from_ocr_output(full_text: str) -> List[Tuple[int, str]]:
    """
    Parse OCR output into a list of (page_num, page_text).

    IMPORTANT: '--- PAGE N ---' is a footer that terminates the page text
    preceding it (not a header starting the next page).
    """
    body = extract_ocr_body(full_text).replace("\r\n", "\n")

    matches = list(_PAGE_FOOTER_RE.finditer(body))
    if not matches:
        return []

    out: List[Tuple[int, str]] = []
    start = 0

    for m in matches:
        page_num_str = m.group(1)
        try:
            page_num = int(page_num_str)
        except ValueError:
            start = m.end()
            continue

        page_text = body[start : m.start()].strip()
        out.append((page_num, page_text))

        # next page starts after the footer line
        start = m.end()

    return out


def build_metadata(source_name: str, page_label: str) -> Dict[str, str]:
    """
    Build metadata for a chunk.

    Args:
        source_name: The source document name (usually PDF filename).
        page_label: Page label (e.g., "12").

    Returns:
        Metadata dict.
    """
    return {"source": source_name, "page_label": page_label, "extraction": "ocr"}


def make_chunk_header(source_name: str) -> str:
    """
    Build the chunk header block to prepend to every chunk.

    Format requested:
        ---
        source_file: <pdf file name>
        ---

    Args:
        source_name: The PDF filename (or other source identifier).

    Returns:
        Header string (ending with two newlines).
    """
    return f"---\nsource_file: {source_name}\n---\n\n"


def ocr_output_text_to_chunks(
    full_text: str,
    source_name: str,
    max_chunk_size: int = 1500,
    overlap: int = 100,
    add_header: bool = True,
) -> List[OcrChunk]:
    """
    Convert OCR output text into a list of OcrChunk objects.

    Strategy:
    - Parse OCR output into per-page blocks.
    - Chunk each page independently (keeps page_label meaningful).
    - Attach metadata: {source, page_label, extraction=ocr}.
    - Optionally prepend a stable header containing ONLY the source file name:
        ---
        source_file: <source_name>
        ---

    Args:
        full_text: Full OCR output text (e.g., content of output.txt).
        source_name: A short source name (usually the original PDF filename).
        max_chunk_size: Chunk size in characters.
        overlap: Chunk overlap in characters.
        add_header: If True, prepends the header block to each chunk.

    Returns:
        List of OcrChunk objects.
    """
    pages = parse_pages_from_ocr_output(full_text)
    chunks: List[OcrChunk] = []
    header = make_chunk_header(source_name) if add_header else ""

    for page_num, page_text in pages:
        page_label = str(page_num)

        # Keep page-level chunking stable even if page is placeholder/empty
        parts = chunk_text(page_text, max_chunk_size=max_chunk_size, overlap=overlap)

        for part in parts:
            txt = part.strip()
            if not txt:
                continue

            if add_header:
                txt = header + txt

            md = build_metadata(source_name, page_label)
            chunks.append(
                OcrChunk(
                    text=txt,
                    source_name=source_name,
                    page_label=page_label,
                    chunk_index=len(chunks),
                    metadata=md,
                )
            )

    return chunks


def ocr_output_file_to_chunks(
    ocr_output_path: str | Path,
    source_name: Optional[str] = None,
    max_chunk_size: int = 1500,
    overlap: int = 100,
    add_header: bool = True,
    encoding: str = "utf-8",
) -> List[OcrChunk]:
    """
    Read an OCR output file (output.txt) and convert it into chunks.

    Args:
        ocr_output_path: Path to the OCR output text file.
        source_name: Optional source name; if None, tries to infer it:
            - first tries to parse 'SOURCE PDF:' line and use its basename
            - otherwise uses the output file name
        max_chunk_size: Chunk size in characters.
        overlap: Chunk overlap in characters.
        add_header: If True, prepends the header block to each chunk.
        encoding: File encoding.

    Returns:
        List of OcrChunk objects.
    """
    p = Path(ocr_output_path).expanduser().resolve()
    text = p.read_text(encoding=encoding, errors="replace")

    inferred = source_name
    if not inferred:
        inferred = _infer_source_name_from_ocr_header(text) or p.name

    return ocr_output_text_to_chunks(
        full_text=text,
        source_name=inferred,
        max_chunk_size=max_chunk_size,
        overlap=overlap,
        add_header=add_header,
    )


def chunks_to_langchain_documents(chunks: List[OcrChunk]) -> List[Document]:
    """
    Convert OcrChunk objects to LangChain Document objects.

    Args:
        chunks: List of OcrChunk.

    Returns:
        List of langchain_core.documents.Document.
    """
    docs: List[Document] = []
    for ch in chunks:
        docs.append(Document(page_content=ch.text, metadata=dict(ch.metadata)))
    return docs


def _infer_source_name_from_ocr_header(full_text: str) -> Optional[str]:
    """
    Infer the original PDF filename from the OCR header line:
        SOURCE PDF: /path/to/file.pdf

    Args:
        full_text: OCR output text.

    Returns:
        Basename of the source PDF if found, else None.
    """
    # Keep it simple and robust: scan first ~50 lines
    head = "\n".join(full_text.splitlines()[:50])
    m = re.search(
        r"^\s*SOURCE\s+PDF:\s*(.+)\s*$",
        head,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not m:
        return None
    src = m.group(1).strip()
    if not src:
        return None
    try:
        return Path(src).name
    except Exception:
        return None
