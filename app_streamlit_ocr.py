"""
Streamlit UI for scanned PDF -> OCR (multimodal LLM) -> single text output

Enhancements:
- Robust PDF viewer using pdf.js (streamlit-pdf-viewer)
- After upload, classify the PDF as TEXT_PDF / SCANNED_PDF / MIXED_OR_UNKNOWN
  using classify_pdf.py and display the type under the PDF viewer.
- "Describe figures" (Level 1) -> appends a [FIGURES] section at end of each page
  (requires text_from_pdf_scanner.py supporting OcrConfig.describe_figures)
- NEW: Chunk & Load step:
  - Takes OCR output text from session_state (output.txt content)
  - Chunks it using ocr_output_chunking_utils.py
  - Provides a stub hook to load chunks into Oracle Vector Store (you will add code)

Author: Luigi Saetta
Python: 3.11+
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import streamlit as st

from streamlit_pdf_viewer import pdf_viewer

from text_from_pdf_scanner import OcrConfig, run_ocr_pipeline
from classify_pdf import ClassifyConfig, classify_pdf

from ocr_output_chunking_utils import (
    ocr_output_text_to_chunks,
    chunks_to_langchain_documents,
)
from db_utils import get_db_connection
from oraclevs_4_db_loading import OracleVS4DBLoading
from oci_models import get_embedding_model
from utils import get_console_logger

from config import DEBUG, COLLECTION_NAME

logger = get_console_logger()

# ---- Configure your available models here ----
# check that you have the right to use the models you list here
# license, availability in your tenant, etc.
MODEL_IDS = [
    "openai.gpt-5.2",
    "meta.llama-4-maverick-17b-128e-instruct-fp8",
    "google.gemini-2.5-pro",
    "xai.grok-4-1-fast-non-reasoning",
    "cohere.command-a-vision v1.0",
    # add others you support via get_llm(...)
]

st.set_page_config(page_title="PDF Scanner OCR", layout="wide")
st.title("PDF Scanner → OCR with Multimodal LLM")


def type_badge(_label: str) -> None:
    """Clean, non-noisy rendering of the classification label."""
    if _label == "TEXT_PDF":
        st.success("Type detected: TEXT_PDF (extractable text)")
    elif _label == "SCANNED_PDF":
        st.warning("Type detected: SCANNED_PDF (scanned/images)")
    elif _label == "MIXED_OR_UNKNOWN":
        st.info("Type detected: MIXED/UNKNOWN (hybrid or ambiguous)")
    else:
        st.info(f"Type detected: {_label}")


@st.cache_data(show_spinner=False)
def classify_uploaded_pdf(tmp_pdf_path_str: str) -> tuple[str, str]:
    """
    Classify using the same heuristics as classify_pdf.py.
    Cached per temp path string (within a session run).
    """
    _cfg = ClassifyConfig(
        sample_pages=10,
        min_text_chars_doc=200,
        min_text_chars_page=50,
        scanned_if_image_pages_ratio_ge=0.6,
        strong_text_chars=5000,
    )
    _label, _reason = classify_pdf(Path(tmp_pdf_path_str), _cfg)
    return _label, (_reason or "")


def print_chunks_loaded(langchain_docs) -> None:
    """Debug: print loaded chunks."""
    for i, doc in enumerate(langchain_docs):
        print("----------------------------")
        print("Chunk n. ", i + 1)
        print("")
        print(f"Doc page_content:\n{doc.page_content}")
        print("")
        print(f"Doc metadata:\n{doc.metadata}")
        print("----------------------------")
        print("")


def oracle_vector_store_load(langchain_docs) -> None:
    """
    hook for Oracle Vector Store loading.

    Replace the body of this function with your actual code, e.g.:
      - create/connect to Oracle VS collection
      - upsert documents with embeddings
      - handle batching, retries, etc.

    Args:
        langchain_docs: List[langchain_core.documents.Document]
    """
    # replace with your real loader
    if len(langchain_docs) > 0:
        with get_db_connection() as conn:
            logger.info("Loading chunks in DB...")

            oracle_vs = OracleVS4DBLoading(
                client=conn,
                table_name=COLLECTION_NAME,
                embedding_function=get_embedding_model(),
            )

            oracle_vs.add_documents(langchain_docs)

            if DEBUG:
                # print the chunks added
                print_chunks_loaded(langchain_docs)

    logger.info("oracle_vector_store_load called with %s chunks.", len(langchain_docs))


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])

    with st.expander("🖥️ PDF Viewer", expanded=False):
        show_pdf_preview = st.checkbox("Show PDF preview", value=True)
        pdf_view_height = st.slider("Viewer height (px)", 300, 1200, 700, 50)

    st.header("LLM")
    model_id = st.selectbox("MODEL_ID", options=MODEL_IDS, index=0)

    with st.expander("⚙️ Rendering & OCR Settings", expanded=False):
        st.subheader("Rendering")
        dpi = st.slider("DPI", min_value=120, max_value=300, value=200, step=10)
        max_pages = st.number_input("Max pages (0 = all)", min_value=0, value=0, step=1)

        st.subheader("Blank page detection")
        white_threshold = st.slider(
            "White threshold", min_value=220, max_value=255, value=245, step=1
        )
        min_nonwhite_ratio = st.slider(
            "Min non-white ratio",
            min_value=0.001,
            max_value=0.050,
            value=0.010,
            step=0.001,
            format="%.3f",
        )
        center_crop = st.checkbox("Use center crop (ignore margins)", value=True)
        blank_placeholder = st.text_input(
            "Blank placeholder text", value="[BLANK PAGE SKIPPED]"
        )

        st.subheader("Image payload")
        max_side = st.slider(
            "Max image side (px)",
            min_value=800,
            max_value=2200,
            value=1600,
            step=100,
        )
        jpeg_quality = st.slider(
            "JPEG quality", min_value=50, max_value=95, value=85, step=1
        )

        st.subheader("Figures (Level 1)")
        describe_figures = st.checkbox(
            "Describe figures (append [FIGURES] per page)",
            value=True,
            help=(
                "Adds a second multimodal call per page to describe diagrams/drawings. "
                "Tables are ignored."
            ),
        )

    st.header("Output")
    out_path_str = st.text_input("Output file path", value="./out_ocr/output.txt")
    save_images = st.checkbox("Save rendered images", value=False)
    images_dir_str = st.text_input("Images dir (optional)", value="./out_ocr/images")

    extra_prompt = st.text_area("Extra prompt (optional)", value="", height=120)

    run_btn = st.button("Run OCR", type="primary", use_container_width=True)

    st.divider()

    st.header("Chunk & Load (OCR text)")
    chunk_size = st.slider("Chunk size (chars)", 600, 3000, 2048, 100)
    chunk_overlap = st.slider("Chunk overlap (chars)", 0, 600, 100, 20)
    add_chunk_header = st.checkbox("Add chunk header (source/page)", value=True)

    # Note: this button operates on session_state["output_text"] (not on the file)
    chunk_load_btn = st.button(
        "Chunk & Load to Vector Store",
        type="secondary",
        use_container_width=True,
        help=(
            "Chunks the current OCR output text (shown on the right) "
            "and calls your Vector Store loader."
        ),
    )


# ----------------------------
# Main layout
# ----------------------------
left, right = st.columns([1, 2], gap="large")

# stable state keys
if "pdf_type_label" not in st.session_state:
    st.session_state["pdf_type_label"] = None
if "pdf_type_reason" not in st.session_state:
    st.session_state["pdf_type_reason"] = None
if "output_text" not in st.session_state:
    st.session_state["output_text"] = None
if "out_path" not in st.session_state:
    st.session_state["out_path"] = None
if "chunks_count" not in st.session_state:
    st.session_state["chunks_count"] = None
if "last_chunk_error" not in st.session_state:
    st.session_state["last_chunk_error"] = None


with left:
    st.subheader("PDF preview / Status")

    if uploaded is None:
        st.info("Upload a PDF to start.")
    else:
        st.success(f"Uploaded: {uploaded.name} ({uploaded.size} bytes)")

        # Save uploaded PDF to a temp file for classification
        # (so classification appears immediately).
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_pdf_path = Path(tmpdir) / uploaded.name
            tmp_pdf_path.write_bytes(uploaded.getbuffer())

            label, reason = classify_uploaded_pdf(str(tmp_pdf_path))
            st.session_state["pdf_type_label"] = label
            st.session_state["pdf_type_reason"] = reason

        if show_pdf_preview:
            pdf_viewer(
                uploaded.getvalue(),
                height=pdf_view_height,
                pages_vertical_spacing=8,
            )
        else:
            st.caption("PDF preview is disabled (enable it from the sidebar).")

        st.download_button(
            "Download original PDF",
            data=uploaded.getvalue(),
            file_name=uploaded.name,
            mime="application/pdf",
            use_container_width=True,
        )

        st.divider()

        if st.session_state["pdf_type_label"]:
            type_badge(st.session_state["pdf_type_label"])
            with st.expander("Details (why?)", expanded=False):
                st.code(st.session_state.get("pdf_type_reason") or "-", language="text")

        st.divider()
        st.subheader("Chunk/Load status")
        if st.session_state.get("chunks_count") is not None:
            st.success(
                f"Last chunk run produced: {st.session_state['chunks_count']} chunks"
            )
        if st.session_state.get("last_chunk_error"):
            st.error(st.session_state["last_chunk_error"])


with right:
    st.subheader("Output text")
    if st.session_state.get("output_text"):
        st.text_area(
            "Extracted text",
            st.session_state["output_text"],
            height=650,
        )
        st.download_button(
            "Download output.txt",
            data=st.session_state["output_text"].encode("utf-8"),
            file_name=Path(
                st.session_state.get("out_path", "output.txt") or "output.txt"
            ).name,
            mime="text/plain",
            use_container_width=True,
        )
    else:
        st.caption("Run OCR to see extracted text here.")


# ----------------------------
# Run OCR pipeline
# ----------------------------
if run_btn:
    if uploaded is None:
        st.error("Please upload a PDF first.")
        st.stop()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_pdf_path = Path(tmpdir) / uploaded.name
        tmp_pdf_path.write_bytes(uploaded.getbuffer())

        out_path = Path(out_path_str).expanduser()
        images_dir = (
            Path(images_dir_str).expanduser() if images_dir_str.strip() else None
        )

        cfg = OcrConfig(
            model_id=model_id,
            out_path=out_path,
            dpi=int(dpi),
            max_pages=None if int(max_pages) == 0 else int(max_pages),
            extra_prompt=extra_prompt,
            save_images=save_images,
            images_dir=images_dir if save_images else None,
            blank_white_threshold=int(white_threshold),
            blank_min_nonwhite_ratio=float(min_nonwhite_ratio),
            blank_use_center_crop=bool(center_crop),
            blank_placeholder=blank_placeholder,
            max_side=int(max_side),
            jpeg_quality=int(jpeg_quality),
            describe_figures=bool(describe_figures),
        )

        with st.spinner("Running text-extraction pipeline..."):
            try:
                TEXT = run_ocr_pipeline(tmp_pdf_path, cfg)
            except Exception as e:
                st.error(f"OCR failed: {type(e).__name__}: {e}")
                st.stop()

        st.session_state["output_text"] = TEXT
        st.session_state["out_path"] = str(out_path)
        st.session_state["chunks_count"] = None
        st.session_state["last_chunk_error"] = None

        st.success(f"Done. Written to: {out_path}")
        st.rerun()


# ----------------------------
# Chunk & Load
# ----------------------------
if chunk_load_btn:
    if uploaded is None:
        st.error("Please upload a PDF first.")
        st.stop()

    if not st.session_state.get("output_text"):
        st.error("No OCR output text found. Run OCR first.")
        st.stop()

    with st.spinner("Chunking OCR output text..."):
        try:
            chunks = ocr_output_text_to_chunks(
                full_text=st.session_state["output_text"],
                source_name=uploaded.name,
                max_chunk_size=int(chunk_size),
                overlap=int(chunk_overlap),
                add_header=bool(add_chunk_header),
            )
            docs = chunks_to_langchain_documents(chunks)

            # ---- Your Vector Store load goes here ----
            # Replace oracle_vector_store_load_stub with your real implementation.
            oracle_vector_store_load(docs)

            st.session_state["chunks_count"] = len(docs)
            st.session_state["last_chunk_error"] = None
            st.success(f"Loaded {len(docs)} chunks to Vector Store.")
        except Exception as e:
            st.session_state["chunks_count"] = None
            st.session_state["last_chunk_error"] = (
                f"Chunk/Load failed: {type(e).__name__}: {e}"
            )
            st.error(st.session_state["last_chunk_error"])

    st.rerun()
