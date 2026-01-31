"""
Streamlit UI for scanned PDF -> OCR (multimodal LLM) -> single text output

Enhancements:
- Robust PDF viewer using pdf.js (streamlit-pdf-viewer)
- After upload, classify the PDF as TEXT_PDF / SCANNED_PDF / MIXED_OR_UNKNOWN
  using classify_pdf.py and display the type under the PDF viewer.
- "Describe figures" (Level 1) -> appends a [FIGURES] section at end of each page
  (requires text_from_pdf_scanner.py supporting OcrConfig.describe_figures)
- Chunk & Load step:
  - Takes OCR output text from session_state (output.txt content)
  - Chunks it using ocr_output_chunking_utils.py
  - Loads chunks into Oracle Vector Store.

DB / Collection Inspector page:
- Shows DB connection parameters read from config_private.py (password masked)
- Sidebar button to check DB connection (runs SELECT 1 FROM dual)
- Shows list of documents contained in the COLLECTION_NAME collection
  with number of chunks per document.

Fix:
- When a NEW PDF is uploaded, clear previous OCR output and chunk/load status.

Author: Luigi Saetta
Python: 3.11+
License: MIT
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

from classify_pdf import ClassifyConfig, classify_pdf
from config import COLLECTION_NAME, DEBUG

from oci_models import get_embedding_model
from ocr_output_chunking_utils import (
    chunks_to_langchain_documents,
    ocr_output_text_to_chunks,
)
from oraclevs_4_db_loading import OracleVS4DBLoading
from text_from_pdf_scanner import OcrConfig, run_ocr_pipeline
from db_utils import get_db_connection, get_connection_params, check_db_connection
from utils import get_console_logger

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


# ----------------------------
# Helpers
# ----------------------------
def type_badge(pdf_type_label: str) -> None:
    """Clean, non-noisy rendering of the classification label."""
    if pdf_type_label == "TEXT_PDF":
        st.success("Type detected: TEXT_PDF (extractable text)")
    elif pdf_type_label == "SCANNED_PDF":
        st.warning("Type detected: SCANNED_PDF (scanned/images)")
    elif pdf_type_label == "MIXED_OR_UNKNOWN":
        st.info("Type detected: MIXED/UNKNOWN (hybrid or ambiguous)")
    else:
        st.info(f"Type detected: {pdf_type_label}")


@st.cache_data(show_spinner=False)
def classify_uploaded_pdf(tmp_pdf_path_str: str) -> tuple[str, str]:
    """
    Classify using the same heuristics as classify_pdf.py.
    Cached per temp path string (within a session run).
    """
    classify_cfg = ClassifyConfig(
        sample_pages=10,
        min_text_chars_doc=200,
        min_text_chars_page=50,
        scanned_if_image_pages_ratio_ge=0.6,
        strong_text_chars=5000,
    )
    detected_label, detected_reason = classify_pdf(Path(tmp_pdf_path_str), classify_cfg)
    return detected_label, (detected_reason or "")


def print_chunks_loaded(langchain_docs: list[Any]) -> None:
    """Debug: print loaded chunks."""
    for idx, doc in enumerate(langchain_docs):
        print("----------------------------")
        print("Chunk n. ", idx + 1)
        print("")
        print(f"Doc page_content:\n{doc.page_content}")
        print("")
        print(f"Doc metadata:\n{doc.metadata}")
        print("----------------------------")
        print("")


def oracle_vector_store_load(langchain_docs: list[Any]) -> None:
    """
    Load chunks into Oracle Vector Store.

    Args:
        langchain_docs: List[langchain_core.documents.Document]
    """
    if not langchain_docs:
        logger.info("oracle_vector_store_load called with 0 chunks.")
        return

    with get_db_connection() as conn:
        logger.info("Loading chunks in DB...")

        oracle_vs = OracleVS4DBLoading(
            client=conn,
            table_name=COLLECTION_NAME,
            embedding_function=get_embedding_model(),
        )

        oracle_vs.add_documents(langchain_docs)

        if DEBUG:
            print_chunks_loaded(langchain_docs)

    logger.info("oracle_vector_store_load called with %s chunks.", len(langchain_docs))


def reset_outputs_for_new_upload() -> None:
    """
    Clear OCR/chunk outputs when the user uploads a different PDF.

    This prevents showing stale OCR text and stale chunk/load status for a new file.
    """
    st.session_state["output_text"] = None
    st.session_state["out_path"] = None
    st.session_state["chunks_count"] = None
    st.session_state["last_chunk_error"] = None


def list_collection_documents_real(collection_name: str) -> list[dict[str, Any]]:
    """
    Return list of {"document": <source>, "n_chunks": <count>} for the given collection.

    Implementation:
      - validate identifier by calling OracleVS4DBLoading.list_documents_in_collection()
      - do a single GROUP BY query using METADATA.source

    Notes:
      - expects the table has a column named METADATA containing JSON with $.source
    """
    with get_db_connection() as conn:
        safe_table_name = collection_name.strip().upper()

        # Validate identifier (raises if invalid)
        _ = OracleVS4DBLoading.list_documents_in_collection(conn, safe_table_name)

        sql = f"""
            SELECT
                json_value(METADATA, '$.source') AS document,
                COUNT(*) AS n_chunks
            FROM {safe_table_name}
            WHERE json_value(METADATA, '$.source') IS NOT NULL
            GROUP BY json_value(METADATA, '$.source')
            ORDER BY document ASC
        """

        with conn.cursor() as cur:
            cur.execute(sql)
            fetched_rows = cur.fetchall()

    return [{"document": r[0], "n_chunks": int(r[1])} for r in fetched_rows]


def init_session_state() -> None:
    """Initialize Streamlit session_state keys used by this app."""
    defaults: dict[str, Any] = {
        "pdf_type_label": None,
        "pdf_type_reason": None,
        "output_text": None,
        "out_path": None,
        "chunks_count": None,
        "last_chunk_error": None,
        "uploaded_file_key": None,
        "db_check_ok": None,
        "db_check_msg": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def build_sidebar_inputs(current_page: str) -> dict[str, Any]:
    """
    Build the sidebar UI for the current page and return a dict of UI values.

    Key idea:
      - Avoid "sometimes-defined" variables by always returning the same dict keys.
      - Avoid pylint W0621 by not reusing common names (label, cfg, rows, ok, params, etc.)
        across scopes.
    """
    ui: dict[str, Any] = {
        "uploaded_file": None,
        "show_pdf_preview": False,
        "pdf_view_height": 700,
        "model_id": MODEL_IDS[0],
        "extra_prompt": "",
        "dpi": 200,
        "max_pages": 0,
        "white_threshold": 245,
        "min_nonwhite_ratio": 0.010,
        "center_crop": True,
        "blank_placeholder": "[BLANK PAGE SKIPPED]",
        "max_side": 1600,
        "jpeg_quality": 85,
        "describe_figures": True,
        "out_path_str": "./out_ocr/output.txt",
        "save_images": False,
        "images_dir_str": "./out_ocr/images",
        "run_btn": False,
        "chunk_size": 2048,
        "chunk_overlap": 100,
        "add_chunk_header": True,
        "chunk_load_btn": False,
        "check_db_btn": False,
    }

    if current_page == "OCR & Load":
        st.header("Input")
        ui["uploaded_file"] = st.file_uploader("Upload PDF", type=["pdf"])

        with st.expander("🖥️ PDF Viewer", expanded=False):
            ui["show_pdf_preview"] = st.checkbox("Show PDF preview", value=True)
            ui["pdf_view_height"] = st.slider("Viewer height (px)", 300, 1200, 700, 50)

        st.header("LLM")
        ui["model_id"] = st.selectbox("MODEL_ID", options=MODEL_IDS, index=0)
        ui["extra_prompt"] = st.text_area(
            "Extra prompt (optional)", value="", height=120
        )

        with st.expander("⚙️ Rendering & OCR Settings", expanded=False):
            st.subheader("Rendering")
            ui["dpi"] = st.slider(
                "DPI", min_value=120, max_value=300, value=200, step=10
            )
            ui["max_pages"] = st.number_input(
                "Max pages (0 = all)", min_value=0, value=0, step=1
            )

            st.subheader("Blank page detection")
            ui["white_threshold"] = st.slider(
                "White threshold", min_value=220, max_value=255, value=245, step=1
            )
            ui["min_nonwhite_ratio"] = st.slider(
                "Min non-white ratio",
                min_value=0.001,
                max_value=0.050,
                value=0.010,
                step=0.001,
                format="%.3f",
            )
            ui["center_crop"] = st.checkbox(
                "Use center crop (ignore margins)", value=True
            )
            ui["blank_placeholder"] = st.text_input(
                "Blank placeholder text", value="[BLANK PAGE SKIPPED]"
            )

            st.subheader("Image payload")
            ui["max_side"] = st.slider(
                "Max image side (px)",
                min_value=800,
                max_value=2200,
                value=1600,
                step=100,
            )
            ui["jpeg_quality"] = st.slider(
                "JPEG quality", min_value=50, max_value=95, value=85, step=1
            )

            st.subheader("Figures (Level 1)")
            ui["describe_figures"] = st.checkbox(
                "Describe figures (append [FIGURES] per page)",
                value=True,
                help=(
                    "Adds a second multimodal call per page to describe diagrams/drawings. "
                    "Tables are ignored."
                ),
            )

        st.header("Output")
        ui["out_path_str"] = st.text_input(
            "Output file path", value="./out_ocr/output.txt"
        )
        ui["save_images"] = st.checkbox("Save rendered images", value=False)
        ui["images_dir_str"] = st.text_input(
            "Images dir (optional)", value="./out_ocr/images"
        )

        ui["run_btn"] = st.button("Run OCR", type="primary", use_container_width=True)

        st.divider()

        st.header("Chunk & Load (OCR text)")
        ui["chunk_size"] = st.slider("Chunk size (chars)", 600, 3000, 2048, 100)
        ui["chunk_overlap"] = st.slider("Chunk overlap (chars)", 0, 600, 100, 20)
        ui["add_chunk_header"] = st.checkbox(
            "Add chunk header (source/page)", value=True
        )

        ui["chunk_load_btn"] = st.button(
            "Chunk & Load to Vector Store",
            type="secondary",
            use_container_width=True,
            help=(
                "Chunks the current OCR output text (shown on the right) "
                "and calls your Vector Store loader."
            ),
        )

    else:
        # second page: DB connection and data inspector
        st.header("DB Connection")
        conn_params = get_connection_params()
        st.caption("Connection parameters:")
        st.code(
            "\n".join([f"{k} = {v}" for k, v in conn_params.items()]),
            language="text",
        )

        st.divider()

        st.header("Actions")
        ui["check_db_btn"] = st.button(
            "Check DB connection",
            type="primary",
            use_container_width=True,
            help="Tries to open a DB connection and run a simple SELECT.",
        )

        if ui["check_db_btn"]:
            check_ok, check_msg = check_db_connection()
            st.session_state["db_check_ok"] = check_ok
            st.session_state["db_check_msg"] = check_msg

        if st.session_state["db_check_ok"] is True:
            st.success(st.session_state["db_check_msg"])
        elif st.session_state["db_check_ok"] is False:
            st.error(st.session_state["db_check_msg"])

    return ui


# ----------------------------
# App start
# ----------------------------
st.set_page_config(page_title="PDF Scanner OCR", layout="wide")
st.title("Multimodal OCR & Figure Extraction for Technical PDF")

init_session_state()

with st.sidebar:
    st.header("Navigation")
    nav_page = st.radio(
        "Go to",
        options=["OCR & Load", "DB / Collection Inspector"],
        index=0,
        label_visibility="collapsed",
    )
    st.divider()

    ui_state = build_sidebar_inputs(nav_page)

# ----------------------------
# Main layout
# ----------------------------
left, right = st.columns([1, 2], gap="large")

# ----------------------------
# OCR PAGE
# ----------------------------
if nav_page == "OCR & Load":
    # exytract from pdf, chunk and load
    uploaded_file = ui_state["uploaded_file"]

    # Detect upload changes and reset outputs
    current_upload_key = None
    if uploaded_file is not None:
        current_upload_key = f"{uploaded_file.name}:{uploaded_file.size}"

    if current_upload_key != st.session_state["uploaded_file_key"]:
        st.session_state["uploaded_file_key"] = current_upload_key
        reset_outputs_for_new_upload()

    with left:
        st.subheader("PDF preview / Status")

        if uploaded_file is None:
            st.info("Upload a PDF to start.")
        else:
            st.success(f"Uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")

            # Save uploaded PDF to a temp file for classification
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_pdf_path = Path(tmpdir) / uploaded_file.name
                tmp_pdf_path.write_bytes(uploaded_file.getbuffer())

                detected_type_label, detected_type_reason = classify_uploaded_pdf(
                    str(tmp_pdf_path)
                )
                st.session_state["pdf_type_label"] = detected_type_label
                st.session_state["pdf_type_reason"] = detected_type_reason

            if ui_state["show_pdf_preview"]:
                pdf_viewer(
                    uploaded_file.getvalue(),
                    height=int(ui_state["pdf_view_height"]),
                    pages_vertical_spacing=8,
                )
            else:
                st.caption("PDF preview is disabled (enable it from the sidebar).")

            st.download_button(
                "Download original PDF",
                data=uploaded_file.getvalue(),
                file_name=uploaded_file.name,
                mime="application/pdf",
                use_container_width=True,
            )

            st.divider()

            stored_type_label = st.session_state.get("pdf_type_label") or ""
            if stored_type_label:
                type_badge(stored_type_label)
                with st.expander("Details (why?)", expanded=False):
                    st.code(
                        st.session_state.get("pdf_type_reason") or "-",
                        language="text",
                    )

            st.divider()
            st.subheader("Chunk/Load status")
            last_chunks_count = st.session_state.get("chunks_count")
            if last_chunks_count is not None:
                st.success(f"Last chunk run produced: {last_chunks_count} chunks")

            last_chunk_error = st.session_state.get("last_chunk_error")
            if last_chunk_error:
                st.error(last_chunk_error)

    with right:
        st.subheader("Output text")
        output_text = st.session_state.get("output_text")
        if output_text:
            st.text_area("Extracted text", output_text, height=650)
            out_name = Path(st.session_state.get("out_path") or "output.txt").name
            st.download_button(
                "Download output.txt",
                data=output_text.encode("utf-8"),
                file_name=out_name,
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.caption("Run OCR to see extracted text and figures description here.")

    # ----------------------------
    # Run OCR pipeline
    # ----------------------------
    if ui_state["run_btn"]:
        if uploaded_file is None:
            st.error("Please upload a PDF first.")
            st.stop()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_pdf_path = Path(tmpdir) / uploaded_file.name
            tmp_pdf_path.write_bytes(uploaded_file.getbuffer())

            out_path = Path(ui_state["out_path_str"]).expanduser()
            images_dir = (
                Path(ui_state["images_dir_str"]).expanduser()
                if ui_state["images_dir_str"].strip()
                else None
            )

            ocr_cfg = OcrConfig(
                model_id=ui_state["model_id"],
                out_path=out_path,
                dpi=int(ui_state["dpi"]),
                max_pages=(
                    None
                    if int(ui_state["max_pages"]) == 0
                    else int(ui_state["max_pages"])
                ),
                extra_prompt=ui_state["extra_prompt"],
                save_images=bool(ui_state["save_images"]),
                images_dir=images_dir if ui_state["save_images"] else None,
                blank_white_threshold=int(ui_state["white_threshold"]),
                blank_min_nonwhite_ratio=float(ui_state["min_nonwhite_ratio"]),
                blank_use_center_crop=bool(ui_state["center_crop"]),
                blank_placeholder=ui_state["blank_placeholder"],
                max_side=int(ui_state["max_side"]),
                jpeg_quality=int(ui_state["jpeg_quality"]),
                describe_figures=bool(ui_state["describe_figures"]),
                text_extraction_mode="auto",
                input_pdf_type=st.session_state.get(
                    "pdf_type_label"
                ),  # TEXT_PDF / SCANNED_PDF / MIXED_OR_UNKNOWN
                min_text_chars_page=50,
            )

            with st.spinner("Running text-extraction pipeline..."):
                try:
                    extracted_text = run_ocr_pipeline(tmp_pdf_path, ocr_cfg)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    st.error(f"OCR failed: {type(exc).__name__}: {exc}")
                    st.stop()

            st.session_state["output_text"] = extracted_text
            st.session_state["out_path"] = str(out_path)
            st.session_state["chunks_count"] = None
            st.session_state["last_chunk_error"] = None

            st.success(f"Done. Written to: {out_path}")
            st.rerun()

    # ----------------------------
    # Chunk & Load
    # ----------------------------
    if ui_state["chunk_load_btn"]:
        if uploaded_file is None:
            st.error("Please upload a PDF first.")
            st.stop()

        output_text = st.session_state.get("output_text")
        if not output_text:
            st.error("No OCR output text found. Run OCR first.")
            st.stop()

        with st.spinner("Chunking OCR output text..."):
            try:
                chunks = ocr_output_text_to_chunks(
                    full_text=output_text,
                    source_name=uploaded_file.name,
                    max_chunk_size=int(ui_state["chunk_size"]),
                    overlap=int(ui_state["chunk_overlap"]),
                    add_header=bool(ui_state["add_chunk_header"]),
                )
                docs = chunks_to_langchain_documents(chunks)

                oracle_vector_store_load(docs)

                st.session_state["chunks_count"] = len(docs)
                st.session_state["last_chunk_error"] = None
                st.success(f"Loaded {len(docs)} chunks to Vector Store.")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                st.session_state["chunks_count"] = None
                st.session_state["last_chunk_error"] = (
                    f"Chunk/Load failed: {type(exc).__name__}: {exc}"
                )
                st.error(st.session_state["last_chunk_error"])

        st.rerun()

# ----------------------------
# DB / COLLECTION INSPECTOR PAGE
# ----------------------------
else:
    with left:
        st.subheader("DB / Collection status")

        inspector_params = get_connection_params()
        st.write(f"**Collection:** `{inspector_params['COLLECTION_NAME']}`")

        db_check_ok = st.session_state.get("db_check_ok", None)
        if db_check_ok is True:
            st.success("DB connection: OK")
        elif db_check_ok is False:
            st.error("DB connection: FAILED")
        else:
            st.info("Run **Check DB connection** from the sidebar.")

        st.divider()
        st.caption("Tip: if this fails, verify DSN / wallet / network ACLs.")

    with right:
        st.subheader(f"Documents in collection: {COLLECTION_NAME}")

        try:
            collection_rows = list_collection_documents_real(COLLECTION_NAME)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            st.error(
                f"Failed to read documents from collection: {type(exc).__name__}: {exc}"
            )
            collection_rows = []

        if not collection_rows:
            st.warning("No documents found in collection.")
        else:
            st.dataframe(collection_rows, use_container_width=True, hide_index=True)

            total_docs = len(collection_rows)
            total_chunks = sum(int(r.get("n_chunks", 0)) for r in collection_rows)
            st.caption(
                f"Total documents: **{total_docs}** · Total chunks: **{total_chunks}**"
            )
