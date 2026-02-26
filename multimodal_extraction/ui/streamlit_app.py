"""
Author: Luigi Saetta
Date last modified: 2026-01-30
Python Version: 3.11
License: MIT

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

"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

from multimodal_extraction.pdf.classify_pdf import ClassifyConfig, classify_pdf
from multimodal_extraction.config import (
    COLLECTION_NAME,
    MODEL_IDS,
    DEBUG,
    ENABLE_MODEL_COMPARISON,
    REFERENCE_MODEL_ID,
    MODEL_COMPARISON_CACHE_DIR,
)

from multimodal_extraction.models.oci_models import get_embedding_model
from multimodal_extraction.chunking.ocr_output_chunking_utils import (
    chunks_to_langchain_documents,
    ocr_output_text_to_chunks,
)
from multimodal_extraction.db.oraclevs_admin import OracleVSAdmin
from multimodal_extraction.ocr.text_from_pdf_scanner import OcrConfig, run_ocr_pipeline
from multimodal_extraction.db.db_utils import get_db_connection, get_connection_params, check_db_connection
from multimodal_extraction.utils import get_console_logger, print_chunks_loaded

logger = get_console_logger()


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


def oracle_vector_store_load(langchain_docs: list[Any], collection_name: str) -> None:
    """
    Load chunks into Oracle Vector Store.

    Args:
        langchain_docs: List[langchain_core.documents.Document]
    """
    if not langchain_docs:
        logger.info("oracle_vector_store_load called with 0 chunks.")
        return

    with get_db_connection() as _conn:
        logger.info("Loading chunks in DB...")

        oracle_vs = OracleVSAdmin(
            client=_conn,
            table_name=collection_name,
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
    st.session_state["comparison_result"] = None


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
        "available_collections": [],
        "selected_collection": COLLECTION_NAME,
        "ocr_target_collection": COLLECTION_NAME,
        "collection_rows": None,
        "collection_load_msg": None,
        "collection_rows_for": None,
        "collection_create_ok": None,
        "collection_create_msg": None,
        "drop_confirm_name": "",
        "collection_drop_ok": None,
        "collection_drop_msg": None,
        "viewer_collection": COLLECTION_NAME,
        "viewer_document": "",
        "viewer_rows": None,
        "viewer_load_msg": None,
        "viewer_rows_for_collection": None,
        "viewer_rows_for_document": None,
        "comparison_result": None,
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
        "page_selection_mode": "All pages",
        "single_page": 1,
        "range_start_page": 1,
        "range_end_page": 1,
        "white_threshold": 245,
        "min_nonwhite_ratio": 0.010,
        "center_crop": True,
        "blank_placeholder": "[BLANK PAGE SKIPPED]",
        "max_side": 1600,
        "image_format": "jpeg",
        "jpeg_quality": 85,
        "describe_figures": True,
        "force_vlm": False,
        "enable_model_comparison": bool(ENABLE_MODEL_COMPARISON),
        "out_path_str": "./out_ocr/output.txt",
        "save_images": False,
        "images_dir_str": "./out_ocr/images",
        "run_btn": False,
        "chunk_by_page": True,
        "chunk_size": 2048,
        "chunk_overlap": 100,
        "add_chunk_header": True,
        "chunk_load_btn": False,
        "check_db_btn": False,
        "show_docs_btn": False,
        "selected_collection": None,
        "ocr_target_collection": None,
        "new_collection_name": "",
        "create_collection_btn": False,
        "drop_collection_btn": False,
        "check_image_payload_btn": False,
        "viewer_collection": None,
        "viewer_document": "",
        "load_document_btn": False,
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
            ui["page_selection_mode"] = st.radio(
                "Pages to process",
                options=["All pages", "Single page", "Range"],
                horizontal=True,
            )
            if ui["page_selection_mode"] == "Single page":
                ui["single_page"] = st.number_input(
                    "Page number (1-based)",
                    min_value=1,
                    value=1,
                    step=1,
                )
            elif ui["page_selection_mode"] == "Range":
                col_start, col_end = st.columns(2)
                with col_start:
                    ui["range_start_page"] = st.number_input(
                        "Start page",
                        min_value=1,
                        value=1,
                        step=1,
                    )
                with col_end:
                    ui["range_end_page"] = st.number_input(
                        "End page",
                        min_value=1,
                        value=1,
                        step=1,
                    )
                if int(ui["range_start_page"]) > int(ui["range_end_page"]):
                    st.error("Invalid range: start page must be <= end page.")

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
            ui["image_format"] = st.selectbox(
                "Image format sent to VLM",
                options=["jpeg", "png"],
                index=0,
                help="Default is JPEG. PNG sends an uncompressed image payload.",
            )
            ui["jpeg_quality"] = st.slider(
                "JPEG quality", min_value=50, max_value=95, value=85, step=1
            )
            ui["check_image_payload_btn"] = st.button(
                "Check image payload settings",
                type="secondary",
                width="stretch",
            )
            if ui["check_image_payload_btn"]:
                if ui["image_format"] == "jpeg":
                    st.success(
                        f"Image payload check OK: JPEG mode (quality={ui['jpeg_quality']})."
                    )
                else:
                    st.success(
                        "Image payload check OK: PNG mode (uncompressed, larger payload)."
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
            st.subheader("Text extraction strategy")
            ui["force_vlm"] = st.checkbox(
                "Force VLM OCR (ignore PDF classification)",
                value=False,
                help=(
                    "If enabled, always uses multimodal OCR. "
                    "If disabled, mode is chosen from classification."
                ),
            )
            st.subheader("Model comparison")
            ui["enable_model_comparison"] = st.checkbox(
                "Enable WER comparison vs reference model",
                value=bool(ENABLE_MODEL_COMPARISON),
                help=(
                    "If enabled, computes page-level WER against cached output from the "
                    f"reference model configured in code ({REFERENCE_MODEL_ID})."
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

        ui["run_btn"] = st.button("Run OCR", type="primary", width="stretch")

        st.divider()

        st.header("Chunk & Load (OCR text)")
        ui["chunk_by_page"] = st.checkbox(
            "Chunk by page (default)",
            value=True,
            help=(
                "Recommended mode: one chunk per page. "
                "Disable to use size-based chunking."
            ),
        )
        ui["chunk_size"] = st.slider(
            "Chunk size (chars)",
            600,
            3000,
            2048,
            100,
            disabled=bool(ui["chunk_by_page"]),
        )
        ui["chunk_overlap"] = st.slider(
            "Chunk overlap (chars)",
            0,
            600,
            100,
            20,
            disabled=bool(ui["chunk_by_page"]),
        )
        if ui["chunk_by_page"]:
            st.caption("Page mode active: size/overlap controls are disabled.")
        ui["add_chunk_header"] = st.checkbox(
            "Add chunk header (source/page)", value=True
        )
        ocr_collections = []
        try:
            with get_db_connection() as conn:
                ocr_collections = OracleVSAdmin.list_collections(conn)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            st.caption(f"Cannot load collections for dropdown: {type(exc).__name__}: {exc}")

        current_ocr_target = st.session_state.get("ocr_target_collection", COLLECTION_NAME)
        if ocr_collections:
            if current_ocr_target not in ocr_collections:
                current_ocr_target = ocr_collections[0]
            ui["ocr_target_collection"] = st.selectbox(
                "Target collection",
                options=ocr_collections,
                index=ocr_collections.index(current_ocr_target),
                help="Collection where chunks will be loaded.",
            )
            st.session_state["ocr_target_collection"] = ui["ocr_target_collection"]
        else:
            st.warning(
                "No collections available. Create one in DB / Collection Inspector first."
            )
            st.session_state["ocr_target_collection"] = current_ocr_target

        ui["chunk_load_btn"] = st.button(
            "Chunk & Load to Vector Store",
            type="secondary",
            width="stretch",
            help=(
                "Chunks the current OCR output text (shown on the right) "
                "and calls your Vector Store loader."
            ),
        )

    elif current_page == "DB / Collection Inspector":
        # second page: DB connection and data inspector
        st.header("DB Connection")
        conn_params = get_connection_params()
        default_collection = conn_params.get("COLLECTION_NAME", COLLECTION_NAME)
        conn_params_to_show = {
            "DB_USER": conn_params.get("DB_USER"),
            "DB_PASSWORD": conn_params.get("DB_PASSWORD"),
            "DB_DSN": conn_params.get("DB_DSN"),
            "DB_WALLET_DIR": conn_params.get("DB_WALLET_DIR"),
            "DEFAULT_COLLECTION": default_collection,
        }
        st.caption("Connection parameters:")
        st.code(
            "\n".join([f"{k} = {v}" for k, v in conn_params_to_show.items()]),
            language="text",
        )

        st.divider()

        st.header("Actions")
        ui["check_db_btn"] = st.button(
            "Check DB connection",
            type="primary",
            width="stretch",
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

        st.divider()
        st.subheader("Create new collection")
        ui["new_collection_name"] = st.text_input(
            "New collection name",
            value=st.session_state.get("new_collection_name", ""),
            help="Oracle identifier, e.g. COLL02",
        )
        st.session_state["new_collection_name"] = ui["new_collection_name"]
        ui["create_collection_btn"] = st.button(
            "Create empty collection",
            type="secondary",
            width="stretch",
            help=(
                "Creates a new OracleVS collection table with a temporary "
                "bootstrap row, then deletes it."
            ),
        )

        if ui["create_collection_btn"]:
            try:
                with get_db_connection() as conn:
                    created_name = OracleVSAdmin.create_empty_collection(
                        connection=conn,
                        collection_name=ui["new_collection_name"],
                        embedding=get_embedding_model(),
                    )
                st.session_state["collection_create_ok"] = True
                st.session_state["collection_create_msg"] = (
                    f"Collection created: {created_name}"
                )
                st.session_state["selected_collection"] = created_name
            except Exception as exc:  # pylint: disable=broad-exception-caught
                st.session_state["collection_create_ok"] = False
                st.session_state["collection_create_msg"] = (
                    f"Create failed: {type(exc).__name__}: {exc}"
                )

        if st.session_state.get("collection_create_ok") is True:
            st.success(st.session_state.get("collection_create_msg", ""))
        elif st.session_state.get("collection_create_ok") is False:
            st.error(st.session_state.get("collection_create_msg", ""))

        available_cols: list[str] = []
        if st.session_state["db_check_ok"] is True:
            try:
                with get_db_connection() as conn:
                    available_cols = OracleVSAdmin.list_collections(conn)
                st.session_state["available_collections"] = available_cols
            except Exception as exc:  # pylint: disable=broad-exception-caught
                st.session_state["available_collections"] = []
                st.error(f"Cannot load collections: {type(exc).__name__}: {exc}")
        else:
            st.session_state["available_collections"] = []

        selected_collection = st.session_state.get("selected_collection", COLLECTION_NAME)
        if available_cols:
            if selected_collection not in available_cols:
                selected_collection = available_cols[0]
            ui["selected_collection"] = st.selectbox(
                "Collection",
                options=available_cols,
                index=available_cols.index(selected_collection),
                help="Choose the collection to inspect.",
            )
            st.session_state["selected_collection"] = ui["selected_collection"]
            ui["show_docs_btn"] = st.button(
                "Show documents",
                type="secondary",
                width="stretch",
            )

            if ui["show_docs_btn"]:
                try:
                    with get_db_connection() as conn:
                        rows_loaded = OracleVSAdmin.list_documents_with_chunk_counts(
                            conn, ui["selected_collection"]
                        )
                    st.session_state["collection_rows"] = rows_loaded
                    st.session_state["collection_rows_for"] = ui["selected_collection"]
                    st.session_state["collection_load_msg"] = (
                        f"Loaded {len(rows_loaded)} documents from collection "
                        f"{ui['selected_collection']}."
                    )
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    st.session_state["collection_rows"] = []
                    st.session_state["collection_rows_for"] = ui["selected_collection"]
                    st.session_state["collection_load_msg"] = (
                        f"Load failed for {ui['selected_collection']}: "
                        f"{type(exc).__name__}: {exc}"
                    )

            st.divider()
            st.subheader("Danger zone")
            st.caption(
                "Drop deletes the whole selected collection table and all chunks."
            )
            st.session_state["drop_confirm_name"] = st.text_input(
                "Type selected collection name to confirm drop",
                value=st.session_state.get("drop_confirm_name", ""),
            )
            ui["drop_collection_btn"] = st.button(
                "Drop selected collection",
                type="secondary",
                width="stretch",
            )
            if ui["drop_collection_btn"]:
                selected_for_drop = ui["selected_collection"]
                confirm_name = (
                    st.session_state.get("drop_confirm_name", "").strip().upper()
                )
                if confirm_name != selected_for_drop:
                    st.session_state["collection_drop_ok"] = False
                    st.session_state["collection_drop_msg"] = (
                        "Drop blocked: confirmation text does not match "
                        "the selected collection."
                    )
                else:
                    try:
                        with get_db_connection() as conn:
                            OracleVSAdmin.drop_collection(conn, selected_for_drop)
                            refreshed = OracleVSAdmin.list_collections(conn)

                        st.session_state["collection_drop_ok"] = True
                        st.session_state["collection_drop_msg"] = (
                            f"Collection dropped: {selected_for_drop}"
                        )
                        st.session_state["available_collections"] = refreshed
                        st.session_state["drop_confirm_name"] = ""
                        st.session_state["collection_rows"] = None
                        st.session_state["collection_rows_for"] = None
                        st.session_state["collection_load_msg"] = None
                        if refreshed:
                            st.session_state["selected_collection"] = refreshed[0]
                        else:
                            st.session_state["selected_collection"] = COLLECTION_NAME
                        st.rerun()
                    except Exception as exc:  # pylint: disable=broad-exception-caught
                        st.session_state["collection_drop_ok"] = False
                        st.session_state["collection_drop_msg"] = (
                            f"Drop failed: {type(exc).__name__}: {exc}"
                        )

            if st.session_state.get("collection_drop_ok") is True:
                st.success(st.session_state.get("collection_drop_msg", ""))
            elif st.session_state.get("collection_drop_ok") is False:
                st.error(st.session_state.get("collection_drop_msg", ""))
        elif st.session_state["db_check_ok"] is True:
            st.info("No vector collections found in current schema.")
    else:
        # third page: document viewer
        st.header("Document Viewer")

        viewer_collections: list[str] = []
        try:
            with get_db_connection() as conn:
                viewer_collections = OracleVSAdmin.list_collections(conn)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            st.error(f"Cannot load collections: {type(exc).__name__}: {exc}")

        if not viewer_collections:
            st.info("No vector collections found in current schema.")
            return ui

        current_viewer_collection = st.session_state.get(
            "viewer_collection", COLLECTION_NAME
        )
        if current_viewer_collection not in viewer_collections:
            current_viewer_collection = viewer_collections[0]

        ui["viewer_collection"] = st.selectbox(
            "Collection",
            options=viewer_collections,
            index=viewer_collections.index(current_viewer_collection),
            help="Choose the collection containing the document.",
        )
        st.session_state["viewer_collection"] = ui["viewer_collection"]

        viewer_documents: list[str] = []
        try:
            with get_db_connection() as conn:
                viewer_documents = OracleVSAdmin.list_documents_in_collection(
                    conn, ui["viewer_collection"]
                )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            st.error(f"Cannot load documents: {type(exc).__name__}: {exc}")

        if not viewer_documents:
            st.info("No documents found in selected collection.")
            return ui

        current_viewer_document = st.session_state.get("viewer_document", "")
        if current_viewer_document not in viewer_documents:
            current_viewer_document = viewer_documents[0]

        ui["viewer_document"] = st.selectbox(
            "Document",
            options=viewer_documents,
            index=viewer_documents.index(current_viewer_document),
            help="Choose the document source (METADATA.source).",
        )
        st.session_state["viewer_document"] = ui["viewer_document"]

        ui["load_document_btn"] = st.button(
            "Load document",
            type="primary",
            width="stretch",
        )

    return ui


# ----------------------------
# App start
# ----------------------------
st.set_page_config(page_title="PDF Scanner OCR", layout="wide")
st.title("Multimodal Text & Figure Extraction for Technical PDF")

init_session_state()

with st.sidebar:
    st.header("Navigation")
    nav_page = st.radio(
        "Go to",
        options=["OCR & Load", "DB / Collection Inspector", "Document Viewer"],
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
                width="stretch",
            )

            st.divider()

            stored_type_label = st.session_state.get("pdf_type_label") or ""
            if stored_type_label:
                type_badge(stored_type_label)
                if ui_state["force_vlm"]:
                    st.warning("Text extraction override active: forcing VLM OCR.")
                with st.expander("Details (why?)", expanded=False):
                    st.code(
                        st.session_state.get("pdf_type_reason") or "-",
                        language="text",
                    )

            st.divider()
            st.subheader("Model comparison (WER)")
            comparison_result = st.session_state.get("comparison_result")
            if comparison_result and comparison_result.get("enabled"):
                mean_wer = comparison_result.get("mean_wer")
                if mean_wer is None:
                    st.info(
                        "Comparison enabled but no WER available yet "
                        f"(reference model: {comparison_result.get('reference_model_id')})."
                    )
                else:
                    st.success(
                        "Reference: "
                        f"{comparison_result.get('reference_model_id')} | "
                        f"Mean WER: {mean_wer:.4f} | "
                        f"Pages: {comparison_result.get('pages_evaluated')} | "
                        f"Cache hits: {comparison_result.get('cache_hits')} | "
                        f"Cache misses: {comparison_result.get('cache_misses')} | "
                        f"Errors: {comparison_result.get('errors')}"
                    )
            else:
                st.caption(
                    "Model comparison is disabled. Enable it in sidebar "
                    "under Rendering & OCR Settings."
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
            # 01/02/2026: you can now chhose between markdoen and raw text view
            st.subheader("Output text")
            output_text = st.session_state.get("output_text")

            if output_text:
                render_markdown = st.checkbox(
                    "Render Markdown",
                    value=False,
                    help=(
                        "Render markdown formatting (tables/headings). "
                        "Use 'Raw text' to copy the original content."
                    ),
                )

                if render_markdown:
                    st.markdown(output_text)

                    with st.expander("Raw text (copy/paste)", expanded=False):
                        st.text_area("Raw", output_text, height=260)
                else:
                    st.text_area("Extracted text", output_text, height=650)

                out_name = Path(st.session_state.get("out_path") or "output.txt").name
                st.download_button(
                    "Download output.txt",
                    data=output_text.encode("utf-8"),
                    file_name=out_name,
                    mime="text/plain",
                    width="stretch",
                )
            else:
                st.caption(
                    "Run OCR to see extracted text and figures description here."
                )

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
            selected_mode = ui_state["page_selection_mode"]
            start_page = None
            end_page = None
            if selected_mode == "Single page":
                start_page = int(ui_state["single_page"])
                end_page = int(ui_state["single_page"])
            elif selected_mode == "Range":
                start_page = int(ui_state["range_start_page"])
                end_page = int(ui_state["range_end_page"])
                if start_page > end_page:
                    st.error("Invalid range: start page must be <= end page.")
                    st.stop()

            ocr_cfg = OcrConfig(
                model_id=ui_state["model_id"],
                out_path=out_path,
                dpi=int(ui_state["dpi"]),
                start_page=start_page,
                end_page=end_page,
                extra_prompt=ui_state["extra_prompt"],
                save_images=bool(ui_state["save_images"]),
                images_dir=images_dir if ui_state["save_images"] else None,
                blank_white_threshold=int(ui_state["white_threshold"]),
                blank_min_nonwhite_ratio=float(ui_state["min_nonwhite_ratio"]),
                blank_use_center_crop=bool(ui_state["center_crop"]),
                blank_placeholder=ui_state["blank_placeholder"],
                max_side=int(ui_state["max_side"]),
                image_format=str(ui_state["image_format"]).lower(),
                jpeg_quality=int(ui_state["jpeg_quality"]),
                describe_figures=bool(ui_state["describe_figures"]),
                enable_model_comparison=bool(ui_state["enable_model_comparison"]),
                reference_model_id=REFERENCE_MODEL_ID,
                comparison_cache_dir=Path(MODEL_COMPARISON_CACHE_DIR),
                text_extraction_mode="vlm" if ui_state["force_vlm"] else "auto",
                input_pdf_type=(
                    None
                    if ui_state["force_vlm"]
                    else st.session_state.get("pdf_type_label")
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
            st.session_state["comparison_result"] = ocr_cfg.comparison_result

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
                    chunk_by_page=bool(ui_state["chunk_by_page"]),
                    add_header=bool(ui_state["add_chunk_header"]),
                )
                docs = chunks_to_langchain_documents(chunks)

                target_collection = st.session_state.get(
                    "ocr_target_collection", COLLECTION_NAME
                )
                with get_db_connection() as conn:
                    existing_cols = OracleVSAdmin.list_collections(conn)
                if target_collection not in existing_cols:
                    raise ValueError(
                        f"Target collection not found: {target_collection}. "
                        "Create/select a valid collection first."
                    )

                oracle_vector_store_load(docs, target_collection)

                st.session_state["chunks_count"] = len(docs)
                st.session_state["last_chunk_error"] = None
                st.success(
                    f"Loaded {len(docs)} chunks to Vector Store ({target_collection})."
                )
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
elif nav_page == "DB / Collection Inspector":
    with left:
        st.subheader("DB / Collection status")

        inspector_params = get_connection_params()
        st.write(f"**Default collection:** `{inspector_params['COLLECTION_NAME']}`")
        selected_collection = st.session_state.get("selected_collection")
        if selected_collection:
            st.write(f"**Selected collection:** `{selected_collection}`")

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
        shown_collection = st.session_state.get("collection_rows_for")
        if shown_collection:
            st.subheader(f"Documents in collection: {shown_collection}")
        else:
            st.subheader("Documents in collection")

        db_ok = st.session_state.get("db_check_ok", None)
        if db_ok is not True:
            st.info("Push **Check DB connection** in the sidebar to load documents.")
            st.stop()

        load_msg = st.session_state.get("collection_load_msg")
        if load_msg:
            # If load_msg contains "failed" you may want st.error; keep simple:
            st.caption(load_msg)

        collection_rows = st.session_state.get("collection_rows")

        if collection_rows is None:
            st.info("Choose a collection in the sidebar, then push **Show documents**.")
            st.stop()

        if not collection_rows:
            st.warning("No documents found in collection.")
        else:
            st.dataframe(collection_rows, width="stretch", hide_index=True)

            total_docs = len(collection_rows)
            total_chunks = sum(int(r.get("n_chunks", 0)) for r in collection_rows)
            st.caption(
                f"Total documents: **{total_docs}** · Total chunks: **{total_chunks}**"
            )

# ----------------------------
# DOCUMENT VIEWER PAGE
# ----------------------------
else:
    if ui_state.get("load_document_btn"):
        selected_collection = st.session_state.get("viewer_collection", COLLECTION_NAME)
        selected_document = st.session_state.get("viewer_document", "")
        try:
            with get_db_connection() as conn:
                viewer_rows = OracleVSAdmin.get_document_chunks(
                    conn, selected_collection, selected_document
                )
            st.session_state["viewer_rows"] = viewer_rows
            st.session_state["viewer_rows_for_collection"] = selected_collection
            st.session_state["viewer_rows_for_document"] = selected_document
            st.session_state["viewer_load_msg"] = (
                f"Loaded {len(viewer_rows)} chunks from {selected_collection} / "
                f"{selected_document}."
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            st.session_state["viewer_rows"] = None
            st.session_state["viewer_rows_for_collection"] = selected_collection
            st.session_state["viewer_rows_for_document"] = selected_document
            st.session_state["viewer_load_msg"] = (
                f"Load failed: {type(exc).__name__}: {exc}"
            )

    with left:
        st.subheader("Selection")
        shown_collection = st.session_state.get("viewer_rows_for_collection")
        shown_document = st.session_state.get("viewer_rows_for_document")

        if shown_collection:
            st.write(f"**Collection:** `{shown_collection}`")
        if shown_document:
            st.write(f"**Document:** `{shown_document}`")

        load_msg = st.session_state.get("viewer_load_msg")
        if load_msg:
            if "failed" in load_msg.lower():
                st.error(load_msg)
            else:
                st.success(load_msg)
        else:
            st.info("Select collection/document in the sidebar, then push **Load document**.")

        viewer_rows = st.session_state.get("viewer_rows")
        if viewer_rows:
            st.caption(f"Chunks loaded: {len(viewer_rows)}")

    with right:
        st.subheader("Document content")
        viewer_rows = st.session_state.get("viewer_rows")

        if viewer_rows is None:
            st.info("No document loaded yet.")
            st.stop()

        if not viewer_rows:
            st.warning("No chunks found for selected document.")
            st.stop()

        render_markdown = st.checkbox(
            "Render Markdown",
            value=False,
            help="Turn on if chunks contain markdown tables/headings.",
        )

        text_parts = []
        for row in viewer_rows:
            page_label = (row.get("page_label") or "").strip()
            chunk_text = row.get("text") or ""
            if page_label:
                text_parts.append(f"## Page {page_label}\n\n{chunk_text}")
            else:
                text_parts.append(chunk_text)
        full_document_text = "\n\n".join(text_parts).strip()

        if render_markdown:
            st.markdown(full_document_text)
            with st.expander("Raw text", expanded=False):
                st.text_area("Raw", full_document_text, height=260)
        else:
            st.text_area("Document", full_document_text, height=720)
