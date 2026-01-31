"""
Minimal Gemini OCR debug script (STREAMING, clean output).

- Load a PNG image from disk
- Use the SAME OCR prompt used in the pipeline
- Call Gemini via get_llm()
- Stream ONLY the text (no LangChain object dumps)
- At end, print finish_reason if available

Author: Luigi Saetta
Python: 3.11+
"""

import argparse
import base64
import io
from pathlib import Path
from typing import Optional, Dict, Any

from PIL import Image
from langchain_core.messages import HumanMessage

from oci_models import get_llm
from prompts import build_ocr_text_prompt


def pil_to_data_url(
    img: Image.Image,
    max_side: int = 1600,
    fmt: str = "PNG",
    jpeg_quality: int = 85,
    png_compress_level: int = 6,
) -> str:
    fmt = fmt.upper()

    if fmt in ("JPEG", "JPG") and img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))

    buf = io.BytesIO()

    if fmt in ("JPEG", "JPG"):
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        mime = "image/jpeg"
    elif fmt == "PNG":
        img.save(buf, format="PNG", compress_level=png_compress_level, optimize=True)
        mime = "image/png"
    else:
        raise ValueError(f"Unsupported fmt={fmt}. Use PNG or JPEG.")

    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def stream_ocr(
    image_path: Path,
    model_id: str,
    extra_prompt: str = "",
    max_side: int = 1600,
    debug_meta: bool = False,
):
    img = Image.open(image_path)
    data_url = pil_to_data_url(img, max_side=max_side, fmt="PNG")
    prompt_text = build_ocr_text_prompt(extra_prompt=extra_prompt)

    llm = get_llm(model_id=model_id)

    msg = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    )

    finish_reason: Optional[str] = None
    last_meta: Dict[str, Any] = {}

    print("\n================ OCR OUTPUT (STREAMING) ================\n")

    full_parts = []

    for chunk in llm.stream([msg]):
        # 1) Print ONLY actual streamed text
        text_piece = getattr(chunk, "content", None)
        if text_piece:
            print(text_piece, end="", flush=True)
            full_parts.append(text_piece)

        # 2) Capture metadata if present (do NOT print unless debug_meta)
        # Many wrappers store finish info in additional_kwargs or response_metadata
        add_kw = getattr(chunk, "additional_kwargs", None) or {}
        resp_meta = getattr(chunk, "response_metadata", None) or {}

        if add_kw or resp_meta:
            last_meta = {"additional_kwargs": add_kw, "response_metadata": resp_meta}
            finish_reason = (
                add_kw.get("finish_reason")
                or resp_meta.get("finish_reason")
                or finish_reason
            )

        if debug_meta:
            # Print only meaningful metadata lines (not the whole object dump)
            if add_kw.get("finish_reason") or resp_meta.get("finish_reason"):
                print(f"\n[DEBUG] finish_reason update: {finish_reason}\n", flush=True)

    print("\n\n================ END OUTPUT ================\n")

    full_text = "".join(full_parts)
    print(f"[INFO] chars={len(full_text)}")

    if finish_reason:
        print(f"[INFO] finish_reason={finish_reason}")

    if debug_meta and last_meta:
        print(f"[DEBUG] last_meta={last_meta}")

    return full_text, finish_reason


def main():
    parser = argparse.ArgumentParser(
        description="Debug Gemini OCR on a single image (streaming, clean)"
    )
    parser.add_argument("image", type=str, help="Path to PNG image")
    parser.add_argument("--model-id", type=str, required=True, help="Gemini model id")
    parser.add_argument("--extra-prompt", type=str, default="")
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument(
        "--debug-meta",
        action="store_true",
        help="Print minimal metadata (finish_reason etc.)",
    )
    args = parser.parse_args()

    stream_ocr(
        image_path=Path(args.image),
        model_id=args.model_id,
        extra_prompt=args.extra_prompt,
        max_side=args.max_side,
        debug_meta=bool(args.debug_meta),
    )


if __name__ == "__main__":
    main()
