"""
Sample 100 random pages from scanned PDFs in a directory, render each page to JPEG,
call OCI GenAI OpenAI-compatible Responses API with a multimodal model, and compute
the distribution of input/output tokens from response.usage.

Usage:
  python sample_pdf_pages_tokens.py /path/to/pdf_dir
"""

from __future__ import annotations

import argparse
import base64
import bisect
import io
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Any

import pypdfium2 as pdfium
from openai import OpenAI

from config_private import OPENAI_API_KEY

# -----------------------
# Configuration (edit as needed)
# -----------------------
BASE_URL = "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com/20231130/actions/v1"
MODEL_ID = "openai.gpt-5.2"
PROMPT = (
    "Extract all the text in the image and provide a detailed description of all images "
    "embedded in the page, if present."
)

SAMPLE_PAGES = 10
RANDOM_SEED = 42

# Rendering settings
RENDER_SCALE = 2.0  # higher => larger images, more tokens/cost
JPEG_QUALITY = 85
MAX_DIM = 2000  # downscale so max(width,height) <= MAX_DIM (None to disable)

# API call retry
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 2.0


# -----------------------
# Helpers
# -----------------------
def require_api_key() -> str:
    """
    For safety: do NOT hardcode keys in code.
    Export it as: export OCI_OPENAI_API_KEY="..."
    """
    return OPENAI_API_KEY


def list_pdfs(dir_path: Path) -> List[Path]:
    pdfs = sorted(
        [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
    )
    return pdfs


def get_pdf_page_count(pdf_path: Path) -> int:
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        return len(pdf)
    finally:
        pdf.close()


def compute_prefix_counts(pdfs: List[Path]) -> Tuple[List[int], List[int]]:
    """
    Returns:
      counts: page counts per pdf
      prefix: cumulative prefix sums (same length as pdfs), where prefix[i] = sum(counts[:i+1])
    """
    counts: List[int] = []
    prefix: List[int] = []
    running = 0
    for p in pdfs:
        n = get_pdf_page_count(p)
        counts.append(n)
        running += n
        prefix.append(running)
    return counts, prefix


def sample_global_page_indices(total_pages: int, k: int, seed: int) -> List[int]:
    random.seed(seed)
    if total_pages <= 0:
        return []
    if total_pages <= k:
        return list(range(total_pages))
    return random.sample(range(total_pages), k)


def map_global_index_to_pdf_page(
    global_idx: int, pdfs: List[Path], prefix: List[int]
) -> Tuple[Path, int]:
    """
    Given a global page index in [0, total_pages), map it to (pdf_path, page_index_in_pdf).
    """
    pdf_i = bisect.bisect_right(prefix, global_idx)
    prev = prefix[pdf_i - 1] if pdf_i > 0 else 0
    page_in_pdf = global_idx - prev
    return pdfs[pdf_i], page_in_pdf


def downscale_pil_image(img, max_dim: Optional[int]):
    if not max_dim:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= max_dim:
        return img
    scale = max_dim / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h))


def render_pdf_page_to_jpeg_bytes(pdf_path: Path, page_index: int) -> bytes:
    """
    Render a single PDF page to JPEG bytes.
    """
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf.get_page(page_index)
        try:
            pil_img = page.render(scale=RENDER_SCALE).to_pil()
            pil_img = pil_img.convert("RGB")
            pil_img = downscale_pil_image(pil_img, MAX_DIM)

            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
            return buf.getvalue()
        finally:
            page.close()
    finally:
        pdf.close()


def b64_data_url_from_jpeg_bytes(jpeg_bytes: bytes) -> str:
    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def extract_usage_tokens(
    usage_obj: Any,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Attempts to extract input_tokens, output_tokens, total_tokens from response.usage.

    The object might be a pydantic model with attributes, or a dict-like structure
    depending on SDK version.
    """

    def get_field(obj, name: str):
        if obj is None:
            return None
        if hasattr(obj, name):
            return getattr(obj, name)
        if isinstance(obj, dict):
            return obj.get(name)
        return None

    input_tokens = get_field(usage_obj, "input_tokens")
    output_tokens = get_field(usage_obj, "output_tokens")
    total_tokens = get_field(usage_obj, "total_tokens")
    return input_tokens, output_tokens, total_tokens


def percentile(values: List[int], p: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    # linear interpolation between closest ranks
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return float(xs[f])
    return xs[f] + (k - f) * (xs[c] - xs[f])


def summarize_distribution(name: str, values: List[int]) -> str:
    if not values:
        return f"{name}: (no data)"
    xs = sorted(values)
    n = len(xs)
    mean = sum(xs) / n
    med = xs[n // 2] if n % 2 == 1 else (xs[n // 2 - 1] + xs[n // 2]) / 2
    p10 = percentile(xs, 10)
    p25 = percentile(xs, 25)
    p75 = percentile(xs, 75)
    p90 = percentile(xs, 90)
    return (
        f"{name} (n={n})\n"
        f"  min={xs[0]}  p10={p10:.1f}  p25={p25:.1f}  median={med:.1f}  "
        f"p75={p75:.1f}  p90={p90:.1f}  max={xs[-1]}  mean={mean:.1f}"
    )


@dataclass
class SampleResult:
    pdf_path: Path
    page_index: int
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    ok: bool
    error: Optional[str] = None


def call_multimodal_responses_api(
    client: OpenAI,
    model: str,
    prompt: str,
    image_data_url: str,
) -> Any:
    """
    Calls the OpenAI Responses API (OpenAI-compatible) with text + image.
    Retries on transient failures.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.responses.create(
                model=model,
                store=False,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": image_data_url,
                                "detail": "high",
                            },
                        ],
                    }
                ],
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            last_exc = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC * attempt)
            else:
                raise
    raise RuntimeError("Unreachable") from last_exc


# -----------------------
# Main
# -----------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample PDF pages, call multimodal model, compute token distributions."
    )
    parser.add_argument("nome_dir", help="Directory containing scanned PDF files")
    args = parser.parse_args()

    dir_path = Path(args.nome_dir).expanduser().resolve()
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"ERROR: '{dir_path}' is not a directory.", file=sys.stderr)
        return 2

    pdfs = list_pdfs(dir_path)
    if not pdfs:
        print(f"ERROR: No PDF files found in '{dir_path}'.", file=sys.stderr)
        return 2

    print(f"Found {len(pdfs)} PDF files.")

    counts, prefix = compute_prefix_counts(pdfs)
    total_pages = prefix[-1] if prefix else 0
    if total_pages <= 0:
        print("ERROR: Total pages is 0 (are the PDFs readable?).", file=sys.stderr)
        return 2

    print(f"Total pages across all PDFs: {total_pages}")
    sampled_globals = sample_global_page_indices(total_pages, SAMPLE_PAGES, RANDOM_SEED)
    print(f"Sampling {len(sampled_globals)} pages (seed={RANDOM_SEED}).")

    api_key = require_api_key()
    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    results: List[SampleResult] = []
    for i, gidx in enumerate(sampled_globals, start=1):
        pdf_path, page_idx = map_global_index_to_pdf_page(gidx, pdfs, prefix)
        tag = f"[{i}/{len(sampled_globals)}] {pdf_path.name} page={page_idx + 1}"

        try:
            jpeg_bytes = render_pdf_page_to_jpeg_bytes(pdf_path, page_idx)
            data_url = b64_data_url_from_jpeg_bytes(jpeg_bytes)

            resp = call_multimodal_responses_api(client, MODEL_ID, PROMPT, data_url)
            in_tok, out_tok, tot_tok = extract_usage_tokens(
                getattr(resp, "usage", None)
            )

            results.append(
                SampleResult(
                    pdf_path=pdf_path,
                    page_index=page_idx,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    total_tokens=tot_tok,
                    ok=True,
                )
            )
            print(f"{tag} -> input={in_tok} output={out_tok} total={tot_tok}")

        except Exception as exc:  # pylint: disable=broad-exception-caught
            results.append(
                SampleResult(
                    pdf_path=pdf_path,
                    page_index=page_idx,
                    input_tokens=None,
                    output_tokens=None,
                    total_tokens=None,
                    ok=False,
                    error=str(exc),
                )
            )
            print(f"{tag} -> ERROR: {exc}", file=sys.stderr)

    ok_results = [
        r
        for r in results
        if r.ok and r.input_tokens is not None and r.output_tokens is not None
    ]
    in_values = [int(r.input_tokens) for r in ok_results if r.input_tokens is not None]
    out_values = [
        int(r.output_tokens) for r in ok_results if r.output_tokens is not None
    ]
    tot_values = [int(r.total_tokens) for r in ok_results if r.total_tokens is not None]

    print("\n=== Token distributions (from response.usage) ===")
    print(summarize_distribution("input_tokens", in_values))
    print(summarize_distribution("output_tokens", out_values))
    if tot_values:
        print(summarize_distribution("total_tokens", tot_values))

    failed = [r for r in results if not r.ok]
    if failed:
        print(f"\nFailures: {len(failed)} / {len(results)}", file=sys.stderr)
        # show up to first 10 failures
        for r in failed[:10]:
            print(
                f"  - {r.pdf_path.name} page={r.page_index + 1}: {r.error}",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
