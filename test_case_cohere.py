"""
Manual test cohere command-vision
Usage:
    python test_case_cohere.py path/to/image.jpg (or png)
"""

from pathlib import Path
import sys
import io
import base64
from typing import Optional

from PIL import Image
from langchain_oci import ChatOCIGenAI
from langchain_core.messages import HumanMessage

#
# Configs
#
REGION = "eu-frankfurt-1"
SERVICE_ENDPOINT = f"https://inference.generativeai.{REGION}.oci.oraclecloud.com"
COMPARTMENT_ID = "ocid1.compartment.oc1..aaaaaaaaushuwb2evpuf7rcpl4r7ugmqoe7ekmaiik3ra3m7gec3d234eknq"


def pil_to_data_url(img: Image.Image, max_side: int = 1600, quality: int = 85) -> str:
    """Convert PIL image to JPEG data URL (base64), resizing to keep payload manageable."""
    if img.mode != "RGB":
        img = img.convert("RGB")

    width, height = img.size
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        img = img.resize((int(width * scale), int(height * scale)))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def get_llm(model_id="cohere.command-a-vision", temperature=0, max_tokens=4096):
    _model_kwargs = {"temperature": temperature, "max_tokens": max_tokens}

    llm = ChatOCIGenAI(
        auth_type="API_KEY",
        model_id=model_id,
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
        is_stream=False,
        model_kwargs=_model_kwargs,
    )
    return llm


# -------------------------------------------------
# FUNCTION UNDER TEST
# -------------------------------------------------
def call_multimodal_llm_text_only(
    llm,
    page_img: Image.Image,
    extra_prompt: str,
    max_side: int,
    jpeg_quality: int,
) -> Optional[str]:
    data_url = pil_to_data_url(
        page_img,
        max_side=max_side,
        quality=jpeg_quality,
    )

    prompt_text = "Extract all the text in the image."

    msg = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    )

    try:
        res = llm.invoke([msg])
    except Exception as exc:
        print("Error extracting text: ", exc)
        return None

    return str(getattr(res, "content", res)).strip()


# -------------------------------------------------
# Test driver
# -------------------------------------------------
MODEL_NAME = "cohere.command-a-vision"  # cambia qui per Gemini / Llama
MAX_SIDE = 2000
JPEG_QUALITY = 90
EXTRA_PROMPT = ""


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python test_case_cohere.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    # Load image
    img = Image.open(image_path).convert("RGB")

    # Init LLM
    llm = get_llm(MODEL_NAME)

    # Call OCR
    text = call_multimodal_llm_text_only(
        llm=llm,
        page_img=img,
        extra_prompt=EXTRA_PROMPT,
        max_side=MAX_SIDE,
        jpeg_quality=JPEG_QUALITY,
    )

    print("\n===== OCR OUTPUT =====\n")
    print(text or "[NO OUTPUT]")


if __name__ == "__main__":
    main()
