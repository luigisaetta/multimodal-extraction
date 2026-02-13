#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

python -m scripts.ingest.extract_one_page ../work-iren/pdf_scanned/PRG_1_UNI_7133-4_Maggio2019.pdf 1 --text-mode vlm --model-id google.gemini-2.5-flash
echo "...."
python -m scripts.ingest.extract_one_page ../work-iren/pdf_scanned/PRG_1_UNI_7133-4_Maggio2019.pdf 2 --text-mode vlm --model-id google.gemini-2.5-flash
echo "...."
python -m scripts.ingest.extract_one_page ../work-iren/pdf_scanned/PRG_1_UNI_7133-4_Maggio2019.pdf 3 --text-mode vlm --model-id google.gemini-2.5-flash
echo "...."
python -m scripts.ingest.extract_one_page ../work-iren/pdf_scanned/PRG_1_UNI_7133-4_Maggio2019.pdf 4 --text-mode vlm --model-id google.gemini-2.5-flash
echo "...."
python -m scripts.ingest.extract_one_page ../work-iren/pdf_scanned/PRG_1_UNI_7133-4_Maggio2019.pdf 5 --text-mode vlm --model-id google.gemini-2.5-flash
echo "...."
python -m scripts.ingest.extract_one_page ../work-iren/pdf_scanned/PRG_1_UNI_7133-4_Maggio2019.pdf 6 --text-mode vlm --model-id google.gemini-2.5-flash
echo "...."
python -m scripts.ingest.extract_one_page ../work-iren/pdf_scanned/PRG_1_UNI_7133-4_Maggio2019.pdf 7 --text-mode vlm --model-id google.gemini-2.5-flash
echo "...."
python -m scripts.ingest.extract_one_page ../work-iren/pdf_scanned/PRG_1_UNI_7133-4_Maggio2019.pdf 8 --text-mode vlm --model-id google.gemini-2.5-flash
echo "...."
python -m scripts.ingest.extract_one_page ../work-iren/pdf_scanned/PRG_1_UNI_7133-4_Maggio2019.pdf 9 --text-mode vlm --model-id google.gemini-2.5-flash
echo "...."
python -m scripts.ingest.extract_one_page ../work-iren/pdf_scanned/PRG_1_UNI_7133-4_Maggio2019.pdf 10 --text-mode vlm --model-id google.gemini-2.5-flash
echo "...."
