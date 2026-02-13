#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

python -m scripts.ingest.extract_one_page ../work-iren/pdf_scanned/PRG_1_Norma_UNI_EN_12186_Novembre2014.pdf 5 --text-mode vlm --model-id google.gemini-2.5-pro
