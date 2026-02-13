#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1


python -m scripts.ingest.extract_one_page $HOME/Progetti/work-iren/pdf_scanned/PRG_1_Norma_UNI_EN_12186_Novembre2014.pdf 22 --text-mode vlm --model-id meta.llama-4-maverick-17b-128e-instruct-fp8
