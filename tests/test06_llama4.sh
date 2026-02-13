#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1


python -m scripts.ingest.extract_one_page $HOME/Progetti/work-iren/pdf_scanned/PRG_1_UNI_10576_Luglio2018.pdf 14 --text-mode vlm --model-id meta.llama-4-maverick-17b-128e-instruct-fp8
