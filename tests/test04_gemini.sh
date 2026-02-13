#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1


python -m scripts.ingest.extract_one_page $HOME/Progetti/work-iren/pdf_scanned/PRG_1_021-LG-CIG-2024.pdf 8 --text-mode vlm --model-id google.gemini-2.5-pro
