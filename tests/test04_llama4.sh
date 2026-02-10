PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

export PYTHONPATH="$PROJECT_ROOT"

python ../extract_one_page.py $HOME/Progetti/work-iren/pdf_scanned/PRG_1_021-LG-CIG-2024.pdf 8 --text-mode vlm --model-id meta.llama-4-maverick-17b-128e-instruct-fp8