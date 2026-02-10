PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

export PYTHONPATH="$PROJECT_ROOT"

python ../extract_one_page.py $HOME/Progetti/work-iren/pdf_scanned/PRG_1_UNI_10576_Luglio2018.pdf 14 --text-mode vlm --model-id meta.llama-4-maverick-17b-128e-instruct-fp8