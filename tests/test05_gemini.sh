PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

export PYTHONPATH="$PROJECT_ROOT"

python ../extract_one_page.py $HOME/Progetti/work-iren/pdf_scanned/PRG_1_UNI_10576_Luglio2018.pdf 11 --text-mode vlm --model-id google.gemini-2.5-pro