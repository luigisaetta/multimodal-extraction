PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

export PYTHONPATH="$PROJECT_ROOT"

python ../extract_one_page.py $HOME/Progetti/work-iren/pdf_scanned/PRG_1_Norma_UNI_EN_12186_Novembre2014.pdf 22 --text-mode vlm --model-id cohere.command-a-vision