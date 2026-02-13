# Auto mode (try Docling/pypdf, fallback to VLM if weak) + figures ON by default
python -m scripts.ingest.extract_one_page ./docs/my.pdf 19 --model-id openai.gpt-5.2

# Force VLM OCR for text (always multimodal) + figures
python -m scripts.ingest.extract_one_page ./docs/scanned.pdf 24 --text-mode vlm --model-id openai.gpt-5.2

# Disable figures if you want only text
python -m scripts.ingest.extract_one_page ./docs/my.pdf 10 --no-describe-figures
