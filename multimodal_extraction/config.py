"""
Author: Luigi Saetta
Date last modified: 2026-01-28
Python Version: 3.11
License: MIT

Description:
    Configuration settings
"""

DEBUG = False

# General OCI
AUTH = "API_KEY"

# LLM configs
# MODEL_ID defined from the UI selection

#

DEFAULT_MODEL_ID = "google.gemini-2.5-pro"
TEMPERATURE = 0.0
MAX_TOKENS = 4000
# for Gemini
MAX_OUTPUT_TOKENS = 32000

# embeddings
EMBED_MODEL_ID = "cohere.embed-v4.0"

# REGION = "eu-frankfurt-1"
REGION = "us-chicago-1"
SERVICE_ENDPOINT = f"https://inference.generativeai.{REGION}.oci.oraclecloud.com"

# list  of models available to the app
# you can choose the one you prefer in the app UI

# ---- Configure your available models here ----
# check that you have the right to use the models you list here
# license, availability in your tenant, etc.

if REGION == "us-chicago-1":
    MODEL_IDS = [
        "openai.gpt-5.2",
        "meta.llama-4-maverick-17b-128e-instruct-fp8",
        "google.gemini-2.5-pro",
        "google.gemini-2.5-flash",
        "xai.grok-4-1-fast-non-reasoning",
        "cohere.command-a-vision",
        # add others you support via get_llm(...)
    ]
else:
    # FRA
    MODEL_IDS = [
        "openai.gpt-5.2",
        "google.gemini-2.5-pro",
        "google.gemini-2.5-flash",
        "cohere.command-a-vision",
        # add others you support via get_llm(...)
    ]

# docling and post processing cleanup
DOCLING_ENABLED = True
DOCLING_TIMEOUT_SEC = 180
DOCLING_FALLBACK_TO_PYPDF = True
DOCLING_MAX_CHUNK_PAGES = 32
ENABLE_CLEANUP = True

# Optional model-to-model OCR comparison (off by default)
ENABLE_MODEL_COMPARISON = False
REFERENCE_MODEL_ID = "openai.gpt-5.2"
MODEL_COMPARISON_CACHE_DIR = "./outputs/model_comparison_cache"

# Chunking parameters, you can change from UI
CHUNK_SIZE = 2048
CHUNK_OVERLAP = 100

# section for scanned pdf loading
COLLECTION_NAME = "COLL01"
# COLLECTION_NAME = "REALE"
