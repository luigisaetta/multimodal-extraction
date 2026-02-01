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

# list  of models available to the app
# you can choose the one you prefer in the app UI

# ---- Configure your available models here ----
# check that you have the right to use the models you list here
# license, availability in your tenant, etc.

MODEL_IDS = [
    "openai.gpt-5.2",
    "meta.llama-4-maverick-17b-128e-instruct-fp8",
    "google.gemini-2.5-pro",
    "xai.grok-4-1-fast-non-reasoning",
    "cohere.command-a-vision v1.0",
    # add others you support via get_llm(...)
]

DEFAULT_MODEL_ID = "google.gemini-2.5-pro"
TEMPERATURE = 0.0
MAX_TOKENS = 8000

# embeddings
EMBED_MODEL_ID = "cohere.embed-v4.0"

# REGION = "eu-frankfurt-1"
REGION = "us-chicago-1"
SERVICE_ENDPOINT = f"https://inference.generativeai.{REGION}.oci.oraclecloud.com"

DOCKLING_ENABLED = True

# Chunking parameters, you can change from UI
CHUNK_SIZE = 2048
CHUNK_OVERLAP = 100

# section for scanned pdf loading
# COLLECTION_NAME = "CIGDOCS01"
COLLECTION_NAME = "BOOKS"
