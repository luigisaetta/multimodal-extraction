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
DEFAULT_MODEL_ID = "google.gemini-2.5-pro"
TEMPERATURE = 0.0
MAX_TOKENS = 8000

# embeddings
EMBED_MODEL_ID = "cohere.embed-v4.0"

# REGION = "eu-frankfurt-1"
REGION = "us-chicago-1"
SERVICE_ENDPOINT = f"https://inference.generativeai.{REGION}.oci.oraclecloud.com"

# Chunking parameters, you can change from UI
CHUNK_SIZE = 2048
CHUNK_OVERLAP = 100

# section for scanned pdf loading
COLLECTION_NAME = "CIGDOCS01"
