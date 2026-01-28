"""
Author: Luigi Saetta
Date last modified: 2026-01-14
Python Version: 3.11
License: MIT

Description:
    Configuration settings
"""

DEBUG = False

# General OCI
AUTH = "API_KEY"

# LLM
# for tests
LLM_MODEL_ID = "openai.gpt-oss-120b"
# LLM_MODEL_ID = "cohere.command-a-03-2025"
TEMPERATURE = 0.0
MAX_TOKENS = 8000

# embeddings
# EMBED_MODEL_ID = "openai.text-embedding-3-large"
# EMBED_MODEL_ID = "cohere.embed-multilingual-v3.0"
EMBED_MODEL_ID = "cohere.embed-v4.0"

# REGION = "eu-frankfurt-1"
REGION = "us-chicago-1"
SERVICE_ENDPOINT = f"https://inference.generativeai.{REGION}.oci.oraclecloud.com"

# Chunking parameters
CHUNK_SIZE = 2048
CHUNK_OVERLAP = 100

# section for scanned pdf loading
COLLECTION_NAME = "CIGDOCS01"
