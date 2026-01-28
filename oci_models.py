"""
Author: Luigi Saetta
Date last modified: 2026-01-14
Python Version: 3.11
License: MIT

Description:
    Contains utility functions to get access to Models
    in OCI GenAI service
"""

from langchain_oci import ChatOCIGenAI
from langchain_oci import OCIGenAIEmbeddings

from config import (
    AUTH,
    EMBED_MODEL_ID,
    LLM_MODEL_ID,
    DEBUG,
    SERVICE_ENDPOINT,
    TEMPERATURE,
    MAX_TOKENS,
)
from config_private import COMPARTMENT_ID
from utils import get_console_logger

logger = get_console_logger()


def get_model_provider(model_name: str) -> str:
    """
    Extract provider name from an OCI-style model identifier.

    Examples:
        "openai.gpt-oss-120b" -> "openai"
        "cohere.command-r-plus" -> "cohere"
        "meta.llama-3.1-70b" -> "meta"

    Fallback:
        - returns "unknown" if input is invalid or malformed
    """
    if not model_name or not isinstance(model_name, str):
        return "unknown"

    model_name = model_name.strip()
    if "." not in model_name:
        return "unknown"

    provider = model_name.split(".", 1)[0].strip().lower()
    return provider if provider else "unknown"


def get_llm(model_id=LLM_MODEL_ID, temperature=TEMPERATURE, max_tokens=MAX_TOKENS):
    """
    Initialize and return an instance of ChatOCIGenAI with the specified configuration.

    Returns:
        ChatOCIGenAI: An instance of the OCI GenAI language model.
    """

    # identify the provider
    provider = get_model_provider(model_id)

    if provider == "openai":
        _model_kwargs = {
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
    elif provider == "gemini":
        # fix for Gemini truncating outputs if using "max_tokens"
        _model_kwargs = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
    else:
        _model_kwargs = {"temperature": temperature, "max_tokens": max_tokens}

    logger.info("Using endpoint: %s...", SERVICE_ENDPOINT)

    llm = ChatOCIGenAI(
        auth_type=AUTH,
        model_id=model_id,
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
        is_stream=False,
        model_kwargs=_model_kwargs,
    )

    return llm


def get_embedding_model(model_id=EMBED_MODEL_ID):
    """
    Initialize and return an instance of OCIGenAIEmbeddings with the specified configuration.
    Returns:
        OCIGenAIEmbeddings: An instance of the OCI GenAI embeddings model.
    """
    embed_model = None

    embed_model = OCIGenAIEmbeddings(
        auth_type=AUTH,
        model_id=model_id,
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
    )

    if DEBUG:
        logger.info("Embedding model is: %s", EMBED_MODEL_ID)

    return embed_model
