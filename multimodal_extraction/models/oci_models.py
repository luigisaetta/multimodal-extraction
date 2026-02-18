"""
Author: Luigi Saetta
Date last modified: 2026-02-17
Python Version: 3.11
License: MIT

Description:
    Contains utility functions to get access to Models
    in OCI GenAI service
"""

from typing import Any

from langchain_oci import ChatOCIGenAI
from langchain_oci import OCIGenAIEmbeddings

from multimodal_extraction.config import (
    AUTH,
    EMBED_MODEL_ID,
    EMBED_SERVICE_ENDPOINT,
    DEFAULT_MODEL_ID,
    DEBUG,
    SERVICE_ENDPOINT,
    TEMPERATURE,
    MAX_TOKENS,
    DEFAULT_SEED,
    MAX_OUTPUT_TOKENS,
)
from config_private import COMPARTMENT_ID
from multimodal_extraction.utils import get_console_logger

logger = get_console_logger()


SUPPORTED_PROVIDERS = {"openai", "google", "cohere", "meta", "xai"}


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


def _build_model_kwargs(
    *,
    model_id: str,
    provider: str,
    temperature: float,
    max_tokens: int,
    seed: int | None,
) -> dict[str, Any]:
    if provider == "openai":
        kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed
        return kwargs

    if provider == "google":
        kwargs = {
            "temperature": temperature,
            # Fix for Gemini truncating outputs with low max_tokens.
            "max_tokens": MAX_OUTPUT_TOKENS,
        }
        if seed is not None:
            kwargs["seed"] = seed
        return kwargs

    if provider in {"cohere", "meta", "xai"}:
        return {"temperature": temperature, "max_tokens": max_tokens}

    raise ValueError(
        f"Unsupported model provider '{provider}' for model_id '{model_id}'. "
        f"Supported providers: {sorted(SUPPORTED_PROVIDERS)}"
    )


def get_llm(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    seed: int | None = DEFAULT_SEED,
) -> ChatOCIGenAI:
    """
    Initialize and return an instance of ChatOCIGenAI with the specified configuration.

    Returns:
        ChatOCIGenAI: An instance of the OCI GenAI language model.
    """

    provider = get_model_provider(model_id)
    _model_kwargs = _build_model_kwargs(
        model_id=model_id,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
    )

    logger.info(
        "Using endpoint: %s...",
        SERVICE_ENDPOINT
    )

    llm = ChatOCIGenAI(
        auth_type=AUTH,
        model_id=model_id,
        service_endpoint=SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
        is_stream=False,
        model_kwargs=_model_kwargs,
    )

    return llm


def get_embedding_model(*, model_id: str = EMBED_MODEL_ID) -> OCIGenAIEmbeddings:
    """
    Initialize and return an instance of OCIGenAIEmbeddings with the specified configuration.
    Returns:
        OCIGenAIEmbeddings: An instance of the OCI GenAI embeddings model.
    """
    embed_model = OCIGenAIEmbeddings(
        auth_type=AUTH,
        model_id=model_id,
        service_endpoint=EMBED_SERVICE_ENDPOINT,
        compartment_id=COMPARTMENT_ID,
    )

    if DEBUG:
        logger.info("Embedding model is: %s", model_id)
        logger.info("Embedding endpoint is: %s", EMBED_SERVICE_ENDPOINT)

    return embed_model
