"""
Embedding utilities for the RAG pipeline.

Wraps Anthropic's Voyage embedding models via LangChain.
Handles batching, retries, and token counting.
"""

import time
import logging
from typing import List
from functools import lru_cache

import tiktoken
from langchain_anthropic import AnthropicEmbeddings

logger = logging.getLogger(__name__)

# Voyage-3 supports up to 32k tokens per request
VOYAGE_MAX_TOKENS = 32_000
VOYAGE_BATCH_SIZE = 128  # Max docs per API call


def get_embeddings_model(api_key: str, model: str = "voyage-3") -> AnthropicEmbeddings:
    """
    Return a configured AnthropicEmbeddings instance.

    Args:
        api_key: Anthropic API key
        model: Voyage model name (default: voyage-3)

    Returns:
        AnthropicEmbeddings instance ready for use
    """
    return AnthropicEmbeddings(
        model=model,
        anthropic_api_key=api_key,
    )


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """
    Estimate token count using tiktoken.
    Used to check if a chunk is within embedding limits.

    Args:
        text: Input text
        model: Tiktoken encoding to use

    Returns:
        Approximate token count
    """
    try:
        enc = tiktoken.get_encoding(model)
        return len(enc.encode(text))
    except Exception:
        # Fallback: rough estimate (1 token ≈ 4 chars)
        return len(text) // 4


def split_into_batches(texts: List[str], batch_size: int = VOYAGE_BATCH_SIZE) -> List[List[str]]:
    """
    Split a list of texts into batches for embedding.

    Args:
        texts: List of text strings to embed
        batch_size: Maximum items per batch

    Returns:
        List of batches (each batch is a list of strings)
    """
    return [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]


def embed_with_retry(
    embeddings_model: AnthropicEmbeddings,
    texts: List[str],
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> List[List[float]]:
    """
    Embed a list of texts with exponential backoff retry.

    Args:
        embeddings_model: Configured AnthropicEmbeddings instance
        texts: Texts to embed
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries (doubles each attempt)

    Returns:
        List of embedding vectors

    Raises:
        RuntimeError: If all retries are exhausted
    """
    batches = split_into_batches(texts)
    all_embeddings = []

    for batch_idx, batch in enumerate(batches):
        logger.info(f"[Embeddings] Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} texts)")

        for attempt in range(max_retries):
            try:
                vectors = embeddings_model.embed_documents(batch)
                all_embeddings.extend(vectors)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Failed to embed batch {batch_idx + 1} after {max_retries} attempts: {e}"
                    )
                wait = retry_delay * (2 ** attempt)
                logger.warning(f"[Embeddings] Attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)

    return all_embeddings


def validate_chunk_size(text: str, max_tokens: int = VOYAGE_MAX_TOKENS) -> bool:
    """
    Check if a text chunk is within the embedding model's token limit.

    Args:
        text: Text to validate
        max_tokens: Maximum allowed tokens

    Returns:
        True if within limit, False otherwise
    """
    token_count = count_tokens(text)
    if token_count > max_tokens:
        logger.warning(
            f"[Embeddings] Chunk exceeds token limit: {token_count} > {max_tokens} tokens. "
            f"Consider reducing CHUNK_SIZE in your .env"
        )
        return False
    return True
