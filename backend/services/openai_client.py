"""
Azure OpenAI async client with retry logic for rate limits.

Provides embedding (ADA-002) and chat completion (GPT-4o) functionality
with automatic retry on rate limit errors.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
from openai import AsyncAzureOpenAI, RateLimitError, APIError

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import get_settings

# Setup logging
logger = logging.getLogger(__name__)


class AzureOpenAIClient:
    """
    Azure OpenAI async client wrapper with retry logic.

    Handles:
    - Text embeddings with ADA-002
    - Chat completions with GPT-4o
    - Automatic retry on rate limit errors
    """

    def __init__(self):
        """Initialize async Azure OpenAI client with credentials from settings."""
        self.settings = get_settings()

        self.client = AsyncAzureOpenAI(
            api_key=self.settings.AZURE_OPENAI_KEY,
            api_version=self.settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=self.settings.AZURE_OPENAI_ENDPOINT
        )

        logger.info("Async Azure OpenAI client initialized")

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def embed(self, text: str) -> List[float]:
        """
        Embed text using Azure OpenAI ADA-002.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector (1536 dimensions)

        Raises:
            RateLimitError: If rate limit exceeded after retries
            APIError: If API error occurs after retries
        """
        try:
            response = await self.client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )

            embedding = response.data[0].embedding
            logger.debug(f"Successfully embedded text (length: {len(text)} chars)")

            return embedding

        except RateLimitError as e:
            logger.warning(f"Rate limit hit during embedding: {e}")
            raise
        except APIError as e:
            logger.error(f"API error during embedding: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during embedding: {e}")
            raise

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Generate chat completion using Azure OpenAI GPT-4o.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response

        Returns:
            Dictionary with:
                - content: The response text
                - tokens_used: Total tokens consumed
                - finish_reason: Completion finish reason

        Raises:
            RateLimitError: If rate limit exceeded after retries
            APIError: If API error occurs after retries
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            finish_reason = response.choices[0].finish_reason

            logger.debug(
                f"Chat completion successful: {tokens_used} tokens, "
                f"finish_reason: {finish_reason}"
            )

            return {
                "content": content,
                "tokens_used": tokens_used,
                "finish_reason": finish_reason
            }

        except RateLimitError as e:
            logger.warning(f"Rate limit hit during chat completion: {e}")
            raise
        except APIError as e:
            logger.error(f"API error during chat completion: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during chat completion: {e}")
            raise

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts in a single API call.

        More efficient than calling embed() multiple times.

        Args:
            texts: List of texts to embed (max 100)

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If more than 100 texts provided
            RateLimitError: If rate limit exceeded after retries
            APIError: If API error occurs after retries
        """
        if len(texts) > 100:
            raise ValueError("Maximum 100 texts per batch")

        try:
            response = await self.client.embeddings.create(
                input=texts,
                model="text-embedding-ada-002"
            )

            embeddings = [item.embedding for item in response.data]
            logger.debug(f"Successfully embedded {len(texts)} texts in batch")

            return embeddings

        except RateLimitError as e:
            logger.warning(f"Rate limit hit during batch embedding: {e}")
            raise
        except APIError as e:
            logger.error(f"API error during batch embedding: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during batch embedding: {e}")
            raise


# Singleton instance
_openai_client: AzureOpenAIClient = None


def get_openai_client() -> AzureOpenAIClient:
    """
    Get or create the singleton OpenAI client instance.

    Returns:
        AzureOpenAIClient instance
    """
    global _openai_client
    if _openai_client is None:
        _openai_client = AzureOpenAIClient()
    return _openai_client
