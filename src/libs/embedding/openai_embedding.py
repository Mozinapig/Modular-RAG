"""OpenAI Embedding provider implementation."""

from typing import List, Optional, Any
from openai import OpenAI
from src.libs.embedding.base_embedding import BaseEmbedding
from src.core.settings import EmbeddingSettings


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI Embedding provider implementation."""

    def __init__(self, settings: EmbeddingSettings):
        """Initialize OpenAI Embedding with settings."""
        self.settings = settings
        self.client = None

    def _ensure_client(self) -> None:
        """Lazy initialize the OpenAI client."""
        if self.client is None:
            client_kwargs = {"api_key": self.settings.api_key}
            if self.settings.base_url:
                client_kwargs["base_url"] = self.settings.base_url
            self.client = OpenAI(**client_kwargs)

    def validate_config(self) -> None:
        """Validate OpenAI configuration."""
        if not self.settings.api_key or self.settings.api_key.strip() == "":
            raise ValueError("OpenAI api_key is required")
        if not self.settings.model or self.settings.model.strip() == "":
            raise ValueError("OpenAI model is required")

    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI API.

        Args:
            texts: List of text strings to embed
            trace: Optional TraceContext for tracking

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If input validation fails
            RuntimeError: If OpenAI API call fails
        """
        # Validate input
        if not texts:
            raise ValueError("texts list cannot be empty")

        # Ensure client is initialized
        self._ensure_client()

        try:
            # Call OpenAI API
            response = self.client.embeddings.create(
                model=self.settings.model,
                input=texts,
            )

            # Extract embeddings from response
            embeddings = [item.embedding for item in response.data]

            return embeddings

        except Exception as e:
            raise RuntimeError(f"OpenAI Embedding error (openai): {str(e)}") from e
