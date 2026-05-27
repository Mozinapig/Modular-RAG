"""Azure OpenAI Embedding provider implementation."""

from typing import List, Optional, Any
from openai import AzureOpenAI
from src.libs.embedding.base_embedding import BaseEmbedding
from src.core.settings import EmbeddingSettings


class AzureEmbedding(BaseEmbedding):
    """Azure OpenAI Embedding provider implementation."""

    def __init__(self, settings: EmbeddingSettings):
        """Initialize Azure OpenAI Embedding with settings."""
        self.settings = settings
        self.client = None

    def _ensure_client(self) -> None:
        """Lazy initialize the Azure OpenAI client."""
        if self.client is None:
            self.client = AzureOpenAI(
                api_key=self.settings.api_key,
                api_version="2024-02-15-preview",
                azure_endpoint=self.settings.azure_endpoint,
            )

    def validate_config(self) -> None:
        """Validate Azure OpenAI configuration."""
        if not self.settings.api_key or self.settings.api_key.strip() == "":
            raise ValueError("Azure api_key is required")
        if not self.settings.model or self.settings.model.strip() == "":
            raise ValueError("Azure model is required")
        if not self.settings.azure_endpoint or self.settings.azure_endpoint.strip() == "":
            raise ValueError("Azure azure_endpoint is required")

    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using Azure OpenAI API.

        Args:
            texts: List of text strings to embed
            trace: Optional TraceContext for tracking

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If input validation fails
            RuntimeError: If Azure OpenAI API call fails
        """
        # Validate input
        if not texts:
            raise ValueError("texts list cannot be empty")

        # Ensure client is initialized
        self._ensure_client()

        try:
            # Call Azure OpenAI API
            response = self.client.embeddings.create(
                model=self.settings.model,
                input=texts,
            )

            # Extract embeddings from response
            embeddings = [item.embedding for item in response.data]

            return embeddings

        except Exception as e:
            raise RuntimeError(f"Azure Embedding error (azure): {str(e)}") from e
