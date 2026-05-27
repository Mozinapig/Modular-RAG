"""Ollama Embedding provider implementation."""

from typing import List, Optional, Any
import requests
from src.libs.embedding.base_embedding import BaseEmbedding
from src.core.settings import EmbeddingSettings


class OllamaEmbedding(BaseEmbedding):
    """Ollama Embedding provider implementation (local HTTP endpoint)."""

    def __init__(self, settings: EmbeddingSettings):
        """Initialize Ollama Embedding with settings."""
        self.settings = settings

    def _get_base_url(self) -> str:
        """Get the base URL, using default if not provided."""
        if self.settings.base_url:
            return self.settings.base_url
        return "http://localhost:11434"

    def validate_config(self) -> None:
        """Validate Ollama configuration."""
        if not self.settings.api_key or self.settings.api_key.strip() == "":
            raise ValueError("Ollama api_key is required")
        if not self.settings.model or self.settings.model.strip() == "":
            raise ValueError("Ollama model is required")

    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using Ollama API.

        Args:
            texts: List of text strings to embed
            trace: Optional TraceContext for tracking

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If input validation fails
            RuntimeError: If Ollama API call fails
        """
        # Validate input
        if not texts:
            raise ValueError("texts list cannot be empty")

        try:
            # Prepare request
            base_url = self._get_base_url()
            request_params = {
                "model": self.settings.model,
                "input": texts,
            }

            # Call Ollama API
            response = requests.post(
                f"{base_url}/api/embed",
                json=request_params,
                timeout=30
            )
            response.raise_for_status()

            # Extract embeddings from response
            response_data = response.json()
            if "embeddings" not in response_data:
                raise ValueError(
                    f"Invalid response format: missing 'embeddings' field. "
                    f"Response: {response_data}"
                )
            embeddings = response_data["embeddings"]

            if not isinstance(embeddings, list):
                raise ValueError(
                    f"Invalid response format: 'embeddings' must be a list, "
                    f"got {type(embeddings).__name__}"
                )

            return embeddings

        except Exception as e:
            raise RuntimeError(f"Ollama API error (ollama): {str(e)}") from e
