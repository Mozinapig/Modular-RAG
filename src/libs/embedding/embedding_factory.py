"""Embedding Factory for provider routing and instantiation."""

from typing import Dict, Type
import hashlib

from src.libs.embedding.base_embedding import BaseEmbedding
from src.core.settings import EmbeddingSettings


class FakeEmbedding(BaseEmbedding):
    """Fake Embedding implementation for testing."""

    def __init__(self, settings: EmbeddingSettings):
        self.settings = settings
        self.dimensions = settings.dimensions

    def embed(self, texts: list[str], trace=None) -> list[list[float]]:
        """Return deterministic fake embeddings based on text hash."""
        embeddings = []
        for text in texts:
            # Generate deterministic embedding based on text hash
            hash_obj = hashlib.md5(text.encode())
            hash_bytes = hash_obj.digest()

            # Use hash bytes to seed random-like values
            embedding = []
            for i in range(self.dimensions):
                byte_idx = i % len(hash_bytes)
                value = (hash_bytes[byte_idx] / 255.0) - 0.5
                embedding.append(value)

            embeddings.append(embedding)

        return embeddings

    def validate_config(self):
        """Validate config (no-op for fake)."""
        if not self.settings.model:
            raise ValueError("Model name is required")
        if self.settings.dimensions <= 0:
            raise ValueError("Dimensions must be positive")


class EmbeddingFactory:
    """Factory for creating Embedding instances based on settings."""

    # Provider registry mapping
    _providers: Dict[str, Type[BaseEmbedding]] = {
        "fake": FakeEmbedding,
    }

    @classmethod
    def _register_default_providers(cls) -> None:
        """Register default providers (lazy loading to avoid import errors)."""
        if "openai" not in cls._providers:
            from src.libs.embedding.openai_embedding import OpenAIEmbedding
            cls._providers["openai"] = OpenAIEmbedding

        if "azure" not in cls._providers:
            from src.libs.embedding.azure_embedding import AzureEmbedding
            cls._providers["azure"] = AzureEmbedding

    def create(self, settings: EmbeddingSettings) -> BaseEmbedding:
        """
        Create an Embedding instance based on settings.

        Args:
            settings: EmbeddingSettings object with provider configuration

        Returns:
            BaseEmbedding instance

        Raises:
            ValueError: If provider is unknown or configuration is invalid
        """
        # Register default providers on first use
        self._register_default_providers()

        # Validate basic settings
        if not settings.provider:
            raise ValueError("Embedding provider is required")
        if not settings.model:
            raise ValueError("Embedding model is required")
        if not settings.api_key or settings.api_key.strip() == "":
            raise ValueError("Embedding api_key is required")

        provider = settings.provider.lower()

        if provider not in self._providers:
            raise ValueError(
                f"Unknown Embedding provider: {provider}. "
                f"Supported providers: {', '.join(self._providers.keys())}"
            )

        provider_class = self._providers[provider]
        embedding = provider_class(settings)

        # Validate provider-specific config
        embedding.validate_config()

        return embedding

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseEmbedding]) -> None:
        """
        Register a new Embedding provider.

        Args:
            name: Provider name (lowercase)
            provider_class: Class that extends BaseEmbedding
        """
        cls._providers[name.lower()] = provider_class
