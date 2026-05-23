"""Splitter Factory for strategy routing and instantiation."""

from typing import Dict, Type

from src.libs.splitter.base_splitter import BaseSplitter


class FakeSplitter(BaseSplitter):
    """Fake Splitter implementation for testing."""

    def __init__(self, settings):
        self.settings = settings
        self.chunk_size = getattr(settings, 'chunk_size', 512)
        self.overlap = getattr(settings, 'overlap', 0)

    def split_text(self, text: str, trace=None) -> list[str]:
        """Simple split strategy: divide by chunk_size with overlap."""
        if not text or len(text) == 0:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)

            # Move start for next chunk (with overlap)
            start = end - self.overlap

        return chunks

    def validate_config(self):
        """Validate config."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")


class SplitterFactory:
    """Factory for creating Splitter instances based on settings."""

    # Provider registry mapping
    _providers: Dict[str, Type[BaseSplitter]] = {
        "fake": FakeSplitter,
    }

    def create(self, settings) -> BaseSplitter:
        """
        Create a Splitter instance based on settings.

        Args:
            settings: Settings object with provider configuration
                      (should have 'provider' attribute)

        Returns:
            BaseSplitter instance

        Raises:
            ValueError: If provider is unknown or configuration is invalid
        """
        # Validate basic settings
        if not hasattr(settings, 'provider') or not settings.provider:
            raise ValueError("Splitter provider is required")

        provider = settings.provider.lower()

        if provider not in self._providers:
            raise ValueError(
                f"Unknown splitter provider: {provider}. "
                f"Supported providers: {', '.join(self._providers.keys())}"
            )

        provider_class = self._providers[provider]
        splitter = provider_class(settings)

        # Validate provider-specific config
        splitter.validate_config()

        return splitter

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseSplitter]) -> None:
        """
        Register a new Splitter provider.

        Args:
            name: Provider name (lowercase)
            provider_class: Class that extends BaseSplitter
        """
        cls._providers[name.lower()] = provider_class
