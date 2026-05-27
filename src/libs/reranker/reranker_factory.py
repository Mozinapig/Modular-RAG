"""Reranker Factory for provider routing and instantiation."""

from typing import Dict, Type, List, Any, Optional

from src.libs.reranker.base_reranker import BaseReranker
from src.libs.reranker.llm_reranker import LLMReranker


class NoneReranker(BaseReranker):
    """No-op Reranker that preserves original order."""

    def __init__(self, settings):
        self.settings = settings

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Return candidates in original order (no reranking)."""
        return candidates

    def validate_config(self) -> None:
        """Validate config (no-op for NoneReranker)."""
        if not self.settings:
            raise ValueError("Settings object is required")


class RerankerFactory:
    """Factory for creating Reranker instances based on settings."""

    # Provider registry mapping
    _providers: Dict[str, Type[BaseReranker]] = {
        "none": NoneReranker,
        "llm": LLMReranker,
    }

    def create(self, settings) -> BaseReranker:
        """
        Create a Reranker instance based on settings.

        Args:
            settings: RerankerSettings object or settings dict with provider configuration

        Returns:
            BaseReranker instance

        Raises:
            ValueError: If provider is unknown or configuration is invalid
        """
        # Handle dict input - convert to RerankerSettings-like object with provider attribute
        if isinstance(settings, dict):
            # Create a simple object to hold the dict data
            provider = settings.get("backend", "none")
        else:
            # It's a RerankerSettings or similar object
            provider = getattr(settings, "backend", settings.provider if hasattr(settings, "provider") else "none")

        if not provider:
            raise ValueError("Reranker provider is required")

        provider = provider.lower()

        if provider not in self._providers:
            raise ValueError(
                f"Unknown Reranker provider: {provider}. "
                f"Supported providers: {', '.join(self._providers.keys())}"
            )

        provider_class = self._providers[provider]

        # For dict input, we need to wrap it or reconstruct the full settings
        # Pass the original settings object, not the dict
        if isinstance(settings, dict):
            from src.core.settings import RerankerSettings
            settings = RerankerSettings(**settings)

        reranker = provider_class(settings)

        # Validate provider-specific config
        reranker.validate_config()

        return reranker

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseReranker]) -> None:
        """
        Register a new Reranker provider.

        Args:
            name: Provider name (lowercase)
            provider_class: Class that extends BaseReranker
        """
        cls._providers[name.lower()] = provider_class
