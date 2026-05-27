"""LLM Factory for provider routing and instantiation."""

from typing import Dict, Type

from src.libs.llm.base_llm import BaseLLM
from src.core.settings import LLMSettings


class FakeLLM(BaseLLM):
    """Fake LLM implementation for testing."""

    def __init__(self, settings: LLMSettings):
        self.settings = settings
        self.temperature = settings.temperature

    def chat(self, messages, temperature=None, max_tokens=None, trace=None):
        """Return a fixed fake response."""
        from src.libs.llm.base_llm import ChatResponse

        return ChatResponse(
            content="Fake LLM response",
            model=self.settings.model,
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    def validate_config(self):
        """Validate config (no-op for fake)."""
        if not self.settings.model:
            raise ValueError("Model name is required")


class LLMFactory:
    """Factory for creating LLM instances based on settings."""

    # Provider registry mapping
    _providers: Dict[str, Type[BaseLLM]] = {
        "fake": FakeLLM,
    }

    @classmethod
    def _register_default_providers(cls) -> None:
        """Register default providers (lazy loading to avoid import errors)."""
        if "openai" not in cls._providers:
            from src.libs.llm.openai_llm import OpenAILLM
            cls._providers["openai"] = OpenAILLM

        if "azure" not in cls._providers:
            from src.libs.llm.azure_llm import AzureLLM
            cls._providers["azure"] = AzureLLM

        if "deepseek" not in cls._providers:
            from src.libs.llm.deepseek_llm import DeepSeekLLM
            cls._providers["deepseek"] = DeepSeekLLM

        if "ollama" not in cls._providers:
            from src.libs.llm.ollama_llm import OllamaLLM
            cls._providers["ollama"] = OllamaLLM

    def create(self, settings: LLMSettings) -> BaseLLM:
        """
        Create an LLM instance based on settings.

        Args:
            settings: LLMSettings object with provider configuration

        Returns:
            BaseLLM instance

        Raises:
            ValueError: If provider is unknown or configuration is invalid
        """
        # Register default providers on first use
        self._register_default_providers()

        # Validate basic settings
        if not settings.provider:
            raise ValueError("LLM provider is required")
        if not settings.model:
            raise ValueError("LLM model is required")
        if not settings.api_key or settings.api_key.strip() == "":
            raise ValueError("LLM api_key is required")

        provider = settings.provider.lower()

        if provider not in self._providers:
            raise ValueError(
                f"Unknown LLM provider: {provider}. "
                f"Supported providers: {', '.join(self._providers.keys())}"
            )

        provider_class = self._providers[provider]
        llm = provider_class(settings)

        # Validate provider-specific config
        llm.validate_config()

        return llm

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseLLM]) -> None:
        """
        Register a new LLM provider.

        Args:
            name: Provider name (lowercase)
            provider_class: Class that extends BaseLLM
        """
        cls._providers[name.lower()] = provider_class
