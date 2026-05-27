"""LLM Factory for provider routing and instantiation."""

from typing import Dict, Type

from src.libs.llm.base_llm import BaseLLM
from src.libs.llm.base_vision_llm import BaseVisionLLM
from src.core.settings import LLMSettings, VisionLLMSettings


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


class FakeVisionLLM(BaseVisionLLM):
    """Fake Vision LLM implementation for testing."""

    def __init__(self, settings):
        self.settings = settings

    def chat_with_image(self, text, image_path=None, image_base64=None, trace=None):
        """Return a fixed fake response."""
        from src.libs.llm.base_vision_llm import VisionChatResponse

        return VisionChatResponse(
            content="Fake vision response: The image shows [mock analysis]",
            model=self.settings.model,
            finish_reason="stop",
            usage={"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
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

    # Vision LLM provider registry
    _vision_providers: Dict[str, Type[BaseVisionLLM]] = {
        "fake": FakeVisionLLM,
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

    @classmethod
    def _register_default_vision_providers(cls) -> None:
        """Register default vision LLM providers (lazy loading to avoid import errors)."""
        if "openai" not in cls._vision_providers:
            from src.libs.llm.openai_vision_llm import OpenAIVisionLLM
            cls._vision_providers["openai"] = OpenAIVisionLLM

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

    @classmethod
    def register_vision_provider(cls, name: str, provider_class: Type[BaseVisionLLM]) -> None:
        """
        Register a new Vision LLM provider.

        Args:
            name: Provider name (lowercase)
            provider_class: Class that extends BaseVisionLLM
        """
        cls._vision_providers[name.lower()] = provider_class

    def create_vision_llm(self, settings: VisionLLMSettings) -> BaseVisionLLM:
        """
        Create a Vision LLM instance based on settings.

        Args:
            settings: VisionLLMSettings object with provider configuration

        Returns:
            BaseVisionLLM instance

        Raises:
            ValueError: If provider is unknown or configuration is invalid
        """
        # Register default providers on first use
        self._register_default_vision_providers()

        # Validate basic settings
        if not settings.provider:
            raise ValueError("Vision LLM provider is required")
        if not settings.model:
            raise ValueError("Model name is required")
        if not settings.api_key or settings.api_key.strip() == "":
            raise ValueError("Vision LLM api_key is required")

        provider = settings.provider.lower()

        if provider not in self._vision_providers:
            raise ValueError(
                f"Unknown Vision LLM provider: {provider}. "
                f"Supported providers: {', '.join(self._vision_providers.keys())}"
            )

        provider_class = self._vision_providers[provider]
        vision_llm = provider_class(settings)

        # Validate provider-specific config
        vision_llm.validate_config()

        return vision_llm
