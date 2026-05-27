"""Ollama LLM provider implementation."""

from typing import List, Optional, Any
import requests
from src.libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse
from src.core.settings import LLMSettings


class OllamaLLM(BaseLLM):
    """Ollama LLM provider implementation (local HTTP endpoint)."""

    def __init__(self, settings: LLMSettings):
        """Initialize Ollama LLM with settings."""
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

    def chat(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> ChatResponse:
        """
        Send a chat request to Ollama API.

        Args:
            messages: List of chat messages
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            trace: Optional TraceContext for tracking

        Returns:
            ChatResponse object

        Raises:
            ValueError: If input validation fails
            RuntimeError: If Ollama API call fails
        """
        # Validate input
        if not messages:
            raise ValueError("messages list cannot be empty")

        # Convert ChatMessage to dict format for Ollama API
        api_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        try:
            # Prepare request parameters
            base_url = self._get_base_url()
            request_params = {
                "model": self.settings.model,
                "messages": api_messages,
                "temperature": temperature or self.settings.temperature,
                "stream": False,
            }

            if max_tokens:
                request_params["num_predict"] = max_tokens
            elif self.settings.max_tokens:
                request_params["num_predict"] = self.settings.max_tokens

            # Call Ollama API
            response = requests.post(
                f"{base_url}/api/chat",
                json=request_params,
                timeout=30
            )
            response.raise_for_status()

            # Extract response content
            response_data = response.json()
            content = response_data["message"]["content"]

            return ChatResponse(
                content=content,
                model=response_data.get("model", self.settings.model),
                finish_reason="stop",
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            )

        except Exception as e:
            raise RuntimeError(f"Ollama API error (ollama): {str(e)}") from e
