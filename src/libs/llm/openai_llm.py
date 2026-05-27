"""OpenAI LLM provider implementation."""

from typing import List, Optional, Any
from openai import OpenAI
from src.libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse
from src.core.settings import LLMSettings


class OpenAILLM(BaseLLM):
    """OpenAI LLM provider implementation."""

    def __init__(self, settings: LLMSettings):
        """Initialize OpenAI LLM with settings."""
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

    def chat(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> ChatResponse:
        """
        Send a chat request to OpenAI API.

        Args:
            messages: List of chat messages
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            trace: Optional TraceContext for tracking

        Returns:
            ChatResponse object

        Raises:
            ValueError: If input validation fails
            RuntimeError: If OpenAI API call fails
        """
        # Validate input
        if not messages:
            raise ValueError("messages list cannot be empty")

        # Ensure client is initialized
        self._ensure_client()

        # Convert ChatMessage to dict format for OpenAI API
        api_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        try:
            # Prepare request parameters
            request_params = {
                "model": self.settings.model,
                "messages": api_messages,
                "temperature": temperature or self.settings.temperature,
            }

            if max_tokens:
                request_params["max_tokens"] = max_tokens
            elif self.settings.max_tokens:
                request_params["max_tokens"] = self.settings.max_tokens

            # Call OpenAI API
            response = self.client.chat.completions.create(**request_params)

            # Extract response content
            content = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            return ChatResponse(
                content=content,
                model=response.model,
                finish_reason=response.choices[0].finish_reason,
                usage=usage,
            )

        except Exception as e:
            raise RuntimeError(f"OpenAI API error (openai): {str(e)}") from e

