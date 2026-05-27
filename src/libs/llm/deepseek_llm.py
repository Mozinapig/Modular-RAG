"""DeepSeek LLM provider implementation."""

from typing import List, Optional, Any
from openai import OpenAI
from src.libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse
from src.core.settings import LLMSettings


class DeepSeekLLM(BaseLLM):
    """DeepSeek LLM provider implementation (OpenAI-compatible API)."""

    def __init__(self, settings: LLMSettings):
        """Initialize DeepSeek LLM with settings."""
        self.settings = settings
        self.client = None

    def _ensure_client(self) -> None:
        """Lazy initialize the DeepSeek client."""
        if self.client is None:
            self.client = OpenAI(
                api_key=self.settings.api_key,
                base_url=self.settings.base_url or "https://api.deepseek.com",
            )

    def validate_config(self) -> None:
        """Validate DeepSeek configuration."""
        if not self.settings.api_key or self.settings.api_key.strip() == "":
            raise ValueError("DeepSeek api_key is required")
        if not self.settings.model or self.settings.model.strip() == "":
            raise ValueError("DeepSeek model is required")

    def chat(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> ChatResponse:
        """
        Send a chat request to DeepSeek API.

        Args:
            messages: List of chat messages
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            trace: Optional TraceContext for tracking

        Returns:
            ChatResponse object

        Raises:
            ValueError: If input validation fails
            RuntimeError: If DeepSeek API call fails
        """
        # Validate input
        if not messages:
            raise ValueError("messages list cannot be empty")

        # Ensure client is initialized
        self._ensure_client()

        # Convert ChatMessage to dict format for DeepSeek API
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

            # Call DeepSeek API
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
            raise RuntimeError(f"DeepSeek API error (deepseek): {str(e)}") from e

