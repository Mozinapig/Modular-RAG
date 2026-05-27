"""Azure OpenAI LLM provider implementation."""

from typing import List, Optional, Any
from openai import AzureOpenAI
from src.libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse
from src.core.settings import LLMSettings


class AzureLLM(BaseLLM):
    """Azure OpenAI LLM provider implementation."""

    def __init__(self, settings: LLMSettings):
        """Initialize Azure OpenAI LLM with settings."""
        self.settings = settings
        self.client = None

    def _ensure_client(self) -> None:
        """Lazy initialize the Azure OpenAI client."""
        if self.client is None:
            self.client = AzureOpenAI(
                api_key=self.settings.api_key,
                api_version=self.settings.api_version or "2024-02-15-preview",
                azure_endpoint=self.settings.azure_endpoint,
            )

    def validate_config(self) -> None:
        """Validate Azure OpenAI configuration."""
        if not self.settings.api_key or self.settings.api_key.strip() == "":
            raise ValueError("Azure api_key is required")
        if not self.settings.model or self.settings.model.strip() == "":
            raise ValueError("Azure model is required")
        if not self.settings.azure_endpoint or self.settings.azure_endpoint.strip() == "":
            raise ValueError("azure_endpoint is required")
        if not self.settings.azure_deployment or self.settings.azure_deployment.strip() == "":
            raise ValueError("azure_deployment is required")

    def chat(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> ChatResponse:
        """
        Send a chat request to Azure OpenAI API.

        Args:
            messages: List of chat messages
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            trace: Optional TraceContext for tracking

        Returns:
            ChatResponse object

        Raises:
            ValueError: If input validation fails
            RuntimeError: If Azure OpenAI API call fails
        """
        # Validate input
        if not messages:
            raise ValueError("messages list cannot be empty")

        # Ensure client is initialized
        self._ensure_client()

        # Convert ChatMessage to dict format for Azure OpenAI API
        api_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        try:
            # Prepare request parameters
            request_params = {
                "model": self.settings.model,
                "deployment_id": self.settings.azure_deployment,
                "messages": api_messages,
                "temperature": temperature or self.settings.temperature,
            }

            if max_tokens:
                request_params["max_tokens"] = max_tokens
            elif self.settings.max_tokens:
                request_params["max_tokens"] = self.settings.max_tokens

            # Call Azure OpenAI API
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
            raise RuntimeError(f"Azure OpenAI API error (azure): {str(e)}") from e

