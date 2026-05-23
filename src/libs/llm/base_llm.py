"""Base LLM interface for provider abstraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class ChatMessage:
    """Represents a chat message."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ChatResponse:
    """Represents a chat response from an LLM."""
    content: str
    model: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None  # {prompt_tokens, completion_tokens, total_tokens}


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def chat(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> ChatResponse:
        """
        Send a chat request to the LLM.

        Args:
            messages: List of chat messages
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            trace: Optional TraceContext for tracking

        Returns:
            ChatResponse object

        Raises:
            ValueError: If input validation fails
            RuntimeError: If LLM call fails
        """
        pass

    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate provider configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        pass
