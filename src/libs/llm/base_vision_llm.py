"""Base Vision LLM interface for multimodal provider abstraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class VisionChatResponse:
    """Represents a vision chat response from an LLM."""
    content: str
    model: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None  # {prompt_tokens, completion_tokens, total_tokens}


class BaseVisionLLM(ABC):
    """Abstract base class for Vision LLM providers."""

    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image_path: Optional[str] = None,
        image_base64: Optional[str] = None,
        trace: Optional[Any] = None,
    ) -> VisionChatResponse:
        """
        Send a multimodal (text + image) request to the Vision LLM.

        Args:
            text: Text prompt for the image analysis
            image_path: Path to the image file (local file path)
            image_base64: Base64-encoded image data (alternative to image_path)
            trace: Optional TraceContext for tracking

        Returns:
            VisionChatResponse object

        Raises:
            ValueError: If input validation fails (missing text or image)
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
