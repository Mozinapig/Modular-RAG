"""OpenAI-compatible Vision LLM provider implementation."""

import base64
from pathlib import Path
from typing import Optional, Any
from PIL import Image
from io import BytesIO
from openai import OpenAI
from src.libs.llm.base_vision_llm import BaseVisionLLM, VisionChatResponse
from src.core.settings import VisionLLMSettings


class OpenAIVisionLLM(BaseVisionLLM):
    """OpenAI-compatible Vision LLM provider implementation."""

    def __init__(self, settings: VisionLLMSettings):
        """Initialize OpenAI Vision LLM with settings."""
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
        """Validate OpenAI Vision configuration."""
        if not self.settings.api_key or self.settings.api_key.strip() == "":
            raise ValueError("OpenAI Vision api_key is required")
        if not self.settings.model or self.settings.model.strip() == "":
            raise ValueError("OpenAI Vision model is required")

    def _load_image_as_base64(self, image_path: str) -> str:
        """
        Load image from file path and convert to base64.

        Args:
            image_path: Path to the image file

        Returns:
            Base64-encoded image data

        Raises:
            FileNotFoundError: If image file not found
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with open(image_path, "rb") as f:
            image_data = f.read()

        return base64.standard_b64encode(image_data).decode("utf-8")

    def _compress_image(self, image_data: bytes) -> bytes:
        """
        Compress image if it exceeds max_image_size.

        Args:
            image_data: Raw image bytes

        Returns:
            Compressed image bytes (or original if smaller)
        """
        try:
            img = Image.open(BytesIO(image_data))

            max_size = self.settings.max_image_size
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            output = BytesIO()
            fmt = img.format or "PNG"
            img.save(output, format=fmt, quality=85)
            return output.getvalue()
        except Exception:
            return image_data

    def _encode_image(self, image_path: Optional[str] = None,
                      image_base64: Optional[str] = None) -> str:
        """
        Encode image to base64 with compression.

        Args:
            image_path: Path to image file (alternative to image_base64)
            image_base64: Already base64-encoded image data

        Returns:
            Base64-encoded image (possibly compressed)

        Raises:
            ValueError: If neither path nor base64 provided
        """
        if not image_path and not image_base64:
            raise ValueError("Either image_path or image_base64 must be provided")

        if image_path:
            image_data = open(image_path, "rb").read()
        else:
            image_data = base64.b64decode(image_base64)

        compressed = self._compress_image(image_data)
        return base64.standard_b64encode(compressed).decode("utf-8")

    def chat_with_image(
        self,
        text: str,
        image_path: Optional[str] = None,
        image_base64: Optional[str] = None,
        trace: Optional[Any] = None,
    ) -> VisionChatResponse:
        """
        Send a multimodal (text + image) request to OpenAI Vision API.

        Args:
            text: Text prompt for the image analysis
            image_path: Path to the image file (local file path)
            image_base64: Base64-encoded image data (alternative to image_path)
            trace: Optional TraceContext for tracking

        Returns:
            VisionChatResponse object

        Raises:
            ValueError: If input validation fails
            RuntimeError: If API call fails
        """
        # Validate inputs
        if not text or text.strip() == "":
            raise ValueError("text prompt cannot be empty")
        if not image_path and not image_base64:
            raise ValueError("Either image_path or image_base64 must be provided")

        # Ensure client is initialized
        self._ensure_client()

        try:
            # Encode image with compression
            encoded_image = self._encode_image(image_path, image_base64)

            # Determine image media type
            if image_path:
                suffix = Path(image_path).suffix.lower()
                media_type = self._get_media_type(suffix)
            else:
                media_type = "image/png"

            # Prepare message with vision content
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{encoded_image}"
                        },
                    },
                ],
            }

            # Call OpenAI Vision API
            response = self.client.chat.completions.create(
                model=self.settings.model,
                messages=[message],
                max_tokens=2048,
            )

            # Extract response content
            content = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            return VisionChatResponse(
                content=content,
                model=response.model,
                finish_reason=response.choices[0].finish_reason,
                usage=usage,
            )

        except Exception as e:
            raise RuntimeError(f"OpenAI Vision API error (openai): {str(e)}") from e

    def _get_media_type(self, file_suffix: str) -> str:
        """
        Get MIME type from file suffix.

        Args:
            file_suffix: File extension (e.g., '.png')

        Returns:
            MIME type string
        """
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return mime_types.get(file_suffix, "image/png")
