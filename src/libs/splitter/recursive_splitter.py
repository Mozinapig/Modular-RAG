"""Recursive Splitter implementation using LangChain."""

from typing import List, Optional, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.libs.splitter.base_splitter import BaseSplitter


class RecursiveSplitter(BaseSplitter):
    """Recursive text splitter that respects document structure."""

    def __init__(self, settings):
        """
        Initialize RecursiveSplitter.

        Args:
            settings: Settings object with configuration
                     - chunk_size: Maximum chunk size (default: 512)
                     - chunk_overlap: Overlap between chunks (default: 50)
                     - separators: List of separators to try (default: ["\n\n", "\n", " ", ""])
        """
        self.settings = settings
        self.chunk_size = getattr(settings, 'chunk_size', 512)
        self.chunk_overlap = getattr(settings, 'chunk_overlap', 50)
        self.separators = getattr(settings, 'separators', ["\n\n", "\n", " ", ""])

        # Validate before creating the underlying splitter
        self.validate_config()

        # Create the underlying LangChain splitter
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

    def split_text(
        self,
        text: str,
        trace: Optional[Any] = None,
    ) -> List[str]:
        """
        Split text into chunks respecting document structure.

        Args:
            text: Text to split
            trace: Optional TraceContext for tracking

        Returns:
            List of text chunks

        Raises:
            ValueError: If input validation fails
        """
        if not isinstance(text, str):
            raise ValueError("Input text must be a string")

        if not text:
            return []

        # Use LangChain's splitter
        chunks = self._splitter.split_text(text)

        return chunks

    def validate_config(self) -> None:
        """
        Validate splitter configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        if not isinstance(self.separators, list) or len(self.separators) == 0:
            raise ValueError("separators must be a non-empty list")
