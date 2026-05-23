"""Base Splitter interface for text splitting strategies."""

from abc import ABC, abstractmethod
from typing import List, Optional, Any


class BaseSplitter(ABC):
    """Abstract base class for text splitting strategies."""

    @abstractmethod
    def split_text(
        self,
        text: str,
        trace: Optional[Any] = None,
    ) -> List[str]:
        """
        Split text into chunks.

        Args:
            text: Text to split
            trace: Optional TraceContext for tracking

        Returns:
            List of text chunks

        Raises:
            ValueError: If input validation fails
        """
        pass

    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate splitter configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        pass
