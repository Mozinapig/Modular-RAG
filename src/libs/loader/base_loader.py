"""
Abstract base class for document loaders.
Defines the contract for loading documents from various sources.
"""

from abc import ABC, abstractmethod

from src.core.types import Document


class BaseLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, path: str) -> Document:
        """
        Load a document from the given path.

        Args:
            path: Path to the document file

        Returns:
            Document object with text and metadata

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file format is not supported
        """
        pass
