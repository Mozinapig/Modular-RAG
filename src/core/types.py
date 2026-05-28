"""
Core data types and contracts for the entire RAG pipeline.
These types are shared across ingestion, retrieval, and MCP layers.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ImageRef:
    """Reference to an image in document/chunk metadata.

    Attributes:
        id: Global unique image identifier (format: {doc_hash}_{page}_{seq})
        path: Image file storage path (convention: data/images/{collection}/{image_id}.png)
        text_offset: Starting character position of placeholder in text (0-indexed)
        text_length: Character length of placeholder (typically len("[IMAGE: {id}]"))
        page: Page number in original document (optional, for PDF etc)
        position: Physical position info in original document (optional, e.g., PDF coords)
    """
    id: str
    path: str
    text_offset: int
    text_length: int
    page: Optional[int] = None
    position: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)


@dataclass
class Document:
    """Document type for the RAG system.

    Represents a complete document that will be ingested.

    Attributes:
        id: Unique document identifier
        text: Full document text content
        metadata: Document metadata dict
            - source_path: Required, path to source file
            - images: Optional, list of ImageRef objects
            - Other fields allowed for extensibility
    """
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        meta = self.metadata.copy()
        if "images" in meta and meta["images"]:
            meta["images"] = [img.to_dict() if isinstance(img, ImageRef) else img
                            for img in meta["images"]]
        return {
            "id": self.id,
            "text": self.text,
            "metadata": meta
        }


@dataclass
class Chunk:
    """Chunk type for the RAG system.

    Represents a text chunk extracted from a document.

    Attributes:
        id: Unique chunk identifier (format: {doc_id}_{index:04d}_{hash_8chars})
        text: Chunk text content
        metadata: Chunk metadata dict
            - source_path: Required (inherited from document)
            - chunk_index: Chunk sequence number in document
            - source_ref: Parent document ID for traceability
            - image_refs: List of image IDs referenced in this chunk
            - images: List of ImageRef objects for images in this chunk
            - Other fields allowed for extensibility
        start_offset: Optional, character offset in original document
        end_offset: Optional, character offset in original document
    """
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        meta = self.metadata.copy()
        if "images" in meta and meta["images"]:
            meta["images"] = [img.to_dict() if isinstance(img, ImageRef) else img
                            for img in meta["images"]]
        result = {
            "id": self.id,
            "text": self.text,
            "metadata": meta
        }
        if self.start_offset is not None:
            result["start_offset"] = self.start_offset
        if self.end_offset is not None:
            result["end_offset"] = self.end_offset
        return result


@dataclass
class ChunkRecord:
    """ChunkRecord type for storage and retrieval.

    Extends Chunk with vector representations for storage in vector databases.

    Attributes:
        id: Unique chunk identifier
        text: Chunk text content
        metadata: Chunk metadata dict (same as Chunk)
        dense_vector: Optional, dense embedding vector (list of floats)
        sparse_vector: Optional, sparse term weights dict (term -> weight)
    """
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    dense_vector: Optional[List[float]] = None
    sparse_vector: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        meta = self.metadata.copy()
        if "images" in meta and meta["images"]:
            meta["images"] = [img.to_dict() if isinstance(img, ImageRef) else img
                            for img in meta["images"]]
        result = {
            "id": self.id,
            "text": self.text,
            "metadata": meta
        }
        if self.dense_vector is not None:
            result["dense_vector"] = self.dense_vector
        if self.sparse_vector is not None:
            result["sparse_vector"] = self.sparse_vector
        return result


@dataclass
class RetrievalResult:
    """Result from retrieval operations.

    Attributes:
        chunk_id: ID of retrieved chunk
        score: Relevance score (0-1 or higher depending on scoring method)
        text: Chunk text content
        metadata: Chunk metadata dict
    """
    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        meta = self.metadata.copy()
        if "images" in meta and meta["images"]:
            meta["images"] = [img.to_dict() if isinstance(img, ImageRef) else img
                            for img in meta["images"]]
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "text": self.text,
            "metadata": meta
        }
