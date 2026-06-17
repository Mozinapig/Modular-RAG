"""
Document Manager for lifecycle management across storage backends.
Handles list, delete, stats operations across Chroma, BM25, ImageStorage, and FileIntegrity.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime


@dataclass
class DocumentInfo:
    """Basic document information for listing."""
    source_path: str
    collection: str
    chunk_count: int
    image_count: int
    created_at: str
    doc_id: Optional[str] = None


@dataclass
class DocumentDetail:
    """Detailed information about a document."""
    doc_id: str
    source_path: str
    collection: str
    title: Optional[str] = None
    summary: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    chunk_count: int = 0
    image_count: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    chunks: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DeleteResult:
    """Result of document deletion."""
    source_path: str
    collection: str
    chunks_deleted: int = 0
    images_deleted: int = 0
    success: bool = True
    error: Optional[str] = None


@dataclass
class CollectionStats:
    """Statistics for a collection."""
    collection_name: str
    chunk_count: int = 0
    image_count: int = 0
    document_count: int = 0


class DocumentManager:
    """Manages document lifecycle across all storage backends."""

    def __init__(self, chroma_store, bm25_indexer, image_storage, file_integrity):
        """
        Initialize DocumentManager with storage backends.

        Args:
            chroma_store: ChromaStore instance for vector storage
            bm25_indexer: BM25Indexer instance for sparse index
            image_storage: ImageStorage instance for images
            file_integrity: FileIntegrityChecker instance for file tracking
        """
        self.chroma_store = chroma_store
        self.bm25_indexer = bm25_indexer
        self.image_storage = image_storage
        self.file_integrity = file_integrity

    def list_documents(self, collection: Optional[str] = None) -> List[DocumentInfo]:
        """
        List all documents, optionally filtered by collection.

        Args:
            collection: Optional collection name to filter by

        Returns:
            List of DocumentInfo objects
        """
        try:
            # Get all chunks from Chroma
            query_metadata = {}
            if collection:
                query_metadata["collection"] = collection

            chunks = self.chroma_store.get_by_metadata(query_metadata)

            # Group chunks by document and collect info
            docs_map: Dict[str, DocumentInfo] = {}

            for chunk in chunks:
                metadata = chunk.get("metadata", {})
                source_path = metadata.get("source_path", "unknown")
                col = metadata.get("collection", "default")
                doc_id = metadata.get("doc_id", source_path)

                if doc_id not in docs_map:
                    docs_map[doc_id] = DocumentInfo(
                        source_path=source_path,
                        collection=col,
                        chunk_count=0,
                        image_count=0,
                        created_at=datetime.now().isoformat(),
                        doc_id=doc_id,
                    )

                # Count chunks and images
                docs_map[doc_id].chunk_count += 1
                image_refs = metadata.get("image_refs", [])
                docs_map[doc_id].image_count += len(image_refs)

            return list(docs_map.values())

        except Exception as e:
            print(f"Error listing documents: {e}")
            return []

    def get_document_detail(self, doc_id: str) -> Optional[DocumentDetail]:
        """
        Get detailed information about a document.

        Args:
            doc_id: Document ID

        Returns:
            DocumentDetail object or None if not found
        """
        try:
            # Query chunks by doc_id
            chunks = self.chroma_store.get_by_metadata({"doc_id": doc_id})

            if not chunks:
                return None

            # Aggregate metadata from first chunk
            first_chunk = chunks[0]
            metadata = first_chunk.get("metadata", {})

            # Count total images across all chunks
            total_images = 0
            for chunk in chunks:
                chunk_metadata = chunk.get("metadata", {})
                image_refs = chunk_metadata.get("image_refs", [])
                total_images += len(image_refs)

            detail = DocumentDetail(
                doc_id=doc_id,
                source_path=metadata.get("source_path", "unknown"),
                collection=metadata.get("collection", "default"),
                title=metadata.get("title"),
                summary=metadata.get("summary"),
                tags=metadata.get("tags", []),
                chunk_count=len(chunks),
                image_count=total_images,
                created_at=metadata.get("created_at"),
                updated_at=metadata.get("updated_at"),
                chunks=[
                    {
                        "id": chunk.get("id"),
                        "text": chunk.get("text", "")[:200],  # First 200 chars
                        "metadata": chunk.get("metadata", {})
                    }
                    for chunk in chunks
                ]
            )

            return detail

        except Exception as e:
            print(f"Error getting document detail: {e}")
            return None

    def delete_document(self, source_path: str, collection: str) -> DeleteResult:
        """
        Delete a document from all storage backends.

        Args:
            source_path: Path to source document
            collection: Collection name

        Returns:
            DeleteResult with deletion statistics
        """
        result = DeleteResult(
            source_path=source_path,
            collection=collection,
        )

        try:
            # Delete from Chroma
            chunks_deleted = 0
            try:
                chunks_deleted = self.chroma_store.delete_by_metadata({
                    "source_path": source_path,
                    "collection": collection,
                })
            except Exception as e:
                print(f"Error deleting from Chroma: {e}")

            # Delete from BM25
            try:
                self.bm25_indexer.remove_document(source_path)
            except Exception as e:
                print(f"Error deleting from BM25: {e}")

            # Delete images
            images_deleted = 0
            try:
                # Get doc_id from metadata first
                chunks = self.chroma_store.get_by_metadata({
                    "source_path": source_path,
                    "collection": collection,
                })
                if chunks:
                    doc_id = chunks[0].get("metadata", {}).get("doc_id", source_path)
                    images_deleted = self.image_storage.delete_by_doc_id(doc_id)
            except Exception as e:
                print(f"Error deleting images: {e}")

            # Remove from file integrity tracking
            try:
                self.file_integrity.remove_record(source_path)
            except Exception as e:
                print(f"Error updating file integrity: {e}")

            result.chunks_deleted = chunks_deleted
            result.images_deleted = images_deleted
            result.success = True

        except Exception as e:
            result.success = False
            result.error = str(e)

        return result

    def get_collection_stats(self, collection: Optional[str] = None) -> CollectionStats:
        """
        Get statistics for a collection.

        Args:
            collection: Collection name (if None, returns stats for all)

        Returns:
            CollectionStats object
        """
        try:
            stats = CollectionStats(collection_name=collection or "all")

            # Get collection-level stats from Chroma
            chroma_stats = self.chroma_store.get_collection_stats()

            if collection:
                # Find matching collection
                for col_info in chroma_stats.get("collections", []):
                    if col_info.get("name") == collection:
                        stats.chunk_count = col_info.get("chunk_count", 0)
                        stats.image_count = col_info.get("image_count", 0)
                        break
            else:
                # Return totals
                stats.chunk_count = chroma_stats.get("total_chunks", 0)
                stats.image_count = chroma_stats.get("total_images", 0)

            # Count unique documents
            try:
                query_metadata = {}
                if collection:
                    query_metadata["collection"] = collection

                chunks = self.chroma_store.get_by_metadata(query_metadata)
                unique_docs = set()
                for chunk in chunks:
                    source = chunk.get("metadata", {}).get("source_path")
                    if source:
                        unique_docs.add(source)

                stats.document_count = len(unique_docs)
            except Exception:
                pass

            return stats

        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return CollectionStats(collection_name=collection or "all")
