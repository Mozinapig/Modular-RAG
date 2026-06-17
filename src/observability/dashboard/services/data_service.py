"""
DataService - Encapsulates ChromaStore and ImageStorage operations
Provides high-level data access for Dashboard pages
"""
from typing import List, Dict, Any, Optional
from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.ingestion.storage.image_storage import ImageStorage


class DataService:
    """
    Service layer for Dashboard data access.
    Encapsulates ChromaStore and ImageStorage reads, providing clean interfaces.
    """

    def __init__(self, chroma_store: BaseVectorStore, image_storage: ImageStorage):
        """
        Initialize DataService.

        Args:
            chroma_store: ChromaStore instance for vector store access
            image_storage: ImageStorage instance for image retrieval
        """
        self.chroma_store = chroma_store
        self.image_storage = image_storage

    def list_collections(self) -> List[str]:
        """
        List all available collections in ChromaStore.

        Returns:
            List of collection names
        """
        try:
            return self.chroma_store.list_collections()
        except AttributeError:
            # Fallback if list_collections not available
            return []

    def get_documents(self, collection: str) -> List[Dict[str, Any]]:
        """
        Get all documents (grouped by source_path) in a collection.

        Args:
            collection: Collection name

        Returns:
            List of documents with metadata and chunk counts
        """
        try:
            # Retrieve all chunks from collection
            filters = {"collection": collection}
            records = self.chroma_store.get_by_metadata(filters=filters)

            if not records:
                return []

            # Group by source_path
            docs_dict: Dict[str, Dict[str, Any]] = {}
            for record in records:
                metadata = record.get("metadata", {})
                source_path = metadata.get("source_path", "unknown")

                if source_path not in docs_dict:
                    docs_dict[source_path] = {
                        "source_path": source_path,
                        "collection": collection,
                        "chunk_count": 0,
                        "ingestion_time": metadata.get("ingestion_time", ""),
                    }

                docs_dict[source_path]["chunk_count"] += 1

            return list(docs_dict.values())
        except Exception as e:
            # Log error but return empty list
            print(f"Error retrieving documents: {e}")
            return []

    def get_chunks(self, source_path: str, collection: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific document (by source_path).

        Args:
            source_path: Source file path (e.g., "doc1.pdf")
            collection: Collection name

        Returns:
            List of chunks with content and metadata
        """
        try:
            filters = {"source_path": source_path, "collection": collection}
            records = self.chroma_store.get_by_metadata(filters=filters)

            # Convert records to chunk format
            chunks = []
            for record in records:
                chunks.append({
                    "id": record.get("id", ""),
                    "content": record.get("content", ""),
                    "metadata": record.get("metadata", {}),
                })

            return chunks
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            return []

    def get_chunk_images(self, chunk_id: str) -> List[Dict[str, Any]]:
        """
        Get all images associated with a chunk.

        Args:
            chunk_id: Chunk ID

        Returns:
            List of image metadata
        """
        try:
            images = self.image_storage.list_images(chunk_id=chunk_id)
            return images if images else []
        except Exception as e:
            print(f"Error retrieving chunk images: {e}")
            return []

    def get_document_stats(self, source_path: str, collection: str) -> Dict[str, Any]:
        """
        Get statistics for a document.

        Args:
            source_path: Source file path
            collection: Collection name

        Returns:
            Document statistics (chunk count, ingestion time, etc.)
        """
        try:
            chunks = self.get_chunks(source_path, collection)
            if not chunks:
                return {
                    "source_path": source_path,
                    "chunk_count": 0,
                    "collection": collection,
                }

            # Get first chunk's ingestion time
            ingestion_time = chunks[0].get("metadata", {}).get("ingestion_time", "")

            return {
                "source_path": source_path,
                "collection": collection,
                "chunk_count": len(chunks),
                "ingestion_time": ingestion_time,
            }
        except Exception as e:
            print(f"Error calculating document stats: {e}")
            return {
                "source_path": source_path,
                "collection": collection,
                "chunk_count": 0,
            }

    def search_chunks(self, query: str, collection: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search chunks by content.

        Args:
            query: Search query
            collection: Collection name
            top_k: Number of top results

        Returns:
            List of matching chunks
        """
        try:
            # This is a placeholder - actual search would use hybrid search from core
            # For now, just return empty list (to be implemented in G4+)
            return []
        except Exception as e:
            print(f"Error searching chunks: {e}")
            return []
