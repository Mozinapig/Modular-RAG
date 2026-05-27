"""ChromaStore implementation for vector storage using Chroma DB."""

import os
import json
import sys
from typing import List, Dict, Optional, Any
from pathlib import Path

from src.libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord
from src.core.settings import Settings


class ChromaStore(BaseVectorStore):
    """ChromaDB implementation of VectorStore (with pure-Python fallback for Windows compatibility)."""

    def __init__(self, settings: Settings):
        """
        Initialize ChromaStore.

        Args:
            settings: Settings object with vector_store configuration
        """
        self.settings = settings
        self.validate_config()

        # Get persist path from settings
        vector_store_config = settings.vector_store or {}
        self.persist_path = vector_store_config.get("persist_path", "./data/db/chroma")

        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_path, exist_ok=True)

        # Determine whether to use Chroma
        # Skip Chroma on Windows due to Rust binding compatibility issues
        self._use_chroma = sys.platform != "win32"
        self.collection = None
        self.client = None
        self._records: Dict[str, VectorRecord] = {}

        if self._use_chroma:
            try:
                import chromadb
                self.client = chromadb.EphemeralClient()
                self.collection = self.client.get_or_create_collection(
                    name="documents",
                    metadata={"hnsw:space": "cosine"}
                )
                print("ChromaStore: Using Chroma backend")
            except Exception as e:
                print(f"ChromaStore: Chroma initialization failed ({e}), using in-memory fallback")
                self._use_chroma = False
        else:
            print("ChromaStore: Using in-memory fallback (Windows detected)")

        self.records_file = Path(self.persist_path) / "records.jsonl"
        self._load_from_disk()

    def validate_config(self) -> None:
        """
        Validate ChromaStore configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.settings.vector_store:
            raise ValueError("vector_store configuration is required")

        vector_store_config = self.settings.vector_store
        if vector_store_config.get("backend") != "chroma":
            raise ValueError(f"Invalid backend: {vector_store_config.get('backend')}")

    def upsert(
        self,
        records: List[VectorRecord],
        trace: Optional[Any] = None,
    ) -> None:
        """
        Upsert (insert or update) vector records.

        Args:
            records: List of VectorRecord objects to store
            trace: Optional TraceContext for tracking

        Raises:
            ValueError: If input validation fails
            RuntimeError: If storage operation fails
        """
        if not records:
            return

        if self._use_chroma:
            self._upsert_chroma(records, trace)
        else:
            self._upsert_memory(records, trace)

    def _upsert_chroma(self, records: List[VectorRecord], trace: Optional[Any]) -> None:
        """Upsert using Chroma backend."""
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for record in records:
            if not record.id or not record.text or not record.embedding:
                raise ValueError("Each record must have id, text, and embedding")

            ids.append(record.id)
            embeddings.append(record.embedding)
            documents.append(record.text)
            metadatas.append(record.metadata or {})

        try:
            # Delete existing records first for idempotency
            try:
                existing = self.collection.get(ids=ids)
                if existing and existing["ids"]:
                    self.collection.delete(ids=existing["ids"])
            except:
                pass

            # Add new records
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            if trace:
                trace.record_stage("vector_upsert", method="chroma", count=len(records))
        except Exception as e:
            raise RuntimeError(f"Failed to upsert records into Chroma: {str(e)}")

    def _upsert_memory(self, records: List[VectorRecord], trace: Optional[Any]) -> None:
        """Upsert using in-memory fallback."""
        for record in records:
            if not record.id or not record.text or not record.embedding:
                raise ValueError("Each record must have id, text, and embedding")
            self._records[record.id] = record

        # Persist to disk
        self._save_to_disk()

        if trace:
            trace.record_stage("vector_upsert", method="memory", count=len(records))

    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[VectorRecord]:
        """
        Query similar vectors.

        Args:
            vector: Query embedding vector
            top_k: Number of top results to return
            filters: Optional metadata filters
            trace: Optional TraceContext for tracking

        Returns:
            List of similar VectorRecord objects

        Raises:
            ValueError: If input validation fails
            RuntimeError: If query fails
        """
        if not vector:
            raise ValueError("Query vector cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if self._use_chroma:
            return self._query_chroma(vector, top_k, filters, trace)
        else:
            return self._query_memory(vector, top_k, filters, trace)

    def _query_chroma(
        self,
        vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]],
        trace: Optional[Any],
    ) -> List[VectorRecord]:
        """Query using Chroma backend."""
        try:
            where_filter = None
            if filters:
                where_filter = self._build_where_filter(filters)

            results = self.collection.query(
                query_embeddings=[vector],
                n_results=top_k,
                where=where_filter,
            )

            records = []
            if results and results["ids"] and len(results["ids"]) > 0:
                for i, record_id in enumerate(results["ids"][0]):
                    record = VectorRecord(
                        id=record_id,
                        text=results["documents"][0][i] if results["documents"] else "",
                        embedding=results["embeddings"][0][i] if results["embeddings"] else vector,
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    )
                    records.append(record)

            if trace:
                trace.record_stage("vector_query", method="chroma", results_count=len(records))

            return records
        except Exception as e:
            raise RuntimeError(f"Failed to query vectors from Chroma: {str(e)}")

    def _query_memory(
        self,
        vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]],
        trace: Optional[Any],
    ) -> List[VectorRecord]:
        """Query using in-memory fallback."""
        if not self._records:
            return []

        def cosine_similarity(v1: List[float], v2: List[float]) -> float:
            """Calculate cosine similarity."""
            if len(v1) != len(v2):
                return 0.0
            dot = sum(a * b for a, b in zip(v1, v2))
            norm1 = sum(a * a for a in v1) ** 0.5
            norm2 = sum(b * b for b in v2) ** 0.5
            return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

        # Calculate similarities
        similarities = []
        for record_id, record in self._records.items():
            sim = cosine_similarity(vector, record.embedding)
            similarities.append((sim, record))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Apply filters
        if filters:
            similarities = [
                (sim, rec) for sim, rec in similarities
                if rec.metadata and all(rec.metadata.get(k) == v for k, v in filters.items())
            ]

        # Return top_k
        results = [rec for _, rec in similarities[:top_k]]

        if trace:
            trace.record_stage("vector_query", method="memory", results_count=len(results))

        return results

    def _build_where_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build Chroma where filter."""
        if not filters:
            return None

        conditions = []
        for key, value in filters.items():
            conditions.append({key: {"$eq": value}})

        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {"$and": conditions}
        return None

    def _save_to_disk(self) -> None:
        """Save records to disk (JSON Lines format)."""
        try:
            with open(self.records_file, "w") as f:
                for record_id, record in self._records.items():
                    obj = {
                        "id": record.id,
                        "text": record.text,
                        "embedding": record.embedding,
                        "metadata": record.metadata,
                    }
                    f.write(json.dumps(obj) + "\n")
        except Exception as e:
            print(f"Warning: Failed to persist records: {e}")

    def _load_from_disk(self) -> None:
        """Load records from disk."""
        if not self.records_file.exists():
            return

        try:
            with open(self.records_file, "r") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        record = VectorRecord(
                            id=obj["id"],
                            text=obj["text"],
                            embedding=obj["embedding"],
                            metadata=obj.get("metadata"),
                        )
                        self._records[record.id] = record
        except Exception as e:
            print(f"Warning: Failed to load records: {e}")



    def _upsert_memory(self, records: List[VectorRecord], trace: Optional[Any]) -> None:
        """Upsert using in-memory fallback."""
        for record in records:
            if not record.id or not record.text or not record.embedding:
                raise ValueError("Each record must have id, text, and embedding")
            self._records[record.id] = record

        # Persist to disk
        self._save_to_disk()

        if trace:
            trace.record_stage("vector_upsert", method="memory", count=len(records))

    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[VectorRecord]:
        """
        Query similar vectors.

        Args:
            vector: Query embedding vector
            top_k: Number of top results to return
            filters: Optional metadata filters
            trace: Optional TraceContext for tracking

        Returns:
            List of similar VectorRecord objects

        Raises:
            ValueError: If input validation fails
            RuntimeError: If query fails
        """
        if not vector:
            raise ValueError("Query vector cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if self._use_chroma:
            return self._query_chroma(vector, top_k, filters, trace)
        else:
            return self._query_memory(vector, top_k, filters, trace)

    def _query_chroma(
        self,
        vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]],
        trace: Optional[Any],
    ) -> List[VectorRecord]:
        """Query using Chroma backend."""
        try:
            where_filter = None
            if filters:
                where_filter = self._build_where_filter(filters)

            results = self.collection.query(
                query_embeddings=[vector],
                n_results=top_k,
                where=where_filter,
            )

            records = []
            if results and results["ids"] and len(results["ids"]) > 0:
                for i, record_id in enumerate(results["ids"][0]):
                    record = VectorRecord(
                        id=record_id,
                        text=results["documents"][0][i] if results["documents"] else "",
                        embedding=results["embeddings"][0][i] if results["embeddings"] else vector,
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    )
                    records.append(record)

            if trace:
                trace.record_stage("vector_query", method="chroma", results_count=len(records))

            return records
        except Exception as e:
            raise RuntimeError(f"Failed to query vectors from Chroma: {str(e)}")

    def _query_memory(
        self,
        vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]],
        trace: Optional[Any],
    ) -> List[VectorRecord]:
        """Query using in-memory fallback."""
        if not self._records:
            return []

        def cosine_similarity(v1: List[float], v2: List[float]) -> float:
            """Calculate cosine similarity."""
            if len(v1) != len(v2):
                return 0.0
            dot = sum(a * b for a, b in zip(v1, v2))
            norm1 = sum(a * a for a in v1) ** 0.5
            norm2 = sum(b * b for b in v2) ** 0.5
            return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

        # Calculate similarities
        similarities = []
        for record_id, record in self._records.items():
            sim = cosine_similarity(vector, record.embedding)
            similarities.append((sim, record))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Apply filters
        if filters:
            similarities = [
                (sim, rec) for sim, rec in similarities
                if rec.metadata and all(rec.metadata.get(k) == v for k, v in filters.items())
            ]

        # Return top_k
        results = [rec for _, rec in similarities[:top_k]]

        if trace:
            trace.record_stage("vector_query", method="memory", results_count=len(results))

        return results

    def _build_where_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build Chroma where filter."""
        if not filters:
            return None

        conditions = []
        for key, value in filters.items():
            conditions.append({key: {"$eq": value}})

        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {"$and": conditions}
        return None

    def _save_to_disk(self) -> None:
        """Save records to disk (JSON Lines format)."""
        try:
            with open(self.records_file, "w") as f:
                for record_id, record in self._records.items():
                    obj = {
                        "id": record.id,
                        "text": record.text,
                        "embedding": record.embedding,
                        "metadata": record.metadata,
                    }
                    f.write(json.dumps(obj) + "\n")
        except Exception as e:
            print(f"Warning: Failed to persist records: {e}")

    def _load_from_disk(self) -> None:
        """Load records from disk."""
        if not self.records_file.exists():
            return

        try:
            with open(self.records_file, "r") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        record = VectorRecord(
                            id=obj["id"],
                            text=obj["text"],
                            embedding=obj["embedding"],
                            metadata=obj.get("metadata"),
                        )
                        self._records[record.id] = record
        except Exception as e:
            print(f"Warning: Failed to load records: {e}")



