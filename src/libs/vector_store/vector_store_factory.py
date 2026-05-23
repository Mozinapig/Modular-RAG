"""VectorStore Factory for provider routing and instantiation."""

from typing import Dict, Type

from src.libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord


class FakeVectorStore(BaseVectorStore):
    """Fake VectorStore implementation for testing."""

    def __init__(self, settings):
        self.settings = settings
        self.records: Dict[str, VectorRecord] = {}

    def upsert(self, records, trace=None):
        """Store records in memory (simple fake implementation)."""
        for record in records:
            self.records[record.id] = record

    def query(self, vector, top_k=10, filters=None, trace=None):
        """
        Query similar vectors using simple cosine similarity.

        In this fake implementation, we calculate cosine similarity
        and return top_k records sorted by similarity.
        """
        if not self.records:
            return []

        def cosine_similarity(v1, v2):
            """Calculate cosine similarity between two vectors."""
            if len(v1) != len(v2):
                return 0.0

            dot_product = sum(a * b for a, b in zip(v1, v2))
            norm_v1 = sum(a * a for a in v1) ** 0.5
            norm_v2 = sum(b * b for b in v2) ** 0.5

            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0

            return dot_product / (norm_v1 * norm_v2)

        # Calculate similarity for all records
        similarities = []
        for record_id, record in self.records.items():
            similarity = cosine_similarity(vector, record.embedding)
            similarities.append((similarity, record))

        # Sort by similarity (descending) and apply filters if provided
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Apply filters if provided
        if filters:
            similarities = [
                (sim, rec) for sim, rec in similarities
                if rec.metadata and all(
                    rec.metadata.get(k) == v for k, v in filters.items()
                )
            ]

        # Return top_k records
        return [record for _, record in similarities[:top_k]]

    def validate_config(self):
        """Validate config (no-op for fake)."""
        if not self.settings:
            raise ValueError("Settings object is required")


class VectorStoreFactory:
    """Factory for creating VectorStore instances based on settings."""

    # Provider registry mapping
    _providers: Dict[str, Type[BaseVectorStore]] = {
        "fake": FakeVectorStore,
    }

    def create(self, settings) -> BaseVectorStore:
        """
        Create a VectorStore instance based on settings.

        Args:
            settings: VectorStoreSettings object with provider configuration

        Returns:
            BaseVectorStore instance

        Raises:
            ValueError: If provider is unknown or configuration is invalid
        """
        # Validate basic settings
        if not settings.provider:
            raise ValueError("VectorStore provider is required")

        provider = settings.provider.lower()

        if provider not in self._providers:
            raise ValueError(
                f"Unknown VectorStore provider: {provider}. "
                f"Supported providers: {', '.join(self._providers.keys())}"
            )

        provider_class = self._providers[provider]
        store = provider_class(settings)

        # Validate provider-specific config
        store.validate_config()

        return store

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseVectorStore]) -> None:
        """
        Register a new VectorStore provider.

        Args:
            name: Provider name (lowercase)
            provider_class: Class that extends BaseVectorStore
        """
        cls._providers[name.lower()] = provider_class
