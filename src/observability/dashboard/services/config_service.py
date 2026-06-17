"""
Configuration service for Dashboard.
Handles reading and formatting settings for display.
"""
from typing import Dict, Any, Optional
from src.core.settings import Settings


class ConfigService:
    """Service for reading and formatting configuration."""

    def __init__(self, settings: Settings):
        """Initialize ConfigService with settings."""
        self.settings = settings

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration with masked sensitive data."""
        return {
            "provider": self.settings.llm.provider,
            "model": self.settings.llm.model,
            "temperature": self.settings.llm.temperature,
            "api_key": self._mask_sensitive_value(self.settings.llm.api_key),
            **({"base_url": self.settings.llm.base_url} if self.settings.llm.base_url else {}),
        }

    def get_embedding_config(self) -> Dict[str, Any]:
        """Get Embedding configuration with masked sensitive data."""
        return {
            "provider": self.settings.embedding.provider,
            "model": self.settings.embedding.model,
            "dimensions": self.settings.embedding.dimensions,
            "api_key": self._mask_sensitive_value(self.settings.embedding.api_key),
            **({"base_url": self.settings.embedding.base_url} if self.settings.embedding.base_url else {}),
        }

    def get_vector_store_config(self) -> Dict[str, Any]:
        """Get VectorStore configuration."""
        return {
            "backend": self.settings.vector_store.backend,
            "persist_path": self.settings.vector_store.persist_path,
        }

    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get Retrieval configuration."""
        return {
            "sparse_backend": self.settings.retrieval.sparse_backend,
            "fusion_algorithm": self.settings.retrieval.fusion_algorithm,
            "top_k_dense": self.settings.retrieval.top_k_dense,
            "top_k_sparse": self.settings.retrieval.top_k_sparse,
            "top_k_final": self.settings.retrieval.top_k_final,
        }

    def get_reranker_config(self) -> Dict[str, Any]:
        """Get Reranker configuration."""
        return {
            "backend": self.settings.rerank.backend,
            **({"model": self.settings.rerank.model} if self.settings.rerank.model else {}),
            "top_m": self.settings.rerank.top_m,
        }

    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configurations."""
        return {
            "llm": self.get_llm_config(),
            "embedding": self.get_embedding_config(),
            "vector_store": self.get_vector_store_config(),
            "retrieval": self.get_retrieval_config(),
            "reranker": self.get_reranker_config(),
            **({"vision_llm": self._get_vision_llm_config()} if self.settings.vision_llm else {}),
        }

    def _get_vision_llm_config(self) -> Dict[str, Any]:
        """Get Vision LLM configuration."""
        if not self.settings.vision_llm:
            return {}

        return {
            "provider": self.settings.vision_llm.provider,
            "model": self.settings.vision_llm.model,
            "api_key": self._mask_sensitive_value(self.settings.vision_llm.api_key),
            "max_image_size": self.settings.vision_llm.max_image_size,
        }

    @staticmethod
    def _mask_sensitive_value(value: str, is_sensitive: bool = True) -> str:
        """Mask sensitive values like API keys."""
        if not is_sensitive or not value or not isinstance(value, str):
            return str(value) if value else ""

        if len(value) <= 4:
            return value[0] + "*" * (len(value) - 2) + value[-1] if len(value) > 1 else "*" * len(value)

        # Show first 2 and last 2 chars with separator
        return value[:2] + "-" + "***" + "..." + value[-2:]
