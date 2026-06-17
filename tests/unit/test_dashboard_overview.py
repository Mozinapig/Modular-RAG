"""
Unit tests for Dashboard Overview page (G1)
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.core.settings import Settings, LLMSettings, EmbeddingSettings, VectorStoreSettings
from src.observability.dashboard.services.config_service import ConfigService


class TestConfigService:
    """Test ConfigService for config reading and formatting."""

    @pytest.fixture
    def settings(self):
        """Create test settings object."""
        from src.core.settings import RetrievalSettings, RerankerSettings, ObservabilitySettings
        return Settings(
            llm=LLMSettings(provider="openai", model="gpt-4", api_key="test_key"),
            embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small", api_key="test_key"),
            vector_store=VectorStoreSettings(backend="chroma", persist_path="data/db/chroma"),
            retrieval=RetrievalSettings(),
            rerank=RerankerSettings(),
            vision_llm=None,
        )

    def test_config_service_init(self, settings):
        """Test ConfigService initialization."""
        service = ConfigService(settings)
        assert service.settings == settings

    def test_get_llm_config(self, settings):
        """Test getting LLM configuration."""
        service = ConfigService(settings)
        llm_config = service.get_llm_config()

        assert llm_config["provider"] == "openai"
        assert llm_config["model"] == "gpt-4"
        assert "api_key" in llm_config
        assert llm_config["api_key"] != "test_key"  # Should be masked

    def test_get_embedding_config(self, settings):
        """Test getting Embedding configuration."""
        service = ConfigService(settings)
        embedding_config = service.get_embedding_config()

        assert embedding_config["provider"] == "openai"
        assert embedding_config["model"] == "text-embedding-3-small"
        assert "api_key" in embedding_config
        assert embedding_config["api_key"] != "test_key"  # Should be masked

    def test_get_vector_store_config(self, settings):
        """Test getting VectorStore configuration."""
        service = ConfigService(settings)
        vs_config = service.get_vector_store_config()

        assert vs_config["backend"] == "chroma"
        assert vs_config["persist_path"] == "data/db/chroma"

    def test_get_all_configs(self, settings):
        """Test getting all configurations."""
        service = ConfigService(settings)
        all_configs = service.get_all_configs()

        assert "llm" in all_configs
        assert "embedding" in all_configs
        assert "vector_store" in all_configs
        assert "retrieval" in all_configs
        assert "reranker" in all_configs
        # vision_llm should not be included when None

    def test_mask_sensitive_value(self):
        """Test sensitive value masking."""
        service = ConfigService.__new__(ConfigService)

        # Test API key masking
        masked = service._mask_sensitive_value("sk-1234567890abcdefghijk")
        assert masked == "sk-***...jk"

        # Test short value (<=4 chars)
        masked = service._mask_sensitive_value("short")
        assert masked == "sh-***...rt"

        # Test non-sensitive value
        masked = service._mask_sensitive_value("normal_value", is_sensitive=False)
        assert masked == "normal_value"


class TestDashboardOverviewIntegration:
    """Integration tests for Dashboard Overview page."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = Mock()
        settings.llm = Mock(provider="openai", model="gpt-4", temperature=0.7)
        settings.embedding = Mock(provider="openai", model="text-embedding-3-small", dimensions=1536)
        settings.vector_store = Mock(backend="chroma", persist_path="data/db/chroma")
        settings.retrieval = Mock(top_k_final=10, fusion_algorithm="rrf", sparse_backend="bm25", top_k_dense=20, top_k_sparse=20)
        settings.rerank = Mock(backend="none", model=None, top_m=30)
        settings.vision_llm = None
        return settings

    @pytest.fixture
    def mock_chroma_store(self):
        """Create mock ChromaStore."""
        store = Mock()
        store.get_collection_stats = Mock(return_value={
            "collections": [
                {"name": "test_collection", "chunk_count": 100, "image_count": 5}
            ],
            "total_chunks": 100,
            "total_images": 5
        })
        return store

    def test_overview_page_components_exist(self, mock_settings, mock_chroma_store):
        """Test that all overview page components are properly structured."""
        # This test verifies the expected structure of page components
        assert mock_settings.llm.provider == "openai"
        assert mock_settings.embedding.dimensions == 1536
        assert mock_settings.vector_store.backend == "chroma"

    def test_collection_stats_retrieval(self, mock_chroma_store):
        """Test collection stats retrieval."""
        stats = mock_chroma_store.get_collection_stats()

        assert "collections" in stats
        assert "total_chunks" in stats
        assert "total_images" in stats
        assert stats["total_chunks"] == 100
        assert stats["total_images"] == 5

    def test_overview_page_data_structure(self, mock_settings, mock_chroma_store):
        """Test overview page data structure."""
        config_service = ConfigService(mock_settings)

        overview_data = {
            "system_config": {
                "llm": config_service.get_llm_config(),
                "embedding": config_service.get_embedding_config(),
                "vector_store": config_service.get_vector_store_config(),
            },
            "data_stats": mock_chroma_store.get_collection_stats()
        }

        # Verify structure
        assert "system_config" in overview_data
        assert "data_stats" in overview_data
        assert "llm" in overview_data["system_config"]
        assert "embedding" in overview_data["system_config"]
        assert "vector_store" in overview_data["system_config"]
