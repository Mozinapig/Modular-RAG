"""Tests for DenseRetriever."""
import pytest
from unittest.mock import Mock, MagicMock
from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.types import RetrievalResult


class TestDenseRetriever:
    """Test DenseRetriever."""

    @pytest.fixture
    def mock_embedding_client(self):
        """Create a mock embedding client."""
        client = Mock()
        client.embed = Mock(return_value=[[0.1, 0.2, 0.3, 0.4, 0.5]])
        return client

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        from src.libs.vector_store.base_vector_store import VectorRecord

        store = Mock()
        record = VectorRecord(
            id="chunk_1",
            text="Sample text content",
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            metadata={"source": "test.pdf", "page": 1},
            score=0.95,
        )
        store.query = Mock(return_value=[record])
        return store

    def test_init_with_dependencies(self, mock_embedding_client, mock_vector_store):
        """Test DenseRetriever initialization with injected dependencies."""
        retriever = DenseRetriever(
            embedding_client=mock_embedding_client,
            vector_store=mock_vector_store,
        )
        assert retriever is not None

    def test_retrieve_basic(self, mock_embedding_client, mock_vector_store):
        """Test basic retrieval."""
        retriever = DenseRetriever(
            embedding_client=mock_embedding_client,
            vector_store=mock_vector_store,
        )

        results = retriever.retrieve("test query", top_k=10)

        assert isinstance(results, list)
        assert len(results) > 0
        assert isinstance(results[0], RetrievalResult)
        assert results[0].chunk_id == "chunk_1"
        assert results[0].text == "Sample text content"

    def test_retrieve_calls_embedding_client(self, mock_embedding_client, mock_vector_store):
        """Test that retrieve calls embedding client."""
        retriever = DenseRetriever(
            embedding_client=mock_embedding_client,
            vector_store=mock_vector_store,
        )

        retriever.retrieve("test query", top_k=10)

        mock_embedding_client.embed.assert_called_once()
        call_args = mock_embedding_client.embed.call_args
        assert "test query" in call_args[0][0]

    def test_retrieve_calls_vector_store(self, mock_embedding_client, mock_vector_store):
        """Test that retrieve calls vector store."""
        retriever = DenseRetriever(
            embedding_client=mock_embedding_client,
            vector_store=mock_vector_store,
        )

        retriever.retrieve("test query", top_k=10)

        mock_vector_store.query.assert_called_once()

    def test_retrieve_with_filters(self, mock_embedding_client, mock_vector_store):
        """Test retrieve with metadata filters."""
        retriever = DenseRetriever(
            embedding_client=mock_embedding_client,
            vector_store=mock_vector_store,
        )

        results = retriever.retrieve("test query", top_k=10, filters={"collection": "test"})

        assert isinstance(results, list)
        mock_vector_store.query.assert_called_once()
        # Verify filters were passed to vector store
        call_kwargs = mock_vector_store.query.call_args[1]
        assert call_kwargs.get("filters") == {"collection": "test"}

    def test_retrieve_without_filters(self, mock_embedding_client, mock_vector_store):
        """Test retrieve without filters."""
        retriever = DenseRetriever(
            embedding_client=mock_embedding_client,
            vector_store=mock_vector_store,
        )

        results = retriever.retrieve("test query", top_k=10)

        assert isinstance(results, list)
        mock_vector_store.query.assert_called_once()

    def test_retrieve_multiple_results(self, mock_embedding_client):
        """Test retrieve with multiple results."""
        from src.libs.vector_store.base_vector_store import VectorRecord

        store = Mock()
        records = [
            VectorRecord(
                id=f"chunk_{i}",
                text=f"Sample text {i}",
                embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
                metadata={"source": f"test_{i}.pdf"},
            )
            for i in range(3)
        ]
        store.query = Mock(return_value=records)

        retriever = DenseRetriever(
            embedding_client=mock_embedding_client,
            vector_store=store,
        )

        results = retriever.retrieve("test query", top_k=10)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.chunk_id == f"chunk_{i}"

    def test_retrieve_empty_results(self, mock_embedding_client):
        """Test retrieve with no results."""
        store = Mock()
        store.query = Mock(return_value=[])

        retriever = DenseRetriever(
            embedding_client=mock_embedding_client,
            vector_store=store,
        )

        results = retriever.retrieve("test query", top_k=10)

        assert isinstance(results, list)
        assert len(results) == 0

    def test_retrieval_result_structure(self, mock_embedding_client, mock_vector_store):
        """Test that RetrievalResult has all required fields."""
        retriever = DenseRetriever(
            embedding_client=mock_embedding_client,
            vector_store=mock_vector_store,
        )

        results = retriever.retrieve("test query", top_k=10)

        result = results[0]
        assert hasattr(result, "chunk_id")
        assert hasattr(result, "score")
        assert hasattr(result, "text")
        assert hasattr(result, "metadata")

    def test_retrieve_with_custom_top_k(self, mock_embedding_client, mock_vector_store):
        """Test retrieve with custom top_k parameter."""
        retriever = DenseRetriever(
            embedding_client=mock_embedding_client,
            vector_store=mock_vector_store,
        )

        retriever.retrieve("test query", top_k=5)

        mock_vector_store.query.assert_called_once()
        call_kwargs = mock_vector_store.query.call_args[1]
        assert call_kwargs.get("top_k") == 5

    def test_retrieve_normalizes_vector_score(self, mock_embedding_client):
        """Test that vector scores from VectorRecord are correctly used."""
        from src.libs.vector_store.base_vector_store import VectorRecord

        store = Mock()
        # VectorRecord now includes score
        record = VectorRecord(
            id="chunk_1",
            text="Test",
            embedding=[0.1, 0.2, 0.3],
            metadata={},
            score=0.87,  # Real similarity score
        )
        store.query = Mock(return_value=[record])

        retriever = DenseRetriever(
            embedding_client=mock_embedding_client,
            vector_store=store,
        )

        results = retriever.retrieve("test query")

        assert results[0].score == 0.87  # Should use the actual score
        assert isinstance(results[0].score, float)

    def test_retrieve_with_trace_context(self, mock_embedding_client, mock_vector_store):
        """Test retrieve with trace context."""
        retriever = DenseRetriever(
            embedding_client=mock_embedding_client,
            vector_store=mock_vector_store,
        )

        mock_trace = Mock()
        results = retriever.retrieve("test query", top_k=10, trace=mock_trace)

        assert isinstance(results, list)
        # Verify trace was used if passed through
        if mock_trace:
            mock_vector_store.query.assert_called_once()

    def test_retrieve_score_from_vector_record(self, mock_embedding_client):
        """Test that score is correctly taken from VectorRecord."""
        from src.libs.vector_store.base_vector_store import VectorRecord

        store = Mock()
        records = [
            VectorRecord(
                id="chunk_1",
                text="Highly relevant",
                embedding=[0.1, 0.2, 0.3],
                metadata={},
                score=0.95,
            ),
            VectorRecord(
                id="chunk_2",
                text="Less relevant",
                embedding=[0.1, 0.2, 0.3],
                metadata={},
                score=0.72,
            ),
        ]
        store.query = Mock(return_value=records)

        retriever = DenseRetriever(
            embedding_client=mock_embedding_client,
            vector_store=store,
        )

        results = retriever.retrieve("test query")

        assert len(results) == 2
        assert results[0].score == 0.95
        assert results[1].score == 0.72
