"""Unit tests for SparseRetriever."""

import pytest
from unittest.mock import Mock, MagicMock

from src.core.query_engine.sparse_retriever import SparseRetriever
from src.core.types import RetrievalResult
from src.libs.vector_store.base_vector_store import VectorRecord
from src.core.settings import Settings


@pytest.fixture
def mock_bm25_indexer():
    """Create a mock BM25Indexer."""
    indexer = Mock()
    return indexer


@pytest.fixture
def mock_vector_store():
    """Create a mock VectorStore."""
    store = Mock()
    return store


@pytest.fixture
def mock_settings():
    """Create a mock Settings object."""
    settings = Mock(spec=Settings)
    return settings


@pytest.fixture
def sparse_retriever(mock_bm25_indexer, mock_vector_store, mock_settings):
    """Create a SparseRetriever instance for testing."""
    return SparseRetriever(
        settings=mock_settings,
        bm25_indexer=mock_bm25_indexer,
        vector_store=mock_vector_store,
    )


def test_sparse_retriever_init(mock_bm25_indexer, mock_vector_store, mock_settings):
    """Test SparseRetriever initialization."""
    retriever = SparseRetriever(
        settings=mock_settings,
        bm25_indexer=mock_bm25_indexer,
        vector_store=mock_vector_store,
    )
    assert retriever.settings == mock_settings
    assert retriever.bm25_indexer == mock_bm25_indexer
    assert retriever.vector_store == mock_vector_store


def test_sparse_retriever_retrieve_basic(sparse_retriever, mock_bm25_indexer, mock_vector_store):
    """Test basic retrieval with single keyword."""
    # Mock BM25 query results
    mock_bm25_indexer.query.return_value = [
        {"chunk_id": "chunk_001", "score": 5.0},
        {"chunk_id": "chunk_002", "score": 3.5},
    ]

    # Mock vector store get_by_ids results
    mock_vector_store.get_by_ids.return_value = [
        VectorRecord(
            id="chunk_001",
            text="This is chunk 001 content",
            embedding=[0.1, 0.2, 0.3],
            metadata={"source": "doc1.pdf", "page": 1}
        ),
        VectorRecord(
            id="chunk_002",
            text="This is chunk 002 content",
            embedding=[0.4, 0.5, 0.6],
            metadata={"source": "doc1.pdf", "page": 2}
        ),
    ]

    # Retrieve
    results = sparse_retriever.retrieve(
        keywords=["configure"],
        top_k=10,
    )

    # Verify results
    assert len(results) == 2
    assert isinstance(results[0], RetrievalResult)
    assert results[0].chunk_id == "chunk_001"
    assert results[0].text == "This is chunk 001 content"
    assert results[0].score == 5.0
    assert results[0].metadata == {"source": "doc1.pdf", "page": 1}

    # Verify calls
    mock_bm25_indexer.query.assert_called_with(term="configure", top_k=10, trace=None)
    mock_vector_store.get_by_ids.assert_called_with(ids=["chunk_001", "chunk_002"], trace=None)


def test_sparse_retriever_retrieve_multiple_keywords(sparse_retriever, mock_bm25_indexer, mock_vector_store):
    """Test retrieval with multiple keywords (aggregated results)."""
    # First keyword query
    def mock_query(term, top_k=10, trace=None):
        if term == "azure":
            return [{"chunk_id": "chunk_001", "score": 5.0}]
        elif term == "configure":
            return [{"chunk_id": "chunk_002", "score": 3.5}]
        else:
            return []

    mock_bm25_indexer.query.side_effect = mock_query

    # Mock vector store
    mock_vector_store.get_by_ids.return_value = [
        VectorRecord(
            id="chunk_001",
            text="Azure configuration",
            embedding=[0.1, 0.2],
            metadata={"source": "doc1.pdf"}
        ),
        VectorRecord(
            id="chunk_002",
            text="Configuration guide",
            embedding=[0.3, 0.4],
            metadata={"source": "doc1.pdf"}
        ),
    ]

    # Retrieve
    results = sparse_retriever.retrieve(
        keywords=["azure", "configure"],
        top_k=10,
    )

    # Verify results combined and scored
    assert len(results) == 2
    assert results[0].chunk_id in ["chunk_001", "chunk_002"]
    assert all(isinstance(r, RetrievalResult) for r in results)


def test_sparse_retriever_retrieve_empty_keywords(sparse_retriever):
    """Test retrieval with empty keywords."""
    results = sparse_retriever.retrieve(
        keywords=[],
        top_k=10,
    )

    assert len(results) == 0


def test_sparse_retriever_retrieve_no_results(sparse_retriever, mock_bm25_indexer, mock_vector_store):
    """Test retrieval with no matching results."""
    mock_bm25_indexer.query.return_value = []
    mock_vector_store.get_by_ids.return_value = []

    results = sparse_retriever.retrieve(
        keywords=["nonexistent"],
        top_k=10,
    )

    assert len(results) == 0


def test_sparse_retriever_retrieve_with_trace(sparse_retriever, mock_bm25_indexer, mock_vector_store):
    """Test retrieval with trace context."""
    mock_trace = Mock()

    mock_bm25_indexer.query.return_value = [
        {"chunk_id": "chunk_001", "score": 5.0},
    ]

    mock_vector_store.get_by_ids.return_value = [
        VectorRecord(
            id="chunk_001",
            text="Content",
            embedding=[0.1],
            metadata={}
        ),
    ]

    results = sparse_retriever.retrieve(
        keywords=["test"],
        top_k=10,
        trace=mock_trace,
    )

    assert len(results) == 1
    mock_bm25_indexer.query.assert_called_with(term="test", top_k=10, trace=mock_trace)
    mock_vector_store.get_by_ids.assert_called_with(ids=["chunk_001"], trace=mock_trace)


def test_sparse_retriever_retrieve_top_k_respected(sparse_retriever, mock_bm25_indexer, mock_vector_store):
    """Test that top_k parameter is respected."""
    # Return more results than top_k
    mock_bm25_indexer.query.return_value = [
        {"chunk_id": f"chunk_{i:03d}", "score": 5.0 - i*0.5}
        for i in range(20)
    ]

    # Mock vector store to return only top_k
    mock_vector_store.get_by_ids.return_value = [
        VectorRecord(
            id=f"chunk_{i:03d}",
            text=f"Content {i}",
            embedding=[0.1],
            metadata={}
        )
        for i in range(5)
    ]

    results = sparse_retriever.retrieve(
        keywords=["test"],
        top_k=5,
    )

    # Should respect top_k in BM25 query
    mock_bm25_indexer.query.assert_called_with(term="test", top_k=5, trace=None)


def test_sparse_retriever_retrieve_handles_missing_chunks(sparse_retriever, mock_bm25_indexer, mock_vector_store):
    """Test handling when some chunks are not found in vector store."""
    mock_bm25_indexer.query.return_value = [
        {"chunk_id": "chunk_001", "score": 5.0},
        {"chunk_id": "chunk_002", "score": 3.5},
        {"chunk_id": "chunk_003", "score": 2.0},
    ]

    # Only return 2 of 3 chunks (chunk_002 is missing)
    mock_vector_store.get_by_ids.return_value = [
        VectorRecord(
            id="chunk_001",
            text="Content 1",
            embedding=[0.1],
            metadata={}
        ),
        VectorRecord(
            id="chunk_003",
            text="Content 3",
            embedding=[0.3],
            metadata={}
        ),
    ]

    results = sparse_retriever.retrieve(
        keywords=["test"],
        top_k=10,
    )

    # Should return results for chunks that exist
    assert len(results) == 2
    assert results[0].chunk_id == "chunk_001"
    assert results[1].chunk_id == "chunk_003"


def test_sparse_retriever_retrieve_score_merging(sparse_retriever, mock_bm25_indexer, mock_vector_store):
    """Test that scores from BM25 are properly merged with results."""
    mock_bm25_indexer.query.side_effect = lambda term, top_k=10, trace=None: [
        {"chunk_id": "chunk_001", "score": 10.0},
    ]

    mock_vector_store.get_by_ids.return_value = [
        VectorRecord(
            id="chunk_001",
            text="Content",
            embedding=[0.1],
            metadata={"source": "doc.pdf"}
        ),
    ]

    results = sparse_retriever.retrieve(
        keywords=["test"],
        top_k=10,
    )

    assert len(results) == 1
    assert results[0].score == 10.0
    assert results[0].text == "Content"
    assert results[0].metadata == {"source": "doc.pdf"}
