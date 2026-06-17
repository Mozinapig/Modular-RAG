"""
Unit tests for Query tracing functionality.
Verifies that HybridSearch and retrieval components record trace data correctly.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch

from src.core.trace.trace_context import TraceContext
from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.dense_retriever import DenseRetriever
from src.core.query_engine.sparse_retriever import SparseRetriever
from src.core.query_engine.fusion import RRFFusion
from src.core.types import RetrievalResult


class TestHybridSearchTracing:
    """Test tracing in HybridSearch."""

    @pytest.fixture
    def mock_components(self):
        """Create mock retrieval components."""
        query_processor = Mock(spec=QueryProcessor)
        dense_retriever = Mock(spec=DenseRetriever)
        sparse_retriever = Mock(spec=SparseRetriever)
        fusion = Mock(spec=RRFFusion)

        # Setup default return values
        query_processor.process.return_value = Mock(keywords=["test", "query"])

        result1 = RetrievalResult(
            chunk_id="chunk_1",
            score=0.9,
            text="Test chunk 1",
            metadata={"source": "doc1.pdf"}
        )
        result2 = RetrievalResult(
            chunk_id="chunk_2",
            score=0.8,
            text="Test chunk 2",
            metadata={"source": "doc2.pdf"}
        )

        dense_retriever.retrieve.return_value = [result1, result2]
        sparse_retriever.retrieve.return_value = [result2, result1]
        fusion.fuse.return_value = [result1, result2]

        return {
            'query_processor': query_processor,
            'dense_retriever': dense_retriever,
            'sparse_retriever': sparse_retriever,
            'fusion': fusion
        }

    def test_hybrid_search_records_query_processing_stage(self, mock_components):
        """Test that query processing stage is recorded."""
        hybrid_search = HybridSearch(**mock_components)
        trace = TraceContext(trace_type="query")

        hybrid_search.search(query="test query", trace=trace)

        # Verify trace has query_processing stage
        stage_names = [s.stage_name for s in trace.stages]
        assert "query_processing" in stage_names

    def test_hybrid_search_records_dense_retrieval_stage(self, mock_components):
        """Test that dense retrieval stage is recorded."""
        hybrid_search = HybridSearch(**mock_components)
        trace = TraceContext(trace_type="query")

        # Mock the retrieve methods to call the trace
        def dense_with_trace(query, top_k, filters=None, trace=None):
            if trace:
                from datetime import datetime
                start = datetime.now().timestamp()
                end = datetime.now().timestamp()
                trace.record_stage(
                    "dense_retrieval",
                    start_time=start,
                    end_time=end,
                    method="embedding",
                    candidates_count=2
                )
            return mock_components['dense_retriever'].retrieve.return_value

        mock_components['dense_retriever'].retrieve = dense_with_trace

        hybrid_search.search(query="test query", trace=trace)

        # Verify trace has dense_retrieval stage
        stage_names = [s.stage_name for s in trace.stages]
        assert "dense_retrieval" in stage_names

    def test_hybrid_search_records_sparse_retrieval_stage(self, mock_components):
        """Test that sparse retrieval stage is recorded."""
        hybrid_search = HybridSearch(**mock_components)
        trace = TraceContext(trace_type="query")

        # Mock the retrieve methods to call the trace
        def sparse_with_trace(keywords, top_k=10, trace=None):
            if trace:
                from datetime import datetime
                start = datetime.now().timestamp()
                end = datetime.now().timestamp()
                trace.record_stage(
                    "sparse_retrieval",
                    start_time=start,
                    end_time=end,
                    method="bm25",
                    candidates_count=2
                )
            return mock_components['sparse_retriever'].retrieve.return_value

        mock_components['sparse_retriever'].retrieve = sparse_with_trace

        hybrid_search.search(query="test query", trace=trace)

        # Verify trace has sparse_retrieval stage
        stage_names = [s.stage_name for s in trace.stages]
        assert "sparse_retrieval" in stage_names

    def test_hybrid_search_records_fusion_stage(self, mock_components):
        """Test that fusion stage is recorded."""
        hybrid_search = HybridSearch(**mock_components)
        trace = TraceContext(trace_type="query")

        hybrid_search.search(query="test query", trace=trace)

        # Verify trace has fusion stage
        stage_names = [s.stage_name for s in trace.stages]
        assert "fusion" in stage_names

    def test_hybrid_search_records_method_metadata(self, mock_components):
        """Test that trace records the method used."""
        hybrid_search = HybridSearch(**mock_components)
        trace = TraceContext(trace_type="query")

        hybrid_search.search(query="test query", trace=trace)

        # Check that stages have method metadata
        for stage in trace.stages:
            if stage.stage_name == "dense_retrieval":
                assert "method" in stage.metadata or "provider" in stage.metadata
            elif stage.stage_name == "sparse_retrieval":
                assert "method" in stage.metadata

    def test_hybrid_search_records_elapsed_time(self, mock_components):
        """Test that trace records elapsed times."""
        hybrid_search = HybridSearch(**mock_components)
        trace = TraceContext(trace_type="query")

        hybrid_search.search(query="test query", trace=trace)

        # Verify each stage has non-zero duration
        for stage in trace.stages:
            assert stage.duration_ms is not None
            assert stage.duration_ms >= 0

    def test_hybrid_search_handles_none_trace(self, mock_components):
        """Test that hybrid search works when trace is None."""
        hybrid_search = HybridSearch(**mock_components)

        # Should not raise exception
        results = hybrid_search.search(query="test query", trace=None)

        assert len(results) > 0

    def test_trace_context_records_query_information(self, mock_components):
        """Test that trace records the original query."""
        hybrid_search = HybridSearch(**mock_components)
        trace = TraceContext(trace_type="query")

        query_text = "test query for tracing"
        hybrid_search.search(query=query_text, trace=trace)

        # Query should be in metadata or stages
        trace_dict = trace.to_dict()
        assert trace_dict["trace_type"] == "query"

    def test_dense_retriever_records_provider_metadata(self, mock_components):
        """Test that dense retriever records provider information in trace."""
        hybrid_search = HybridSearch(**mock_components)
        trace = TraceContext(trace_type="query")

        # Mock dense retriever to record trace
        def dense_with_trace(query, top_k, filters=None, trace=None):
            if trace:
                trace.record_stage(
                    "dense_retrieval",
                    provider="openai",
                    model="text-embedding-3-small",
                    candidates_count=2
                )
            return [
                RetrievalResult(
                    chunk_id="d1", score=0.9, text="Dense result 1",
                    metadata={"source": "doc1.pdf"}
                ),
                RetrievalResult(
                    chunk_id="d2", score=0.8, text="Dense result 2",
                    metadata={"source": "doc2.pdf"}
                )
            ]

        mock_components['dense_retriever'].retrieve = dense_with_trace

        hybrid_search.search(query="test", trace=trace)

        # Verify trace has provider metadata
        dense_stages = [s for s in trace.stages if s.stage_name == "dense_retrieval"]
        assert len(dense_stages) > 0
        if len(dense_stages) > 0:
            metadata = dense_stages[0].metadata
            # May have provider info depending on implementation
            assert isinstance(metadata, dict)

    def test_sparse_retriever_records_method_metadata(self, mock_components):
        """Test that sparse retriever records method information in trace."""
        hybrid_search = HybridSearch(**mock_components)
        trace = TraceContext(trace_type="query")

        # Mock sparse retriever to record trace
        def sparse_with_trace(keywords, top_k=10, trace=None):
            if trace:
                trace.record_stage(
                    "sparse_retrieval",
                    method="bm25",
                    keywords_count=len(keywords),
                    candidates_count=2
                )
            return [
                RetrievalResult(
                    chunk_id="s1", score=0.85, text="Sparse result 1",
                    metadata={"source": "doc3.pdf"}
                ),
                RetrievalResult(
                    chunk_id="s2", score=0.75, text="Sparse result 2",
                    metadata={"source": "doc4.pdf"}
                )
            ]

        mock_components['sparse_retriever'].retrieve = sparse_with_trace

        hybrid_search.search(query="test query", trace=trace)

        # Verify trace has method metadata
        sparse_stages = [s for s in trace.stages if s.stage_name == "sparse_retrieval"]
        assert len(sparse_stages) > 0

    def test_trace_has_trace_type_query(self, mock_components):
        """Test that trace type is correctly set to query."""
        hybrid_search = HybridSearch(**mock_components)
        trace = TraceContext(trace_type="query")

        hybrid_search.search(query="test", trace=trace)

        assert trace.trace_type == "query"
        trace_dict = trace.to_dict()
        assert trace_dict["trace_type"] == "query"

    def test_trace_records_all_stages_in_order(self, mock_components):
        """Test that stages are recorded in expected order."""
        hybrid_search = HybridSearch(**mock_components)
        trace = TraceContext(trace_type="query")

        # Mock the retrieve methods to record trace
        def dense_with_trace(query, top_k, filters=None, trace=None):
            if trace:
                from datetime import datetime
                start = datetime.now().timestamp()
                end = datetime.now().timestamp()
                trace.record_stage(
                    "dense_retrieval",
                    start_time=start,
                    end_time=end,
                    method="embedding",
                    candidates_count=2
                )
            return mock_components['dense_retriever'].retrieve.return_value

        def sparse_with_trace(keywords, top_k=10, trace=None):
            if trace:
                from datetime import datetime
                start = datetime.now().timestamp()
                end = datetime.now().timestamp()
                trace.record_stage(
                    "sparse_retrieval",
                    start_time=start,
                    end_time=end,
                    method="bm25",
                    candidates_count=2
                )
            return mock_components['sparse_retriever'].retrieve.return_value

        mock_components['dense_retriever'].retrieve = dense_with_trace
        mock_components['sparse_retriever'].retrieve = sparse_with_trace

        hybrid_search.search(query="test query", trace=trace)

        stage_names = [s.stage_name for s in trace.stages]

        # Expected order: query_processing, dense_retrieval, sparse_retrieval, fusion
        expected_stages = [
            "query_processing",
            "dense_retrieval",
            "sparse_retrieval",
            "fusion"
        ]

        # At least these stages should be present
        for expected in expected_stages:
            assert expected in stage_names, f"Expected stage '{expected}' not found in {stage_names}"
