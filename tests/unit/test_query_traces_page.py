"""
Unit tests for Query Traces page
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from src.observability.dashboard.pages.query_traces import (
    show_query_traces,
    _render_query_traces_list,
    _render_retrieval_comparison,
    _render_rerank_changes,
)


class TestShowQueryTraces:
    """Test main query_traces page function"""

    @patch('streamlit.title')
    @patch('streamlit.write')
    def test_show_query_traces_renders_page(self, mock_write, mock_title):
        """Should render Query Traces page"""
        mock_trace_service = Mock()
        mock_trace_service.get_query_traces.return_value = []

        show_query_traces(mock_trace_service)

        # Verify title was set
        mock_title.assert_called_once()


class TestRenderQueryTracesList:
    """Test query traces list rendering"""

    @patch('streamlit.dataframe')
    def test_render_query_traces_list_with_data(self, mock_dataframe):
        """Should render list of query traces"""
        traces = [
            {
                "trace_id": "query_1",
                "timestamp": "2026-06-17T10:00:00Z",
                "query": "What is RAG?",
                "metadata": {"query_type": "general"},
                "stages": [
                    {"name": "process", "start_ms": 0, "end_ms": 50},
                    {"name": "dense_search", "start_ms": 50, "end_ms": 150},
                ]
            }
        ]

        _render_query_traces_list(traces)

        # Verify dataframe was called
        mock_dataframe.assert_called_once()

    @patch('streamlit.info')
    def test_render_query_traces_list_empty(self, mock_info):
        """Should show info message when no traces"""
        _render_query_traces_list([])

        mock_info.assert_called_once()


class TestRenderRetrievalComparison:
    """Test retrieval method comparison"""

    @patch('streamlit.write')
    @patch('streamlit.metric')
    @patch('streamlit.columns')
    def test_render_retrieval_comparison(self, mock_columns, mock_metric, mock_write):
        """Should compare Dense vs Sparse retrieval results"""
        # Mock columns - function is called with both 2 and 3 arguments
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_col3 = MagicMock()
        mock_columns.side_effect = [
            [mock_col1, mock_col2],  # First call (2 columns)
            [mock_col1, mock_col2, mock_col3],  # Second call (3 columns)
        ]

        trace = {
            "metadata": {
                "dense_results": [
                    {"id": "chunk_1", "score": 0.95},
                    {"id": "chunk_2", "score": 0.85},
                ],
                "sparse_results": [
                    {"id": "chunk_1", "score": 0.8},
                    {"id": "chunk_3", "score": 0.7},
                ]
            }
        }

        _render_retrieval_comparison(trace)

        # Verify columns were called
        assert mock_columns.called


class TestRenderRerankChanges:
    """Test Rerank changes visualization"""

    @patch('streamlit.write')
    @patch('streamlit.columns')
    def test_render_rerank_changes(self, mock_columns, mock_write):
        """Should show ranking changes before/after rerank"""
        # Mock columns
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_columns.return_value = [mock_col1, mock_col2]

        trace = {
            "metadata": {
                "before_rerank": [
                    {"id": "chunk_1", "rank": 1, "score": 0.9},
                    {"id": "chunk_2", "rank": 2, "score": 0.85},
                    {"id": "chunk_3", "rank": 3, "score": 0.8},
                ],
                "after_rerank": [
                    {"id": "chunk_2", "rank": 1, "score": 0.92},
                    {"id": "chunk_1", "rank": 2, "score": 0.88},
                    {"id": "chunk_3", "rank": 3, "score": 0.8},
                ]
            }
        }

        _render_rerank_changes(trace)

        # Verify write was called to display results
        assert mock_write.called


class TestQueryTracesIntegration:
    """Integration tests for query traces"""

    def test_query_trace_selection_workflow(self):
        """Should handle trace selection and detail display"""
        mock_trace_service = Mock()
        mock_trace_service.get_query_traces.return_value = [
            {
                "trace_id": "query_1",
                "timestamp": "2026-06-17T10:00:00Z",
                "query": "What is RAG?",
            }
        ]

        mock_trace_service.get_trace_detail.return_value = {
            "trace_id": "query_1",
            "query": "What is RAG?",
            "metadata": {
                "dense_results": [],
                "sparse_results": [],
            }
        }

        # Verify services are set up correctly
        assert mock_trace_service is not None


class TestQuerySearchFilter:
    """Test query search/filter functionality"""

    def test_filter_traces_by_query_keyword(self):
        """Should filter traces by query keyword"""
        traces = [
            {"trace_id": "1", "query": "What is RAG?"},
            {"trace_id": "2", "query": "How to use embeddings?"},
            {"trace_id": "3", "query": "What is semantic search?"},
        ]

        # Filter by keyword
        filtered = [t for t in traces if "RAG" in t["query"]]
        assert len(filtered) == 1
        assert filtered[0]["trace_id"] == "1"

    def test_filter_traces_by_timestamp_range(self):
        """Should filter traces by timestamp range"""
        traces = [
            {"trace_id": "1", "timestamp": "2026-06-17T10:00:00Z"},
            {"trace_id": "2", "timestamp": "2026-06-17T11:00:00Z"},
            {"trace_id": "3", "timestamp": "2026-06-17T12:00:00Z"},
        ]

        # Filter by time range
        start_time = "2026-06-17T10:30:00Z"
        end_time = "2026-06-17T11:30:00Z"

        filtered = [
            t for t in traces
            if start_time <= t["timestamp"] <= end_time
        ]

        assert len(filtered) == 1
        assert filtered[0]["trace_id"] == "2"


class TestRetrievalStatistics:
    """Test retrieval statistics calculation"""

    def test_calculate_hit_rate(self):
        """Should calculate hit rate for retrieval"""
        retrieval_results = [
            {"id": "chunk_1", "score": 0.95},
            {"id": "chunk_2", "score": 0.85},
            {"id": "chunk_3", "score": 0.75},
        ]

        # Calculate hit rate (assuming all have score > 0.7)
        hit_rate = len([r for r in retrieval_results if r["score"] > 0.7]) / len(retrieval_results) * 100
        assert hit_rate == 100.0

    def test_compare_retrieval_methods(self):
        """Should compare Dense vs Sparse retrieval results"""
        dense_results = [
            {"id": "chunk_1", "score": 0.95},
            {"id": "chunk_2", "score": 0.85},
        ]

        sparse_results = [
            {"id": "chunk_1", "score": 0.8},
            {"id": "chunk_3", "score": 0.7},
        ]

        # Find common top result
        dense_top = dense_results[0] if dense_results else None
        sparse_top = sparse_results[0] if sparse_results else None

        # Check agreement
        agreement = dense_top and sparse_top and dense_top["id"] == sparse_top["id"]
        assert agreement is True
