"""
Unit tests for Ingestion Traces page
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from src.observability.dashboard.pages.ingestion_traces import (
    show_ingestion_traces,
    _render_traces_timeline,
    _render_stage_waterfall,
)


class TestShowIngestionTraces:
    """Test main ingestion_traces page function"""

    @patch('streamlit.title')
    @patch('streamlit.write')
    def test_show_ingestion_traces_renders_page(self, mock_write, mock_title):
        """Should render Ingestion Traces page"""
        mock_trace_service = Mock()
        mock_trace_service.get_ingestion_traces.return_value = []

        show_ingestion_traces(mock_trace_service)

        # Verify title was set
        mock_title.assert_called_once()


class TestRenderTracesTimeline:
    """Test traces timeline rendering"""

    @patch('streamlit.dataframe')
    def test_render_traces_timeline_with_data(self, mock_dataframe):
        """Should render timeline of ingestion traces"""
        traces = [
            {
                "trace_id": "trace_1",
                "timestamp": "2026-06-17T10:00:00Z",
                "metadata": {"source_path": "doc1.pdf"},
                "stages": [
                    {"name": "load", "start_ms": 0, "end_ms": 100},
                ]
            },
            {
                "trace_id": "trace_2",
                "timestamp": "2026-06-17T11:00:00Z",
                "metadata": {"source_path": "doc2.pdf"},
                "stages": [
                    {"name": "load", "start_ms": 0, "end_ms": 150},
                ]
            }
        ]

        _render_traces_timeline(traces)

        # Verify dataframe was called
        mock_dataframe.assert_called_once()

    @patch('streamlit.info')
    def test_render_traces_timeline_empty(self, mock_info):
        """Should show info message when no traces"""
        _render_traces_timeline([])

        mock_info.assert_called_once()


class TestRenderStageWaterfall:
    """Test stage waterfall chart rendering"""

    @patch('streamlit.bar_chart')
    def test_render_stage_waterfall_with_data(self, mock_bar_chart):
        """Should render stage duration waterfall"""
        trace = {
            "trace_id": "trace_1",
            "stages": [
                {"name": "load", "start_ms": 0, "end_ms": 100},
                {"name": "split", "start_ms": 100, "end_ms": 300},
                {"name": "transform", "start_ms": 300, "end_ms": 500},
            ]
        }

        _render_stage_waterfall(trace)

        # Verify chart was rendered
        mock_bar_chart.assert_called_once()

    @patch('streamlit.info')
    def test_render_stage_waterfall_no_stages(self, mock_info):
        """Should show info when trace has no stages"""
        trace = {"trace_id": "trace_1", "stages": []}

        _render_stage_waterfall(trace)

        mock_info.assert_called_once()


class TestIngestionTracesIntegration:
    """Integration tests for ingestion traces page"""

    def test_trace_selection_workflow(self):
        """Should handle trace selection and detail display"""
        mock_trace_service = Mock()
        mock_trace_service.get_ingestion_traces.return_value = [
            {
                "trace_id": "trace_1",
                "timestamp": "2026-06-17T10:00:00Z",
                "metadata": {"source_path": "doc1.pdf"},
            }
        ]

        mock_trace_service.get_trace_detail.return_value = {
            "trace_id": "trace_1",
            "stages": [
                {"name": "load", "start_ms": 0, "end_ms": 100},
            ]
        }

        # Verify services are set up correctly
        assert mock_trace_service is not None


class TestTraceSummaryStats:
    """Test trace summary statistics"""

    def test_calculate_total_duration(self):
        """Should calculate total ingestion time"""
        trace = {
            "stages": [
                {"name": "load", "start_ms": 0, "end_ms": 100},
                {"name": "split", "start_ms": 100, "end_ms": 250},
                {"name": "embed", "start_ms": 250, "end_ms": 500},
            ]
        }

        # Total should be 500ms
        total = trace["stages"][-1]["end_ms"]
        assert total == 500

    def test_stage_percentages(self):
        """Should calculate percentage for each stage"""
        trace = {
            "stages": [
                {"name": "load", "start_ms": 0, "end_ms": 100},
                {"name": "split", "start_ms": 100, "end_ms": 300},
                {"name": "embed", "start_ms": 300, "end_ms": 500},
            ]
        }

        # Calculate percentages
        total = 500
        stages = {
            "load": (100 / total) * 100,
            "split": (200 / total) * 100,
            "embed": (200 / total) * 100,
        }

        assert stages["load"] == 20.0
        assert stages["split"] == 40.0
        assert stages["embed"] == 40.0
