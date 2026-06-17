"""
Unit tests for Data Browser page
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.observability.dashboard.pages.data_browser import (
    show_data_browser,
    _render_documents_table,
    _render_chunk_details,
    _render_chunk_images,
)


class TestShowDataBrowser:
    """Test main data_browser page function"""

    @patch('streamlit.title')
    @patch('streamlit.selectbox')
    @patch('streamlit.tabs')
    def test_show_data_browser_renders_sections(self, mock_tabs, mock_selectbox, mock_title):
        """Should render Data Browser with collection selector and tabs"""
        mock_data_service = Mock()
        mock_data_service.list_collections.return_value = ["collection1", "collection2"]
        mock_data_service.get_documents.return_value = []

        # Mock selectbox to return first collection
        mock_selectbox.return_value = "collection1"

        # Mock tabs context manager
        mock_tab1 = MagicMock()
        mock_tab2 = MagicMock()
        mock_tabs.return_value = [mock_tab1, mock_tab2]

        show_data_browser(mock_data_service)

        # Verify page title was set
        mock_title.assert_called_once()

        # Verify collection selector was called
        mock_selectbox.assert_called_once()


class TestRenderDocumentsTable:
    """Test documents table rendering"""

    @patch('streamlit.dataframe')
    def test_render_documents_table_with_data(self, mock_dataframe):
        """Should render dataframe with document list"""
        documents = [
            {
                "source_path": "doc1.pdf",
                "collection": "test_collection",
                "chunk_count": 10,
                "ingestion_time": "2026-06-17T10:00:00Z"
            },
            {
                "source_path": "doc2.pdf",
                "collection": "test_collection",
                "chunk_count": 5,
                "ingestion_time": "2026-06-17T11:00:00Z"
            }
        ]

        _render_documents_table(documents)

        # Verify dataframe was called
        mock_dataframe.assert_called_once()

    @patch('streamlit.info')
    def test_render_documents_table_empty(self, mock_info):
        """Should show info message when no documents"""
        _render_documents_table([])

        mock_info.assert_called_once()


class TestRenderChunkDetails:
    """Test chunk details rendering"""

    @patch('streamlit.write')
    def test_render_chunk_details_with_metadata(self, mock_write):
        """Should display chunk content and metadata"""
        chunk = {
            "id": "chunk_1",
            "content": "This is the chunk content",
            "metadata": {
                "source_path": "doc1.pdf",
                "page": 1,
                "title": "Section 1",
                "tags": ["intro", "content"]
            }
        }

        _render_chunk_details(chunk)

        # Verify write was called to display chunk info
        assert mock_write.called

    @patch('streamlit.expander')
    def test_render_chunk_details_expandable(self, mock_expander):
        """Should show chunk content in expandable section"""
        chunk = {
            "id": "chunk_1",
            "content": "Long content" * 50,  # Long content
            "metadata": {
                "source_path": "doc1.pdf",
                "page": 1
            }
        }

        mock_exp = MagicMock()
        mock_expander.return_value.__enter__.return_value = mock_exp

        _render_chunk_details(chunk)

        # Verify expander was used for content
        mock_expander.assert_called()


class TestRenderChunkImages:
    """Test chunk images rendering"""

    @patch('streamlit.image')
    def test_render_chunk_images_with_data(self, mock_image):
        """Should render images for chunk"""
        images = [
            {
                "image_id": "img_001",
                "path": "data/images/collection/img_001.png"
            },
            {
                "image_id": "img_002",
                "path": "data/images/collection/img_002.png"
            }
        ]

        _render_chunk_images(images)

        # Verify images were rendered
        assert mock_image.call_count >= 1

    @patch('streamlit.info')
    def test_render_chunk_images_empty(self, mock_info):
        """Should show info message when no images"""
        _render_chunk_images([])

        mock_info.assert_called_once()
