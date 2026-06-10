"""
Unit tests for Response Builder and Citation Generator (E3)
"""
from unittest.mock import MagicMock

import pytest

from src.core.response.response_builder import ResponseBuilder
from src.core.response.citation_generator import CitationGenerator


class TestResponseBuilder:
    """Test ResponseBuilder functionality"""

    def test_response_builder_initialization(self):
        """Test ResponseBuilder can be initialized"""
        builder = ResponseBuilder()
        assert builder is not None

    def test_build_with_empty_results(self):
        """Test building response with no results"""
        builder = ResponseBuilder()
        result = builder.build([], "test query")

        assert "No results found" in result

    def test_build_with_single_result(self):
        """Test building response with single result"""
        builder = ResponseBuilder()

        mock_result = MagicMock()
        mock_result.content = "Test content"

        markdown = builder.build([mock_result], "test query")

        assert "[1]" in markdown
        assert "Test content" in markdown

    def test_build_with_multiple_results(self):
        """Test building response with multiple results"""
        builder = ResponseBuilder()

        results = []
        for i in range(3):
            mock_result = MagicMock()
            mock_result.content = f"Content {i}"
            results.append(mock_result)

        markdown = builder.build(results, "test query")

        assert "[1]" in markdown
        assert "[2]" in markdown
        assert "[3]" in markdown

    def test_build_includes_query(self):
        """Test response includes the query"""
        builder = ResponseBuilder()

        mock_result = MagicMock()
        mock_result.content = "Test"

        markdown = builder.build([mock_result], "my search query")

        assert "my search query" in markdown

    def test_build_error_returns_error_text(self):
        """Test building error response"""
        builder = ResponseBuilder()
        error_msg = "Connection failed"

        response = builder.build_error(error_msg)

        assert "Error" in response
        assert error_msg in response


class TestCitationGenerator:
    """Test CitationGenerator functionality"""

    def test_citation_generator_initialization(self):
        """Test CitationGenerator can be initialized"""
        gen = CitationGenerator()
        assert gen is not None

    def test_generate_with_empty_results(self):
        """Test generating citations with no results"""
        gen = CitationGenerator()
        citations = gen.generate([])

        assert citations == []

    def test_generate_creates_citations_with_indices(self):
        """Test generated citations have correct indices"""
        gen = CitationGenerator()

        results = []
        for i in range(3):
            mock_result = MagicMock()
            mock_result.source = f"doc{i}.pdf"
            mock_result.chunk_id = f"chunk_{i}"
            mock_result.score = 0.9 - i * 0.1
            results.append(mock_result)

        citations = gen.generate(results)

        assert len(citations) == 3
        assert citations[0]['index'] == 1
        assert citations[1]['index'] == 2
        assert citations[2]['index'] == 3

    def test_citation_has_required_fields(self):
        """Test each citation has required fields"""
        gen = CitationGenerator()

        mock_result = MagicMock()
        mock_result.source = "test.pdf"
        mock_result.page = 5
        mock_result.chunk_id = "chunk_001"
        mock_result.score = 0.95

        citations = gen.generate([mock_result])

        assert 'index' in citations[0]
        assert 'source' in citations[0]
        assert 'page' in citations[0]
        assert 'chunk_id' in citations[0]
        assert 'score' in citations[0]

    def test_citation_preserves_metadata(self):
        """Test citations preserve all metadata"""
        gen = CitationGenerator()

        mock_result = MagicMock()
        mock_result.source = "guide.pdf"
        mock_result.page = 42
        mock_result.chunk_id = "abc_123"
        mock_result.score = 0.88

        citations = gen.generate([mock_result])

        citation = citations[0]
        assert citation['source'] == "guide.pdf"
        assert citation['page'] == 42
        assert citation['chunk_id'] == "abc_123"
        assert citation['score'] == 0.88

    def test_generate_handles_missing_fields(self):
        """Test generator handles results with missing fields"""
        gen = CitationGenerator()

        mock_result = MagicMock()
        # Only set some attributes
        mock_result.source = "doc.pdf"
        # page, chunk_id, score are not set
        del mock_result.page
        del mock_result.chunk_id
        del mock_result.score

        citations = gen.generate([mock_result])

        # Should still create citation with None values
        assert len(citations) == 1
        assert citations[0]['source'] == "doc.pdf"
