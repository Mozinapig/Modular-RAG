"""Tests for QueryProcessor."""
import pytest
from src.core.query_engine.query_processor import QueryProcessor, ProcessedQuery


class TestProcessedQuery:
    """Test ProcessedQuery data class."""

    def test_processed_query_creation(self):
        """Test creating a ProcessedQuery instance."""
        query = ProcessedQuery(keywords=["hello", "world"], filters={})
        assert query.keywords == ["hello", "world"]
        assert query.filters == {}

    def test_processed_query_with_filters(self):
        """Test ProcessedQuery with filters."""
        filters = {"collection": "test", "doc_type": "pdf"}
        query = ProcessedQuery(keywords=["test"], filters=filters)
        assert query.filters == filters


class TestQueryProcessor:
    """Test QueryProcessor."""

    def test_init(self):
        """Test QueryProcessor initialization."""
        processor = QueryProcessor()
        assert processor is not None

    def test_simple_query(self):
        """Test processing a simple query."""
        processor = QueryProcessor()
        result = processor.process("how to configure azure")

        assert isinstance(result, ProcessedQuery)
        assert len(result.keywords) > 0
        assert isinstance(result.filters, dict)

    def test_query_lowercase_conversion(self):
        """Test that query is converted to lowercase."""
        processor = QueryProcessor()
        result = processor.process("HELLO World")

        # Keywords should be in lowercase
        assert all(kw.islower() for kw in result.keywords)

    def test_stopword_removal(self):
        """Test that common stopwords are removed."""
        processor = QueryProcessor()
        result = processor.process("the quick brown fox jumps")

        # "the" should be filtered out as stopword
        assert "the" not in result.keywords
        # Other words should remain
        assert any(kw in result.keywords for kw in ["quick", "brown", "fox", "jumps"])

    def test_empty_query(self):
        """Test processing an empty query."""
        processor = QueryProcessor()
        result = processor.process("")

        # Should return empty keywords
        assert result.keywords == []
        assert result.filters == {}

    def test_only_stopwords_query(self):
        """Test query with only stopwords."""
        processor = QueryProcessor()
        result = processor.process("the a an")

        # Should return empty keywords
        assert result.keywords == []

    def test_filters_parsing_basic(self):
        """Test basic filters parsing."""
        processor = QueryProcessor()
        # Filters format: key:value pairs
        result = processor.process("search term collection:test")

        assert "search" in result.keywords or "term" in result.keywords
        assert isinstance(result.filters, dict)

    def test_filters_multiple_pairs(self):
        """Test multiple filter pairs."""
        processor = QueryProcessor()
        result = processor.process("query term collection:docs doc_type:pdf")

        assert isinstance(result.filters, dict)

    def test_punctuation_removal(self):
        """Test that punctuation is handled."""
        processor = QueryProcessor()
        result = processor.process("hello, world!")

        # Should still extract keywords, with punctuation handled
        assert len(result.keywords) > 0

    def test_special_characters_handling(self):
        """Test handling of special characters."""
        processor = QueryProcessor()
        result = processor.process("C++ programming language")

        # Should be able to extract relevant keywords
        assert len(result.keywords) > 0

    def test_multiple_spaces(self):
        """Test handling of multiple spaces."""
        processor = QueryProcessor()
        result = processor.process("hello    world    test")

        # Should correctly tokenize despite multiple spaces
        assert len(result.keywords) >= 2

    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        processor = QueryProcessor()
        result = processor.process("如何配置 Azure")

        assert len(result.keywords) > 0
        assert isinstance(result.filters, dict)

    def test_numbers_in_query(self):
        """Test handling of numbers."""
        processor = QueryProcessor()
        result = processor.process("openai gpt-4 model")

        # Should extract meaningful keywords
        assert len(result.keywords) > 0

    def test_consistent_output(self):
        """Test that same input produces same output."""
        processor = QueryProcessor()
        query = "search for azure configuration"

        result1 = processor.process(query)
        result2 = processor.process(query)

        assert result1.keywords == result2.keywords
        assert result1.filters == result2.filters
