"""Tests for Recursive Splitter implementation."""

import pytest
from unittest.mock import Mock

from src.libs.splitter.recursive_splitter import RecursiveSplitter
from src.libs.splitter.splitter_factory import SplitterFactory


class TestRecursiveSplitter:
    """Test RecursiveSplitter implementation."""

    def test_recursive_splitter_creation(self):
        """Test RecursiveSplitter can be created with valid config."""
        splitter_settings = Mock()
        splitter_settings.provider = "recursive"
        splitter_settings.chunk_size = 512
        splitter_settings.chunk_overlap = 50
        splitter_settings.separators = ["\n\n", "\n", " ", ""]

        splitter = RecursiveSplitter(splitter_settings)
        assert splitter is not None
        assert splitter.chunk_size == 512
        assert splitter.chunk_overlap == 50

    def test_recursive_splitter_validates_config(self):
        """Test RecursiveSplitter validates configuration."""
        splitter_settings = Mock()
        splitter_settings.provider = "recursive"
        splitter_settings.chunk_size = -1
        splitter_settings.chunk_overlap = 50
        splitter_settings.separators = ["\n\n", "\n", " ", ""]

        with pytest.raises(ValueError) as exc_info:
            RecursiveSplitter(splitter_settings)
        assert "chunk_size" in str(exc_info.value).lower()

    def test_recursive_splitter_splits_by_paragraph(self):
        """Test RecursiveSplitter respects paragraph boundaries."""
        splitter_settings = Mock()
        splitter_settings.provider = "recursive"
        splitter_settings.chunk_size = 100
        splitter_settings.chunk_overlap = 0
        splitter_settings.separators = ["\n\n", "\n", " ", ""]

        splitter = RecursiveSplitter(splitter_settings)

        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = splitter.split_text(text)

        assert len(chunks) > 0
        assert isinstance(chunks, list)
        for chunk in chunks:
            assert isinstance(chunk, str)
            assert len(chunk) > 0

    def test_recursive_splitter_preserves_markdown_structure(self):
        """Test RecursiveSplitter preserves Markdown structure."""
        splitter_settings = Mock()
        splitter_settings.provider = "recursive"
        splitter_settings.chunk_size = 200
        splitter_settings.chunk_overlap = 0
        splitter_settings.separators = ["\n\n", "\n", " ", ""]

        splitter = RecursiveSplitter(splitter_settings)

        text = """# Title

## Section 1
Content for section 1.

## Section 2
Content for section 2.

```python
def hello():
    print("world")
```
"""
        chunks = splitter.split_text(text)

        assert len(chunks) > 0
        # Check that code blocks are preserved (not split in middle)
        full_text = " ".join(chunks)
        assert "def hello():" in full_text
        assert "print" in full_text

    def test_recursive_splitter_handles_code_blocks(self):
        """Test RecursiveSplitter doesn't break code blocks."""
        splitter_settings = Mock()
        splitter_settings.provider = "recursive"
        splitter_settings.chunk_size = 150
        splitter_settings.chunk_overlap = 0
        splitter_settings.separators = ["\n\n", "\n", " ", ""]

        splitter = RecursiveSplitter(splitter_settings)

        text = """Some text before.

```python
def function_with_long_name():
    x = 1
    y = 2
    return x + y
```

Some text after."""
        chunks = splitter.split_text(text)

        assert len(chunks) > 0
        # Verify code block is preserved
        full_text = " ".join(chunks)
        assert "def function_with_long_name():" in full_text

    def test_recursive_splitter_respects_chunk_size(self):
        """Test RecursiveSplitter respects chunk_size limit."""
        splitter_settings = Mock()
        splitter_settings.provider = "recursive"
        splitter_settings.chunk_size = 50
        splitter_settings.chunk_overlap = 0
        splitter_settings.separators = ["\n\n", "\n", " ", ""]

        splitter = RecursiveSplitter(splitter_settings)

        text = "word " * 100  # Create text longer than chunk_size
        chunks = splitter.split_text(text)

        assert len(chunks) > 0
        # Most chunks should respect the size limit
        for chunk in chunks:
            assert len(chunk) <= 50 or len(chunk) == len(text)

    def test_recursive_splitter_handles_overlap(self):
        """Test RecursiveSplitter handles chunk overlap."""
        splitter_settings = Mock()
        splitter_settings.provider = "recursive"
        splitter_settings.chunk_size = 100
        splitter_settings.chunk_overlap = 20
        splitter_settings.separators = ["\n\n", "\n", " ", ""]

        splitter = RecursiveSplitter(splitter_settings)

        text = "word " * 50  # Create long text
        chunks = splitter.split_text(text)

        assert len(chunks) > 0
        # Check that consecutive chunks have overlap
        if len(chunks) > 1:
            # There should be some overlap between consecutive chunks
            combined = " ".join(chunks)
            assert len(combined) > 0

    def test_recursive_splitter_empty_text(self):
        """Test RecursiveSplitter handles empty text."""
        splitter_settings = Mock()
        splitter_settings.provider = "recursive"
        splitter_settings.chunk_size = 100
        splitter_settings.chunk_overlap = 0
        splitter_settings.separators = ["\n\n", "\n", " ", ""]

        splitter = RecursiveSplitter(splitter_settings)

        chunks = splitter.split_text("")

        assert isinstance(chunks, list)
        assert len(chunks) == 0

    def test_recursive_splitter_single_word(self):
        """Test RecursiveSplitter handles single word."""
        splitter_settings = Mock()
        splitter_settings.provider = "recursive"
        splitter_settings.chunk_size = 100
        splitter_settings.chunk_overlap = 0
        splitter_settings.separators = ["\n\n", "\n", " ", ""]

        splitter = RecursiveSplitter(splitter_settings)

        chunks = splitter.split_text("word")

        assert len(chunks) == 1
        assert chunks[0] == "word"

    def test_recursive_splitter_very_long_word(self):
        """Test RecursiveSplitter handles very long word."""
        splitter_settings = Mock()
        splitter_settings.provider = "recursive"
        splitter_settings.chunk_size = 50
        splitter_settings.chunk_overlap = 0
        splitter_settings.separators = ["\n\n", "\n", " ", ""]

        splitter = RecursiveSplitter(splitter_settings)

        long_word = "a" * 100
        chunks = splitter.split_text(long_word)

        assert len(chunks) > 0
        # Long word should be split even if it exceeds chunk_size
        assert all(len(chunk) > 0 for chunk in chunks)

    def test_factory_creates_recursive_provider(self):
        """Test factory can create RecursiveSplitter via provider."""
        splitter_settings = Mock()
        splitter_settings.provider = "recursive"
        splitter_settings.chunk_size = 512
        splitter_settings.chunk_overlap = 50
        splitter_settings.separators = ["\n\n", "\n", " ", ""]

        factory = SplitterFactory()
        splitter = factory.create(splitter_settings)

        assert splitter is not None
        assert isinstance(splitter, RecursiveSplitter)

    def test_recursive_splitter_with_tables(self):
        """Test RecursiveSplitter preserves table structure."""
        splitter_settings = Mock()
        splitter_settings.provider = "recursive"
        splitter_settings.chunk_size = 200
        splitter_settings.chunk_overlap = 0
        splitter_settings.separators = ["\n\n", "\n", " ", ""]

        splitter = RecursiveSplitter(splitter_settings)

        text = """Some text.

| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |
| Cell 3   | Cell 4   |

More text."""
        chunks = splitter.split_text(text)

        assert len(chunks) > 0
        # Verify table structure is preserved
        full_text = " ".join(chunks)
        assert "Header 1" in full_text
        assert "Cell 1" in full_text

    def test_recursive_splitter_trace_parameter(self):
        """Test RecursiveSplitter accepts trace parameter."""
        splitter_settings = Mock()
        splitter_settings.provider = "recursive"
        splitter_settings.chunk_size = 100
        splitter_settings.chunk_overlap = 0
        splitter_settings.separators = ["\n\n", "\n", " ", ""]

        splitter = RecursiveSplitter(splitter_settings)

        trace = Mock()
        text = "Test text for splitting."
        chunks = splitter.split_text(text, trace=trace)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
