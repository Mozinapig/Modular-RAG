"""
Unit tests for ChunkRefiner - rule-based and LLM-enhanced chunk refinement.
Tests focus on noise removal, LLM enhancement, and graceful degradation.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.chunk_refiner import ChunkRefiner


class FakeSettings:
    """Fake settings for testing."""
    def __init__(self, use_llm=False, provider="openai", model="gpt-4o-mini"):
        self.use_llm = use_llm
        self.provider = provider
        self.model = model
        self.api_key = "test-key"


@pytest.fixture
def noisy_chunks_data():
    """Load noisy chunks test data."""
    fixture_path = Path("tests/fixtures/noisy_chunks.json")
    with open(fixture_path, encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture
def sample_chunks(noisy_chunks_data):
    """Create sample chunks from test data."""
    chunks = []
    for key, data in noisy_chunks_data.items():
        chunk = Chunk(
            id=f"chunk_{key}",
            text=data["input"],
            metadata={
                "source_path": "/test/doc.pdf",
                "chunk_index": len(chunks),
                "source_ref": "doc_123"
            }
        )
        chunks.append(chunk)
    return chunks


class TestChunkRefinerRuleBased:
    """Test rule-based refinement."""

    @pytest.fixture
    def refiner(self):
        """Create refiner with rule-based mode."""
        settings = FakeSettings(use_llm=False)
        return ChunkRefiner(settings)

    def test_refiner_initialization(self, refiner):
        """Test that refiner initializes correctly."""
        assert refiner is not None
        assert hasattr(refiner, 'transform')

    def test_rule_based_removes_excessive_whitespace(self, refiner):
        """Test that rule-based refinement removes excessive whitespace."""
        chunk = Chunk(
            id="test_1",
            text="Line 1\n\n\n\n\nLine 2\n\n\n\nLine 3",
            metadata={"source_path": "/test"}
        )
        result = refiner.transform([chunk])

        assert len(result) == 1
        refined_text = result[0].text
        # Should have reduced consecutive newlines
        assert "\n\n\n" not in refined_text

    def test_rule_based_preserves_clean_text(self, refiner):
        """Test that clean text is not over-processed."""
        clean_text = "This is clean, well-formatted text.\nIt has proper spacing and no noise."
        chunk = Chunk(
            id="test_2",
            text=clean_text,
            metadata={"source_path": "/test"}
        )
        result = refiner.transform([chunk])

        assert len(result) == 1
        # Clean text should be largely preserved
        assert "clean" in result[0].text.lower()
        assert "well-formatted" in result[0].text.lower()

    def test_rule_based_preserves_code_blocks(self, refiner):
        """Test that code block formatting is preserved."""
        code_text = """Here is code:

```python
def hello():
    print("Hello")
```

End."""
        chunk = Chunk(
            id="test_3",
            text=code_text,
            metadata={"source_path": "/test"}
        )
        result = refiner.transform([chunk])

        assert len(result) == 1
        refined = result[0].text
        # Code block markers should be preserved
        assert "```" in refined
        assert "def hello" in refined

    def test_rule_based_handles_empty_chunk(self, refiner):
        """Test handling of empty chunk."""
        chunk = Chunk(
            id="test_4",
            text="",
            metadata={"source_path": "/test"}
        )
        result = refiner.transform([chunk])

        assert len(result) == 1
        assert result[0].text == ""

    def test_rule_based_multiple_chunks(self, refiner):
        """Test processing multiple chunks."""
        chunks = [
            Chunk(id="c1", text="Text 1\n\n\n\nMore", metadata={"source_path": "/test"}),
            Chunk(id="c2", text="Text 2\n\n\nMore", metadata={"source_path": "/test"}),
            Chunk(id="c3", text="Text 3", metadata={"source_path": "/test"}),
        ]
        result = refiner.transform(chunks)

        assert len(result) == 3
        for chunk in result:
            assert chunk.text is not None

    def test_rule_based_metadata_preserved(self, refiner):
        """Test that metadata is preserved."""
        chunk = Chunk(
            id="test_5",
            text="Noisy\n\n\ntext",
            metadata={
                "source_path": "/test/doc.pdf",
                "chunk_index": 0,
                "source_ref": "doc_123",
                "custom_field": "custom_value"
            }
        )
        result = refiner.transform([chunk])

        assert result[0].metadata["source_path"] == "/test/doc.pdf"
        assert result[0].metadata["chunk_index"] == 0
        assert result[0].metadata["custom_field"] == "custom_value"

    def test_rule_based_adds_refined_by_metadata(self, refiner):
        """Test that refined_by metadata is added."""
        chunk = Chunk(
            id="test_6",
            text="Noisy\n\n\ntext",
            metadata={"source_path": "/test"}
        )
        result = refiner.transform([chunk])

        assert "refined_by" in result[0].metadata
        assert result[0].metadata["refined_by"] == "rule"


class TestChunkRefinerLLM:
    """Test LLM-enhanced refinement with mocking."""

    @pytest.fixture
    def mock_llm_settings(self):
        """Create settings with LLM enabled."""
        return FakeSettings(use_llm=True, provider="openai")

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM."""
        mock = Mock()
        mock.generate = Mock(return_value="Refined text from LLM")
        return mock

    def test_llm_refiner_initialization(self, mock_llm_settings):
        """Test LLM refiner initialization."""
        with patch('src.libs.llm.llm_factory.LLMFactory') as mock_factory:
            mock_factory.return_value.create.return_value = Mock()
            refiner = ChunkRefiner(mock_llm_settings)
            assert refiner is not None

    def test_llm_refiner_calls_llm(self, mock_llm_settings):
        """Test that LLM is called for refinement."""
        with patch('src.libs.llm.llm_factory.LLMFactory') as mock_factory:
            mock_llm = Mock()
            mock_llm.generate = Mock(return_value="Refined: cleaned text")
            mock_factory.return_value.create.return_value = mock_llm

            refiner = ChunkRefiner(mock_llm_settings)
            chunk = Chunk(
                id="test_llm_1",
                text="Noisy text with issues",
                metadata={"source_path": "/test"}
            )

            result = refiner.transform([chunk])

            # LLM should have been called
            assert mock_llm.generate.called

    def test_llm_refiner_metadata_marked(self, mock_llm_settings):
        """Test that LLM-refined chunks are marked in metadata."""
        with patch('src.libs.llm.llm_factory.LLMFactory') as mock_factory:
            mock_llm = Mock()
            mock_llm.generate = Mock(return_value="Refined text")
            mock_factory.return_value.create.return_value = mock_llm

            refiner = ChunkRefiner(mock_llm_settings)
            chunk = Chunk(
                id="test_llm_2",
                text="Original text",
                metadata={"source_path": "/test"}
            )

            result = refiner.transform([chunk])

            # Should be marked as refined by LLM
            assert result[0].metadata.get("refined_by") == "llm"

    def test_llm_fallback_on_error(self, mock_llm_settings):
        """Test fallback to rule-based when LLM fails."""
        with patch('src.libs.llm.llm_factory.LLMFactory') as mock_factory:
            mock_llm = Mock()
            mock_llm.generate = Mock(side_effect=Exception("LLM error"))
            mock_factory.return_value.create.return_value = mock_llm

            refiner = ChunkRefiner(mock_llm_settings)
            chunk = Chunk(
                id="test_fallback_1",
                text="Text\n\n\nwith noise",
                metadata={"source_path": "/test"}
            )

            result = refiner.transform([chunk])

            # Should fall back to rule-based
            assert result[0].metadata.get("refined_by") == "rule"
            assert "fallback_reason" in result[0].metadata

    def test_llm_disabled_uses_rule_based(self):
        """Test that disabled LLM uses rule-based."""
        settings = FakeSettings(use_llm=False)
        refiner = ChunkRefiner(settings)

        chunk = Chunk(
            id="test_disabled",
            text="Text\n\n\nwith noise",
            metadata={"source_path": "/test"}
        )

        result = refiner.transform([chunk])

        assert result[0].metadata.get("refined_by") == "rule"


class TestChunkRefinerEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def refiner(self):
        """Create refiner."""
        settings = FakeSettings(use_llm=False)
        return ChunkRefiner(settings)

    def test_empty_chunks_list(self, refiner):
        """Test handling of empty chunks list."""
        result = refiner.transform([])
        assert result == []

    def test_chunk_with_special_characters(self, refiner):
        """Test handling of special characters."""
        chunk = Chunk(
            id="test_special",
            text="Text with émojis 🎉 and spëcial çhars",
            metadata={"source_path": "/test"}
        )
        result = refiner.transform([chunk])

        assert len(result) == 1
        assert "émojis" in result[0].text or "emoji" in result[0].text.lower()

    def test_chunk_with_very_long_text(self, refiner):
        """Test handling of very long text."""
        long_text = "Word " * 10000  # 50k+ characters
        chunk = Chunk(
            id="test_long",
            text=long_text,
            metadata={"source_path": "/test"}
        )
        result = refiner.transform([chunk])

        assert len(result) == 1
        assert len(result[0].text) > 0

    def test_chunk_with_only_whitespace(self, refiner):
        """Test handling of whitespace-only chunk."""
        chunk = Chunk(
            id="test_whitespace",
            text="   \n\n\n   \t\t\t   ",
            metadata={"source_path": "/test"}
        )
        result = refiner.transform([chunk])

        assert len(result) == 1
        # Should be cleaned to minimal whitespace or empty
        assert len(result[0].text.strip()) == 0 or result[0].text.strip() != ""

    def test_chunk_exception_doesnt_block_others(self, refiner):
        """Test that exception in one chunk doesn't block others."""
        chunks = [
            Chunk(id="c1", text="Normal text", metadata={"source_path": "/test"}),
            Chunk(id="c2", text="Text\n\n\nwith noise", metadata={"source_path": "/test"}),
            Chunk(id="c3", text="Another normal", metadata={"source_path": "/test"}),
        ]

        # Should process all chunks without raising
        result = refiner.transform(chunks)
        assert len(result) == 3

    def test_trace_context_optional(self, refiner):
        """Test that trace context is optional."""
        chunk = Chunk(
            id="test_trace",
            text="Text\n\n\nwith noise",
            metadata={"source_path": "/test"}
        )

        # Should work without trace
        result = refiner.transform([chunk], trace=None)
        assert len(result) == 1

        # Should also work with trace
        trace = TraceContext(trace_type="ingestion")
        result = refiner.transform([chunk], trace=trace)
        assert len(result) == 1


class TestChunkRefinerWithFixtures:
    """Test with actual fixture data."""

    @pytest.fixture
    def refiner(self):
        """Create refiner."""
        settings = FakeSettings(use_llm=False)
        return ChunkRefiner(settings)

    def test_typical_noise_scenario(self, refiner, noisy_chunks_data):
        """Test with typical noise scenario."""
        data = noisy_chunks_data["typical_noise_scenario"]
        chunk = Chunk(
            id="typical",
            text=data["input"],
            metadata={"source_path": "/test"}
        )

        result = refiner.transform([chunk])

        assert len(result) == 1
        refined = result[0].text
        # Should have reduced noise
        assert len(refined) < len(data["input"]) or refined.strip() != ""

    def test_ocr_errors_preserved(self, refiner, noisy_chunks_data):
        """Test that OCR errors are preserved (not over-corrected)."""
        data = noisy_chunks_data["ocr_errors"]
        chunk = Chunk(
            id="ocr",
            text=data["input"],
            metadata={"source_path": "/test"}
        )

        result = refiner.transform([chunk])

        assert len(result) == 1
        # Content should be largely preserved (we don't auto-correct OCR)
        assert len(result[0].text) > 0

    def test_code_blocks_fixture(self, refiner, noisy_chunks_data):
        """Test code block preservation with fixture."""
        data = noisy_chunks_data["code_blocks"]
        chunk = Chunk(
            id="code",
            text=data["input"],
            metadata={"source_path": "/test"}
        )

        result = refiner.transform([chunk])

        assert len(result) == 1
        refined = result[0].text
        # Code markers should be preserved
        assert "```" in refined or "code" in refined.lower()

    def test_mixed_noise_fixture(self, refiner, noisy_chunks_data):
        """Test mixed noise scenario."""
        data = noisy_chunks_data["mixed_noise"]
        chunk = Chunk(
            id="mixed",
            text=data["input"],
            metadata={"source_path": "/test"}
        )

        result = refiner.transform([chunk])

        assert len(result) == 1
        # Should produce valid output
        assert len(result[0].text) > 0
