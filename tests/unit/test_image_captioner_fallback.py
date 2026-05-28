"""
Test suite for ImageCaptioner with Vision LLM integration and fallback behavior.
Coverage: Disabled mode, Enabled mode, Fallback scenarios, Edge cases, Integration.
Target: 20+ tests covering graceful degradation and metadata marking.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List

from src.core.types import Chunk, ImageRef
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.image_captioner import ImageCaptioner


class TestImageCapionerDisabled:
    """Tests when Vision LLM is disabled (use_llm=False)."""

    def test_disabled_chunks_without_images_passthrough(self):
        """Chunks without images should pass through unchanged."""
        settings = Mock()
        settings.ingestion.image_captioner.use_llm = False

        chunk = Chunk(
            id="chunk_001",
            text="Sample text without images",
            metadata={"source_path": "/path/to/doc.pdf"}
        )

        captioner = ImageCaptioner(settings)
        result = captioner.transform([chunk])

        assert len(result) == 1
        assert result[0].id == chunk.id
        assert result[0].text == chunk.text
        assert "has_unprocessed_images" not in result[0].metadata

    def test_disabled_chunks_with_images_mark_unprocessed(self):
        """Chunks with images should be marked as unprocessed when disabled."""
        settings = Mock()
        settings.ingestion.image_captioner.use_llm = False

        chunk = Chunk(
            id="chunk_001",
            text="Text with [IMAGE: img_001]",
            metadata={
                "source_path": "/path/to/doc.pdf",
                "images": [
                    {
                        "id": "img_001",
                        "path": "data/images/doc_hash/img_001.png",
                        "page": 1,
                        "text_offset": 10,
                        "text_length": 20
                    }
                ]
            }
        )

        captioner = ImageCaptioner(settings)
        result = captioner.transform([chunk])

        assert len(result) == 1
        assert result[0].metadata["has_unprocessed_images"] is True
        assert "image_captions" not in result[0].metadata
        assert result[0].text == chunk.text

    def test_disabled_multiple_chunks_mixed(self):
        """Mix of chunks with and without images, all marked correctly."""
        settings = Mock()
        settings.ingestion.image_captioner.use_llm = False

        chunks = [
            Chunk(
                id="chunk_001",
                text="No images here",
                metadata={"source_path": "/doc.pdf"}
            ),
            Chunk(
                id="chunk_002",
                text="Has [IMAGE: img_001]",
                metadata={
                    "source_path": "/doc.pdf",
                    "images": [{"id": "img_001", "path": "data/images/img_001.png"}]
                }
            ),
            Chunk(
                id="chunk_003",
                text="Also no images",
                metadata={"source_path": "/doc.pdf"}
            ),
        ]

        captioner = ImageCaptioner(settings)
        result = captioner.transform(chunks)

        assert len(result) == 3
        assert "has_unprocessed_images" not in result[0].metadata
        assert result[1].metadata["has_unprocessed_images"] is True
        assert "has_unprocessed_images" not in result[2].metadata

    def test_disabled_preserves_all_metadata(self):
        """All existing metadata fields should be preserved when disabled."""
        settings = Mock()
        settings.ingestion.image_captioner.use_llm = False

        chunk = Chunk(
            id="chunk_001",
            text="Text with [IMAGE: img_001]",
            metadata={
                "source_path": "/doc.pdf",
                "title": "Chapter 1",
                "summary": "Introduction",
                "tags": ["intro", "chapter"],
                "images": [{"id": "img_001", "path": "data/images/img_001.png"}]
            }
        )

        captioner = ImageCaptioner(settings)
        result = captioner.transform([chunk])

        assert result[0].metadata["source_path"] == "/doc.pdf"
        assert result[0].metadata["title"] == "Chapter 1"
        assert result[0].metadata["summary"] == "Introduction"
        assert result[0].metadata["tags"] == ["intro", "chapter"]
        assert result[0].metadata["has_unprocessed_images"] is True


class TestImageCapionerEnabled:
    """Tests when Vision LLM is enabled (use_llm=True)."""

    def test_enabled_no_images_passthrough(self):
        """Chunks without images should pass through unchanged."""
        settings = Mock()
        settings.ingestion = Mock()
        settings.ingestion.image_captioner = Mock()
        settings.ingestion.image_captioner.use_llm = True

        chunk = Chunk(
            id="chunk_001",
            text="No images",
            metadata={"source_path": "/doc.pdf"}
        )

        with patch('src.ingestion.transform.image_captioner.LLMFactory') as mock_factory:
            mock_vision_llm = Mock()
            mock_factory.return_value.create_vision_llm = Mock(return_value=mock_vision_llm)

            captioner = ImageCaptioner(settings)
            result = captioner.transform([chunk])

        assert len(result) == 1
        assert "image_captions" not in result[0].metadata
        assert "has_unprocessed_images" not in result[0].metadata

    def test_enabled_generates_captions(self):
        """Vision LLM should generate captions for images."""
        settings = Mock()
        settings.ingestion = Mock()
        settings.ingestion.image_captioner = Mock()
        settings.ingestion.image_captioner.use_llm = True

        chunk = Chunk(
            id="chunk_001",
            text="Text with [IMAGE: img_001]",
            metadata={
                "source_path": "/doc.pdf",
                "images": [
                    {
                        "id": "img_001",
                        "path": "data/images/doc_hash/img_001.png"
                    }
                ]
            }
        )

        with patch('src.ingestion.transform.image_captioner.LLMFactory') as mock_factory:
            mock_vision_llm = Mock()
            mock_vision_llm.chat_with_image = Mock(return_value="A diagram showing data flow")
            mock_factory.return_value.create_vision_llm = Mock(return_value=mock_vision_llm)

            captioner = ImageCaptioner(settings)
            result = captioner.transform([chunk])

        assert len(result) == 1
        assert "image_captions" in result[0].metadata
        assert result[0].metadata["image_captions"]["img_001"] == "A diagram showing data flow"
        assert "has_unprocessed_images" not in result[0].metadata

    def test_enabled_multiple_images_per_chunk(self):
        """Multiple images in one chunk should all get captions."""
        settings = Mock()
        settings.ingestion = Mock()
        settings.ingestion.image_captioner = Mock()
        settings.ingestion.image_captioner.use_llm = True

        chunk = Chunk(
            id="chunk_001",
            text="[IMAGE: img_001] and [IMAGE: img_002]",
            metadata={
                "source_path": "/doc.pdf",
                "images": [
                    {"id": "img_001", "path": "data/images/img_001.png"},
                    {"id": "img_002", "path": "data/images/img_002.png"}
                ]
            }
        )

        with patch('src.ingestion.transform.image_captioner.LLMFactory') as mock_factory:
            mock_vision_llm = Mock()
            mock_vision_llm.chat_with_image = Mock(
                side_effect=["First image caption", "Second image caption"]
            )
            mock_factory.return_value.create_vision_llm = Mock(return_value=mock_vision_llm)

            captioner = ImageCaptioner(settings)
            result = captioner.transform([chunk])

        assert result[0].metadata["image_captions"]["img_001"] == "First image caption"
        assert result[0].metadata["image_captions"]["img_002"] == "Second image caption"

    def test_enabled_marks_with_enriched_by(self):
        """Metadata should mark captions were generated by 'vision_llm'."""
        settings = Mock()
        settings.ingestion = Mock()
        settings.ingestion.image_captioner = Mock()
        settings.ingestion.image_captioner.use_llm = True

        chunk = Chunk(
            id="chunk_001",
            text="[IMAGE: img_001]",
            metadata={
                "source_path": "/doc.pdf",
                "images": [{"id": "img_001", "path": "data/images/img_001.png"}]
            }
        )

        with patch('src.ingestion.transform.image_captioner.LLMFactory') as mock_factory:
            mock_vision_llm = Mock()
            mock_vision_llm.chat_with_image = Mock(return_value="Test caption")
            mock_factory.return_value.create_vision_llm = Mock(return_value=mock_vision_llm)

            captioner = ImageCaptioner(settings)
            result = captioner.transform([chunk])

        assert result[0].metadata.get("captions_enriched_by") == "vision_llm"


class TestImageCapionerFallback:
    """Tests fallback behavior when Vision LLM fails or is unavailable."""

    def test_llm_failure_marks_unprocessed(self):
        """When Vision LLM fails, chunk should be marked as unprocessed."""
        settings = Mock()
        settings.ingestion = Mock()
        settings.ingestion.image_captioner = Mock()
        settings.ingestion.image_captioner.use_llm = True

        chunk = Chunk(
            id="chunk_001",
            text="[IMAGE: img_001]",
            metadata={
                "source_path": "/doc.pdf",
                "images": [{"id": "img_001", "path": "data/images/img_001.png"}]
            }
        )

        with patch('src.ingestion.transform.image_captioner.LLMFactory') as mock_factory:
            mock_vision_llm = Mock()
            mock_vision_llm.chat_with_image = Mock(side_effect=Exception("API Error"))
            mock_factory.return_value.create_vision_llm = Mock(return_value=mock_vision_llm)

            captioner = ImageCaptioner(settings)
            result = captioner.transform([chunk])

        assert len(result) == 1
        assert result[0].metadata["has_unprocessed_images"] is True
        assert "image_captions" not in result[0].metadata
        assert result[0].metadata.get("fallback_reason") is not None

    def test_llm_not_available_mark_unprocessed(self):
        """When Vision LLM is not available, chunk should be marked unprocessed."""
        settings = Mock()
        settings.ingestion = Mock()
        settings.ingestion.image_captioner = Mock()
        settings.ingestion.image_captioner.use_llm = True

        chunk = Chunk(
            id="chunk_001",
            text="[IMAGE: img_001]",
            metadata={
                "source_path": "/doc.pdf",
                "images": [{"id": "img_001", "path": "data/images/img_001.png"}]
            }
        )

        with patch('src.ingestion.transform.image_captioner.LLMFactory') as mock_factory:
            mock_factory.return_value.create_vision_llm = Mock(side_effect=Exception("Vision LLM not configured"))

            captioner = ImageCaptioner(settings)
            result = captioner.transform([chunk])

        assert result[0].metadata["has_unprocessed_images"] is True

    def test_partial_failure_fallback(self):
        """When some images fail but others succeed, mark as unprocessed."""
        settings = Mock()
        settings.ingestion = Mock()
        settings.ingestion.image_captioner = Mock()
        settings.ingestion.image_captioner.use_llm = True

        chunk = Chunk(
            id="chunk_001",
            text="[IMAGE: img_001] [IMAGE: img_002]",
            metadata={
                "source_path": "/doc.pdf",
                "images": [
                    {"id": "img_001", "path": "data/images/img_001.png"},
                    {"id": "img_002", "path": "data/images/img_002.png"}
                ]
            }
        )

        with patch('src.ingestion.transform.image_captioner.LLMFactory') as mock_factory:
            mock_vision_llm = Mock()
            mock_vision_llm.chat_with_image = Mock(
                side_effect=["Success", Exception("Timeout")]
            )
            mock_factory.return_value.create_vision_llm = Mock(return_value=mock_vision_llm)

            captioner = ImageCaptioner(settings)
            result = captioner.transform([chunk])

        assert result[0].metadata["has_unprocessed_images"] is True

    def test_never_blocks_pipeline(self):
        """ImageCaptioner should never raise exception that blocks pipeline."""
        settings = Mock()
        settings.ingestion = Mock()
        settings.ingestion.image_captioner = Mock()
        settings.ingestion.image_captioner.use_llm = True

        chunks = [
            Chunk(
                id="chunk_001",
                text="[IMAGE: img_001]",
                metadata={
                    "source_path": "/doc.pdf",
                    "images": [{"id": "img_001", "path": "invalid/path.png"}]
                }
            )
        ]

        with patch('src.ingestion.transform.image_captioner.LLMFactory') as mock_factory:
            mock_vision_llm = Mock()
            mock_vision_llm.chat_with_image = Mock(side_effect=FileNotFoundError("Image not found"))
            mock_factory.return_value.create_vision_llm = Mock(return_value=mock_vision_llm)

            captioner = ImageCaptioner(settings)
            # Should not raise
            result = captioner.transform(chunks)

        assert len(result) == 1
        assert result[0].metadata["has_unprocessed_images"] is True


class TestImageCapionerEdgeCases:
    """Edge case tests."""

    def test_empty_chunk_list(self):
        """Empty chunk list should return empty list."""
        settings = Mock()
        settings.ingestion.image_captioner.use_llm = False

        captioner = ImageCaptioner(settings)
        result = captioner.transform([])

        assert result == []

    def test_chunk_with_empty_images_list(self):
        """Chunk with empty images list should not be marked as unprocessed."""
        settings = Mock()
        settings.ingestion.image_captioner.use_llm = False

        chunk = Chunk(
            id="chunk_001",
            text="Text",
            metadata={"source_path": "/doc.pdf", "images": []}
        )

        captioner = ImageCaptioner(settings)
        result = captioner.transform([chunk])

        assert "has_unprocessed_images" not in result[0].metadata

    def test_placeholder_without_corresponding_image_ref(self):
        """Placeholder in text but no matching image_ref should not crash."""
        settings = Mock()
        settings.ingestion.image_captioner.use_llm = False

        chunk = Chunk(
            id="chunk_001",
            text="Text with [IMAGE: img_missing]",
            metadata={
                "source_path": "/doc.pdf",
                "images": [
                    {"id": "img_001", "path": "data/images/img_001.png"}
                ]
            }
        )

        captioner = ImageCaptioner(settings)
        result = captioner.transform([chunk])

        assert len(result) == 1
        # Should still mark as having images since there are image refs
        assert result[0].metadata["has_unprocessed_images"] is True

    def test_very_long_image_list(self):
        """Chunk with many images should be handled correctly."""
        settings = Mock()
        settings.ingestion = Mock()
        settings.ingestion.image_captioner = Mock()
        settings.ingestion.image_captioner.use_llm = True

        images = [
            {"id": f"img_{i:03d}", "path": f"data/images/img_{i:03d}.png"}
            for i in range(50)
        ]
        placeholders = " ".join([f"[IMAGE: img_{i:03d}]" for i in range(50)])

        chunk = Chunk(
            id="chunk_001",
            text=placeholders,
            metadata={"source_path": "/doc.pdf", "images": images}
        )

        with patch('src.ingestion.transform.image_captioner.LLMFactory') as mock_factory:
            mock_vision_llm = Mock()
            mock_vision_llm.chat_with_image = Mock(
                return_value="Caption"
            )
            mock_factory.return_value.create_vision_llm = Mock(return_value=mock_vision_llm)

            captioner = ImageCaptioner(settings)
            result = captioner.transform([chunk])

        assert len(result[0].metadata["image_captions"]) == 50

    def test_chunk_without_source_path(self):
        """Chunk without source_path should still be processed."""
        settings = Mock()
        settings.ingestion.image_captioner.use_llm = False

        chunk = Chunk(
            id="chunk_001",
            text="[IMAGE: img_001]",
            metadata={"images": [{"id": "img_001", "path": "data/images/img_001.png"}]}
        )

        captioner = ImageCaptioner(settings)
        result = captioner.transform([chunk])

        assert result[0].metadata["has_unprocessed_images"] is True


class TestImageCapionerIntegration:
    """Integration tests with TraceContext and multiple chunks."""

    def test_trace_context_recording(self):
        """Transform should work with trace context."""
        settings = Mock()
        settings.ingestion = Mock()
        settings.ingestion.image_captioner = Mock()
        settings.ingestion.image_captioner.use_llm = False

        chunk = Chunk(
            id="chunk_001",
            text="[IMAGE: img_001]",
            metadata={
                "source_path": "/doc.pdf",
                "images": [{"id": "img_001", "path": "data/images/img_001.png"}]
            }
        )

        trace = TraceContext()
        captioner = ImageCaptioner(settings)
        result = captioner.transform([chunk], trace=trace)

        assert len(result) == 1
        assert result[0].metadata["has_unprocessed_images"] is True

    def test_multiple_chunks_batch_processing(self):
        """Batch processing multiple chunks should handle each independently."""
        settings = Mock()
        settings.ingestion = Mock()
        settings.ingestion.image_captioner = Mock()
        settings.ingestion.image_captioner.use_llm = True

        chunks = [
            Chunk(
                id="chunk_001",
                text="[IMAGE: img_001]",
                metadata={
                    "source_path": "/doc.pdf",
                    "images": [{"id": "img_001", "path": "data/images/img_001.png"}]
                }
            ),
            Chunk(
                id="chunk_002",
                text="No images",
                metadata={"source_path": "/doc.pdf"}
            ),
            Chunk(
                id="chunk_003",
                text="[IMAGE: img_002]",
                metadata={
                    "source_path": "/doc.pdf",
                    "images": [{"id": "img_002", "path": "data/images/img_002.png"}]
                }
            ),
        ]

        with patch('src.ingestion.transform.image_captioner.LLMFactory') as mock_factory:
            mock_vision_llm = Mock()
            mock_vision_llm.chat_with_image = Mock(
                side_effect=["Caption 1", "Caption 2"]
            )
            mock_factory.return_value.create_vision_llm = Mock(return_value=mock_vision_llm)

            captioner = ImageCaptioner(settings)
            result = captioner.transform(chunks)

        assert len(result) == 3
        assert "image_captions" in result[0].metadata
        assert "image_captions" not in result[1].metadata
        assert "image_captions" in result[2].metadata

    def test_idempotent_processing(self):
        """Processing same chunk twice should give identical results."""
        settings = Mock()
        settings.ingestion = Mock()
        settings.ingestion.image_captioner = Mock()
        settings.ingestion.image_captioner.use_llm = False

        chunk = Chunk(
            id="chunk_001",
            text="[IMAGE: img_001]",
            metadata={
                "source_path": "/doc.pdf",
                "images": [{"id": "img_001", "path": "data/images/img_001.png"}]
            }
        )

        captioner = ImageCaptioner(settings)
        result1 = captioner.transform([chunk])

        # Process the result again
        result2 = captioner.transform(result1)

        assert result1[0].metadata == result2[0].metadata

    def test_transform_preserves_chunk_order(self):
        """Chunk order should be preserved after transformation."""
        settings = Mock()
        settings.ingestion = Mock()
        settings.ingestion.image_captioner = Mock()
        settings.ingestion.image_captioner.use_llm = False

        chunks = [
            Chunk(id="chunk_003", text="Third", metadata={"source_path": "/doc.pdf"}),
            Chunk(id="chunk_001", text="First", metadata={"source_path": "/doc.pdf"}),
            Chunk(id="chunk_002", text="Second", metadata={"source_path": "/doc.pdf"}),
        ]

        captioner = ImageCaptioner(settings)
        result = captioner.transform(chunks)

        assert result[0].id == "chunk_003"
        assert result[1].id == "chunk_001"
        assert result[2].id == "chunk_002"

    def test_different_config_switches_behavior(self):
        """Switching use_llm setting should change behavior."""
        chunk = Chunk(
            id="chunk_001",
            text="[IMAGE: img_001]",
            metadata={
                "source_path": "/doc.pdf",
                "images": [{"id": "img_001", "path": "data/images/img_001.png"}]
            }
        )

        # Test with disabled
        settings_disabled = Mock()
        settings_disabled.ingestion = Mock()
        settings_disabled.ingestion.image_captioner = Mock()
        settings_disabled.ingestion.image_captioner.use_llm = False
        captioner_disabled = ImageCaptioner(settings_disabled)
        result_disabled = captioner_disabled.transform([chunk])

        assert result_disabled[0].metadata["has_unprocessed_images"] is True
        assert "image_captions" not in result_disabled[0].metadata

        # Test with enabled (should attempt LLM)
        settings_enabled = Mock()
        settings_enabled.ingestion = Mock()
        settings_enabled.ingestion.image_captioner = Mock()
        settings_enabled.ingestion.image_captioner.use_llm = True

        with patch('src.ingestion.transform.image_captioner.LLMFactory') as mock_factory:
            mock_vision_llm = Mock()
            mock_vision_llm.chat_with_image = Mock(return_value="Test caption")
            mock_factory.return_value.create_vision_llm = Mock(return_value=mock_vision_llm)

            captioner_enabled = ImageCaptioner(settings_enabled)
            result_enabled = captioner_enabled.transform([chunk])

        assert "image_captions" in result_enabled[0].metadata
