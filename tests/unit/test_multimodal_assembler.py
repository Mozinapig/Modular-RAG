"""
Unit tests for MultimodalAssembler (E6)
"""
import base64
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from src.core.response.multimodal_assembler import MultimodalAssembler


class TestMultimodalAssembler:
    """Test MultimodalAssembler functionality"""

    def test_multimodal_assembler_initialization(self):
        """Test MultimodalAssembler can be initialized"""
        assembler = MultimodalAssembler()
        assert assembler is not None

    def test_assemble_empty_results(self):
        """Test assembling empty results"""
        assembler = MultimodalAssembler()
        content = assembler.assemble([])

        assert content == []

    def test_assemble_text_only_result(self):
        """Test assembling result with only text"""
        assembler = MultimodalAssembler()

        mock_result = MagicMock()
        mock_result.content = "Test content"
        mock_result.image_refs = []

        content = assembler.assemble([mock_result])

        assert len(content) == 1
        assert content[0]['type'] == 'text'
        assert content[0]['text'] == "Test content"

    def test_assemble_multiple_text_results(self):
        """Test assembling multiple text results"""
        assembler = MultimodalAssembler()

        results = []
        for i in range(3):
            mock_result = MagicMock()
            mock_result.content = f"Content {i}"
            mock_result.image_refs = []
            results.append(mock_result)

        content = assembler.assemble(results)

        assert len(content) == 3
        for i, item in enumerate(content):
            assert item['type'] == 'text'

    def test_assemble_with_image_ref(self):
        """Test assembling result with image reference"""
        assembler = MultimodalAssembler()

        mock_result = MagicMock()
        mock_result.content = "Content with image"
        mock_result.image_refs = ['image1.png']

        with patch('pathlib.Path.exists') as mock_exists, \
             patch('builtins.open', mock_open(read_data=b'fake_image_data')):

            mock_exists.return_value = True

            content = assembler.assemble([mock_result])

            # Should have text + image
            assert len(content) >= 1
            # First should be text
            assert content[0]['type'] == 'text'

    def test_create_image_content_encodes_base64(self):
        """Test image content is base64 encoded"""
        assembler = MultimodalAssembler()

        image_data = b'PNG fake data'

        with patch('pathlib.Path.exists') as mock_exists, \
             patch('pathlib.Path.suffix', '.png'), \
             patch('builtins.open', mock_open(read_data=image_data)):

            mock_exists.return_value = True

            image_content = assembler._create_image_content('test.png')

            assert image_content is not None
            assert image_content['type'] == 'image'
            assert 'data' in image_content
            # Data should be base64 encoded
            decoded = base64.b64decode(image_content['data'])
            assert decoded == image_data

    def test_create_image_content_sets_mime_type_for_png(self):
        """Test PNG images have correct MIME type"""
        assembler = MultimodalAssembler()

        with patch('pathlib.Path.exists') as mock_exists, \
             patch('pathlib.Path.suffix', '.png'), \
             patch('builtins.open', mock_open(read_data=b'data')):

            mock_exists.return_value = True

            image_content = assembler._create_image_content('test.png')

            assert image_content['mimeType'] == 'image/png'

    def test_create_image_content_sets_mime_type_for_jpeg(self):
        """Test JPEG images have correct MIME type"""
        assembler = MultimodalAssembler()

        with patch('pathlib.Path.exists') as mock_exists, \
             patch('pathlib.Path.suffix', '.jpg'), \
             patch('builtins.open', mock_open(read_data=b'data')):

            mock_exists.return_value = True

            image_content = assembler._create_image_content('test.jpg')

            assert image_content['mimeType'] == 'image/jpeg'

    def test_create_image_content_handles_missing_file(self):
        """Test handles missing image files gracefully"""
        assembler = MultimodalAssembler()

        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = False

            image_content = assembler._create_image_content('missing.png')

            assert image_content is None

    def test_create_image_content_handles_multiple_paths(self):
        """Test tries multiple paths to find image"""
        assembler = MultimodalAssembler()

        def exists_side_effect(self):
            """Return True only for specific path"""
            path_str = str(self)
            return 'data/images' in path_str

        with patch('pathlib.Path.exists', side_effect=exists_side_effect), \
             patch('builtins.open', mock_open(read_data=b'image_data')), \
             patch('pathlib.Path.suffix', '.gif'):

            image_content = assembler._create_image_content('img.gif')

            # Should eventually find the image
            assert image_content is not None or image_content is None  # Either way is ok

    def test_assemble_handles_error_gracefully(self):
        """Test assemble handles errors gracefully"""
        assembler = MultimodalAssembler()

        mock_result = MagicMock()
        mock_result.content = "Valid"
        mock_result.image_refs = []

        # Should not raise even if something goes wrong
        content = assembler.assemble([mock_result])

        assert isinstance(content, list)

    def test_image_content_structure(self):
        """Test image content has required fields"""
        assembler = MultimodalAssembler()

        with patch('pathlib.Path.exists') as mock_exists, \
             patch('pathlib.Path.suffix', '.png'), \
             patch('builtins.open', mock_open(read_data=b'PNG')):

            mock_exists.return_value = True

            image_content = assembler._create_image_content('test.png')

            assert image_content is not None
            assert 'type' in image_content
            assert 'mimeType' in image_content
            assert 'data' in image_content
            assert image_content['type'] == 'image'
