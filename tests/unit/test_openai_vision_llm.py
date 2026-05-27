"""Unit tests for OpenAI Vision LLM implementation."""

import base64
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
import pytest
from src.libs.llm.openai_vision_llm import OpenAIVisionLLM
from src.libs.llm.base_vision_llm import VisionChatResponse
from src.core.settings import VisionLLMSettings


@pytest.fixture
def vision_settings():
    """Create OpenAI Vision settings for testing."""
    return VisionLLMSettings(
        provider="openai",
        model="gpt-4-vision-preview",
        api_key="test-api-key",
        base_url=None,
        max_image_size=2048,
    )


@pytest.fixture
def vision_settings_with_base_url():
    """Create OpenAI Vision settings with custom base_url."""
    return VisionLLMSettings(
        provider="openai",
        model="gpt-4-vision-preview",
        api_key="test-api-key",
        base_url="http://localhost:8000/v1",
        max_image_size=2048,
    )


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a sample test image."""
    from PIL import Image
    img = Image.new('RGB', (100, 100), color='red')
    img_path = tmp_path / "test_image.png"
    img.save(img_path)
    return str(img_path)


def test_openai_vision_llm_init(vision_settings):
    """Test OpenAI Vision LLM initialization."""
    llm = OpenAIVisionLLM(vision_settings)
    assert llm.settings == vision_settings
    assert llm.client is None


def test_openai_vision_llm_validate_config_success(vision_settings):
    """Test successful config validation."""
    llm = OpenAIVisionLLM(vision_settings)
    llm.validate_config()  # Should not raise


def test_openai_vision_llm_validate_config_missing_api_key():
    """Test config validation fails without api_key."""
    settings = VisionLLMSettings(
        provider="openai",
        model="gpt-4-vision-preview",
        api_key="",
    )
    llm = OpenAIVisionLLM(settings)
    with pytest.raises(ValueError, match="api_key is required"):
        llm.validate_config()


def test_openai_vision_llm_validate_config_missing_model():
    """Test config validation fails without model."""
    settings = VisionLLMSettings(
        provider="openai",
        model="",
        api_key="test-key",
    )
    llm = OpenAIVisionLLM(settings)
    with pytest.raises(ValueError, match="model is required"):
        llm.validate_config()


@patch('src.libs.llm.openai_vision_llm.OpenAI')
def test_openai_vision_llm_ensure_client(mock_openai, vision_settings):
    """Test lazy client initialization."""
    llm = OpenAIVisionLLM(vision_settings)
    llm._ensure_client()

    mock_openai.assert_called_once_with(api_key="test-api-key")
    assert llm.client is not None


@patch('src.libs.llm.openai_vision_llm.OpenAI')
def test_openai_vision_llm_ensure_client_with_base_url(mock_openai, vision_settings_with_base_url):
    """Test client initialization with custom base_url."""
    llm = OpenAIVisionLLM(vision_settings_with_base_url)
    llm._ensure_client()

    mock_openai.assert_called_once_with(
        api_key="test-api-key",
        base_url="http://localhost:8000/v1"
    )


def test_openai_vision_llm_load_image_as_base64(vision_settings, sample_image_path):
    """Test loading image from file path."""
    llm = OpenAIVisionLLM(vision_settings)
    encoded = llm._load_image_as_base64(sample_image_path)

    assert isinstance(encoded, str)
    assert len(encoded) > 0
    # Verify it's valid base64
    decoded = base64.b64decode(encoded)
    assert len(decoded) > 0


def test_openai_vision_llm_load_image_file_not_found(vision_settings):
    """Test error when image file not found."""
    llm = OpenAIVisionLLM(vision_settings)
    with pytest.raises(FileNotFoundError):
        llm._load_image_as_base64("/nonexistent/path/image.png")


def test_openai_vision_llm_get_media_type(vision_settings):
    """Test MIME type detection."""
    llm = OpenAIVisionLLM(vision_settings)

    assert llm._get_media_type(".png") == "image/png"
    assert llm._get_media_type(".jpg") == "image/jpeg"
    assert llm._get_media_type(".jpeg") == "image/jpeg"
    assert llm._get_media_type(".gif") == "image/gif"
    assert llm._get_media_type(".webp") == "image/webp"
    assert llm._get_media_type(".unknown") == "image/png"


@patch('src.libs.llm.openai_vision_llm.Image')
def test_openai_vision_llm_compress_image(mock_image_module, vision_settings):
    """Test image compression."""
    llm = OpenAIVisionLLM(vision_settings)

    # Create mock image
    mock_img = MagicMock()
    mock_img.width = 4000
    mock_img.height = 3000
    mock_image_module.open.return_value = mock_img

    test_data = b"fake_image_data"
    compressed = llm._compress_image(test_data)

    # Should call thumbnail when image exceeds max_image_size
    assert mock_img.thumbnail.called


def test_openai_vision_llm_compress_image_small_image(vision_settings):
    """Test compression skips small images."""
    llm = OpenAIVisionLLM(vision_settings)

    # Create a small real image
    from PIL import Image
    small_img = Image.new('RGB', (100, 100), color='blue')
    from io import BytesIO
    buffer = BytesIO()
    small_img.save(buffer, format='PNG')
    small_data = buffer.getvalue()

    # Should return similar size (no compression needed)
    compressed = llm._compress_image(small_data)
    assert len(compressed) > 0


def test_openai_vision_llm_encode_image_from_path(vision_settings, sample_image_path):
    """Test encoding image from file path."""
    llm = OpenAIVisionLLM(vision_settings)
    encoded = llm._encode_image(image_path=sample_image_path)

    assert isinstance(encoded, str)
    assert len(encoded) > 0
    # Verify it's valid base64
    decoded = base64.b64decode(encoded)
    assert len(decoded) > 0


def test_openai_vision_llm_encode_image_from_base64(vision_settings):
    """Test encoding already-encoded base64 image."""
    llm = OpenAIVisionLLM(vision_settings)

    # Create a small test image and encode it
    from PIL import Image
    from io import BytesIO
    img = Image.new('RGB', (100, 100), color='green')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    original_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Encode from base64 (should work same as from path)
    encoded = llm._encode_image(image_base64=original_base64)
    assert isinstance(encoded, str)
    assert len(encoded) > 0


def test_openai_vision_llm_encode_image_missing_input(vision_settings):
    """Test error when both image_path and image_base64 missing."""
    llm = OpenAIVisionLLM(vision_settings)

    with pytest.raises(ValueError, match="Either image_path or image_base64 must be provided"):
        llm._encode_image()


@patch('src.libs.llm.openai_vision_llm.OpenAI')
def test_openai_vision_llm_chat_with_image_success(mock_openai, vision_settings, sample_image_path):
    """Test successful chat with image."""
    # Setup mock
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Image contains a red square"
    mock_response.choices[0].finish_reason = "stop"
    mock_response.model = "gpt-4-vision-preview"
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 120
    mock_client.chat.completions.create.return_value = mock_response

    # Execute
    llm = OpenAIVisionLLM(vision_settings)
    result = llm.chat_with_image(
        text="Describe this image",
        image_path=sample_image_path
    )

    # Verify
    assert isinstance(result, VisionChatResponse)
    assert result.content == "Image contains a red square"
    assert result.model == "gpt-4-vision-preview"
    assert result.finish_reason == "stop"
    assert result.usage["prompt_tokens"] == 100
    assert result.usage["completion_tokens"] == 20
    assert result.usage["total_tokens"] == 120


@patch('src.libs.llm.openai_vision_llm.OpenAI')
def test_openai_vision_llm_chat_with_image_base64(mock_openai, vision_settings):
    """Test chat with image from base64."""
    # Setup
    from PIL import Image
    from io import BytesIO
    img = Image.new('RGB', (100, 100), color='blue')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Image analysis result"
    mock_response.choices[0].finish_reason = "stop"
    mock_response.model = "gpt-4-vision-preview"
    mock_response.usage.prompt_tokens = 50
    mock_response.usage.completion_tokens = 10
    mock_response.usage.total_tokens = 60
    mock_client.chat.completions.create.return_value = mock_response

    # Execute
    llm = OpenAIVisionLLM(vision_settings)
    result = llm.chat_with_image(
        text="What do you see?",
        image_base64=image_base64
    )

    # Verify
    assert result.content == "Image analysis result"
    assert result.usage["total_tokens"] == 60


def test_openai_vision_llm_chat_with_image_missing_text(vision_settings, sample_image_path):
    """Test error when text prompt is missing."""
    llm = OpenAIVisionLLM(vision_settings)

    with pytest.raises(ValueError, match="text prompt cannot be empty"):
        llm.chat_with_image(text="", image_path=sample_image_path)


def test_openai_vision_llm_chat_with_image_missing_image(vision_settings):
    """Test error when neither image_path nor image_base64 provided."""
    llm = OpenAIVisionLLM(vision_settings)

    with pytest.raises(ValueError, match="Either image_path or image_base64 must be provided"):
        llm.chat_with_image(text="Describe the image")


@patch('src.libs.llm.openai_vision_llm.OpenAI')
def test_openai_vision_llm_chat_with_image_api_error(mock_openai, vision_settings, sample_image_path):
    """Test error handling when API call fails."""
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.side_effect = Exception("API Error: rate limit exceeded")

    llm = OpenAIVisionLLM(vision_settings)

    with pytest.raises(RuntimeError, match="OpenAI Vision API error"):
        llm.chat_with_image(
            text="Describe this",
            image_path=sample_image_path
        )


@patch('src.libs.llm.openai_vision_llm.OpenAI')
def test_openai_vision_llm_chat_with_image_constructs_message_correctly(mock_openai, vision_settings, sample_image_path):
    """Test that message structure is correct for API call."""
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Result"
    mock_response.choices[0].finish_reason = "stop"
    mock_response.model = "gpt-4-vision-preview"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    mock_client.chat.completions.create.return_value = mock_response

    llm = OpenAIVisionLLM(vision_settings)
    llm.chat_with_image(
        text="Test prompt",
        image_path=sample_image_path
    )

    # Verify API call structure
    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs["model"] == "gpt-4-vision-preview"
    assert call_args.kwargs["max_tokens"] == 2048

    message = call_args.kwargs["messages"][0]
    assert message["role"] == "user"
    assert len(message["content"]) == 2
    assert message["content"][0]["type"] == "text"
    assert message["content"][0]["text"] == "Test prompt"
    assert message["content"][1]["type"] == "image_url"
    assert "data:image/" in message["content"][1]["image_url"]["url"]
