"""Tests for QWEN Analyzer.

This module tests the QwenAnalyzer class including:
- Lyrics analysis functionality
- Input validation
- Cache integration
- Retry logic
- Error handling
- Context manager protocol
"""

import time
from unittest.mock import Mock, patch

import pytest

from src.analyzers.qwen_analyzer import QwenAnalyzer


# Fixtures
@pytest.fixture
def mock_config():
    """Mock configuration for tests."""
    config = Mock()
    qwen_config = Mock()
    qwen_config.model_name = "qwen-test"
    qwen_config.base_url = "https://test.api"
    qwen_config.api_key = "test_key_12345678"
    qwen_config.temperature = 0.7
    qwen_config.max_tokens = 1000
    qwen_config.timeout = 30
    qwen_config.retry_attempts = 3

    config.analyzers.get_qwen.return_value = qwen_config
    return config


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for tests."""
    client = Mock()

    # Mock successful API response
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = '{"analysis": "test analysis"}'
    response.usage.total_tokens = 100

    client.chat.completions.create.return_value = response

    return client


@patch("src.analyzers.qwen_analyzer.get_config")
@patch("src.analyzers.qwen_analyzer.OpenAI")
@patch("src.analyzers.qwen_analyzer.redis_cache")
def test_analyze_lyrics_success(
    mock_cache, mock_openai, mock_get_config, mock_config, mock_openai_client
):
    """Test successful lyrics analysis."""
    # Arrange
    mock_get_config.return_value = mock_config
    mock_openai.return_value = mock_openai_client
    mock_cache.get_analysis.return_value = None  # No cache

    analyzer = QwenAnalyzer()
    test_lyrics = "Started from the bottom now we here"

    # Act
    result = analyzer.analyze_lyrics(test_lyrics, use_cache=False)

    # Assert
    assert result is not None
    assert result["model"] == "qwen-test"
    assert result["tokens_used"] == 100
    assert "analysis" in result
    assert result.get("failed") is None


# Test 2: Empty lyrics validation
@patch("src.analyzers.qwen_analyzer.get_config")
@patch("src.analyzers.qwen_analyzer.OpenAI")
def test_analyze_lyrics_empty_raises_error(
    mock_openai, mock_get_config, mock_config, mock_openai_client
):
    """Test that empty lyrics raise ValueError."""
    # Arrange
    mock_get_config.return_value = mock_config
    mock_openai.return_value = mock_openai_client

    analyzer = QwenAnalyzer()

    # Act & Assert
    with pytest.raises(ValueError, match="Lyrics cannot be empty"):
        analyzer.analyze_lyrics("")

    with pytest.raises(ValueError, match="Lyrics cannot be empty"):
        analyzer.analyze_lyrics("   ")  # Only whitespace


# Test 3: Invalid temperature validation
@patch("src.analyzers.qwen_analyzer.get_config")
@patch("src.analyzers.qwen_analyzer.OpenAI")
def test_invalid_temperature_raises_error(
    mock_openai, mock_get_config, mock_config, mock_openai_client
):
    """Test that invalid temperature raises ValueError."""
    # Arrange
    mock_get_config.return_value = mock_config
    mock_openai.return_value = mock_openai_client

    analyzer = QwenAnalyzer()

    # Act & Assert - temperature too low
    with pytest.raises(ValueError, match="Temperature must be between"):
        analyzer.analyze_lyrics("test lyrics", temperature=-0.1)

    # Act & Assert - temperature too high
    with pytest.raises(ValueError, match="Temperature must be between"):
        analyzer.analyze_lyrics("test lyrics", temperature=2.1)


# Test 4: Cache integration
@patch("src.analyzers.qwen_analyzer.get_config")
@patch("src.analyzers.qwen_analyzer.OpenAI")
@patch("src.analyzers.qwen_analyzer.redis_cache")
def test_analyze_lyrics_with_cache(
    mock_cache, mock_openai, mock_get_config, mock_config
):
    """Test that cached results are returned."""
    # Arrange
    mock_get_config.return_value = mock_config

    cached_result = {
        "model": "qwen-test",
        "analysis": "cached analysis",
        "tokens_used": 50,
        "timestamp": time.time(),
    }
    mock_cache.get_analysis.return_value = cached_result

    analyzer = QwenAnalyzer()

    # Act
    result = analyzer.analyze_lyrics("test lyrics", use_cache=True)

    # Assert
    assert result == cached_result
    assert result["analysis"] == "cached analysis"
    assert result["tokens_used"] == 50
    mock_cache.get_analysis.assert_called_once()  # Cache was checked


# Test 5: API connection error with retry logic
@patch("src.analyzers.qwen_analyzer.get_config")
@patch("src.analyzers.qwen_analyzer.OpenAI")
@patch("src.analyzers.qwen_analyzer.redis_cache")
@patch("src.analyzers.qwen_analyzer.time.sleep")
def test_api_connection_error_retry(
    mock_sleep,
    mock_cache,
    mock_openai,
    mock_get_config,
    mock_config,
    mock_openai_client,
):
    """Test that retryable errors trigger retry logic."""
    # Arrange
    from openai import APIConnectionError

    mock_get_config.return_value = mock_config
    mock_cache.get_analysis.return_value = None

    # First 2 attempts fail with APIConnectionError, 3rd succeeds
    mock_openai.return_value = mock_openai_client

    # Create mock request for APIConnectionError
    mock_request = Mock()
    mock_request.method = "POST"

    mock_openai_client.chat.completions.create.side_effect = [
        APIConnectionError(message="Connection failed", request=mock_request),
        APIConnectionError(message="Connection failed", request=mock_request),
        mock_openai_client.chat.completions.create.return_value,
    ]

    analyzer = QwenAnalyzer()
    test_lyrics = "Test lyrics for retry"

    # Act
    result = analyzer.analyze_lyrics(test_lyrics, use_cache=False)

    # Assert
    assert result is not None
    assert result["model"] == "qwen-test"
    # Verify that sleep was called for retries (2 retries = 2 sleeps)
    assert mock_sleep.call_count == 2


# Test 6: Authentication error - fail fast (no retry)
@patch("src.analyzers.qwen_analyzer.get_config")
@patch("src.analyzers.qwen_analyzer.OpenAI")
@patch("src.analyzers.qwen_analyzer.redis_cache")
def test_authentication_error_no_retry(
    mock_cache, mock_openai, mock_get_config, mock_config, mock_openai_client
):
    """Test that authentication errors fail fast without retrying."""
    # Arrange
    from openai import AuthenticationError

    mock_get_config.return_value = mock_config
    mock_cache.get_analysis.return_value = None

    mock_openai.return_value = mock_openai_client

    # Create mock response for AuthenticationError
    mock_response = Mock()
    mock_response.status_code = 401

    mock_openai_client.chat.completions.create.side_effect = AuthenticationError(
        "Invalid API key", response=mock_response, body=None
    )

    analyzer = QwenAnalyzer()

    # Act
    result = analyzer.analyze_lyrics("test lyrics", use_cache=False)

    # Assert - should fail immediately
    assert result.get("failed") is True
    assert result.get("error") == "Authentication failed"
    # Verify create was called only once (no retry)
    assert mock_openai_client.chat.completions.create.call_count == 1


# Test 7: Response JSON parsing
@patch("src.analyzers.qwen_analyzer.get_config")
@patch("src.analyzers.qwen_analyzer.OpenAI")
@patch("src.analyzers.qwen_analyzer.redis_cache")
def test_response_parsing_json(mock_cache, mock_openai, mock_get_config, mock_config):
    """Test JSON response parsing."""
    # Arrange
    mock_get_config.return_value = mock_config
    mock_cache.get_analysis.return_value = None

    # Mock JSON response
    json_response = '{"themes": ["struggle", "success"], "quality": 8}'
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = json_response
    response.usage.total_tokens = 120

    # Create mock client for this specific test
    mock_openai_client = Mock()
    mock_openai_client.chat.completions.create.return_value = response
    mock_openai.return_value = mock_openai_client

    analyzer = QwenAnalyzer()

    # Act
    result = analyzer.analyze_lyrics("test lyrics", use_cache=False)

    # Assert
    assert result is not None
    assert "themes" in result  # Parsed from JSON
    assert result["themes"] == ["struggle", "success"]
    assert result["quality"] == 8
    assert result.get("raw_response") is None  # Not marked as raw_response


# Test 8: Context manager cleanup
@patch("src.analyzers.qwen_analyzer.get_config")
@patch("src.analyzers.qwen_analyzer.OpenAI")
def test_context_manager(mock_openai, mock_get_config, mock_config, mock_openai_client):
    """Test context manager cleanup."""
    # Arrange
    mock_get_config.return_value = mock_config
    mock_openai.return_value = mock_openai_client

    # Act
    with QwenAnalyzer() as analyzer:
        assert analyzer is not None
        assert isinstance(analyzer, QwenAnalyzer)

    # Assert - verify cleanup was called
    mock_openai_client.close.assert_called_once()
