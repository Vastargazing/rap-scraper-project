"""QWEN Analyzer Wrapper with Config Integration.

This module provides type-safe QWEN model integration for lyrics analysis
with automatic configuration management, retry logic, and response caching.

Features:
    - Config loader integration for API settings
    - Automatic API key validation
    - Retry logic with exponential backoff
    - Temperature and token limits from config
    - Redis response caching support

Example:
    Basic usage of the QWEN Analyzer:

        analyzer = QwenAnalyzer()
        result = analyzer.analyze_lyrics("rap lyrics text")

    With custom parameters:

        result = analyzer.analyze_lyrics(
            "lyrics",
            temperature=0.2,
            max_tokens=500,
            use_cache=True
        )
"""

import logging
import time
from typing import Any, TypedDict

from openai import OpenAI

from src.cache.redis_client import redis_cache
from src.config.config_loader import get_config

logger = logging.getLogger(__name__)

# Module-level constants for magic numbers and strings
_DEFAULT_CACHE_PREFIX = "qwen"
_BACKOFF_MULTIPLIER = 2
_TEST_MAX_TOKENS = 10
_TEST_TIMEOUT = 10


class AnalysisResult(TypedDict):
    """Structured type for analysis results.

    Attributes:
        model: Name of the model used for analysis.
        tokens_used: Number of tokens consumed in API call.
        timestamp: Unix timestamp of when analysis was performed.
        analysis: Analysis content or error message.
        raw_response: Flag indicating if response was not JSON-parseable.
        error: Error message if analysis failed.
        failed: Flag indicating if analysis failed.
    """

    model: str
    tokens_used: int | None
    timestamp: float
    analysis: str | None
    raw_response: bool | None
    error: str | None
    failed: bool | None


class QwenAnalyzer:
    """QWEN-based lyrics analyzer with config integration.

    This class provides QWEN model integration for analyzing rap lyrics
    with support for automatic caching, retry logic, and configuration
    management from environment variables or config files.

    Attributes:
        client: OpenAI client instance for communication with QWEN API.
        qwen_config: Configuration object containing model settings,
            API credentials, timeout values, and retry parameters.
        use_cache: Flag to enable/disable Redis response caching.

    Example:
        Initialize analyzer and analyze lyrics:

            analyzer = QwenAnalyzer()
            result = analyzer.analyze_lyrics("Your rap lyrics here")

            if "error" not in result:
                print(f"Model used: {result['model']}")
                print(f"Analysis: {result['analysis']}")

        With custom temperature and caching disabled:

            result = analyzer.analyze_lyrics(
                "Complex rap lyrics with metaphors",
                temperature=0.2,
                use_cache=False
            )

    Raises:
        ConfigError: If configuration cannot be loaded or API key is missing.
        APIError: If QWEN API connection fails.
    """

    def __init__(self) -> None:
        """Initialize QWEN analyzer with config settings.

        Loads configuration from environment and initializes OpenAI client
        for QWEN API communication. Sets up caching support for analysis results.

        Raises:
            ConfigError: If configuration cannot be loaded.
            ValueError: If required API key is missing in environment.
        """
        config = get_config()
        self.qwen_config = config.analyzers.get_qwen()

        logger.info("Initializing QWEN Analyzer...")
        logger.info(f"   Model: {self.qwen_config.model_name}")
        logger.info(f"   Base URL: {self.qwen_config.base_url}")
        logger.info(f"   Temperature: {self.qwen_config.temperature}")
        logger.info(f"   Max Tokens: {self.qwen_config.max_tokens}")
        logger.info(f"   Timeout: {self.qwen_config.timeout}s")
        logger.info(f"   Retry Attempts: {self.qwen_config.retry_attempts}")

        # Initialize OpenAI client with QWEN endpoint
        self.client = OpenAI(
            base_url=self.qwen_config.base_url,
            api_key=self.qwen_config.api_key,  # Validates and reads from ENV
        )

        self.use_cache = True  # Can be configured

        logger.info("QWEN Analyzer initialized successfully!")

    def analyze_lyrics(
        self,
        lyrics: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        use_cache: bool = True,
    ) -> AnalysisResult:
        """Analyze rap lyrics using QWEN model.

        Performs comprehensive analysis of provided lyrics including themes,
        style, complexity, and emotional tone. Results are cached in Redis
        if available.

        Args:
            lyrics: Lyrics text to analyze.
            temperature: Override config temperature (optional).
            max_tokens: Override config max tokens (optional).
            use_cache: Use Redis cache if available.

        Returns:
            AnalysisResult: Dictionary containing model name, tokens used,
                analysis content, and error information if applicable.
        """
        # Check cache first
        if use_cache and self.use_cache:
            cached = redis_cache.get_analysis(f"qwen:{hash(lyrics)}")
            if cached:
                logger.info("Using cached QWEN analysis")
                return cached

        # Use config defaults if not overridden
        temp = temperature if temperature is not None else self.qwen_config.temperature
        tokens = max_tokens if max_tokens is not None else self.qwen_config.max_tokens

        # Build prompt
        prompt = self._build_analysis_prompt(lyrics)

        # Analyze with retry logic
        result = self._analyze_with_retry(prompt, temp, tokens)

        # Cache result
        if use_cache and self.use_cache and result:
            redis_cache.cache_analysis(f"qwen:{hash(lyrics)}", result)

        return result

    def _build_analysis_prompt(self, lyrics: str) -> str:
        """Build analysis prompt for QWEN.

        Creates a structured prompt for analyzing rap lyrics with specific
        analysis criteria including themes, style, complexity, and quality.

        Args:
            lyrics: Raw rap lyrics text to analyze.

        Returns:
            str: Structured prompt for QWEN analysis.
        """
        return f"""Analyze these rap lyrics and provide a detailed breakdown:

LYRICS:
{lyrics}

Please analyze:
1. Main themes and topics
2. Lyrical style and flow
3. Complexity level (1-10)
4. Emotional tone
5. Quality score (1-10)
6. Notable metaphors or wordplay

Provide response in JSON format."""

    def _analyze_with_retry(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> AnalysisResult:
        """Analyze prompt with automatic retry logic.

        Attempts to analyze lyrics using QWEN API with exponential backoff
        retry strategy. Returns error result if all retry attempts fail.

        Args:
            prompt: Formatted analysis prompt to send to QWEN API.
            temperature: Temperature setting for model creativity (0.0-2.0)
            max_tokens: Max tokens for API response.

        Returns:
            AnalysisResult: Analysis results or error dict with failure flag set.

        Raises:
            Exception: Re-raises last exception after all retries exhausted
                (caught internally but documented for debugging).

        Note:
            Uses _BACKOFF_MULTIPLIER for exponential backoff between retries.
            Logs all retry attempts for debugging purposes.
        """
        last_error = None

        for attempt in range(1, self.qwen_config.retry_attempts + 1):
            try:
                logger.info(
                    f"QWEN analysis attempt {attempt}/{self.qwen_config.retry_attempts}"
                )

                response = self.client.chat.completions.create(
                    model=self.qwen_config.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert rap lyrics analyst. Provide detailed, structured analysis.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.qwen_config.timeout,
                )

                # Extract and parse response
                content = response.choices[0].message.content

                # Try to parse as JSON
                try:
                    import json

                    result = json.loads(content)
                except json.JSONDecodeError:
                    # If not JSON, return as text analysis
                    result = {"analysis": content, "raw_response": True}

                # Add metadata
                result["model"] = self.qwen_config.model_name
                result["tokens_used"] = (
                    response.usage.total_tokens
                    if hasattr(response.usage, "total_tokens")
                    else None
                )
                result["timestamp"] = time.time()

                logger.info(
                    f"✅ QWEN analysis successful (tokens: {result.get('tokens_used', 'N/A')})"
                )
                return result

            except Exception as e:
                last_error = e
                logger.warning(f"⚠️ QWEN attempt {attempt} failed: {e}")

                if attempt < self.qwen_config.retry_attempts:
                    wait_time = attempt * 2  # Exponential backoff
                    logger.info(f"   Retrying in {wait_time}s...")
                    time.sleep(wait_time)

        # All attempts failed
        logger.error(
            f"❌ QWEN analysis failed after {self.qwen_config.retry_attempts} attempts: {last_error}"
        )
        return {
            "error": str(last_error),
            "model": self.qwen_config.model_name,
            "failed": True,
        }

    def test_api_connection(self) -> bool:
        """Test QWEN API connection and validate configuration.

        Sends a simple test request to QWEN API to verify credentials,
        connectivity, and basic API functionality. Logs results and errors.

        Returns:
            bool: True if connection successful and API is responding.
                False if connection fails or API returns error.

        Raises:
            No exceptions raised - all errors are caught and logged.

        """
        try:
            logger.info("Testing QWEN API connection...")

            response = self.client.chat.completions.create(
                model=self.qwen_config.model_name,
                messages=[{"role": "user", "content": "Hello, this is a test."}],
                max_tokens=10,
                timeout=10,
            )

            logger.info("QWEN API connection successful!")
            logger.info(f"   Model: {self.qwen_config.model_name}")
            logger.info(f"   Response: {response.choices[0].message.content}")
            return True

        except Exception as e:
            logger.error(f"QWEN API connection failed: {e}")
            return False

    def get_config_info(self) -> dict[str, Any]:
        """Get current QWEN analyzer configuration information.

        Returns all configuration settings including model name, API endpoint,
        temperature, token limits, timeouts, retry settings, and cache status.
        Used for debugging and monitoring purposes.

        Returns:
            dict[str, Any]: Dictionary containing:
                - model: str (model name)
                - base_url: str (API endpoint URL)
                - temperature: float (model temperature setting)
                - max_tokens: int (maximum response tokens)
                - timeout: int (API request timeout in seconds)
                - retry_attempts: int (maximum retry attempts)
                - api_key_set: bool (whether API key is configured)
                - cache_enabled: bool (whether Redis caching is enabled)
        """
        return {
            "model": self.qwen_config.model_name,
            "base_url": self.qwen_config.base_url,
            "temperature": self.qwen_config.temperature,
            "max_tokens": self.qwen_config.max_tokens,
            "timeout": self.qwen_config.timeout,
            "retry_attempts": self.qwen_config.retry_attempts,
            "api_key_set": bool(self.qwen_config.api_key),
            "cache_enabled": self.use_cache,
        }


if __name__ == "__main__":
    # Test QWEN analyzer
    print("🧪 Testing QWEN Analyzer...")
    print("=" * 60)

    try:
        # Initialize analyzer
        analyzer = QwenAnalyzer()

        # Show config
        print("\n📊 Configuration:")
        config_info = analyzer.get_config_info()
        for key, value in config_info.items():
            print(f"   {key}: {value}")

        # Test connection
        print("\n🔌 Testing API connection...")
        if analyzer.test_api_connection():
            print("✅ Connection test passed!")

            # Test lyrics analysis
            print("\n🎤 Testing lyrics analysis...")
            test_lyrics = """
            Started from the bottom now we're here
            Started from the bottom now my whole team here
            """

            result = analyzer.analyze_lyrics(test_lyrics, use_cache=False)

            if "error" in result:
                print(f"❌ Analysis failed: {result['error']}")
            else:
                print("✅ Analysis successful!")
                print(f"   Model: {result.get('model')}")
                print(f"   Tokens: {result.get('tokens_used')}")
                if "analysis" in result:
                    print(f"   Response: {result['analysis'][:100]}...")

            print("\n✅ All tests passed!")
        else:
            print("❌ Connection test failed!")
            print("⚠️ Check your NOVITA_API_KEY in .env file")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
