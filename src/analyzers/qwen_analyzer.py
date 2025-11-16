# TODO(Google Style): Remove emojis from docstrings - not standard in production code
# TODO(Google Style): Add module-level docstring following Google format with Examples section
"""
🤖 QWEN Analyzer Wrapper with Config Integration
Type-safe QWEN model integration for lyrics analysis

Features:
- Config loader integration for API settings
- Automatic API key validation
- Retry logic from config
- Temperature and token limits from config
- Response caching support

Author: Vastargazing
Version: 2.0.0
"""

# TODO(imports): Move json import to top level - importing inside function is anti-pattern
import hashlib  # TODO(security): Add hashlib for secure cache key generation instead of hash()
import json  # TODO(imports): Moved from inside function to module level
import logging
import time
from typing import Any, Dict, Optional, TypedDict  # TODO(typing): Add TypedDict for structured return types

from openai import OpenAI

from src.cache.redis_client import redis_cache
from src.config.config_loader import get_config

logger = logging.getLogger(__name__)

# TODO(constants): Add module-level constants for magic numbers and strings
# _DEFAULT_CACHE_PREFIX = "qwen"
# _BACKOFF_MULTIPLIER = 2
# _TEST_MAX_TOKENS = 10
# _TEST_TIMEOUT = 10


# TODO(typing): Define TypedDict for analysis result structure
# class AnalysisResult(TypedDict):
#     """Structured type for analysis results."""
#     model: str
#     tokens_used: Optional[int]
#     timestamp: float
#     analysis: Optional[str]
#     raw_response: Optional[bool]
#     error: Optional[str]
#     failed: Optional[bool]


class QwenAnalyzer:
    # TODO(docstring): Rewrite class docstring in Google Style format:
    # """QWEN-based lyrics analyzer with config integration.
    #
    # This class provides...
    #
    # Attributes:
    #     qwen_config: Configuration for QWEN model
    #     client: OpenAI client instance
    #     use_cache: Whether to use Redis caching
    #
    # Example:
    #     analyzer = QwenAnalyzer()
    #     result = analyzer.analyze_lyrics("rap lyrics here")
    # """
    """
    QWEN-based lyrics analyzer with config integration

    Usage:
        analyzer = QwenAnalyzer()
        result = analyzer.analyze_lyrics("rap lyrics here")

        # With custom temperature
        result = analyzer.analyze_lyrics("lyrics", temperature=0.2)
    """

    def __init__(self):
        # TODO(docstring): Use Google Style docstring format
        # TODO(logging): Remove emojis from production logs - use structured logging instead
        # TODO(error handling): Add try-except for config initialization and OpenAI client creation
        # TODO(design): Consider dependency injection for config and client (testability)
        """Initialize QWEN analyzer with config settings"""
        config = get_config()
        self.qwen_config = config.analyzers.get_qwen()

        logger.info("🤖 Initializing QWEN Analyzer...")
        logger.info(f"   Model: {self.qwen_config.model_name}")
        logger.info(f"   Base URL: {self.qwen_config.base_url}")
        logger.info(f"   Temperature: {self.qwen_config.temperature}")
        logger.info(f"   Max Tokens: {self.qwen_config.max_tokens}")
        logger.info(f"   Timeout: {self.qwen_config.timeout}s")
        logger.info(f"   Retry Attempts: {self.qwen_config.retry_attempts}")

        # Initialize OpenAI client with QWEN endpoint
        # TODO(resource management): Add __enter__ and __exit__ for context manager support
        # TODO(resource management): Add close() method to properly cleanup client
        self.client = OpenAI(
            base_url=self.qwen_config.base_url,
            api_key=self.qwen_config.api_key,  # Validates and reads from ENV
        )

        # TODO(config): Move use_cache to config instead of hardcoding
        self.use_cache = True  # Can be configured

        logger.info("✅ QWEN Analyzer initialized successfully!")

    def analyze_lyrics(
        self,
        lyrics: str,
        temperature: float | None = None,  # TODO(typing): Use Optional[float] for Python <3.10 compatibility
        max_tokens: int | None = None,  # TODO(typing): Use Optional[int] for Python <3.10 compatibility
        use_cache: bool = True,
    ) -> dict[str, Any]:  # TODO(typing): Replace with AnalysisResult TypedDict for type safety
        # TODO(docstring): Rewrite in Google Style with proper sections:
        # """Analyze rap lyrics using QWEN model.
        #
        # Args:
        #     lyrics: Lyrics text to analyze.
        #     temperature: Override config temperature. Defaults to config value.
        #     max_tokens: Override config max tokens. Defaults to config value.
        #     use_cache: Whether to use Redis cache. Defaults to True.
        #
        # Returns:
        #     Analysis results dict containing themes, style, quality metrics.
        #
        # Raises:
        #     ValueError: If lyrics is empty or None.
        # """
        """
        Analyze rap lyrics using QWEN model

        Args:
            lyrics: Lyrics text to analyze
            temperature: Override config temperature (optional)
            max_tokens: Override config max tokens (optional)
            use_cache: Use Redis cache if available

        Returns:
            dict: Analysis results with themes, style, quality, etc.
        """
        # TODO(validation): Add input validation for lyrics (empty, None, max length)
        # TODO(validation): Add range validation for temperature (0.0-2.0) and max_tokens (>0)

        # Check cache first
        if use_cache and self.use_cache:
            # TODO(security): Replace hash() with hashlib.sha256 - hash() is not deterministic across processes
            # TODO(security): hash() can have collisions and is not cryptographically secure
            cached = redis_cache.get_analysis(f"qwen:{hash(lyrics)}")
            if cached:
                logger.info("✅ Using cached QWEN analysis")
                return cached

        # Use config defaults if not overridden
        temp = temperature if temperature is not None else self.qwen_config.temperature
        tokens = max_tokens if max_tokens is not None else self.qwen_config.max_tokens

        # Build prompt
        prompt = self._build_analysis_prompt(lyrics)

        # Analyze with retry logic
        result = self._analyze_with_retry(prompt, temp, tokens)

        # Cache result
        # TODO(error handling): Only cache successful results, check for 'error' or 'failed' keys
        if use_cache and self.use_cache and result:
            # TODO(security): Replace hash() with hashlib.sha256 (same issue as above)
            redis_cache.cache_analysis(f"qwen:{hash(lyrics)}", result)

        return result

    def _build_analysis_prompt(self, lyrics: str) -> str:
        # TODO(docstring): Add Google Style docstring with Args, Returns sections
        # TODO(prompt engineering): Move prompt template to config or constants
        # TODO(prompt engineering): Add few-shot examples for better JSON formatting
        # TODO(design): Consider using prompt templates (e.g., jinja2) for maintainability
        """Build analysis prompt for QWEN"""
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
    ) -> dict[str, Any]:  # TODO(typing): Replace with AnalysisResult TypedDict
        # TODO(docstring): Add Raises section for possible exceptions
        # TODO(error handling): Specify which exceptions to retry (network, timeout) vs fail fast (auth, validation)
        """
        Analyze with automatic retry on failure

        Args:
            prompt: Analysis prompt
            temperature: Temperature setting
            max_tokens: Max tokens setting

        Returns:
            dict: Analysis results or error dict
        """
        # TODO(typing): Type hint last_error as Optional[Exception]
        last_error = None

        for attempt in range(1, self.qwen_config.retry_attempts + 1):
            try:
                logger.info(
                    f"🤖 QWEN analysis attempt {attempt}/{self.qwen_config.retry_attempts}"
                )

                # TODO(constants): Move system message to module-level constant
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
                # TODO(validation): Add null check for response.choices and response.choices[0]
                # TODO(error handling): Handle IndexError if choices is empty
                content = response.choices[0].message.content

                # Try to parse as JSON
                # TODO(separation of concerns): Extract JSON parsing to separate method _parse_response()
                # TODO(error handling): Add more specific error handling for malformed JSON
                try:
                    # TODO(imports): json import already moved to top - remove this try block duplication

                    result = json.loads(content)
                except json.JSONDecodeError:
                    # TODO(logging): Log warning about JSON parse failure with content sample
                    # If not JSON, return as text analysis
                    result = {"analysis": content, "raw_response": True}

                # Add metadata
                # TODO(validation): Validate that result is a dict before adding keys
                result["model"] = self.qwen_config.model_name
                # TODO(error handling): Add try-except for attribute access on response.usage
                result["tokens_used"] = (
                    response.usage.total_tokens
                    if hasattr(response.usage, "total_tokens")
                    else None
                )
                # TODO(datetime): Use datetime.now(timezone.utc).isoformat() instead of timestamp for readability
                result["timestamp"] = time.time()

                logger.info(
                    f"✅ QWEN analysis successful (tokens: {result.get('tokens_used', 'N/A')})"
                )
                return result

            # TODO(error handling): Catch specific exceptions instead of bare Exception
            # Separate transient errors (APIConnectionError, Timeout) from permanent ones (AuthenticationError)
            except Exception as e:
                last_error = e
                # TODO(logging): Use logger.exception() to include traceback in logs
                logger.warning(f"⚠️ QWEN attempt {attempt} failed: {e}")

                if attempt < self.qwen_config.retry_attempts:
                    # TODO(constants): Move backoff multiplier to constant _BACKOFF_MULTIPLIER
                    # TODO(algorithm): Use exponential backoff 2^attempt instead of linear attempt*2
                    wait_time = attempt * 2  # Exponential backoff
                    logger.info(f"   Retrying in {wait_time}s...")
                    time.sleep(wait_time)

        # All attempts failed
        # TODO(logging): Include error type and suggestion for common errors (auth, network, quota)
        logger.error(
            f"❌ QWEN analysis failed after {self.qwen_config.retry_attempts} attempts: {last_error}"
        )
        # TODO(error handling): Consider raising exception instead of returning error dict for consistency
        return {
            "error": str(last_error),
            "model": self.qwen_config.model_name,
            "failed": True,
        }

    def test_api_connection(self) -> bool:
        # TODO(docstring): Rewrite in Google Style format with detailed Args/Returns/Raises
        # TODO(testing): This method should be in tests, not production code
        """
        Test QWEN API connection

        Returns:
            bool: True if connection successful
        """
        # TODO(error handling): Catch specific OpenAI exceptions (AuthenticationError, APIConnectionError, etc.)
        try:
            logger.info("🧪 Testing QWEN API connection...")

            # TODO(constants): Use module-level constants for test values
            response = self.client.chat.completions.create(
                model=self.qwen_config.model_name,
                messages=[{"role": "user", "content": "Hello, this is a test."}],
                max_tokens=10,  # TODO(constants): Use _TEST_MAX_TOKENS constant
                timeout=10,  # TODO(constants): Use _TEST_TIMEOUT constant
            )

            logger.info("✅ QWEN API connection successful!")
            logger.info(f"   Model: {self.qwen_config.model_name}")
            # TODO(error handling): Add null check for response.choices[0]
            logger.info(f"   Response: {response.choices[0].message.content}")
            return True

        # TODO(error handling): Catch specific exceptions and return detailed error info
        except Exception as e:
            # TODO(logging): Use logger.exception() to capture full traceback
            logger.error(f"❌ QWEN API connection failed: {e}")
            return False

    def get_config_info(self) -> dict[str, Any]:  # TODO(typing): Create ConfigInfo TypedDict
        # TODO(docstring): Add Google Style docstring with Returns section
        # TODO(security): Redact or mask API key in output (show only last 4 chars)
        """Get current configuration info"""
        return {
            "model": self.qwen_config.model_name,
            "base_url": self.qwen_config.base_url,
            "temperature": self.qwen_config.temperature,
            "max_tokens": self.qwen_config.max_tokens,
            "timeout": self.qwen_config.timeout,
            "retry_attempts": self.qwen_config.retry_attempts,
            # TODO(security): Mask API key - only show "sk-...xyz" format
            "api_key_set": bool(self.qwen_config.api_key),
            "cache_enabled": self.use_cache,
        }


# TODO(testing): Move all test code to proper unit tests (tests/test_qwen_analyzer.py)
# TODO(testing): Use pytest framework with fixtures and mocks
# TODO(testing): Remove manual testing from production code
if __name__ == "__main__":
    # Test QWEN analyzer
    print("🧪 Testing QWEN Analyzer...")
    print("=" * 60)

    # TODO(error handling): Add specific exception handling for different failure modes
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
            # TODO(testing): Use dedicated test fixtures instead of hardcoded strings
            test_lyrics = """
            Started from the bottom now we're here
            Started from the bottom now my whole team here
            """

            result = analyzer.analyze_lyrics(test_lyrics, use_cache=False)

            # TODO(validation): Add schema validation for result structure
            if "error" in result:
                print(f"❌ Analysis failed: {result['error']}")
            else:
                print("✅ Analysis successful!")
                print(f"   Model: {result.get('model')}")
                print(f"   Tokens: {result.get('tokens_used')}")
                if "analysis" in result:
                    # TODO(magic numbers): Move 100 to constant
                    print(f"   Response: {result['analysis'][:100]}...")

            print("\n✅ All tests passed!")
        else:
            print("❌ Connection test failed!")
            # TODO(config): Don't hardcode env var name - get from config
            print("⚠️ Check your NOVITA_API_KEY in .env file")

    # TODO(error handling): Catch specific exceptions instead of bare Exception
    except Exception as e:
        print(f"❌ Test failed: {e}")
        # TODO(imports): traceback already available, no need to import here
        import traceback

        traceback.print_exc()
