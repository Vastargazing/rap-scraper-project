#!/usr/bin/env python3
# TODO(code-quality): Replace emojis in docstrings with standard text per Google style guide
# TODO(documentation): Add proper Google-style docstring with Args, Returns, Raises sections
"""Unified mass Qwen analyzer (PostgreSQL + embedded Qwen).

This module provides a mass analyzer for rap lyrics using Qwen AI model integrated
with PostgreSQL database. It supports batch processing, checkpoint recovery, and
detailed analytics.

Typical usage example:
    python src/analyzers/mass_qwen_analysis.py --test
    python src/analyzers/mass_qwen_analysis.py --stats
    python src/analyzers/mass_qwen_analysis.py --batch 100
    python src/analyzers/mass_qwen_analysis.py --resume

Author: AI Assistant
Date: September 2025
Version: 3.0 (Unified)
"""

# TODO(imports): Reorganize imports per Google style - group by stdlib, third-party, local
# TODO(imports): Sort imports alphabetically within each group
import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any  # TODO(typing): Use more specific types instead of Any where possible

from dotenv import load_dotenv

# TODO(i18n): Remove Russian comments, use English for international collaboration
# Load environment variables
load_dotenv()

# TODO(architecture): Avoid modifying sys.path at module level - use proper package structure
# TODO(magic-numbers): Document why we need 3 levels of os.path.dirname
# Add project root to path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

# TODO(i18n): Remove Russian comments, use English
# Conditional import for OpenAI-compatible client
try:
    import openai

    HAS_OPENAI = True  # TODO(naming): Use SCREAMING_SNAKE_CASE for module-level constants
except ImportError:
    HAS_OPENAI = False

# TODO(error-handling): Catch more specific exceptions than bare ImportError
# TODO(logging): Use logger instead of print() for error messages
# TODO(i18n): Remove emojis and Russian text from user-facing messages
try:
    from src.core.app import create_app
    from src.database.postgres_adapter import PostgreSQLManager
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("💡 Убедитесь, что вы запускаете скрипт из корневой папки проекта")
    sys.exit(1)

# TODO(logging): Configure logger format, level, and handlers at module initialization
logger = logging.getLogger(__name__)


# ============================================================================
# TODO(i18n): Translate Russian section headers to English
# EMBEDDED QWEN ANALYZER (from archive/qwen_analyzer.py)
# ============================================================================


# TODO(documentation): Add Google-style docstring with attributes documentation
# TODO(typing): Replace dict[str, Any] with TypedDict or more specific types
# TODO(design): Consider using slots=True for memory efficiency
@dataclass
class AnalysisResult:
    """Analysis result for compatibility with mass analyzer.

    Attributes:
        artist: The artist name.
        title: The song title.
        analyzer_type: Type of analyzer used (e.g., 'qwen').
        confidence: Confidence score between 0 and 1.
        metadata: Additional metadata about the analysis.
        raw_output: Raw analysis data from the model.
        processing_time: Time taken for analysis in seconds.
        timestamp: ISO format timestamp of when analysis was performed.
    """

    artist: str
    title: str
    analyzer_type: str
    confidence: float  # TODO(validation): Add range validation (0.0-1.0)
    metadata: dict[str, Any]
    raw_output: dict[str, Any]
    processing_time: float  # TODO(validation): Ensure non-negative
    timestamp: str  # TODO(typing): Use datetime instead of str for type safety


# TODO(documentation): Add comprehensive Google-style docstring with examples
# TODO(design): Consider extracting configuration to a separate Config dataclass
class EmbeddedQwenAnalyzer:
    """Embedded Qwen analyzer (unified version from archive/qwen_analyzer.py).

    This analyzer uses Qwen model via Novita AI API to analyze rap lyrics.

    Attributes:
        config: Configuration dictionary for the analyzer.
        model_name: Name of the Qwen model to use.
        base_url: API base URL.
        temperature: Sampling temperature for model.
        max_tokens: Maximum tokens for response.
        timeout: Request timeout in seconds.
        api_key: API key for authentication.
        available: Whether the analyzer is available for use.
        client: OpenAI client instance.
    """

    # TODO(constants): Extract magic values to module-level constants
    _DEFAULT_MODEL = "qwen/qwen3-4b-fp8"
    _DEFAULT_BASE_URL = "https://api.novita.ai/openai/v1"
    _DEFAULT_TEMPERATURE = 0.1
    _DEFAULT_MAX_TOKENS = 1500
    _DEFAULT_TIMEOUT = 30

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize embedded Qwen analyzer.

        Args:
            config: Optional configuration dictionary. If None, uses defaults.

        Note:
            Requires NOVITA_API_KEY environment variable to be set.
        """
        # TODO(validation): Validate config structure and types
        self.config = config or {}

        # TODO(i18n): Remove Russian comments
        # Model settings
        self.model_name = self.config.get("model_name", self._DEFAULT_MODEL)
        self.base_url = self.config.get("base_url", self._DEFAULT_BASE_URL)
        self.temperature = self.config.get("temperature", self._DEFAULT_TEMPERATURE)
        self.max_tokens = self.config.get("max_tokens", self._DEFAULT_MAX_TOKENS)
        self.timeout = self.config.get("timeout", self._DEFAULT_TIMEOUT)

        # TODO(security): Avoid storing API key as instance variable - use secure vault
        # API key
        self.api_key = self.config.get("api_key") or os.getenv("NOVITA_API_KEY")

        # TODO(design): Move initialization check to separate method, not in __init__
        # Availability check
        self.available = self._check_availability()

        # TODO(error-handling): Raise exception instead of silent failure if unavailable
        if self.available:
            self.client = openai.OpenAI(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
            )
            # TODO(i18n): Remove emojis from log messages
            logger.info(
                f"✅ Встроенный Qwen анализатор инициализирован: {self.model_name}"
            )
        else:
            # TODO(logging): Use logger.error instead of warning for critical failures
            logger.warning("⚠️ Встроенный Qwen анализатор недоступен")

    def _check_availability(self) -> bool:
        """Check Novita AI API availability.

        Returns:
            True if API is available and responsive, False otherwise.
        """
        # TODO(i18n): Remove emojis and Russian text from log messages
        if not HAS_OPENAI:
            logger.error("❌ openai не установлен. Установите: pip install openai")
            return False

        if not self.api_key:
            logger.error(
                "❌ NOVITA_API_KEY не найден в конфигурации или переменных окружения"
            )
            return False

        # TODO(error-handling): Catch specific exceptions (APIError, ConnectionError, etc.)
        # TODO(testing): Extract API test logic to separate method for better testability
        # TODO(performance): Consider making this async to not block initialization
        try:
            # TODO(i18n): Remove Russian comment
            # Test connection
            # TODO(constants): Extract test message to constant
            # TODO(magic-numbers): Extract magic numbers (10, 0.1) to constants
            client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Test"},
                ],
                max_tokens=10,
                temperature=0.1,
            )

            if response.choices and response.choices[0].message:
                logger.info("✅ Novita AI Qwen API успешно протестирован")
                return True
            logger.error("❌ Получен пустой ответ от Qwen API")
            return False

        # TODO(error-handling): Replace bare except with specific exception types
        except Exception as e:
            logger.error(f"❌ Ошибка проверки Qwen API: {e}")
            return False

    def validate_input(self, artist: str, title: str, lyrics: str) -> bool:
        """Validate input data.

        Args:
            artist: Artist name to validate.
            title: Song title to validate.
            lyrics: Lyrics text to validate.

        Returns:
            True if all inputs are valid, False otherwise.
        """
        # TODO(validation): Add more specific validation for each field
        # TODO(validation): Raise ValueError with specific error message instead of returning False
        # TODO(magic-numbers): Extract minimum lyrics length (10) to constant
        if not all([artist, title, lyrics]):
            return False
        if len(lyrics.strip()) < 10:
            return False
        return True

    def preprocess_lyrics(self, lyrics: str) -> str:
        """Preprocess song lyrics text.

        Performs text normalization including:
        - Removing excess whitespace
        - Limiting repeated characters
        - Normalizing line breaks
        - Removing URLs and special characters

        Args:
            lyrics: Raw lyrics text to preprocess.

        Returns:
            Preprocessed and normalized lyrics text.
        """
        # TODO(imports): Move re import to module level instead of function level
        # TODO(i18n): Remove Russian comments
        lyrics = lyrics.strip()

        # Remove excess whitespace
        import re

        lyrics = re.sub(r"\s+", " ", lyrics)

        # TODO(magic-numbers): Extract magic number {3,} to constant (MAX_CHAR_REPETITION)
        # Limit repeated characters
        lyrics = re.sub(r"(.)\1{3,}", r"\1\1\1", lyrics)

        # TODO(magic-numbers): Extract {3,} to constant (MAX_NEWLINES)
        # Normalize line breaks
        lyrics = re.sub(r"\n{3,}", "\n\n", lyrics)

        # TODO(security): Use more specific regex to avoid removing legitimate content
        # Remove URLs and special characters
        lyrics = re.sub(r"http[s]?://\S+", "", lyrics)
        lyrics = re.sub(r'[^\w\s\n.,!?\'"-]', "", lyrics)

        return lyrics.strip()

    def analyze_song(self, artist: str, title: str, lyrics: str) -> AnalysisResult:
        """Analyze song using Qwen model.

        Args:
            artist: The artist name.
            title: The song title.
            lyrics: The song lyrics text.

        Returns:
            AnalysisResult object containing analysis data.

        Raises:
            ValueError: If input parameters are invalid.
            RuntimeError: If analyzer is not available or API call fails.
        """
        # TODO(performance): Consider caching results to avoid duplicate API calls
        # TODO(metrics): Add timing metrics for each step (validation, preprocessing, API call)
        start_time = time.time()

        # TODO(i18n): Remove Russian comments
        # Validate input data
        if not self.validate_input(artist, title, lyrics):
            # TODO(error-handling): Provide specific error message about what is invalid
            raise ValueError("Invalid input parameters")

        if not self.available:
            raise RuntimeError("Qwen analyzer is not available")

        # TODO(i18n): Remove Russian comment
        # Preprocess text
        processed_lyrics = self.preprocess_lyrics(lyrics)

        # TODO(error-handling): Catch specific exceptions (APIError, TimeoutError, etc.)
        try:
            # TODO(i18n): Remove Russian comments
            # Create prompt
            system_prompt, user_prompt = self._create_analysis_prompts(
                artist, title, processed_lyrics
            )

            # TODO(i18n): Remove Russian comment
            # TODO(retry): Add retry logic with exponential backoff for API failures
            # TODO(rate-limiting): Implement rate limiting to avoid API quota issues
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            if not response.choices or not response.choices[0].message.content:
                raise RuntimeError("Empty response from Qwen model")

            # TODO(i18n): Remove Russian comments
            # Parse result
            analysis_data = self._parse_response(response.choices[0].message.content)

            # Calculate confidence
            confidence = self._calculate_confidence(analysis_data)

            processing_time = time.time() - start_time

            # TODO(constants): Extract "qwen-3-4b-fp8" and "Novita AI" to module constants
            # TODO(i18n): Remove Russian comment "Правильное поле!"
            return AnalysisResult(
                artist=artist,
                title=title,
                analyzer_type="qwen",
                confidence=confidence,
                metadata={
                    "model_name": self.model_name,
                    "model_version": "qwen3-4b-fp8",
                    "processing_date": datetime.now().isoformat(),
                    "lyrics_length": len(processed_lyrics),
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "provider": "Novita AI",
                    # TODO(error-handling): Add proper null checking with explicit defaults
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens
                        if response.usage
                        else 0,
                        "completion_tokens": response.usage.completion_tokens
                        if response.usage
                        else 0,
                        "total_tokens": response.usage.total_tokens
                        if response.usage
                        else 0,
                    },
                },
                raw_output=analysis_data,
                processing_time=processing_time,
                timestamp=datetime.now().isoformat(),
            )

        # TODO(error-handling): Replace bare except with specific exception types
        except Exception as e:
            # TODO(i18n): Remove emojis and Russian text from error messages
            logger.error(
                f"❌ Ошибка анализа встроенного Qwen для {artist} - {title}: {e}"
            )
            raise RuntimeError(f"Qwen analysis failed: {e}") from e

    def _create_analysis_prompts(
        self, artist: str, title: str, lyrics: str
    ) -> tuple[str, str]:
        """Create system and user prompts for Qwen model.

        Args:
            artist: Artist name to include in prompt.
            title: Song title to include in prompt.
            lyrics: Song lyrics to analyze.

        Returns:
            Tuple of (system_prompt, user_prompt).
        """
        # TODO(i18n): Remove Russian comment
        # TODO(constants): Extract max_lyrics_length to class/module constant
        # TODO(documentation): Document why 2000 is chosen as the limit
        # Limit text length for API
        max_lyrics_length = 2000
        if len(lyrics) > max_lyrics_length:
            # TODO(data-loss): Log warning when truncating lyrics
            # TODO(design): Consider using smarter truncation (e.g., by sentences)
            lyrics = lyrics[:max_lyrics_length] + "..."

        # TODO(design): Extract prompts to separate template files or constants
        # TODO(maintainability): Use jinja2 or similar for prompt templates
        # TODO(testing): Create test cases to verify prompt structure produces valid JSON
        system_prompt = """You are a rap lyrics analyzer. You MUST respond with ONLY a JSON object, no other text.

CRITICAL: Do not include ANY explanations, thoughts, or text outside the JSON.
NO <think> tags, NO explanations, NO additional text.
Start your response with { and end with }.

Analyze rap songs and return JSON with this structure only."""

        # TODO(security): Sanitize artist/title to prevent prompt injection attacks
        # TODO(design): Use f-string with proper escaping for JSON safety
        user_prompt = f"""Artist: {artist}
Title: {title}
Lyrics: {lyrics}

Return ONLY this JSON structure (fill with actual analysis):
{{
    "genre_analysis": {{
        "primary_genre": "rap",
        "subgenre": "string",
        "confidence": 0.9
    }},
    "mood_analysis": {{
        "primary_mood": "confident",
        "emotional_intensity": "high",
        "energy_level": "high", 
        "valence": "positive"
    }},
    "content_analysis": {{
        "explicit_content": false,
        "explicit_level": "none",
        "main_themes": ["success"],
        "narrative_style": "boastful"
    }},
    "technical_analysis": {{
        "rhyme_scheme": "complex",
        "flow_pattern": "varied",
        "complexity_level": "advanced",
        "wordplay_quality": "excellent",
        "metaphor_usage": "moderate",
        "structure": "traditional"
    }},
    "quality_metrics": {{
        "lyrical_creativity": 0.8,
        "technical_skill": 0.9,
        "authenticity": 0.9,
        "commercial_appeal": 0.8,
        "originality": 0.8,
        "overall_quality": 0.8,
        "ai_generated_likelihood": 0.1
    }},
    "cultural_context": {{
        "era_estimate": "2020s",
        "regional_style": "mainstream",
        "cultural_references": [],
        "social_commentary": false
    }}
}}"""

        return system_prompt, user_prompt

    def _parse_response(self, response_text: str) -> dict[str, Any]:
        """Parse response from Qwen model.

        Args:
            response_text: Raw response text from the model.

        Returns:
            Parsed analysis data dictionary.
        """
        # TODO(error-handling): Return Result type instead of dict for better error handling
        # TODO(complexity): Split this method into smaller helper methods (Single Responsibility)
        try:
            # TODO(i18n): Remove Russian comments
            # Clean excess text
            response_text = response_text.strip()

            # TODO(imports): Move re import to module level
            # Remove <think>...</think> tags
            import re

            response_text = re.sub(
                r"<think>.*?</think>", "", response_text, flags=re.DOTALL
            )
            response_text = response_text.strip()

            # TODO(i18n): Remove Russian comment and emoji
            # TODO(magic-numbers): Extract 500 to constant (MAX_LOG_PREVIEW_LENGTH)
            # Log raw response for debugging
            logger.debug(f"Cleaned response (first 500 chars): {response_text[:500]}")

            # TODO(complexity): Extract JSON parsing logic to separate method
            # TODO(i18n): Remove Russian comment
            # If response is already JSON, parse directly
            if response_text.startswith("{") and response_text.endswith("}"):
                try:
                    analysis_data = json.loads(response_text)
                    # TODO(i18n): Remove emoji from log
                    logger.debug("✅ Direct JSON parsing successful")
                except json.JSONDecodeError as e:
                    logger.warning(f"Direct JSON parsing failed: {e}")
                    # TODO(i18n): Remove Russian comment
                    # Try to fix and parse again
                    fixed_json = self._fix_common_json_issues(response_text)
                    try:
                        analysis_data = json.loads(fixed_json)
                        # TODO(i18n): Remove emoji
                        logger.debug("✅ JSON parsing successful after fixes")
                    except json.JSONDecodeError:
                        logger.error("JSON still invalid after fixes")
                        analysis_data = self._create_fallback_analysis()
            else:
                # TODO(i18n): Remove Russian comment
                # Find JSON block in text
                json_start = response_text.find("{")
                # TODO(magic-numbers): Document why we add 1 to json_end
                json_end = response_text.rfind("}") + 1

                # TODO(magic-numbers): Extract -1 and 0 checks to named constants
                if json_start == -1 or json_end == 0:
                    logger.error(
                        f"No JSON found in response. Full response: {response_text}"
                    )
                    # TODO(i18n): Remove Russian comment
                    # Create basic response
                    analysis_data = self._create_fallback_analysis()
                else:
                    json_str = response_text[json_start:json_end]
                    try:
                        analysis_data = json.loads(json_str)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON in response: {json_str}")
                        analysis_data = self._create_fallback_analysis()

            # TODO(i18n): Remove Russian comment
            # Validate structure
            self._validate_analysis_structure(analysis_data)

            return analysis_data

        except json.JSONDecodeError as e:
            # TODO(i18n): Remove emojis and Russian text
            # TODO(magic-numbers): Extract 500 to constant
            logger.error(f"❌ Ошибка парсинга JSON: {e}")
            logger.error(f"Ответ модели: {response_text[:500]}...")
            return self._create_fallback_analysis()

        # TODO(error-handling): Replace bare except with specific exception types
        except Exception as e:
            # TODO(i18n): Remove emojis and Russian text
            logger.error(f"❌ Ошибка обработки ответа: {e}")
            return self._create_fallback_analysis()

    def _fix_common_json_issues(self, json_str: str) -> str:
        """Fix common JSON issues.

        Args:
            json_str: JSON string with potential issues.

        Returns:
            Fixed JSON string.
        """
        # TODO(imports): Move re import to module level
        # TODO(testing): Add unit tests for various JSON edge cases
        import re

        # TODO(i18n): Remove Russian comments
        # Remove trailing commas
        json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)

        # TODO(i18n): Remove Russian comment
        # TODO(edge-cases): This regex may incorrectly replace single quotes in content
        # Fix single quotes to double quotes (if any)
        json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)

        # TODO(i18n): Remove Russian comment
        # Remove excess trailing characters
        json_str = json_str.strip()

        return json_str

    def _validate_analysis_structure(self, data: dict[str, Any]) -> None:
        """Validate analysis result structure.

        Ensures all required sections are present and metrics have valid values.
        Mutates the data dict by adding missing sections.

        Args:
            data: Analysis data dictionary to validate.
        """
        # TODO(constants): Extract required sections list to class/module constant
        # TODO(design): Consider raising ValidationError instead of silently fixing
        required_sections = [
            "genre_analysis",
            "mood_analysis",
            "content_analysis",
            "technical_analysis",
            "quality_metrics",
            "cultural_context",
        ]

        for section in required_sections:
            if section not in data:
                logger.warning(f"Missing section: {section}")
                # TODO(side-effects): Document that this method mutates input data
                data[section] = {}

        # TODO(i18n): Remove Russian comment
        # TODO(constants): Extract metric names to constant
        # Check quality metrics
        quality_metrics = data.get("quality_metrics", {})
        for metric in [
            "lyrical_creativity",
            "technical_skill",
            "authenticity",
            "commercial_appeal",
            "originality",
            "overall_quality",
        ]:
            if metric in quality_metrics:
                value = quality_metrics[metric]
                # TODO(validation): Actually fix invalid values instead of just logging
                if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                    logger.warning(f"Invalid metric value for {metric}: {value}")

    def _calculate_confidence(self, analysis_data: dict[str, Any]) -> float:
        """Calculate confidence in analysis results.

        Args:
            analysis_data: Analysis data dictionary.

        Returns:
            Confidence score between 0 and 1.
        """
        # TODO(algorithm): Document confidence calculation algorithm
        # TODO(testing): Add unit tests for edge cases (empty data, partial data, etc.)
        confidence_factors = []

        # TODO(i18n): Remove Russian comments
        # TODO(magic-numbers): Extract 6 to constant (REQUIRED_SECTIONS_COUNT)
        # TODO(constants): Reuse section names from _validate_analysis_structure
        # Check analysis completeness
        sections_completed = 0
        total_sections = 6

        for section_name in [
            "genre_analysis",
            "mood_analysis",
            "content_analysis",
            "technical_analysis",
            "quality_metrics",
            "cultural_context",
        ]:
            if analysis_data.get(section_name):
                sections_completed += 1

        completeness_score = sections_completed / total_sections
        confidence_factors.append(completeness_score)

        # TODO(i18n): Remove Russian comment
        # Check genre confidence
        genre_analysis = analysis_data.get("genre_analysis", {})
        if "confidence" in genre_analysis:
            genre_confidence = genre_analysis["confidence"]
            if (
                isinstance(genre_confidence, (int, float))
                and 0 <= genre_confidence <= 1
            ):
                confidence_factors.append(genre_confidence)

        # TODO(i18n): Remove Russian comments
        # Check quality metrics
        quality_metrics = analysis_data.get("quality_metrics", {})
        if quality_metrics:
            # Average confidence by metrics
            valid_metrics = []
            for metric_value in quality_metrics.values():
                if isinstance(metric_value, (int, float)) and 0 <= metric_value <= 1:
                    valid_metrics.append(metric_value)

            if valid_metrics:
                avg_quality = sum(valid_metrics) / len(valid_metrics)
                confidence_factors.append(avg_quality)

        # TODO(i18n): Remove Russian comment
        # TODO(magic-numbers): Extract 0.5 to constant (DEFAULT_CONFIDENCE)
        # Overall confidence
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        return 0.5  # Average confidence when no data available

    def _create_fallback_analysis(self) -> dict[str, Any]:
        """Create basic analysis in case of parsing error.

        Returns:
            Default analysis data dictionary with low confidence values.
        """
        # TODO(constants): Extract all magic values to constants
        # TODO(design): Consider loading fallback from JSON config file
        return {
            "genre_analysis": {
                "primary_genre": "rap",
                "subgenre": "unknown",
                "confidence": 0.3,
            },
            "mood_analysis": {
                "primary_mood": "neutral",
                "emotional_intensity": "unknown",
                "energy_level": "unknown",
                "valence": "neutral",
            },
            "content_analysis": {
                "explicit_content": False,
                "explicit_level": "unknown",
                "main_themes": ["general"],
                "narrative_style": "unknown",
            },
            "technical_analysis": {
                "rhyme_scheme": "unknown",
                "flow_pattern": "unknown",
                "complexity_level": "unknown",
                "wordplay_quality": "unknown",
                "metaphor_usage": "unknown",
                "structure": "unknown",
            },
            "quality_metrics": {
                "lyrical_creativity": 0.3,
                "technical_skill": 0.3,
                "authenticity": 0.3,
                "commercial_appeal": 0.3,
                "originality": 0.3,
                "overall_quality": 0.3,
                "ai_generated_likelihood": 0.5,
            },
            "cultural_context": {
                "era_estimate": "unknown",
                "regional_style": "unknown",
                "cultural_references": [],
                "social_commentary": False,
            },
        }


# ============================================================================
# TODO(i18n): Translate Russian section headers to English
# MASS ANALYZER (from src/analyzers/mass_qwen_analysis.py)
# ============================================================================


# TODO(documentation): Add comprehensive Google-style docstring
# TODO(design): Consider adding slots=True for memory efficiency
@dataclass
class AnalysisStats:
    """Analysis statistics.

    Attributes:
        total_records: Total number of records to process.
        processed: Number of successfully processed records.
        errors: Number of errors encountered.
        skipped: Number of skipped records.
        start_time: When analysis started.
        current_batch: Current batch number being processed.
        total_batches: Total number of batches.
    """

    total_records: int = 0
    processed: int = 0
    errors: int = 0
    skipped: int = 0
    start_time: datetime | None = None
    current_batch: int = 0
    total_batches: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage.

        Returns:
            Success rate as percentage (0-100).
        """
        # TODO(i18n): Remove Russian docstring
        # TODO(magic-numbers): Extract 100 to constant
        # TODO(edge-case): Consider what should happen when total_records is 0
        return (self.processed / max(self.total_records, 1)) * 100

    @property
    def processing_rate(self) -> float:
        """Calculate processing rate in records per minute.

        Returns:
            Number of records processed per minute.
        """
        # TODO(i18n): Remove Russian docstring
        # TODO(magic-numbers): Extract 60 (seconds) to constant
        if not self.start_time:
            return 0.0
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return (self.processed / max(elapsed, 1)) * 60

    @property
    def estimated_remaining(self) -> timedelta:
        """Estimate remaining time to completion.

        Returns:
            Estimated time remaining as timedelta.
        """
        # TODO(i18n): Remove Russian docstring
        remaining = self.total_records - self.processed
        if remaining <= 0 or self.processing_rate == 0:
            return timedelta(0)
        minutes = remaining / self.processing_rate
        return timedelta(minutes=minutes)


# TODO(documentation): Add comprehensive Google-style docstring with examples
# TODO(design): Consider dependency injection for app, analyzer, db_manager
class UnifiedQwenMassAnalyzer:
    """Unified mass analyzer with embedded Qwen.

    This class orchestrates batch analysis of tracks using the Qwen model,
    with checkpoint support and progress tracking.

    Attributes:
        app: Application instance.
        analyzer: EmbeddedQwenAnalyzer instance.
        db_manager: PostgreSQL database manager.
        stats: Analysis statistics tracker.
        last_processed_id: ID of last successfully processed track.
        checkpoint_file: Path to checkpoint file for resume support.
    """

    # TODO(constants): Extract checkpoint filename to constant
    _CHECKPOINT_DIR = Path("results")
    _CHECKPOINT_FILENAME = "qwen_analysis_checkpoint.json"

    def __init__(self):
        """Initialize UnifiedQwenMassAnalyzer.

        Note:
            Call initialize() async method before using the analyzer.
        """
        # TODO(typing): Add type hints for these attributes
        self.app = None
        self.analyzer = None
        self.db_manager = None
        self.stats = AnalysisStats()
        self.last_processed_id = 0
        # TODO(i18n): Remove Russian comment
        # Save checkpoint to results folder
        self.checkpoint_file = self._CHECKPOINT_DIR / self._CHECKPOINT_FILENAME

    async def initialize(self) -> bool:
        """Initialize components.

        Returns:
            True if initialization successful, False otherwise.
        """
        # TODO(error-handling): Catch specific exceptions instead of bare except
        # TODO(logging): Use logger instead of print statements
        try:
            # TODO(i18n): Remove emojis and Russian text
            print("🔧 Инициализация системы...")

            # TODO(i18n): Remove Russian comment
            # Initialize application
            self.app = create_app()

            # TODO(i18n): Remove Russian comments
            # Get embedded analyzer
            self.analyzer = EmbeddedQwenAnalyzer()
            if not self.analyzer or not self.analyzer.available:
                # TODO(i18n): Remove emojis and Russian text
                print("❌ Встроенный Qwen анализатор недоступен!")
                print("💡 Проверьте NOVITA_API_KEY в .env файле")
                return False

            # TODO(i18n): Remove emojis and Russian text
            print(f"✅ Встроенный Qwen анализатор готов: {self.analyzer.model_name}")

            # TODO(i18n): Remove Russian comment
            # Connect to PostgreSQL
            self.db_manager = PostgreSQLManager()
            await self.db_manager.initialize()

            # TODO(i18n): Remove emojis and Russian text
            print("✅ PostgreSQL подключение установлено")
            return True

        # TODO(error-handling): Replace bare except with specific exception types
        except Exception as e:
            # TODO(i18n): Remove emojis and Russian text
            # TODO(logging): Use logger.error instead of print
            print(f"❌ Ошибка инициализации: {e}")
            return False

    async def get_database_stats(self) -> dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with database statistics (total_tracks, tracks_with_lyrics, etc.).
        """
        # TODO(error-handling): Catch specific database exceptions
        # TODO(logging): Use logger instead of print
        try:
            async with self.db_manager.connection_pool.acquire() as conn:
                # TODO(sql): Consider using parameterized queries or query builder
                # TODO(sql): Extract SQL to separate file or constant for maintainability
                # TODO(sql): Add index hints if performance becomes an issue
                stats_query = """
                SELECT
                    COUNT(*) as total_tracks,
                    COUNT(CASE WHEN lyrics IS NOT NULL AND lyrics != '' THEN 1 END) as tracks_with_lyrics,
                    COUNT(DISTINCT ar.track_id) as qwen_analyzed,
                    COUNT(CASE WHEN ar.track_id IS NULL THEN 1 END) as unanalyzed
                FROM tracks t
                LEFT JOIN analysis_results ar ON t.id = ar.track_id
                    AND ar.analyzer_type LIKE '%qwen%'
                WHERE t.lyrics IS NOT NULL AND t.lyrics != ''
                """

                result = await conn.fetchrow(stats_query)
                return dict(result) if result else {}

        # TODO(error-handling): Replace bare except with specific exception types
        except Exception as e:
            # TODO(i18n): Remove emojis and Russian text
            # TODO(logging): Use logger.error instead of print
            print(f"❌ Ошибка получения статистики: {e}")
            return {}

    async def load_checkpoint(self) -> bool:
        """Load checkpoint to resume work.

        Returns:
            True if checkpoint loaded successfully, False otherwise.
        """
        # TODO(error-handling): Catch specific file/JSON exceptions
        # TODO(logging): Use logger instead of print
        try:
            if not self.checkpoint_file.exists():
                return False

            # TODO(security): Validate checkpoint data structure and values
            with open(self.checkpoint_file, encoding="utf-8") as f:
                data = json.load(f)

            # TODO(magic-numbers): Extract default value 0 to constant
            self.last_processed_id = data.get("last_processed_id", 0)
            # TODO(i18n): Remove emojis and Russian text
            print(
                f"📍 Загружен чекпоинт: последняя обработанная запись ID {self.last_processed_id}"
            )
            return True

        # TODO(error-handling): Replace bare except with specific exception types
        except Exception as e:
            # TODO(i18n): Remove emojis and Russian text
            # TODO(logging): Use logger.warning instead of print
            print(f"⚠️ Ошибка загрузки чекпоинта: {e}")
            return False

    async def save_checkpoint(self):
        """Save checkpoint.

        Saves current progress to checkpoint file for resume capability.
        """
        # TODO(error-handling): Catch specific file/IO exceptions
        # TODO(logging): Use logger instead of print
        # TODO(typing): Add return type annotation (-> None or -> bool)
        try:
            # TODO(i18n): Remove Russian comment
            # TODO(magic-numbers): Extract exist_ok=True rationale to comment
            # Create results folder if it doesn't exist
            self.checkpoint_file.parent.mkdir(exist_ok=True)

            # TODO(validation): Validate data before saving
            data = {
                "last_processed_id": self.last_processed_id,
                "timestamp": datetime.now().isoformat(),
                "processed": self.stats.processed,
                "errors": self.stats.errors,
            }

            # TODO(atomicity): Use atomic write (write to temp file, then rename)
            # TODO(magic-numbers): Document why indent=2 is chosen
            with open(self.checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        # TODO(error-handling): Replace bare except with specific exception types
        except Exception as e:
            # TODO(i18n): Remove emojis and Russian text
            # TODO(logging): Use logger.error instead of print
            print(f"⚠️ Ошибка сохранения чекпоинта: {e}")

    async def get_unanalyzed_records(
        self, limit: int | None = None, resume: bool = False
    ) -> list[dict[str, Any]]:
        """Get unanalyzed records from database.

        Args:
            limit: Maximum number of records to retrieve.
            resume: Whether to resume from last checkpoint.

        Returns:
            List of unanalyzed track records.
        """
        # TODO(error-handling): Catch specific database exceptions
        # TODO(logging): Use logger instead of print
        try:
            async with self.db_manager.connection_pool.acquire() as conn:
                # TODO(i18n): Remove Russian comments
                # TODO(sql): Extract SQL to constant or query builder
                # Base query
                query = """
                SELECT t.id, t.artist, t.title, t.lyrics
                FROM tracks t
                WHERE t.lyrics IS NOT NULL
                AND t.lyrics != ''
                AND t.id NOT IN (
                    SELECT DISTINCT track_id
                    FROM analysis_results
                    WHERE analyzer_type LIKE '%qwen%'
                )
                """

                # TODO(sql-injection): CRITICAL! Use parameterized query instead of f-string
                # TODO(security): This is a SQL injection vulnerability!
                # TODO(i18n): Remove Russian comment
                # Condition for resume mode
                if resume and self.last_processed_id > 0:
                    query += f" AND t.id > {self.last_processed_id}"

                query += " ORDER BY t.id"

                # TODO(sql-injection): CRITICAL! Use parameterized query instead of f-string
                # TODO(security): This is a SQL injection vulnerability!
                # TODO(i18n): Remove Russian comment
                # Limit
                if limit:
                    query += f" LIMIT {limit}"

                records = await conn.fetch(query)
                return [dict(record) for record in records]

        # TODO(error-handling): Replace bare except with specific exception types
        except Exception as e:
            # TODO(i18n): Remove emojis and Russian text
            # TODO(logging): Use logger.error instead of print
            print(f"❌ Ошибка получения записей: {e}")
            return []

    async def analyze_single_record(self, record: dict[str, Any]) -> bool:
        """Analyze single record.

        Args:
            record: Track record dictionary containing id, artist, title, lyrics.

        Returns:
            True if analysis successful, False otherwise.
        """
        # TODO(validation): Validate record structure and handle KeyError
        # TODO(error-handling): Catch specific exceptions
        # TODO(i18n): Remove Russian docstring
        track_id = record["id"]
        artist = record["artist"]  # TODO(error-handling): Use .get() with default
        title = record["title"]  # TODO(error-handling): Use .get() with default
        lyrics = record["lyrics"]

        try:
            # TODO(i18n): Remove Russian comment
            # Analyze text (embedded analyzer)
            result = self.analyzer.analyze_song(artist, title, lyrics)

            if result is None:
                return False

            # TODO(i18n): Remove Russian comment
            # TODO(constants): Extract "qwen-3-4b-fp8" to module constant
            # TODO(magic-numbers): Extract 0.5, 1000 to constants
            # Prepare data for PostgreSQL
            analysis_data = {
                "track_id": track_id,
                "analyzer_type": "qwen-3-4b-fp8",
                "sentiment": self._extract_sentiment(result),
                "confidence": result.confidence or 0.5,
                "complexity_score": self._extract_complexity(result),
                "themes": self._extract_themes(result),
                "analysis_data": result.raw_output or {},
                "processing_time_ms": int((result.processing_time or 0) * 1000),
                "model_version": result.metadata.get("model_name", "qwen-3-4b-fp8"),
            }

            # TODO(i18n): Remove Russian comment
            # Save to database
            success = await self._save_analysis_to_database(analysis_data)

            if success:
                self.last_processed_id = track_id
                return True
            return False

        # TODO(error-handling): Replace bare except with specific exception types
        except Exception as e:
            # TODO(i18n): Remove emojis and Russian text
            # TODO(logging): Use logger.error instead of print
            print(f"❌ Ошибка анализа записи {track_id}: {e}")
            return False

    async def _save_analysis_to_database(self, analysis_data: dict[str, Any]) -> bool:
        """Save analysis result to database.

        Args:
            analysis_data: Analysis data dictionary to save.

        Returns:
            True if save successful, False otherwise.
        """
        # TODO(error-handling): Catch specific database exceptions
        # TODO(logging): Use logger instead of print
        # TODO(i18n): Remove Russian docstring
        try:
            async with self.db_manager.connection_pool.acquire() as conn:
                # TODO(i18n): Remove Russian comment
                # TODO(sql): Extract SQL queries to constants
                # Check if analysis already exists for this track
                existing = await conn.fetchrow(
                    "SELECT id FROM analysis_results WHERE track_id = $1 AND analyzer_type = $2",
                    analysis_data["track_id"],
                    analysis_data["analyzer_type"],
                )

                if existing:
                    # TODO(i18n): Remove emojis and Russian text
                    # TODO(logging): Use logger.info instead of print
                    print(
                        f"  ⚠️ Анализ уже существует для трека {analysis_data['track_id']}"
                    )
                    return True

                # TODO(i18n): Remove Russian comment
                # TODO(sql): Extract SQL to constant
                # Insert new analysis
                insert_query = """
                INSERT INTO analysis_results
                (track_id, analyzer_type, analysis_data, confidence, sentiment, complexity_score, themes, processing_time_ms, model_version, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                RETURNING id
                """

                # TODO(error-handling): Validate analysis_data structure before using
                result = await conn.fetchrow(
                    insert_query,
                    analysis_data["track_id"],
                    analysis_data["analyzer_type"],
                    json.dumps(analysis_data["analysis_data"]),
                    analysis_data["confidence"],
                    analysis_data["sentiment"],
                    analysis_data["complexity_score"],
                    json.dumps(analysis_data["themes"]),
                    analysis_data["processing_time_ms"],
                    analysis_data["model_version"],
                )

                return result is not None

        # TODO(error-handling): Replace bare except with specific exception types
        except Exception as e:
            # TODO(i18n): Remove emojis and Russian text
            # TODO(logging): Use logger.error instead of print
            print(f"❌ Ошибка сохранения анализа: {e}")
            return False

    def _extract_sentiment(self, result) -> str:
        """Extract sentiment from analysis result.

        Args:
            result: AnalysisResult object.

        Returns:
            Sentiment string (e.g., 'neutral', 'positive', etc.).
        """
        # TODO(typing): Add type hint for result parameter (AnalysisResult)
        # TODO(error-handling): Replace bare except with specific exception types
        # TODO(constants): Extract "neutral" default to constant
        # TODO(i18n): Remove Russian docstring
        try:
            if hasattr(result, "raw_output") and result.raw_output:
                mood_analysis = result.raw_output.get("mood_analysis", {})
                return mood_analysis.get("primary_mood", "neutral")
            return "neutral"
        except:  # TODO(error-handling): Specify exception type
            return "neutral"

    def _extract_complexity(self, result) -> float:
        """Extract complexity score from analysis result.

        Args:
            result: AnalysisResult object.

        Returns:
            Complexity score (0.0-5.0).
        """
        # TODO(typing): Add type hint for result parameter (AnalysisResult)
        # TODO(error-handling): Replace bare except with specific exception types
        # TODO(magic-numbers): Extract 0.5, 5.0, 3.0 to constants with documentation
        # TODO(i18n): Remove Russian docstring
        try:
            if hasattr(result, "raw_output") and result.raw_output:
                quality_metrics = result.raw_output.get("quality_metrics", {})
                return float(quality_metrics.get("overall_quality", 0.5)) * 5.0
            return 3.0
        except:  # TODO(error-handling): Specify exception type
            return 3.0

    def _extract_themes(self, result) -> list[str]:
        """Extract themes from analysis result.

        Args:
            result: AnalysisResult object.

        Returns:
            List of theme strings.
        """
        # TODO(typing): Add type hint for result parameter (AnalysisResult)
        # TODO(error-handling): Replace bare except with specific exception types
        # TODO(constants): Extract ["general"] default to constant
        # TODO(i18n): Remove Russian docstring
        try:
            if hasattr(result, "raw_output") and result.raw_output:
                content_analysis = result.raw_output.get("content_analysis", {})
                return content_analysis.get("main_themes", ["general"])
            return ["general"]
        except:  # TODO(error-handling): Specify exception type
            return ["general"]

    async def process_batch(self, batch: list[dict[str, Any]]) -> tuple[int, int]:
        """Process batch of records.

        Args:
            batch: List of track records to process.

        Returns:
            Tuple of (processed_count, error_count).
        """
        # TODO(i18n): Remove Russian docstring
        # TODO(logging): Use logger instead of print
        processed = 0
        errors = 0

        # TODO(magic-numbers): Extract 1 start index to constant or use enumerate(batch, start=1)
        for i, record in enumerate(batch, 1):
            # TODO(validation): Validate record structure
            track_id = record["id"]
            # TODO(constants): Extract "Unknown" to constant
            artist = record.get("artist", "Unknown")
            title = record.get("title", "Unknown")

            # TODO(i18n): Remove Russian comment and emojis
            # Progress within batch
            print(f"  🎵 [{i}/{len(batch)}] {artist} - {title} (ID: {track_id})")

            # TODO(error-handling): Catch specific exceptions
            try:
                if await self.analyze_single_record(record):
                    processed += 1
                    # TODO(i18n): Remove emojis and Russian text
                    # TODO(logging): Use logger.info instead of print
                    print("    ✅ Успешно проанализировано")
                else:
                    errors += 1
                    # TODO(i18n): Remove emojis and Russian text
                    # TODO(logging): Use logger.error instead of print
                    print("    ❌ Ошибка анализа")

                # TODO(i18n): Remove Russian comment
                # TODO(magic-numbers): Extract 0.5 to constant (INTER_REQUEST_DELAY_SECONDS)
                # TODO(rate-limiting): Implement smarter rate limiting (e.g., token bucket)
                # Pause between requests to prevent rate limiting
                await asyncio.sleep(0.5)

            # TODO(error-handling): Replace bare except with specific exception types
            except Exception as e:
                errors += 1
                # TODO(i18n): Remove emojis and Russian text
                # TODO(logging): Use logger.exception to include traceback
                print(f"    ❌ Исключение: {e}")

        return processed, errors

    def print_progress(self):
        """Print progress information.

        Displays current processing statistics.
        """
        # TODO(i18n): Remove Russian docstring and emojis
        # TODO(logging): Use logger.info instead of print
        # TODO(design): Consider using a progress bar library (tqdm, rich)
        # TODO(typing): Add return type annotation (-> None)
        print("\n📊 ПРОГРЕСС:")
        print(f"  📈 Обработано: {self.stats.processed}/{self.stats.total_records}")
        # TODO(magic-numbers): Extract .1f precision to constant
        print(f"  ✅ Успешность: {self.stats.success_rate:.1f}%")
        print(f"  ⚡ Скорость: {self.stats.processing_rate:.1f} записей/мин")
        print(f"  ⏱️  Осталось: {self.stats.estimated_remaining}")
        print(f"  📦 Батч: {self.stats.current_batch}/{self.stats.total_batches}")

    async def run_analysis(
        self,
        batch_size: int = 100,
        max_records: int | None = None,
        resume: bool = False,
        test_mode: bool = False,
    ) -> dict[str, Any]:
        """Run mass analysis.

        Args:
            batch_size: Number of records to process per batch.
            max_records: Maximum number of records to process (None for all).
            resume: Whether to resume from last checkpoint.
            test_mode: Enable test mode (limited records).

        Returns:
            Dictionary with analysis results and statistics.
        """
        # TODO(i18n): Remove Russian docstring
        # TODO(logging): Use logger instead of print for all output
        # TODO(complexity): This method is too long (200+ lines) - split into smaller methods
        # TODO(magic-numbers): Extract 100, 10, 5, 70, 15, 60 to constants

        # TODO(i18n): Remove emojis and Russian text
        # TODO(magic-numbers): Extract 70 to constant (SEPARATOR_LENGTH)
        print("🎵 Объединенный Qwen массовый анализатор (PostgreSQL v3.0)")
        print("=" * 70)

        # TODO(i18n): Remove Russian comment
        # Load checkpoint if needed
        if resume:
            await self.load_checkpoint()

        # TODO(i18n): Remove Russian comments
        # Get database statistics
        db_stats = await self.get_database_stats()
        # TODO(i18n): Remove emojis and Russian text
        # TODO(magic-numbers): Extract 0 default to constant
        print("📊 База данных:")
        print(f"  📁 Всего треков: {db_stats.get('total_tracks', 0)}")
        print(f"  📝 С текстами: {db_stats.get('tracks_with_lyrics', 0)}")
        print(f"  🤖 Уже проанализировано Qwen: {db_stats.get('qwen_analyzed', 0)}")
        print(f"  ⏳ Ожидает анализа: {db_stats.get('unanalyzed', 0)}")

        # TODO(i18n): Remove Russian comment
        # TODO(constants): Extract 10, 5 to TEST_MODE_MAX_RECORDS, TEST_MODE_BATCH_SIZE
        # Determine limit for test mode
        if test_mode:
            max_records = 10
            batch_size = 5
            # TODO(i18n): Remove emojis and Russian text
            print(f"\n🧪 ТЕСТОВЫЙ РЕЖИМ: анализируем только {max_records} записей")

        # TODO(i18n): Remove Russian comment and emojis
        # Get records for analysis
        print("\n🔍 Загрузка записей для анализа...")
        records = await self.get_unanalyzed_records(limit=max_records, resume=resume)

        if not records:
            # TODO(i18n): Remove emojis and Russian text
            print("✅ Все записи уже проанализированы!")
            return {"status": "completed", "message": "No records to process"}

        # TODO(i18n): Remove Russian comment
        # Initialize statistics
        self.stats.total_records = len(records)
        self.stats.start_time = datetime.now()
        # TODO(magic-numbers): Document ceiling division formula
        self.stats.total_batches = (len(records) + batch_size - 1) // batch_size

        # TODO(i18n): Remove Russian comments and emojis
        # TODO(magic-numbers): Extract 15, 60 to constants (AVG_SECONDS_PER_RECORD, SECONDS_PER_MINUTE)
        print("\n🎯 План анализа:")
        print(f"  📊 Записей к обработке: {len(records)}")
        print(f"  📦 Размер батча: {batch_size}")
        print(f"  🔢 Количество батчей: {self.stats.total_batches}")
        print(f"  ⏱️  Примерное время: {(len(records) * 15) // 60} минут")
        print("  🆓 Бесплатная модель Qwen через Novita AI - без затрат!")

        if not test_mode:
            # TODO(i18n): Remove emojis and Russian text
            # TODO(magic-numbers): Extract 3 to constant (STARTUP_DELAY_SECONDS)
            print("\n⏳ Начинаем через 3 секунды...")
            await asyncio.sleep(3)

        # TODO(i18n): Remove Russian comment and emojis
        # TODO(magic-numbers): Extract 50 to constant (SEPARATOR_LENGTH)
        # Mass processing by batches
        print("\n🚀 НАЧИНАЕМ МАССОВЫЙ АНАЛИЗ")
        print("=" * 50)

        for i in range(0, len(records), batch_size):
            self.stats.current_batch += 1
            batch = records[i : i + batch_size]
            batch_start = time.time()

            print(f"\n📦 Батч {self.stats.current_batch}/{self.stats.total_batches}")
            print(
                f"📊 Записи {i + 1}-{min(i + batch_size, len(records))} из {len(records)}"
            )

            # Обработка батча
            batch_processed, batch_errors = await self.process_batch(batch)

            # Обновление статистики
            self.stats.processed += batch_processed
            self.stats.errors += batch_errors

            # Сохранение чекпоинта
            await self.save_checkpoint()

            # Статистика батча
            batch_time = time.time() - batch_start
            print(f"  ⏱️  Батч завершен за {batch_time:.1f}с")
            print(f"  ✅ Успешно: {batch_processed}")
            print(f"  ❌ Ошибок: {batch_errors}")

            # Общий прогресс
            self.print_progress()

            # Пауза между батчами (кроме последнего)
            if i + batch_size < len(records):
                print("  ⏸️  Пауза между батчами...")
                await asyncio.sleep(2)

        # Финальная статистика
        total_time = (datetime.now() - self.stats.start_time).total_seconds()

        print("\n🏆 АНАЛИЗ ЗАВЕРШЕН!")
        print("=" * 50)
        print(f"✅ Успешно проанализировано: {self.stats.processed}")
        print(f"❌ Ошибок: {self.stats.errors}")
        print(f"📊 Всего записей: {self.stats.total_records}")
        print(f"🎯 Успешность: {self.stats.success_rate:.1f}%")
        print(f"⏱️  Общее время: {total_time // 60:.0f}м {total_time % 60:.0f}с")
        print(f"⚡ Средняя скорость: {self.stats.processing_rate:.1f} записей/мин")

        # Обновленная статистика базы
        final_db_stats = await self.get_database_stats()
        print("\n📈 Обновленная статистика базы:")
        print(f"  🤖 Qwen проанализировано: {final_db_stats.get('qwen_analyzed', 0)}")
        print(f"  ⏳ Осталось: {final_db_stats.get('unanalyzed', 0)}")

        # Удаление чекпоинта при успешном завершении
        if self.stats.errors == 0 and self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            print("🗑️  Чекпоинт удален (анализ завершен без ошибок)")

        return {
            "status": "completed",
            "processed": self.stats.processed,
            "errors": self.stats.errors,
            "success_rate": self.stats.success_rate,
            "total_time": total_time,
            "processing_rate": self.stats.processing_rate,
        }

    async def show_stats_only(self) -> dict[str, Any]:
        """Показать только статистику без анализа"""
        print("📊 Статистика объединенного Qwen анализа базы данных")
        print("=" * 60)

        db_stats = await self.get_database_stats()

        print(f"📁 Всего треков: {db_stats.get('total_tracks', 0)}")
        print(f"📝 С текстами: {db_stats.get('tracks_with_lyrics', 0)}")
        print(f"🤖 Проанализировано Qwen: {db_stats.get('qwen_analyzed', 0)}")
        print(f"⏳ Ожидает анализа: {db_stats.get('unanalyzed', 0)}")

        if db_stats.get("tracks_with_lyrics", 0) > 0:
            coverage = (
                db_stats.get("qwen_analyzed", 0) / db_stats.get("tracks_with_lyrics", 1)
            ) * 100
            print(f"📈 Покрытие анализом: {coverage:.1f}%")

        # Проверка чекпоинта
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file) as f:
                    checkpoint = json.load(f)
                print("\n📍 Найден чекпоинт:")
                print(
                    f"  📄 Последняя обработанная запись: {checkpoint.get('last_processed_id', 0)}"
                )
                print(f"  📅 Дата: {checkpoint.get('timestamp', 'unknown')}")
                print(f"  ✅ Обработано в сессии: {checkpoint.get('processed', 0)}")
                print(f"  ❌ Ошибок в сессии: {checkpoint.get('errors', 0)}")
            except:
                print("\n⚠️ Найден поврежденный чекпоинт")

        return db_stats

    async def cleanup(self):
        """Clean up resources.

        Closes database connections and releases resources.
        """
        # TODO(i18n): Remove Russian docstring and emojis
        # TODO(error-handling): Add try-except for cleanup operations
        # TODO(typing): Add return type annotation (-> None)
        # TODO(logging): Use logger instead of print
        if self.db_manager:
            await self.db_manager.close()
        # TODO(i18n): Remove emojis and Russian text
        print("🧹 Ресурсы освобождены")


# TODO(documentation): Add comprehensive Google-style docstring
# TODO(error-handling): Add proper exception handling and logging
# TODO(design): Consider moving argument parsing to separate function
async def main():
    """Main function with extended argument parsing.

    Parses command-line arguments and runs the mass analyzer.
    """
    # TODO(i18n): Remove Russian docstring and text
    # TODO(i18n): Translate help messages and examples to English
    parser = argparse.ArgumentParser(
        description="Объединенный массовый анализ Qwen (PostgreSQL v3.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python src/analyzers/mass_qwen_analysis.py                    # Полный анализ
  python src/analyzers/mass_qwen_analysis.py --batch 50         # Батч 50 записей
  python src/analyzers/mass_qwen_analysis.py --test             # Тестовый режим
  python src/analyzers/mass_qwen_analysis.py --max 1000         # Лимит 1000 записей
  python src/analyzers/mass_qwen_analysis.py --resume           # Продолжить с чекпоинта
  python src/analyzers/mass_qwen_analysis.py --stats            # Только статистика
        """,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=100,
        metavar="N",
        help="Размер батча (default: 100)",
    )
    parser.add_argument(
        "--max",
        type=int,
        metavar="N",
        help="Максимальное количество записей для анализа",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Тестовый режим (только 10 записей с батчем 5)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Продолжить с последнего чекпоинта"
    )
    parser.add_argument(
        "--stats", action="store_true", help="Показать только статистику базы данных"
    )

    args = parser.parse_args()

    # Создание и инициализация анализатора
    analyzer = UnifiedQwenMassAnalyzer()

    try:
        # Режим только статистики
        if args.stats:
            if await analyzer.initialize():
                await analyzer.show_stats_only()
            return

        # Основной режим анализа
        if not await analyzer.initialize():
            print("❌ Не удалось инициализировать систему")
            return

        # Запуск анализа с параметрами
        result = await analyzer.run_analysis(
            batch_size=args.batch,
            max_records=args.max,
            resume=args.resume,
            test_mode=args.test,
        )

        print(f"\n🎯 Результат: {result['status']}")

    # TODO(error-handling): Catch specific exceptions instead of bare except
    except KeyboardInterrupt:
        # TODO(i18n): Remove emojis and Russian text
        # TODO(logging): Use logger instead of print
        print("\n\n⏹️  Анализ прерван пользователем")
        print("💡 Используйте --resume для продолжения с последней позиции")
    # TODO(error-handling): Catch specific exceptions instead of bare except
    except Exception as e:
        # TODO(i18n): Remove emojis and Russian text
        # TODO(logging): Use logger.exception instead of print to include traceback
        print(f"\n❌ Критическая ошибка: {e}")
    finally:
        await analyzer.cleanup()


if __name__ == "__main__":
    # TODO(logging): Configure root logger before running main
    # TODO(error-handling): Add sys.exit() with proper exit codes
    asyncio.run(main())


# ==============================================================================
# CODE REVIEW SUMMARY - CRITICAL ISSUES TO FIX (Google Standards)
# ==============================================================================
#
# PRIORITY 1 - CRITICAL SECURITY ISSUES (Fix immediately):
# --------------------------------------------------------
# 1. SQL INJECTION VULNERABILITIES (lines 1038-1052):
#    - get_unanalyzed_records() uses f-strings for SQL queries
#    - MUST use parameterized queries with $1, $2 placeholders
#    - Example: query += " AND t.id > $1" with params=[self.last_processed_id]
#
# 2. SECURITY - API Key Storage (line 153):
#    - API key stored as instance variable
#    - Consider using secure vault or environment-only access
#
# 3. SECURITY - Prompt Injection (lines 422-423):
#    - artist/title inserted directly into prompts without sanitization
#    - Could allow prompt injection attacks
#    - Sanitize or escape user-provided content
#
# PRIORITY 2 - CODE QUALITY & MAINTAINABILITY:
# --------------------------------------------
# 1. ERROR HANDLING:
#    - Replace ALL bare `except Exception:` with specific exceptions
#    - Lines with bare except: 218, 380, 560, 801-802, 1203-1204, 1224-1225, 1245-1246
#    - Use specific exceptions: asyncpg.PostgresError, openai.APIError, etc.
#
# 2. INTERNATIONALIZATION (i18n):
#    - Remove ALL Russian text and emojis throughout the file
#    - Use English for all comments, docstrings, and user-facing messages
#    - Consider using gettext for internationalization if needed
#
# 3. LOGGING vs PRINT:
#    - Replace ALL print() statements with proper logger calls
#    - Use logger.info(), logger.error(), logger.warning(), logger.debug()
#    - Configure logging with proper format, level, and handlers
#
# 4. MAGIC NUMBERS:
#    - Extract all magic numbers to module-level constants with SCREAMING_SNAKE_CASE
#    - Examples: 0.1, 1500, 30, 2000, 10, 100, 0.5, 3, 50, 60, etc.
#    - Add docstrings explaining why each constant has its value
#
# 5. TYPE HINTS:
#    - Replace dict[str, Any] with TypedDict or more specific types
#    - Add missing return type annotations (-> None where applicable)
#    - Add type hints for all function parameters (especially 'result' params)
#
# PRIORITY 3 - DESIGN & ARCHITECTURE:
# -----------------------------------
# 1. COMPLEXITY:
#    - run_analysis() method is 200+ lines - split into smaller methods
#    - _parse_response() has deep nesting - extract to helper methods
#    - Follow Single Responsibility Principle
#
# 2. IMPORTS:
#    - Move ALL regex (`import re`) to module level (appears 5+ times)
#    - Reorganize imports: stdlib, third-party, local (alphabetically)
#
# 3. CONSTANTS & CONFIGURATION:
#    - Extract prompts to separate template files or use jinja2
#    - Extract SQL queries to constants or query builder
#    - Create Config dataclass for analyzer settings
#
# 4. VALIDATION:
#    - Add input validation for all public methods
#    - Validate data structures before database operations
#    - Use Pydantic or dataclasses with validators
#
# 5. TESTING:
#    - Add unit tests for all methods (currently 0% coverage)
#    - Add integration tests for database operations
#    - Mock external API calls in tests
#
# PRIORITY 4 - PERFORMANCE & BEST PRACTICES:
# ------------------------------------------
# 1. RATE LIMITING:
#    - Implement smarter rate limiting (token bucket algorithm)
#    - Current sleep(0.5) is too simplistic
#
# 2. RETRY LOGIC:
#    - Add exponential backoff for API failures
#    - Handle transient errors gracefully
#
# 3. ATOMIC OPERATIONS:
#    - Use atomic file writes for checkpoints (temp file + rename)
#
# 4. RESOURCE MANAGEMENT:
#    - Add proper context managers for cleanup
#    - Ensure database connections are always closed
#
# 5. DOCUMENTATION:
#    - Add Google-style docstrings to ALL public methods
#    - Include Args, Returns, Raises, Examples sections
#    - Document WHY, not just WHAT
#
# ESTIMATED EFFORT:
# ----------------
# - Priority 1 (Security): ~4-6 hours
# - Priority 2 (Code Quality): ~8-12 hours
# - Priority 3 (Design): ~12-16 hours
# - Priority 4 (Performance): ~6-8 hours
# TOTAL: ~30-42 hours for complete refactoring
#
# RECOMMENDED APPROACH:
# --------------------
# 1. Fix Priority 1 security issues immediately (SQL injection!)
# 2. Add comprehensive unit tests before making other changes
# 3. Refactor in small, tested increments
# 4. Use linters: pylint, mypy, bandit for security
# 5. Consider code review with security expert for API key handling
#
# ==============================================================================
