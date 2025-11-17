#!/usr/bin/env python3
# TODO(code-review): Remove emojis from docstrings - not Google style compliant
# TODO(code-review): Translate all Russian text to English for international team
# TODO(code-review): Convert to Google-style docstring format with Args/Returns/Raises
"""Advanced Emotion Analyzer - Production Ready.

This module provides emotion analysis for rap lyrics using Hugging Face transformers.

IMPROVEMENTS V2.0:
- Async-first architecture with proper context management
- PostgreSQL integration through database abstraction
- Advanced emotional patterns for rap music
- Caching and batch optimization
- Comprehensive error handling and graceful degradation
- Memory management and resource cleanup
- Extended quality metrics

Usage:
    python src/analyzers/emotion_analyzer.py --test
    # Or via main.py interface with PostgreSQL backend

Requirements:
    - Python 3.8+
    - transformers >= 4.21.0
    - torch >= 1.12.0
    - PostgreSQL connection (via project config)

Author: Enhanced by Claude | Date: September 2025
"""

import sys
from pathlib import Path

# TODO(code-review): Avoid modifying sys.path at module level - use proper package structure
# TODO(code-review): This creates implicit dependencies and makes testing harder
# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import functools
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# TODO(code-review): Translate Russian comments to English
# TODO(code-review): Avoid module-level mutable globals - use proper configuration class
# TODO(code-review): Consider using TYPE_CHECKING for optional imports
# Conditional imports for graceful degradation
try:
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        pipeline,
    )

    HAS_TRANSFORMERS = True
except ImportError as e:
    HAS_TRANSFORMERS = False
    TRANSFORMERS_ERROR = str(e)

try:
    from interfaces.analyzer_interface import (
        AnalysisResult,
        BaseAnalyzer,
        register_analyzer,
    )

    HAS_INTERFACE = True
except ImportError:
    HAS_INTERFACE = False

    # TODO(code-review): Add type hints to fallback classes
    # TODO(code-review): Document why fallback is needed and when it's used
    # Fallback for independent testing
    class BaseAnalyzer:
        def __init__(self, config=None):
            self.config = config or {}
            self.name = "emotion_analyzer"

    def register_analyzer(name):
        def decorator(cls):
            return cls

        return decorator

    @dataclass
    class AnalysisResult:
        artist: str
        title: str
        analysis_type: str
        confidence: float
        metadata: dict[str, Any]
        raw_output: dict[str, Any]
        processing_time: float
        timestamp: str


# PostgreSQL integration
try:
    from src.database.postgres_adapter import DatabaseConfig, PostgreSQLManager

    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False
    PostgreSQLManager = None
    DatabaseConfig = None


# TODO(code-review): Add comprehensive docstring with field descriptions
# TODO(code-review): Consider using Pydantic for validation instead of dataclass
# TODO(code-review): Document valid ranges for scores (e.g., 0.0-1.0)
# Enhanced result with detailed metrics
@dataclass
class EmotionAnalysisResult:
    """Extended emotion analysis result.

    Attributes:
        analyzer_name: Name of the analyzer that produced this result.
        sentiment_score: Overall sentiment from 0.0 (negative) to 1.0 (positive).
        confidence: Confidence level of the analysis (0.0-1.0).
        dominant_emotion: The most prominent detected emotion.
        emotion_scores: Dictionary mapping emotion names to scores.
        genre_prediction: Predicted rap subgenre.
        intensity: Overall emotional intensity (0.0-1.0).
        analysis_time: Time taken for analysis in seconds.
        metadata: Additional metadata about the analysis.
        aggression_level: Rap-specific aggression metric (0.0-1.0).
        energy_level: Rap-specific energy metric (0.0-1.0).
        authenticity_score: Rap-specific authenticity metric (0.0-1.0).
        complexity_score: Rap-specific complexity metric (0.0-1.0).
    """

    analyzer_name: str
    sentiment_score: float  # 0.0-1.0 (negative to positive)
    confidence: float
    dominant_emotion: str
    emotion_scores: dict[str, float]
    genre_prediction: str
    intensity: float  # Overall emotional intensity
    analysis_time: float
    metadata: dict[str, Any] = field(default_factory=dict)

    # Rap-specific metrics
    aggression_level: float = 0.0
    energy_level: float = 0.0
    authenticity_score: float = 0.0
    complexity_score: float = 0.0


# TODO(code-review): This singleton pattern is NOT thread-safe for synchronous code
# TODO(code-review): _lock is created at module import, should be created in __init__
# TODO(code-review): Use proper locking mechanism or dependency injection instead
# TODO(code-review): Document memory implications of caching models
# Model cache to prevent repeated model loading
class ModelCache:
    """Thread-safe model caching singleton.

    WARNING: This implementation has thread-safety issues.
    The asyncio.Lock is created at class definition time.
    """

    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache = {}
            cls._instance._initialized = False
        return cls._instance

    async def get_model(self, model_name: str, device: str = "auto"):
        # TODO(code-review): Add Google-style docstring with Args/Returns/Raises
        # TODO(code-review): Don't catch bare Exception - specify expected exceptions
        # TODO(code-review): Return None on error is error-prone - raise exception instead
        # TODO(code-review): Add type hints for return value (Optional[Pipeline])
        """Get cached model or create new one.

        Args:
            model_name: Name of the Hugging Face model to load.
            device: Device to load model on ('auto', 'cpu', 'cuda').

        Returns:
            The loaded model pipeline or None on failure.
        """
        cache_key = f"{model_name}_{device}"

        async with self._lock:
            if cache_key not in self._cache:
                try:
                    self._cache[cache_key] = await self._create_model(
                        model_name, device
                    )
                except Exception as e:
                    logging.error(f"Failed to create model {model_name}: {e}")
                    return None

            return self._cache[cache_key]

    async def _create_model(self, model_name: str, device: str):
        # TODO(code-review): Add proper docstring with Args/Returns/Raises
        # TODO(code-review): Magic numbers: 0, -1 should be named constants
        # TODO(code-review): Add input validation for device parameter
        # TODO(code-review): This method is not actually async - remove async or make it truly async
        """Create new model instance.

        Args:
            model_name: Name of the Hugging Face model to load.
            device: Device to load model on ('auto', 'cpu', 'cuda').

        Returns:
            A text classification pipeline instance.
        """
        # Determine optimal device
        if device == "auto" or device == "cuda":
            device_id = 0 if torch.cuda.is_available() else -1
        else:
            device_id = -1

        # Create pipeline with optimization
        classifier = pipeline(
            "text-classification",
            model=model_name,
            device=device_id,
            top_k=None,  # return all scores - replaces return_all_scores
            model_kwargs={
                "torch_dtype": torch.float16
                if torch.cuda.is_available()
                else torch.float32,
                "low_cpu_mem_usage": True,
            },
        )

        return classifier


logger = logging.getLogger(__name__)


@register_analyzer("emotion_analyzer")
class EmotionAnalyzer(BaseAnalyzer):
    # TODO(code-review): Class is too large (1000+ lines) - violates Single Responsibility Principle
    # TODO(code-review): Split into: EmotionAnalyzer, RapMetricsCalculator, DatabaseService
    # TODO(code-review): Translate all Russian text to English in docstring
    # TODO(code-review): Add Examples section to docstring showing usage
    """Production-ready emotion analyzer for rap lyrics.

    NEW FEATURES V2.0:
    - Async-first with proper resource management
    - Rap-specific emotion patterns and metrics
    - Advanced caching and batch processing
    - PostgreSQL-ready with structured metadata
    - Enhanced error handling and monitoring
    - Memory-efficient model loading
    """

    # TODO(code-review): Class constants should be UPPER_CASE per PEP8
    # TODO(code-review): Consider moving these to a separate config module
    # Rap-specific emotion mappings
    RAP_EMOTION_PATTERNS = {
        "aggression": ["anger", "dominance", "power"],
        "authenticity": ["sadness", "pain", "struggle"],
        "celebration": ["joy", "pride", "success"],
        "romance": ["love", "desire", "attraction"],
        "reflection": ["contemplation", "wisdom", "growth"],
        "energy": ["excitement", "hype", "motivation"],
    }

    # TODO(code-review): Class constants should use UPPER_CASE naming
    # TODO(code-review): Consider loading keywords from external config file for easier updates
    # TODO(code-review): Add content warning comment for explicit language in data
    # Extended keywords for rap analysis
    RAP_KEYWORDS = {
        "aggression": {
            "high": ["fuck", "shit", "damn", "bitch", "kill", "murder", "war", "fight"],
            "medium": ["mad", "angry", "hate", "rage", "wild", "crazy", "beast"],
            "low": ["tough", "hard", "strong", "power", "boss", "king"],
        },
        "authenticity": {
            "high": ["struggle", "pain", "real", "truth", "hood", "street", "broke"],
            "medium": ["life", "story", "journey", "hustle", "grind", "survive"],
            "low": ["experience", "learn", "grow", "change", "move"],
        },
        "success": {
            "high": [
                "money",
                "cash",
                "rich",
                "millionaire",
                "gold",
                "diamond",
                "luxury",
            ],
            "medium": ["success", "win", "top", "best", "champion", "star"],
            "low": ["good", "great", "nice", "cool", "awesome", "amazing"],
        },
    }

    def __init__(self, config: dict[str, Any] | None = None):
        # TODO(code-review): Add comprehensive docstring with Args/Raises sections
        # TODO(code-review): Validate config values (e.g., batch_size > 0, max_length > 0)
        # TODO(code-review): Consider using Pydantic or dataclass for typed config
        """Initialize enhanced emotion analyzer.

        Args:
            config: Optional configuration dictionary with keys:
                - model_name: Hugging Face model name
                - device: Device to use ('auto', 'cpu', 'cuda')
                - max_length: Maximum text length for processing
                - batch_size: Batch size for processing
                - cache_enabled: Enable model caching
                - fallback_enabled: Enable keyword-based fallback
                - rap_analysis_enabled: Enable rap-specific analysis
                - postgres_enabled: Enable PostgreSQL integration
        """
        super().__init__(config)

        # Enhanced configuration
        self.model_name = self.config.get(
            "model_name", "j-hartmann/emotion-english-distilroberta-base"
        )
        self.device = self.config.get("device", "auto")
        self.max_length = self.config.get("max_length", 512)
        self.batch_size = self.config.get("batch_size", 16)
        self.cache_enabled = self.config.get("cache_enabled", True)
        self.fallback_enabled = self.config.get("fallback_enabled", True)
        self.rap_analysis_enabled = self.config.get("rap_analysis_enabled", True)

        # PostgreSQL integration
        self.postgres_enabled = self.config.get("postgres_enabled", True)
        self.db_manager = None

        # Performance settings
        self.model_cache = ModelCache() if self.cache_enabled else None
        self.session_stats = {
            "total_analyzed": 0,
            "total_time": 0.0,
            "cache_hits": 0,
            "fallback_uses": 0,
            "errors": 0,
        }

        # State management
        self._classifier = None
        self._is_available = False
        self._initialization_error = None

        # Initialize in background
        self._init_task = None

    async def initialize(self) -> bool:
        # TODO(code-review): Add proper docstring with Returns/Raises sections
        # TODO(code-review): Don't catch bare Exception - specify expected exceptions
        # TODO(code-review): Consider raising exception instead of returning False
        # TODO(code-review): Add timeout to prevent infinite waiting
        """Async initialization of the analyzer.

        Returns:
            True if initialization successful, False otherwise.

        Raises:
            Should raise specific exceptions instead of returning False.
        """
        if self._init_task is None:
            self._init_task = asyncio.create_task(self._async_initialize())

        try:
            await self._init_task
            return self._is_available
        except Exception as e:
            logger.error(f"Failed to initialize EmotionAnalyzer: {e}")
            self._initialization_error = str(e)
            return False

    async def _async_initialize(self):
        # TODO(code-review): Add proper docstring
        # TODO(code-review): Method is too long (60+ lines) - split into smaller methods
        # TODO(code-review): Too many nested if statements - refactor for readability
        # TODO(code-review): Line length exceeds 80 characters (Google style)
        """Async model initialization.

        Initializes PostgreSQL connection and transformer model.
        """
        # Initialize PostgreSQL connection first (independent of model loading)
        logger.info(
            f"PostgreSQL initialization check: enabled={self.postgres_enabled}, "
            f"has_postgres={HAS_POSTGRES}, "
            f"manager_class={PostgreSQLManager is not None}, "
            f"config_class={DatabaseConfig is not None}"
        )

        if (
            self.postgres_enabled
            and HAS_POSTGRES
            and PostgreSQLManager
            and DatabaseConfig
        ):
            try:
                logger.info("Attempting to initialize PostgreSQL connection...")
                config = DatabaseConfig.from_env()
                logger.info(
                    f"Database config: {config.host}:{config.port}/{config.database}"
                )
                self.db_manager = PostgreSQLManager(config)
                init_success = await self.db_manager.initialize()
                if init_success:
                    logger.info("✅ PostgreSQL connection established")
                else:
                    logger.warning("PostgreSQL connection failed")
                    self.db_manager = None
            except Exception as e:
                logger.error(f"PostgreSQL initialization failed: {e}")
                import traceback

                logger.error(f"Traceback: {traceback.format_exc()}")
                self.db_manager = None
        else:
            logger.warning("PostgreSQL not enabled or dependencies missing")

        # Initialize transformer model
        if not HAS_TRANSFORMERS:
            logger.warning(f"Transformers not available: {TRANSFORMERS_ERROR}")
            self._is_available = False
            return

        try:
            logger.info(f"Loading emotion model: {self.model_name}")

            if self.model_cache:
                self._classifier = await self.model_cache.get_model(
                    self.model_name, self.device
                )
            else:
                self._classifier = await self._create_standalone_model()

            if self._classifier:
                self._is_available = True
                logger.info(f"✅ Emotion analyzer ready on {self._get_device_info()}")

                # Warm-up test
                await self._warmup_model()
            else:
                raise Exception("Failed to create classifier")

        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            self._is_available = False
            self._initialization_error = str(e)

    async def _create_standalone_model(self):
        """Create model without caching"""
        device_id = 0 if torch.cuda.is_available() and self.device != "cpu" else -1

        return pipeline(
            "text-classification",
            model=self.model_name,
            device=device_id,
            top_k=None,  # return all scores - replaces return_all_scores
            model_kwargs={"low_cpu_mem_usage": True},
        )

    async def _warmup_model(self):
        """Warm up model with sample text"""
        try:
            sample_text = "This is a test for model warmup"
            await self._analyze_with_model(sample_text)
            logger.debug("Model warmup completed successfully")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    def _get_device_info(self) -> str:
        """Get current device information"""
        if torch.cuda.is_available():
            return f"GPU ({torch.cuda.get_device_name(0)})"
        return "CPU"

    async def analyze_song(
        self, artist: str, title: str, lyrics: str
    ) -> AnalysisResult:
        """
        Main interface for song analysis - PostgreSQL compatible

        Args:
            artist: Artist name
            title: Song title
            lyrics: Song lyrics text

        Returns:
            AnalysisResult compatible with PostgreSQL storage
        """
        # Ensure initialization
        if not await self.initialize():
            return self._create_error_result(
                artist, title, "Analyzer initialization failed"
            )

        # Perform analysis
        emotion_result = await self._analyze_emotion_enhanced(lyrics)

        # Convert to standardized format
        return self._convert_to_analysis_result(artist, title, emotion_result)

    async def _analyze_emotion_enhanced(self, text: str) -> EmotionAnalysisResult:
        """Enhanced emotion analysis with rap-specific features"""
        start_time = datetime.now()

        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)

            # Model-based analysis
            if self._is_available:
                emotion_scores = await self._analyze_with_model(processed_text)
                fallback_used = False
            else:
                emotion_scores = self._fallback_emotion_analysis(processed_text)
                fallback_used = True
                self.session_stats["fallback_uses"] += 1

            # Enhanced rap-specific analysis
            rap_metrics = (
                self._analyze_rap_patterns(text) if self.rap_analysis_enabled else {}
            )

            # Calculate derived metrics
            sentiment = self._calculate_enhanced_sentiment(emotion_scores, rap_metrics)
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            confidence = max(emotion_scores.values())
            intensity = self._calculate_emotional_intensity(emotion_scores)
            genre_prediction = self._predict_rap_subgenre(emotion_scores, rap_metrics)

            analysis_time = (datetime.now() - start_time).total_seconds()

            # Update session statistics
            self.session_stats["total_analyzed"] += 1
            self.session_stats["total_time"] += analysis_time

            result = EmotionAnalysisResult(
                analyzer_name=self.name,
                sentiment_score=sentiment,
                confidence=confidence,
                dominant_emotion=dominant_emotion,
                emotion_scores=emotion_scores,
                genre_prediction=genre_prediction,
                intensity=intensity,
                analysis_time=analysis_time,
                aggression_level=rap_metrics.get("aggression_level", 0.0),
                energy_level=rap_metrics.get("energy_level", 0.0),
                authenticity_score=rap_metrics.get("authenticity_score", 0.0),
                complexity_score=rap_metrics.get("complexity_score", 0.0),
                metadata={
                    "text_length": len(text),
                    "processed_length": len(processed_text),
                    "model_name": self.model_name,
                    "device": self._get_device_info(),
                    "fallback_used": fallback_used,
                    "rap_analysis": self.rap_analysis_enabled,
                    "session_id": id(self),
                    "timestamp": datetime.now().isoformat(),
                },
            )

            return result

        except Exception as e:
            logger.error(f"Enhanced emotion analysis failed: {e}")
            self.session_stats["errors"] += 1
            return self._create_error_emotion_result(text, start_time, str(e))

    def _preprocess_text(self, text: str) -> str:
        # TODO(code-review): Add comprehensive docstring with Args/Returns
        # TODO(code-review): Magic number 2.5 should be a named constant
        # TODO(code-review): Magic number 2 (target_length = max - 2) needs explanation
        # TODO(code-review): Don't catch bare Exception - specify expected exceptions
        # TODO(code-review): Duplicated truncation logic - extract to helper method
        # TODO(code-review): Consider adding unit tests for edge cases
        """Enhanced text preprocessing for rap lyrics with proper tokenization.

        Args:
            text: The input text to preprocess.

        Returns:
            The preprocessed and potentially truncated text.
        """
        if not text:
            return ""

        # Remove excessive whitespace and normalize
        text = " ".join(text.split())

        # Handle rap-specific patterns
        text = re.sub(r"\b(yeah|yo|uh|huh|ay)\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\[.*?\]", "", text)  # Remove annotations like [Verse 1]
        text = re.sub(r"\(.*?\)", "", text)  # Remove parenthetical notes

        # Properly truncate to model's max_length using tokenizer
        if hasattr(self._classifier, "tokenizer") and self._classifier.tokenizer:
            try:
                # Tokenize and check length
                tokens = self._classifier.tokenizer.encode(
                    text, add_special_tokens=True
                )

                # If too long, truncate to max_length with buffer for special tokens
                if len(tokens) > self.max_length:
                    # More aggressive truncation - leave room for special tokens
                    target_length = self.max_length - 2  # Room for [CLS] and [SEP]
                    truncated_tokens = tokens[:target_length]
                    text = self._classifier.tokenizer.decode(
                        truncated_tokens, skip_special_tokens=True
                    )
                    logger.debug(
                        f"Text truncated from {len(tokens)} to "
                        f"{len(truncated_tokens)} tokens"
                    )
            except Exception as e:
                logger.warning(f"Failed to use tokenizer for truncation: {e}")
                # More aggressive fallback truncation
                if len(text) > self.max_length * 2.5:  # Reduced multiplier
                    text = text[: int(self.max_length * 2.5)]
        # More aggressive fallback truncation if tokenizer not available
        elif len(text) > self.max_length * 2.5:
            text = text[: int(self.max_length * 2.5)]

        return text.strip()

    async def _analyze_with_model(self, text: str) -> dict[str, float]:
        """Model-based emotion analysis with proper error handling"""
        try:
            predictions = self._classifier(text)

            emotion_scores = {}
            for pred_list in predictions:  # Handle batch results
                for pred in pred_list:
                    emotion_name = pred["label"].lower()
                    score = float(pred["score"])
                    emotion_scores[emotion_name] = score

            return emotion_scores

        except Exception as e:
            logger.error(f"Model analysis failed: {e}")
            raise

    def _fallback_emotion_analysis(self, text: str) -> dict[str, float]:
        # TODO(code-review): Method is too long (100+ lines) - split into smaller methods
        # TODO(code-review): Hardcoded keyword dictionaries should be class constants
        # TODO(code-review): Add comprehensive docstring
        # TODO(code-review): Magic numbers in normalization (10.0) should be constants
        # TODO(code-review): Consider moving to separate FallbackAnalyzer class
        """Enhanced fallback analysis with rap-specific patterns.

        Args:
            text: The input text to analyze.

        Returns:
            Dictionary mapping emotion names to normalized scores (0.0-1.0).
        """
        text_lower = text.lower()
        emotion_scores = {}

        # Base emotion keywords (from original)
        emotion_keywords = {
            "joy": [
                "happy",
                "joy",
                "celebration",
                "party",
                "fun",
                "good",
                "great",
                "amazing",
            ],
            "anger": ["angry", "mad", "hate", "damn", "shit", "fuck", "rage", "pissed"],
            "fear": ["scared", "afraid", "fear", "worry", "anxious", "paranoid"],
            "sadness": ["sad", "cry", "tear", "pain", "hurt", "depression", "lonely"],
            "love": ["love", "heart", "baby", "girl", "woman", "kiss", "romance"],
            "surprise": ["wow", "amazing", "incredible", "unbelievable", "shocking"],
        }

        # Enhanced with rap-specific terms
        rap_emotion_keywords = {
            "aggression": ["beef", "diss", "enemy", "war", "battle", "destroy", "kill"],
            "confidence": [
                "boss",
                "king",
                "crown",
                "throne",
                "champion",
                "winner",
                "best",
            ],
            "struggle": [
                "hustle",
                "grind",
                "struggle",
                "poverty",
                "broke",
                "street",
                "hood",
            ],
            "success": [
                "money",
                "cash",
                "rich",
                "gold",
                "diamond",
                "mansion",
                "luxury",
            ],
            "loyalty": [
                "family",
                "crew",
                "team",
                "brothers",
                "loyalty",
                "trust",
                "real",
            ],
            "party": ["club", "dance", "party", "vibe", "turn up", "lit", "fire"],
        }

        # Combine emotion categories
        all_keywords = {**emotion_keywords, **rap_emotion_keywords}

        # Calculate scores with position weighting
        for emotion, keywords in all_keywords.items():
            total_score = 0
            word_positions = []

            for keyword in keywords:
                positions = [
                    m.start()
                    for m in re.finditer(r"\b" + re.escape(keyword) + r"\b", text_lower)
                ]
                word_positions.extend(positions)
                total_score += len(positions)

            # Weight by position (earlier = more important for hooks/intro)
            if word_positions:
                text_len = len(text_lower)
                position_weight = sum(
                    1 - (pos / text_len) for pos in word_positions
                ) / len(word_positions)
                total_score *= 1 + position_weight

            # Normalize score
            emotion_scores[emotion] = min(total_score / 10.0, 1.0)

        # Map rap emotions to standard emotions
        emotion_mapping = {
            "aggression": "anger",
            "confidence": "joy",
            "struggle": "sadness",
            "success": "joy",
            "loyalty": "love",
            "party": "joy",
        }

        # Merge mapped emotions
        final_scores = {}
        standard_emotions = ["joy", "anger", "fear", "sadness", "love", "surprise"]

        for emotion in standard_emotions:
            score = emotion_scores.get(emotion, 0.0)

            # Add mapped rap emotions
            for rap_emotion, standard_emotion in emotion_mapping.items():
                if standard_emotion == emotion:
                    score += emotion_scores.get(rap_emotion, 0.0)

            final_scores[emotion] = min(score, 1.0)

        return final_scores

    def _analyze_rap_patterns(self, text: str) -> dict[str, float]:
        # TODO(code-review): Add comprehensive docstring with Args/Returns
        # TODO(code-review): Magic numbers everywhere (3, 2, 1, 20.0, 15.0, 10.0) - use constants
        # TODO(code-review): Hardcoded weight dictionaries - extract to class constants
        # TODO(code-review): Use word boundaries in count() to avoid partial matches
        # TODO(code-review): Consider using Counter from collections for efficiency
        """Analyze rap-specific patterns and themes.

        Args:
            text: The input text to analyze.

        Returns:
            Dictionary with rap-specific metrics (all values 0.0-1.0):
                - aggression_level
                - authenticity_score
                - energy_level
                - complexity_score
        """
        text_lower = text.lower()

        metrics = {}

        # Aggression level
        aggression_score = 0
        for level, keywords in self.RAP_KEYWORDS["aggression"].items():
            weight = {"high": 3, "medium": 2, "low": 1}[level]
            for keyword in keywords:
                aggression_score += text_lower.count(keyword) * weight
        metrics["aggression_level"] = min(aggression_score / 20.0, 1.0)

        # Authenticity (street credibility)
        authenticity_score = 0
        for level, keywords in self.RAP_KEYWORDS["authenticity"].items():
            weight = {"high": 3, "medium": 2, "low": 1}[level]
            for keyword in keywords:
                authenticity_score += text_lower.count(keyword) * weight
        metrics["authenticity_score"] = min(authenticity_score / 15.0, 1.0)

        # Success themes
        success_score = 0
        for level, keywords in self.RAP_KEYWORDS["success"].items():
            weight = {"high": 3, "medium": 2, "low": 1}[level]
            for keyword in keywords:
                success_score += text_lower.count(keyword) * weight

        # Energy level (based on rhythm markers and exclamations)
        energy_indicators = [
            "!",
            "yeah",
            "yo",
            "ay",
            "uh",
            "turn up",
            "let's go",
            "fire",
            "lit",
        ]
        energy_score = sum(
            text_lower.count(indicator) for indicator in energy_indicators
        )
        metrics["energy_level"] = min(energy_score / 10.0, 1.0)

        # Complexity (vocabulary diversity, wordplay indicators)
        words = text_lower.split()
        unique_words = set(words)
        metrics["complexity_score"] = len(unique_words) / len(words) if words else 0.0

        # Wordplay indicators
        wordplay_indicators = ["like", "as", "than", "double", "triple", "word", "play"]
        wordplay_score = sum(
            text_lower.count(indicator) for indicator in wordplay_indicators
        )
        metrics["complexity_score"] = min(
            metrics["complexity_score"] + wordplay_score / 20.0, 1.0
        )

        return metrics

    def _calculate_enhanced_sentiment(
        self, emotion_scores: dict[str, float], rap_metrics: dict[str, float]
    ) -> float:
        # TODO(code-review): Add comprehensive docstring with Args/Returns
        # TODO(code-review): Magic numbers (0.3, 0.7, 0.2, 0.1, 0.5) should be named constants
        # TODO(code-review): Hardcoded emotion lists should be class constants
        # TODO(code-review): Add unit tests for different scenarios
        # TODO(code-review): Document the sentiment calculation algorithm
        """Enhanced sentiment calculation with rap context.

        Args:
            emotion_scores: Dictionary of emotion names to scores.
            rap_metrics: Dictionary of rap-specific metrics.

        Returns:
            Sentiment score from 0.0 (negative) to 1.0 (positive).
            Returns 0.5 if no emotions detected.
        """
        # Standard sentiment calculation
        positive_emotions = ["joy", "love", "surprise"]
        negative_emotions = ["anger", "fear", "sadness"]

        positive_score = sum(
            emotion_scores.get(emotion, 0.0) for emotion in positive_emotions
        )
        negative_score = sum(
            emotion_scores.get(emotion, 0.0) for emotion in negative_emotions
        )

        # Rap adjustments
        if rap_metrics:
            # Success themes are positive in rap context
            success_boost = rap_metrics.get("success_level", 0.0) * 0.3
            positive_score += success_boost

            # High aggression can be positive in rap (confidence/power)
            aggression = rap_metrics.get("aggression_level", 0.0)
            if aggression > 0.7:  # Very high aggression = confidence
                positive_score += aggression * 0.2
            elif aggression > 0.3:  # Medium aggression = slight negative
                negative_score += aggression * 0.1

        total_score = positive_score + negative_score
        if total_score > 0:
            return positive_score / total_score
        return 0.5

    def _calculate_emotional_intensity(self, emotion_scores: dict[str, float]) -> float:
        """Calculate overall emotional intensity"""
        return (
            sum(emotion_scores.values()) / len(emotion_scores)
            if emotion_scores
            else 0.0
        )

    def _predict_rap_subgenre(
        self, emotion_scores: dict[str, float], rap_metrics: dict[str, float]
    ) -> str:
        # TODO(code-review): Add comprehensive docstring with Args/Returns
        # TODO(code-review): All thresholds (0.7, 0.6, 0.5, etc.) should be named constants
        # TODO(code-review): Consider using a more maintainable approach (config-driven or ML)
        # TODO(code-review): Add type hints for genre strings (use Literal or Enum)
        # TODO(code-review): Document the classification logic and thresholds
        """Predict rap subgenre based on emotional patterns.

        Args:
            emotion_scores: Dictionary of emotion names to scores.
            rap_metrics: Dictionary of rap-specific metrics.

        Returns:
            Predicted rap subgenre name (one of: hardcore_rap, r&b_rap,
            conscious_rap, party_rap, gangsta_rap, mainstream_hip_hop,
            alternative_rap).
        """
        anger = emotion_scores.get("anger", 0.0)
        joy = emotion_scores.get("joy", 0.0)
        love = emotion_scores.get("love", 0.0)
        sadness = emotion_scores.get("sadness", 0.0)

        aggression = rap_metrics.get("aggression_level", 0.0)
        authenticity = rap_metrics.get("authenticity_score", 0.0)
        energy = rap_metrics.get("energy_level", 0.0)

        # Genre classification logic
        if aggression > 0.7 and anger > 0.6:
            return "hardcore_rap"
        if love > 0.5 or (joy > 0.4 and energy < 0.3):
            return "r&b_rap"
        if authenticity > 0.6 and sadness > 0.4:
            return "conscious_rap"
        if energy > 0.7 and joy > 0.4:
            return "party_rap"
        if anger > 0.4 and aggression > 0.4:
            return "gangsta_rap"
        if joy > 0.3 and energy > 0.3:
            return "mainstream_hip_hop"
        return "alternative_rap"

    def _convert_to_analysis_result(
        self, artist: str, title: str, emotion_result: EmotionAnalysisResult
    ) -> AnalysisResult:
        """Convert to standardized AnalysisResult format"""
        raw_output = {
            "emotion_scores": emotion_result.emotion_scores,
            "sentiment": emotion_result.sentiment_score,
            "confidence": emotion_result.confidence,
            "genre": emotion_result.genre_prediction,
            "analysis_time": emotion_result.analysis_time,
            "rap_specific": {
                "aggression": emotion_result.aggression_level,
                "energy": emotion_result.energy_level,
                "authenticity": emotion_result.authenticity_score,
                "complexity": emotion_result.complexity_score,
            },
        }
        base = {
            "artist": artist,
            "title": title,
            "confidence": float(emotion_result.confidence),
            "processing_time": float(emotion_result.analysis_time),
            "metadata": {
                "analyzer_version": "2.0.0",
                "processing_date": datetime.now().isoformat(),
                "dominant_emotion": emotion_result.dominant_emotion,
                "emotion_scores": emotion_result.emotion_scores,
                "sentiment_score": emotion_result.sentiment_score,
                "intensity": emotion_result.intensity,
                "genre_prediction": emotion_result.genre_prediction,
                "rap_metrics": {
                    "aggression_level": emotion_result.aggression_level,
                    "energy_level": emotion_result.energy_level,
                    "authenticity_score": emotion_result.authenticity_score,
                    "complexity_score": emotion_result.complexity_score,
                },
                "technical_info": emotion_result.metadata,
                "session_stats": self.session_stats.copy(),
            },
            "raw_output": raw_output,
            "timestamp": datetime.now().isoformat(),
        }

        return self._build_compatible_analysis_result(
            base, analyzer_type_value="emotional", analysis_data_value=raw_output
        )

    def _create_error_result(
        self, artist: str, title: str, error: str
    ) -> AnalysisResult:
        """Create error result in standard format"""
        base = {
            "artist": artist,
            "title": title,
            "confidence": 0.0,
            "processing_time": 0.0,
            "metadata": {
                "error": error,
                "analyzer_version": "2.0.0",
                "processing_date": datetime.now().isoformat(),
                "available": self._is_available,
                "initialization_error": self._initialization_error,
            },
            "raw_output": {"error": error},
            "timestamp": datetime.now().isoformat(),
        }

        return self._build_compatible_analysis_result(
            base, analyzer_type_value="emotional", analysis_data_value={"error": error}
        )

    def _build_compatible_analysis_result(
        self,
        base: dict[str, Any],
        analyzer_type_value: str,
        analysis_data_value: dict[str, Any],
    ) -> AnalysisResult:
        """Build AnalysisResult compatible with either current interface or fallback definition.

        Uses dataclass fields introspection to decide whether to pass 'analyzer_type'/'analysis_data' or 'analysis_type'/'raw_output'.
        """
        try:
            fields = set(getattr(AnalysisResult, "__dataclass_fields__", {}).keys())
        except Exception:
            fields = set()

        kwargs = base.copy()

        # prefer new names
        if "analyzer_type" in fields:
            kwargs["analyzer_type"] = analyzer_type_value
            # new schema uses analysis_data
            kwargs["analysis_data"] = analysis_data_value
        elif "analysis_type" in fields:
            # fallback legacy schema
            kwargs["analysis_type"] = analyzer_type_value
            kwargs["raw_output"] = analysis_data_value
        else:
            # best-effort: include both
            kwargs["analyzer_type"] = analyzer_type_value
            kwargs["analysis_data"] = analysis_data_value

        # Remove keys not accepted by the dataclass constructor to avoid TypeError
        try:
            accepted = set(getattr(AnalysisResult, "__dataclass_fields__", {}).keys())
            filtered = {k: v for k, v in kwargs.items() if k in accepted}
            return AnalysisResult(**filtered)
        except Exception:
            # Fallback: try to call with all kwargs and let Python raise if incompatible
            return AnalysisResult(**kwargs)

    def _create_error_emotion_result(
        self, text: str, start_time: datetime, error: str
    ) -> EmotionAnalysisResult:
        """Create error emotion result"""
        analysis_time = (datetime.now() - start_time).total_seconds()

        return EmotionAnalysisResult(
            analyzer_name=self.name,
            sentiment_score=0.5,
            confidence=0.0,
            dominant_emotion="unknown",
            emotion_scores={},
            genre_prediction="unknown",
            intensity=0.0,
            analysis_time=analysis_time,
            metadata={"error": error, "text_length": len(text), "fallback_used": True},
        )

    async def batch_analyze(
        self, lyrics_list: list[str], **kwargs
    ) -> list[EmotionAnalysisResult]:
        """Optimized batch analysis"""
        if not await self.initialize():
            return [
                self._create_error_emotion_result(
                    lyrics, datetime.now(), "Analyzer not available"
                )
                for lyrics in lyrics_list
            ]

        results = []
        batch_size = self.batch_size

        for i in range(0, len(lyrics_list), batch_size):
            batch = lyrics_list[i : i + batch_size]
            batch_tasks = [self._analyze_emotion_enhanced(lyrics) for lyrics in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    error_result = self._create_error_emotion_result(
                        "", datetime.now(), str(result)
                    )
                    results.append(error_result)
                else:
                    results.append(result)

        return results

    async def analyze_from_database(
        self, limit: int = 100, analyzer_type: str = "emotion_analyzer_v2"
    ) -> list[AnalysisResult]:
        """
        Analyze songs from PostgreSQL database that haven't been analyzed yet

        Args:
            limit: Maximum number of songs to analyze
            analyzer_type: Type of analysis to perform

        Returns:
            List of AnalysisResult objects
        """
        if not self.db_manager:
            logger.error("PostgreSQL manager not available")
            return []

        try:
            # Get tracks that need analysis
            tracks = await self.db_manager.get_tracks_for_analysis(limit, analyzer_type)

            if not tracks:
                logger.info("No tracks found that need analysis")
                return []

            logger.info(f"Found {len(tracks)} tracks to analyze")

            results = []
            for track in tracks:
                try:
                    # Analyze the song
                    result = await self.analyze_song(
                        track["artist"], track["title"], track["lyrics"]
                    )

                    # Store result in database
                    await self._save_analysis_to_database(track["id"], result)

                    results.append(result)

                except Exception as e:
                    logger.error(f"Failed to analyze track {track['id']}: {e}")
                    continue

            logger.info(f"Successfully analyzed {len(results)} tracks")
            return results

        except Exception as e:
            logger.error(f"Database analysis failed: {e}")
            return []

    async def _save_analysis_to_database(self, track_id: int, result: AnalysisResult):
        """Save analysis result to PostgreSQL database"""
        if not self.db_manager:
            return

        try:
            # Prepare analysis data
            analyzer_type = getattr(result, "analyzer_type", None) or getattr(
                result, "analysis_type", "emotion_analyzer_v2"
            )
            analysis_data = {
                "track_id": track_id,
                "analyzer_type": analyzer_type,
                "sentiment": result.metadata.get("sentiment_score", 0.0),
                "confidence": result.confidence,
                "complexity_score": result.metadata.get("rap_metrics", {}).get(
                    "complexity_score", 0.0
                ),
                "themes": result.metadata.get("dominant_emotion", "unknown"),
                "analysis_data": result.raw_output,
                "processing_time_ms": int(result.processing_time * 1000),
                "model_version": result.metadata.get("analyzer_version", "2.0.0"),
            }

            # Save to database
            result_id = await self.db_manager.save_analysis_result(analysis_data)

            if result_id:
                logger.debug(f"Saved analysis result for track {track_id}")
            else:
                logger.warning(f"Failed to save analysis result for track {track_id}")

        except Exception as e:
            logger.error(f"Error saving analysis to database: {e}")

    async def batch_analyze_from_database(
        self, batch_size: int = 50, max_batches: int = 10
    ) -> dict[str, Any]:
        # TODO(code-review): Add input validation (batch_size > 0, max_batches > 0)
        # TODO(code-review): Method is too long (90+ lines) - split into smaller methods
        # TODO(code-review): Uses print() instead of logger - inconsistent with rest of code
        # TODO(code-review): Magic strings in progress output - extract to constants
        # TODO(code-review): Add proper error recovery for partial batch failures
        """Perform batch analysis of songs from database.

        Args:
            batch_size: Number of songs per batch (must be > 0).
            max_batches: Maximum number of batches to process (must be > 0).

        Returns:
            Statistics dictionary with keys:
                - total_processed: Total number of tracks processed
                - total_analyzed: Successfully analyzed tracks
                - total_saved: Successfully saved results
                - batches_processed: Number of batches completed
                - errors: Number of errors encountered
                - start_time: Processing start time
                - end_time: Processing end time
                - duration: Total processing time in seconds
        """
        if not self.db_manager:
            logger.error("PostgreSQL manager not available")
            return {"error": "PostgreSQL not available"}

        stats = {
            "total_processed": 0,
            "total_analyzed": 0,
            "total_saved": 0,
            "batches_processed": 0,
            "errors": 0,
            "start_time": datetime.now(),
        }

        try:
            for batch_num in range(max_batches):
                logger.info(f"Processing batch {batch_num + 1}/{max_batches}")

                # Get batch of tracks
                tracks = await self.db_manager.get_tracks_for_analysis(
                    batch_size, "emotion_analyzer_v2"
                )

                if not tracks:
                    logger.info("No more tracks to analyze")
                    break

                stats["batches_processed"] += 1
                batch_results = []

                # Analyze batch with progress
                for i, track in enumerate(tracks, 1):
                    try:
                        # TODO(code-review): Remove emojis from production code
                        # TODO(code-review): Use logger instead of print()
                        # TODO(code-review): Extract progress formatting to separate method
                        # Progress indicator
                        progress = (
                            f"[Batch {batch_num + 1}/{max_batches}] {i}/{len(tracks)}"
                        )
                        print(
                            f"\r🎵 {progress} Analyzing: {track['artist'][:20]}... ",
                            end="",
                            flush=True,
                        )

                        result = await self.analyze_song(
                            track["artist"], track["title"], track["lyrics"]
                        )
                        batch_results.append((track["id"], result))
                        stats["total_analyzed"] += 1

                    except Exception as e:
                        logger.error(f"Failed to analyze track {track['id']}: {e}")
                        stats["errors"] += 1
                        continue

                # Save batch results
                saved_count = 0
                for track_id, result in batch_results:
                    try:
                        await self._save_analysis_to_database(track_id, result)
                        saved_count += 1
                    except Exception as e:
                        logger.error(f"Failed to save result for track {track_id}: {e}")

                stats["total_saved"] += saved_count
                stats["total_processed"] += len(tracks)

                # Clear progress line and show batch summary
                print(
                    f"\r✅ Batch {batch_num + 1}: {len(tracks)} processed, {saved_count} saved"
                    + " " * 20
                )

                # Small delay between batches
                await asyncio.sleep(0.1)

            stats["end_time"] = datetime.now()
            stats["duration"] = (
                stats["end_time"] - stats["start_time"]
            ).total_seconds()

            logger.info(f"Batch analysis completed: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            stats["error"] = str(e)
            return stats

    def is_available(self) -> bool:
        """Check if analyzer is ready"""
        return self._is_available

    def get_session_stats(self) -> dict[str, Any]:
        """Get current session statistics"""
        stats = self.session_stats.copy()
        if stats["total_analyzed"] > 0:
            stats["avg_analysis_time"] = stats["total_time"] / stats["total_analyzed"]
            stats["success_rate"] = 1 - (stats["errors"] / stats["total_analyzed"])
        return stats

    async def get_database_stats(self) -> dict[str, Any]:
        """Get analysis statistics from PostgreSQL database"""
        if not self.db_manager:
            return {"error": "PostgreSQL not available"}

        try:
            return await self.db_manager.get_table_stats()
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {"error": str(e)}

    async def get_analysis_summary(self, limit: int = 100) -> dict[str, Any]:
        """Get summary of recent analysis results"""
        if not self.db_manager:
            return {"error": "PostgreSQL not available"}

        try:
            # This would require a custom query to get analysis summary
            # For now, return basic stats
            db_stats = await self.get_database_stats()
            session_stats = self.get_session_stats()

            return {
                "database_stats": db_stats,
                "session_stats": session_stats,
                "analyzer_status": {
                    "available": self.is_available(),
                    "postgres_connected": self.db_manager is not None,
                    "model_loaded": self._classifier is not None,
                },
            }
        except Exception as e:
            logger.error(f"Error getting analysis summary: {e}")
            return {"error": str(e)}

    async def cleanup(self):
        """Cleanup resources"""
        if self._classifier and hasattr(self._classifier, "model"):
            # Clean up model memory
            if hasattr(self._classifier.model, "cpu"):
                self._classifier.model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Close PostgreSQL connection
        if self.db_manager:
            try:
                await self.db_manager.close()
                logger.info("PostgreSQL connection closed")
            except Exception as e:
                logger.warning(f"Error closing PostgreSQL connection: {e}")

        # Clear references
        self._classifier = None
        self._is_available = False
        self.db_manager = None

        logger.info("EmotionAnalyzer cleanup completed")

    def get_analyzer_info(self) -> dict[str, Any]:
        """Get comprehensive analyzer information"""
        return {
            "name": "EmotionAnalyzer",
            "version": "2.0.0",
            "description": "Advanced emotion detection with rap-specific analysis using Hugging Face transformers",
            "author": "Rap Scraper Project - Enhanced",
            "type": self.analyzer_type,
            "supported_features": self.supported_features,
            "model_info": {
                "name": self.model_name,
                "available": self.is_available(),
                "device": self._get_device_info() if self._is_available else "N/A",
                "initialization_error": self._initialization_error,
            },
            "capabilities": {
                "standard_emotions": [
                    "joy",
                    "anger",
                    "fear",
                    "sadness",
                    "surprise",
                    "love",
                ],
                "rap_specific_metrics": [
                    "aggression_level",
                    "energy_level",
                    "authenticity_score",
                    "complexity_score",
                ],
                "genre_prediction": [
                    "hardcore_rap",
                    "r&b_rap",
                    "conscious_rap",
                    "party_rap",
                    "gangsta_rap",
                    "mainstream_hip_hop",
                    "alternative_rap",
                ],
                "batch_processing": True,
                "async_support": True,
                "fallback_mode": self.fallback_enabled,
                "caching": self.cache_enabled,
            },
            "config_options": {
                "model_name": "Hugging Face model for emotion classification",
                "device": "Computation device (auto, cpu, cuda)",
                "max_length": "Maximum text length for processing",
                "batch_size": "Batch size for processing multiple texts",
                "cache_enabled": "Enable model caching for performance",
                "fallback_enabled": "Enable keyword-based fallback analysis",
                "rap_analysis_enabled": "Enable rap-specific pattern analysis",
            },
            "session_stats": self.get_session_stats(),
            "requirements": [
                "transformers >= 4.21.0",
                "torch >= 1.12.0",
                "numpy >= 1.21.0",
            ],
        }

    @property
    def analyzer_type(self) -> str:
        """Return analyzer type classification"""
        return "ai_enhanced"

    @property
    def supported_features(self) -> list[str]:
        """Return comprehensive feature list"""
        return [
            "emotion_detection",
            "sentiment_analysis",
            "rap_genre_classification",
            "confidence_scoring",
            "batch_processing",
            "async_processing",
            "fallback_mode",
            "rap_specific_metrics",
            "emotional_intensity",
            "authenticity_scoring",
            "aggression_analysis",
            "energy_detection",
            "complexity_analysis",
            "model_caching",
            "session_statistics",
            "memory_management",
        ]


# TODO(code-review): Decorator is defined but never used - remove or use it
# TODO(code-review): Translate Russian to English
# TODO(code-review): Add comprehensive docstring with Args/Returns/Examples
# TODO(code-review): Don't catch bare Exception - be more specific
# TODO(code-review): Consider using contextlib.contextmanager or existing profiling tools
# Performance monitoring decorator
def monitor_performance(func):
    """Decorator for performance monitoring.

    Args:
        func: The async function to monitor.

    Returns:
        Wrapped function that logs execution time and success status.
    """

    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        start_time = datetime.now()
        try:
            result = await func(self, *args, **kwargs)
            success = True
        except Exception as e:
            logger.error(f"Function {func.__name__} failed: {e}")
            success = False
            raise
        finally:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.debug(
                f"{func.__name__} executed in {execution_time:.3f}s, "
                f"success: {success}"
            )

        return result

    return wrapper


# TODO(code-review): Test functions should be in separate test file, not in main module
# TODO(code-review): Remove emojis from all print statements
# TODO(code-review): Use proper testing framework (pytest) instead of manual tests
# TODO(code-review): Translate all Russian text to English
# Utility functions for testing and integration
async def test_analyzer_comprehensive():
    # TODO(code-review): Add comprehensive docstring
    # TODO(code-review): Use logger instead of print()
    # TODO(code-review): Add assertions to validate results
    # TODO(code-review): Function is too long (100+ lines) - split into smaller tests
    """Comprehensive analyzer testing.

    Tests the EmotionAnalyzer with various rap lyrics samples representing
    different genres and emotional patterns.
    """
    print("🎯 Testing Enhanced Emotion Analyzer V2.0")
    print("=" * 60)

    # Configuration for testing
    config = {
        "model_name": "j-hartmann/emotion-english-distilroberta-base",
        "device": "auto",
        "max_length": 512,
        "batch_size": 4,
        "cache_enabled": True,
        "fallback_enabled": True,
        "rap_analysis_enabled": True,
    }

    analyzer = EmotionAnalyzer(config)

    # Test initialization
    print("📋 Initializing analyzer...")
    init_success = await analyzer.initialize()
    print(f"   ✅ Initialization: {'Success' if init_success else 'Failed'}")

    if not init_success and analyzer._initialization_error:
        print(f"   ❌ Error: {analyzer._initialization_error}")

    # Test texts with different emotional patterns
    test_cases = [
        {
            "name": "Aggressive Rap",
            "artist": "Test Artist",
            "title": "Hard Track",
            "lyrics": "I'm the king of this game, fuck all my enemies, I'll destroy anyone who tries to step to me, this is war!",
        },
        {
            "name": "Conscious Rap",
            "artist": "Deep Artist",
            "title": "Real Talk",
            "lyrics": "Growing up in the hood, seen too much pain and struggle, but I keep grinding, trying to make it out, stay real to my family",
        },
        {
            "name": "Party Rap",
            "artist": "Hype Artist",
            "title": "Turn Up",
            "lyrics": "Party all night, turn up the music, dancing with my crew, feeling good, living life to the fullest, yeah!",
        },
        {
            "name": "Love Rap",
            "artist": "Romantic Artist",
            "title": "My Queen",
            "lyrics": "Baby you're my everything, my heart belongs to you, I love you more than words can say, you're my queen",
        },
        {
            "name": "Success Rap",
            "artist": "Rich Artist",
            "title": "Made It",
            "lyrics": "Started from the bottom, now I'm rich, money in the bank, diamonds on my wrist, living luxury life",
        },
    ]

    print(f"\n📊 Running {len(test_cases)} test cases...")
    print("-" * 60)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   Text: {test_case['lyrics'][:50]}...")

        try:
            result = await analyzer.analyze_song(
                test_case["artist"], test_case["title"], test_case["lyrics"]
            )

            print("   ✅ Analysis completed")
            print(f"   📈 Sentiment: {result.metadata.get('sentiment_score', 0):.3f}")
            print(f"   🎯 Confidence: {result.confidence:.3f}")
            print(
                f"   🎭 Dominant: {result.metadata.get('dominant_emotion', 'unknown')}"
            )
            print(f"   🎵 Genre: {result.metadata.get('genre_prediction', 'unknown')}")

            # Rap-specific metrics
            rap_metrics = result.metadata.get("rap_metrics", {})
            if rap_metrics:
                print(f"   🔥 Aggression: {rap_metrics.get('aggression_level', 0):.3f}")
                print(f"   ⚡ Energy: {rap_metrics.get('energy_level', 0):.3f}")
                print(
                    f"   💯 Authenticity: {rap_metrics.get('authenticity_score', 0):.3f}"
                )
                print(f"   🧠 Complexity: {rap_metrics.get('complexity_score', 0):.3f}")

            print(f"   ⏱️  Time: {result.processing_time:.3f}s")

        except Exception as e:
            print(f"   ❌ Analysis failed: {e}")

    # Session statistics
    print("\n📊 Session Statistics:")
    stats = analyzer.get_session_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Analyzer info
    print("\n🔍 Analyzer Information:")
    info = analyzer.get_analyzer_info()
    print(f"   Version: {info['version']}")
    print(f"   Available: {info['model_info']['available']}")
    print(f"   Device: {info['model_info']['device']}")
    print(f"   Features: {len(info['supported_features'])}")

    # Cleanup
    await analyzer.cleanup()
    print("\n✅ Testing completed successfully!")


# PostgreSQL integration test
async def test_postgresql_integration():
    """Test integration with PostgreSQL database"""
    print("\n🗄️ Testing PostgreSQL Integration")
    print("-" * 40)

    try:
        # Try importing PostgreSQL components
        from src.database.postgres_adapter import PostgreSQLManager

        print("   ✅ PostgreSQL adapter available")

        # Test database connection (if configured)
        try:
            db_manager = PostgreSQLManager()
            await db_manager.initialize()
            print("   ✅ PostgreSQL connection successful")

            # Test storing analysis result
            analyzer = EmotionAnalyzer()
            if await analyzer.initialize():
                result = await analyzer.analyze_song(
                    "Test Artist",
                    "Test Song",
                    "This is a test song with happy vibes and good energy",
                )

                # Here you would store the result to PostgreSQL
                # await db_manager.store_analysis_result(result)
                print("   ✅ Analysis result ready for PostgreSQL storage")

                await analyzer.cleanup()

            await db_manager.close()

        except Exception as e:
            print(f"   ⚠️  PostgreSQL connection failed: {e}")
            print("   💡 Make sure PostgreSQL is configured in .env")

    except ImportError:
        print("   ⚠️  PostgreSQL adapter not available")
        print("   💡 This analyzer can work without PostgreSQL")


# Batch processing test
async def test_batch_processing():
    """Test batch processing capabilities"""
    print("\n⚡ Testing Batch Processing")
    print("-" * 30)

    analyzer = EmotionAnalyzer({"batch_size": 3, "rap_analysis_enabled": True})

    if not await analyzer.initialize():
        print("   ❌ Analyzer initialization failed")
        return

    # Sample lyrics for batch testing
    lyrics_batch = [
        "I'm feeling happy and joyful today, life is beautiful",
        "I'm so angry and mad, this situation pisses me off",
        "Baby I love you so much, you mean everything to me",
        "I'm scared and afraid of what might happen next",
        "Money, success, I made it to the top, living luxury",
    ]

    print(f"   Processing {len(lyrics_batch)} texts in batch...")

    start_time = datetime.now()
    results = await analyzer.batch_analyze(lyrics_batch)
    batch_time = (datetime.now() - start_time).total_seconds()

    print(f"   ✅ Batch completed in {batch_time:.3f}s")
    print(f"   📊 Average per text: {batch_time / len(lyrics_batch):.3f}s")

    for i, result in enumerate(results, 1):
        if hasattr(result, "dominant_emotion"):
            print(f"   {i}. {result.dominant_emotion} ({result.confidence:.3f})")

    await analyzer.cleanup()


# TODO(code-review): Interactive menu should be in separate CLI module
# TODO(code-review): Remove all emojis from UI text
# TODO(code-review): Translate all Russian text to English
# TODO(code-review): Use proper CLI framework (click, argparse) instead of raw input()
# TODO(code-review): Add input validation and error handling
# Interactive menu functions
async def create_interactive_menu():
    # TODO(code-review): Add comprehensive docstring
    # TODO(code-review): Function is too long (100+ lines) - split into smaller functions
    # TODO(code-review): Use enum for menu options instead of string comparison
    """Create interactive menu for analysis mode selection.

    Presents a menu-driven interface for different analyzer operations
    including testing, database analysis, and custom text analysis.
    """
    print("\n" + "=" * 70)
    print("🎯 ENHANCED EMOTION ANALYZER V2.0 - ИНТЕРАКТИВНОЕ МЕНЮ")
    print("=" * 70)
    print("Выберите режим работы:")
    print()
    print("1. 🧪 Тест анализатора (5 тестовых треков)")
    print("2. 🎵 Анализ одного трека из базы данных")
    print("3. 🗄️ Массовый анализ всей базы данных")
    print("4. 📊 Показать статистику базы данных")
    print("5. ⚡ Batch-анализ (50 треков)")
    print("6. 📝 Анализ произвольного текста")
    print("7. 🔧 Тест PostgreSQL подключения")
    print("0. ❌ Выход")
    print("-" * 70)

    while True:
        try:
            choice = input("👉 Введите номер опции (0-7): ").strip()

            if choice == "0":
                print("👋 До свидания!")
                return
            if choice == "1":
                await test_analyzer_comprehensive()
                await pause_for_user()
            elif choice == "2":
                await analyze_single_track_interactive()
                await pause_for_user()
            elif choice == "3":
                await analyze_all_database_interactive()
                await pause_for_user()
            elif choice == "4":
                await show_database_stats()
                await pause_for_user()
            elif choice == "5":
                await batch_analyze_interactive()
                await pause_for_user()
            elif choice == "6":
                await analyze_custom_text_interactive()
                await pause_for_user()
            elif choice == "7":
                await test_postgresql_integration()
                await pause_for_user()
            else:
                print("❌ Некорректный выбор. Попробуйте снова.")
                continue

        except KeyboardInterrupt:
            print("\n👋 Анализ прерван пользователем")
            return
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            await pause_for_user()


async def pause_for_user():
    """Пауза для просмотра результатов"""
    print("\n" + "-" * 50)
    input("📄 Нажмите Enter для продолжения...")
    print()


async def analyze_single_track_interactive():
    """Интерактивный анализ одного трека"""
    print("\n🎵 АНАЛИЗ ОДНОГО ТРЕКА")
    print("-" * 40)

    # Инициализация анализатора
    analyzer = EmotionAnalyzer({"postgres_enabled": True})
    if not await analyzer.initialize():
        print("❌ Не удалось инициализировать анализатор")
        return

    if not analyzer.db_manager:
        print("❌ PostgreSQL недоступен")
        await analyzer.cleanup()
        return

    try:
        print("Выберите способ поиска трека:")
        print("1. По ID трека")
        print("2. Поиск по исполнителю")
        print("3. Поиск по названию")
        print("4. Случайный трек")

        search_type = input("👉 Введите номер (1-4): ").strip()

        track = None

        if search_type == "1":
            track_id = input("Введите ID трека: ").strip()
            try:
                track_id = int(track_id)
                track = await get_track_by_id(analyzer.db_manager, track_id)
            except ValueError:
                print("❌ ID должен быть числом")
                return

        elif search_type == "2":
            artist_name = input("Введите имя исполнителя (частично): ").strip()
            tracks = await search_tracks_by_artist(analyzer.db_manager, artist_name)
            track = await select_from_tracks_list(tracks, "исполнителю")

        elif search_type == "3":
            song_title = input("Введите название песни (частично): ").strip()
            tracks = await search_tracks_by_title(analyzer.db_manager, song_title)
            track = await select_from_tracks_list(tracks, "названию")

        elif search_type == "4":
            track = await get_random_track(analyzer.db_manager)

        else:
            print("❌ Некорректный выбор")
            return

        if not track:
            print("❌ Трек не найден")
            return

        # Анализ трека
        print(f"\n📀 Анализируется: {track['artist']} - {track['title']}")
        print("-" * 60)

        result = await analyzer.analyze_song(
            track["artist"], track["title"], track["lyrics"] or "No lyrics available"
        )

        # Отображение результатов
        await display_analysis_result(result, track)

        # Сохранение в базу данных
        if track.get("id"):
            await analyzer._save_analysis_to_database(track["id"], result)
            print("✅ Результат сохранен в базу данных")

    except Exception as e:
        print(f"❌ Ошибка анализа: {e}")

    finally:
        await analyzer.cleanup()


async def analyze_all_database_interactive():
    """Интерактивный массовый анализ базы данных"""
    print("\n🗄️ МАССОВЫЙ АНАЛИЗ БАЗЫ ДАННЫХ")
    print("-" * 50)

    analyzer = EmotionAnalyzer({"postgres_enabled": True})
    if not await analyzer.initialize():
        print("❌ Не удалось инициализировать анализатор")
        return

    if not analyzer.db_manager:
        print("❌ PostgreSQL недоступен")
        await analyzer.cleanup()
        return

    try:
        # Получить статистику
        stats = await get_unanalyzed_tracks_count(analyzer.db_manager)
        total_tracks = stats.get("total_tracks", 0)
        unanalyzed = stats.get("unanalyzed_tracks", 0)

        print(f"📊 Всего треков в базе: {total_tracks}")
        print(f"🎯 Не проанализированных: {unanalyzed}")

        if unanalyzed == 0:
            print("✅ Все треки уже проанализированы!")
            return

        print(f"\n⚠️ Будет проанализировано {unanalyzed} треков")
        print("Это может занять продолжительное время...")

        confirm = input("Продолжить? (y/N): ").strip().lower()
        if confirm not in ["y", "yes", "да", "д"]:
            print("❌ Анализ отменен")
            return

        # Запуск массового анализа с progress bar
        print("\n🚀 Начинаем массовый анализ...")
        stats = await analyzer.batch_analyze_from_database(
            batch_size=20,  # Уменьшенный batch size для более частых обновлений
            max_batches=max(1, unanalyzed // 20 + 1),
        )

        # Отображение результатов
        print("\n📊 РЕЗУЛЬТАТЫ МАССОВОГО АНАЛИЗА:")
        print(f"   Всего обработано: {stats.get('total_processed', 0)}")
        print(f"   Успешно проанализировано: {stats.get('total_analyzed', 0)}")
        print(f"   Сохранено в БД: {stats.get('total_saved', 0)}")
        print(f"   Ошибок: {stats.get('errors', 0)}")
        print(f"   Время выполнения: {stats.get('duration', 0):.1f} сек")

        if stats.get("total_analyzed", 0) > 0:
            avg_time = stats.get("duration", 0) / stats.get("total_analyzed", 1)
            print(f"   Среднее время на трек: {avg_time:.3f} сек")

    except Exception as e:
        print(f"❌ Ошибка массового анализа: {e}")

    finally:
        await analyzer.cleanup()


async def batch_analyze_interactive():
    """Интерактивный batch-анализ (ограниченное количество)"""
    print("\n⚡ BATCH-АНАЛИЗ (ОГРАНИЧЕННЫЙ)")
    print("-" * 40)

    try:
        limit = input("Количество треков для анализа (по умолчанию 50): ").strip()
        limit = int(limit) if limit.isdigit() else 50

        analyzer = EmotionAnalyzer({"postgres_enabled": True})
        if not await analyzer.initialize():
            print("❌ Не удалось инициализировать анализатор")
            return

        if not analyzer.db_manager:
            print("❌ PostgreSQL недоступен")
            await analyzer.cleanup()
            return

        print(f"🎯 Анализируем {limit} треков...")
        results = await analyzer.analyze_from_database(limit=limit)

        print(f"✅ Проанализировано {len(results)} треков")

        # Показать несколько примеров результатов
        if results:
            print("\n📝 Примеры результатов:")
            for i, result in enumerate(results[:3], 1):
                print(f"   {i}. {result.artist} - {result.title}")
                print(
                    f"      Эмоция: {result.metadata.get('dominant_emotion', 'unknown')}"
                )
                print(f"      Уверенность: {result.confidence:.3f}")

        await analyzer.cleanup()

    except ValueError:
        print("❌ Некорректное количество")
    except Exception as e:
        print(f"❌ Ошибка batch-анализа: {e}")


async def analyze_custom_text_interactive():
    """Интерактивный анализ произвольного текста"""
    print("\n📝 АНАЛИЗ ПРОИЗВОЛЬНОГО ТЕКСТА")
    print("-" * 40)

    analyzer = EmotionAnalyzer()
    if not await analyzer.initialize():
        print("❌ Не удалось инициализировать анализатор")
        return

    try:
        print(
            "Введите текст для анализа (можно многострочный, закончите пустой строкой):"
        )
        lines = []
        while True:
            line = input()
            if not line.strip():
                break
            lines.append(line)

        text = "\n".join(lines)
        if not text.strip():
            print("❌ Текст не введен")
            return

        print(f"\n🔍 Анализируем текст ({len(text)} символов)...")

        result = await analyzer.analyze_song("Custom", "User Input", text)

        # Отображение результатов
        print("\n📊 РЕЗУЛЬТАТЫ АНАЛИЗА:")
        print(
            f"   📈 Настроение: {result.metadata.get('sentiment_score', 0):.3f} (0=негативное, 1=позитивное)"
        )
        print(f"   🎯 Уверенность: {result.confidence:.3f}")
        print(
            f"   🎭 Доминирующая эмоция: {result.metadata.get('dominant_emotion', 'unknown')}"
        )
        print(
            f"   🎵 Предполагаемый жанр: {result.metadata.get('genre_prediction', 'unknown')}"
        )
        print(f"   ⚡ Интенсивность: {result.metadata.get('intensity', 0):.3f}")

        # Rap-specific метрики
        rap_metrics = result.metadata.get("rap_metrics", {})
        if rap_metrics:
            print("\n🎤 RAP-СПЕЦИФИЧНЫЕ МЕТРИКИ:")
            print(f"   🔥 Агрессивность: {rap_metrics.get('aggression_level', 0):.3f}")
            print(f"   ⚡ Энергия: {rap_metrics.get('energy_level', 0):.3f}")
            print(
                f"   💯 Аутентичность: {rap_metrics.get('authenticity_score', 0):.3f}"
            )
            print(f"   🧠 Сложность: {rap_metrics.get('complexity_score', 0):.3f}")

        # Детальные эмоции
        emotion_scores = result.metadata.get("emotion_scores", {})
        if emotion_scores:
            print("\n🎭 ДЕТАЛЬНЫЙ АНАЛИЗ ЭМОЦИЙ:")
            for emotion, score in sorted(
                emotion_scores.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"   {emotion.capitalize()}: {score:.3f}")

        print(f"\n⏱️ Время анализа: {result.processing_time:.3f} сек")

    except Exception as e:
        print(f"❌ Ошибка анализа текста: {e}")

    finally:
        await analyzer.cleanup()


async def show_database_stats():
    """Показать статистику базы данных"""
    print("\n📊 СТАТИСТИКА БАЗЫ ДАННЫХ")
    print("-" * 40)

    analyzer = EmotionAnalyzer({"postgres_enabled": True})
    if not await analyzer.initialize():
        print("❌ Не удалось инициализировать анализатор")
        return

    if not analyzer.db_manager:
        print("❌ PostgreSQL недоступен")
        await analyzer.cleanup()
        return

    try:
        # Основная статистика
        stats = await analyzer.get_database_stats()
        summary = await analyzer.get_analysis_summary()

        print("🗄️ ОСНОВНАЯ СТАТИСТИКА:")
        if "error" not in stats:
            print(f"   Всего треков: {stats.get('total_tracks', 'N/A')}")
            print(f"   Треков с текстами: {stats.get('tracks_with_lyrics', 'N/A')}")
            print(f"   Всего анализов: {stats.get('total_analyses', 'N/A')}")

        # Статистика по анализаторам
        analyzer_stats = await get_analyzer_stats(analyzer.db_manager)
        if analyzer_stats:
            print("\n🤖 СТАТИСТИКА ПО АНАЛИЗАТОРАМ:")
            for analyzer_type, count in analyzer_stats.items():
                print(f"   {analyzer_type}: {count} анализов")

        # Статистика неанализированных треков
        unanalyzed_stats = await get_unanalyzed_tracks_count(analyzer.db_manager)
        print("\n🎯 ПОТРЕБНОСТЬ В АНАЛИЗЕ:")
        print(f"   Всего треков: {unanalyzed_stats.get('total_tracks', 0)}")
        print(
            f"   Проанализировано emotion_analyzer_v2: {unanalyzed_stats.get('analyzed_tracks', 0)}"
        )
        print(f"   Требует анализа: {unanalyzed_stats.get('unanalyzed_tracks', 0)}")

        # Статистика сессии
        session_stats = analyzer.get_session_stats()
        print("\n📈 СТАТИСТИКА ТЕКУЩЕЙ СЕССИИ:")
        for key, value in session_stats.items():
            print(f"   {key}: {value}")

    except Exception as e:
        print(f"❌ Ошибка получения статистики: {e}")

    finally:
        await analyzer.cleanup()


# Utility functions для работы с базой данных
async def get_track_by_id(db_manager, track_id: int):
    # TODO(code-review): Add type hints for db_manager parameter
    # TODO(code-review): Add type hint for return value (Optional[dict])
    # TODO(code-review): Translate Russian to English
    # TODO(code-review): Don't catch bare Exception - be specific
    # TODO(code-review): Use logger instead of print()
    # TODO(code-review): Consider raising exception instead of returning None
    """Get track by ID from database.

    Args:
        db_manager: Database manager instance.
        track_id: The track ID to retrieve.

    Returns:
        Track dictionary with keys: id, artist, title, lyrics.
        Returns None if track not found or on error.
    """
    try:
        query = "SELECT id, artist, title, lyrics FROM tracks WHERE id = %s"
        result = await db_manager.execute_query(query, (track_id,))
        if result:
            return dict(result[0])
        return None
    except Exception as e:
        print(f"Error retrieving track: {e}")
        return None


async def search_tracks_by_artist(db_manager, artist_name: str, limit: int = 10):
    """Поиск треков по исполнителю"""
    try:
        query = """
        SELECT id, artist, title, lyrics 
        FROM tracks 
        WHERE artist ILIKE %s AND lyrics IS NOT NULL
        LIMIT %s
        """
        result = await db_manager.execute_query(query, (f"%{artist_name}%", limit))
        return [dict(row) for row in result] if result else []
    except Exception as e:
        print(f"Ошибка поиска по исполнителю: {e}")
        return []


async def search_tracks_by_title(db_manager, title: str, limit: int = 10):
    """Поиск треков по названию"""
    try:
        query = """
        SELECT id, artist, title, lyrics 
        FROM tracks 
        WHERE title ILIKE %s AND lyrics IS NOT NULL
        LIMIT %s
        """
        result = await db_manager.execute_query(query, (f"%{title}%", limit))
        return [dict(row) for row in result] if result else []
    except Exception as e:
        print(f"Ошибка поиска по названию: {e}")
        return []


async def get_random_track(db_manager):
    """Получить случайный трек"""
    try:
        query = """
        SELECT id, artist, title, lyrics 
        FROM tracks 
        WHERE lyrics IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 1
        """
        result = await db_manager.execute_query(query)
        if result:
            return dict(result[0])
        return None
    except Exception as e:
        print(f"Ошибка получения случайного трека: {e}")
        return None


async def select_from_tracks_list(tracks, search_type: str):
    """Выбрать трек из списка найденных"""
    if not tracks:
        print(f"❌ Треки по {search_type} не найдены")
        return None

    if len(tracks) == 1:
        return tracks[0]

    print(f"\n📋 Найдено {len(tracks)} треков по {search_type}:")
    for i, track in enumerate(tracks, 1):
        print(f"   {i}. {track['artist']} - {track['title']}")

    while True:
        try:
            choice = input(f"Выберите трек (1-{len(tracks)}): ").strip()
            index = int(choice) - 1
            if 0 <= index < len(tracks):
                return tracks[index]
            print("❌ Некорректный номер")
        except ValueError:
            print("❌ Введите число")


async def display_analysis_result(result, track):
    """Отобразить результаты анализа"""
    print("\n📊 РЕЗУЛЬТАТЫ АНАЛИЗА:")
    print(f"   🎵 Трек: {track['artist']} - {track['title']}")
    print(f"   📈 Настроение: {result.metadata.get('sentiment_score', 0):.3f}")
    print(f"   🎯 Уверенность: {result.confidence:.3f}")
    print(
        f"   🎭 Доминирующая эмоция: {result.metadata.get('dominant_emotion', 'unknown')}"
    )
    print(f"   🎵 Жанр: {result.metadata.get('genre_prediction', 'unknown')}")
    print(f"   ⚡ Интенсивность: {result.metadata.get('intensity', 0):.3f}")

    # Rap-specific метрики
    rap_metrics = result.metadata.get("rap_metrics", {})
    if rap_metrics:
        print(f"   🔥 Агрессивность: {rap_metrics.get('aggression_level', 0):.3f}")
        print(f"   ⚡ Энергия: {rap_metrics.get('energy_level', 0):.3f}")
        print(f"   💯 Аутентичность: {rap_metrics.get('authenticity_score', 0):.3f}")
        print(f"   🧠 Сложность: {rap_metrics.get('complexity_score', 0):.3f}")

    print(f"   ⏱️ Время анализа: {result.processing_time:.3f} сек")


async def get_unanalyzed_tracks_count(db_manager):
    # TODO(code-review): Add type hints for all parameters and return value
    # TODO(code-review): Add comprehensive docstring
    # TODO(code-review): Translate Russian comments to English
    # TODO(code-review): Don't catch bare Exception - be specific
    # TODO(code-review): Use logger instead of print()
    # TODO(code-review): Hardcoded 'emotion_analyzer_v2' should be a constant
    """Get count of unanalyzed tracks from database.

    Args:
        db_manager: Database manager instance.

    Returns:
        Dictionary with keys:
            - total_tracks: Total tracks with lyrics
            - analyzed_tracks: Tracks already analyzed
            - unanalyzed_tracks: Tracks needing analysis
    """
    try:
        # Total tracks with lyrics
        total_query = "SELECT COUNT(*) as count FROM tracks WHERE lyrics IS NOT NULL"
        total_result = await db_manager.execute_query(total_query)
        total_tracks = total_result[0]["count"] if total_result else 0

        # Analyzed tracks with emotion_analyzer_v2
        analyzed_query = """
        SELECT COUNT(DISTINCT track_id) as count
        FROM analysis_results
        WHERE analyzer_type = 'emotion_analyzer_v2'
        """
        analyzed_result = await db_manager.execute_query(analyzed_query)
        analyzed_tracks = analyzed_result[0]["count"] if analyzed_result else 0

        return {
            "total_tracks": total_tracks,
            "analyzed_tracks": analyzed_tracks,
            "unanalyzed_tracks": total_tracks - analyzed_tracks,
        }
    except Exception as e:
        print(f"Error retrieving statistics: {e}")
        return {"total_tracks": 0, "analyzed_tracks": 0, "unanalyzed_tracks": 0}


async def get_analyzer_stats(db_manager):
    """Получить статистику по анализаторам"""
    try:
        query = """
        SELECT analyzer_type, COUNT(*) as count 
        FROM analysis_results 
        GROUP BY analyzer_type 
        ORDER BY count DESC
        """
        result = await db_manager.execute_query(query)
        return {row["analyzer_type"]: row["count"] for row in result} if result else {}
    except Exception as e:
        print(f"Ошибка получения статистики анализаторов: {e}")
        return {}


# TODO(code-review): Main execution block should be minimal - move logic to main() function
# TODO(code-review): Translate all Russian comments to English
# TODO(code-review): Add proper CLI framework instead of manual argument parsing
# TODO(code-review): Add configuration file support instead of only env vars
# Main execution
if __name__ == "__main__":
    import argparse
    import sys

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Command line arguments
    parser = argparse.ArgumentParser(
        description="Enhanced Emotion Analyzer V2.0 with PostgreSQL support"
    )
    parser.add_argument("--test", action="store_true", help="Run comprehensive tests")
    parser.add_argument("--batch", action="store_true", help="Test batch processing")
    parser.add_argument(
        "--postgres", action="store_true", help="Test PostgreSQL integration"
    )
    parser.add_argument(
        "--analyze-db",
        action="store_true",
        help="Analyze songs from PostgreSQL database",
    )
    parser.add_argument(
        "--batch-db", action="store_true", help="Run batch analysis from database"
    )
    parser.add_argument(
        "--db-stats", action="store_true", help="Show database statistics"
    )
    parser.add_argument("--text", type=str, help="Analyze specific text")
    parser.add_argument(
        "--limit", type=int, default=100, help="Limit for database operations"
    )
    parser.add_argument(
        "--all", action="store_true", help="Analyze all unanalyzed tracks in database"
    )
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--menu", action="store_true", help="Launch interactive menu")

    args = parser.parse_args()

    async def main():
        # TODO(code-review): Add comprehensive docstring
        # TODO(code-review): Function is too long (70+ lines) - split into smaller functions
        # TODO(code-review): Too many if statements - use command pattern or dispatch table
        # TODO(code-review): Add proper error handling and exit codes
        """Main execution function.

        Processes command line arguments and executes the appropriate
        analyzer operation (testing, database analysis, etc.).
        """
        # If no arguments or --menu specified, launch interactive menu
        if args.menu or (len(sys.argv) == 1):
            await create_interactive_menu()
            return

        if args.test:
            await test_analyzer_comprehensive()

        if args.batch:
            await test_batch_processing()

        if args.postgres:
            await test_postgresql_integration()

        if args.analyze_db:
            # Determine limit: use --all for unlimited, otherwise use --limit
            limit = None if args.all else args.limit
            limit_text = "all unanalyzed tracks" if args.all else f"limit: {args.limit}"

            print(f"\n🗄️ Analyzing songs from PostgreSQL database ({limit_text})...")
            analyzer = EmotionAnalyzer({"postgres_enabled": True})
            if await analyzer.initialize():
                if args.all:
                    # For --all, use a very large limit or batch processing
                    results = await analyzer.analyze_from_database(
                        limit=1000000
                    )  # Very large limit
                else:
                    results = await analyzer.analyze_from_database(limit=args.limit)
                print(f"✅ Analyzed {len(results)} songs from database")
                await analyzer.cleanup()
            else:
                print("❌ Failed to initialize analyzer")

        if args.batch_db:
            print("\n⚡ Running batch analysis from database...")
            analyzer = EmotionAnalyzer({"postgres_enabled": True})
            if await analyzer.initialize():
                stats = await analyzer.batch_analyze_from_database(
                    batch_size=50, max_batches=10
                )
                print(f"📊 Batch analysis completed: {stats}")
                await analyzer.cleanup()
            else:
                print("❌ Failed to initialize analyzer")

        if args.db_stats:
            print("\n📊 Database Statistics:")
            analyzer = EmotionAnalyzer({"postgres_enabled": True})
            if await analyzer.initialize():
                db_stats = await analyzer.get_database_stats()
                print(f"Database stats: {db_stats}")
                summary = await analyzer.get_analysis_summary()
                print(f"Analysis summary: {summary}")
                await analyzer.cleanup()
            else:
                print("❌ Failed to initialize analyzer")

        if args.text:
            print("\n📝 Analyzing custom text...")
            analyzer = EmotionAnalyzer()
            if await analyzer.initialize():
                result = await analyzer.analyze_song("Custom", "Analysis", args.text)
                print(f"Result: {result.metadata}")
                await analyzer.cleanup()

    # TODO(code-review): Remove emoji from user-facing messages
    # TODO(code-review): Use logger instead of print()
    # TODO(code-review): Add proper cleanup on shutdown
    # Run main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
