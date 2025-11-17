"""Multi-model AI analyzer for rap lyrics with safety validation and interpretability.

This module provides comprehensive AI-powered analysis of rap song lyrics using
multiple provider models with automatic fallback, safety validation, and hallucination
detection. It supports local models (Ollama), cloud APIs (Google Gemma), and mock
providers for testing.

Key Features:
    - Multi-provider AI analysis with automatic fallback (Ollama -> Gemma -> Mock)
    - Safety validation and hallucination detection to ensure reliable results
    - Interpretable analysis with decision explanations and confidence scores
    - Batch processing with PostgreSQL storage and async support
    - Cost optimization (prioritizes free local models)
    - Comprehensive quality metrics and authenticity scoring

Architecture:
    - ModelProvider: Base class for AI providers (Ollama, Gemma, Mock)
    - MultiModelAnalyzer: Main analyzer with fallback logic
    - SafetyValidator: Validates analysis reliability and detects hallucinations
    - InterpretableAnalyzer: Generates explanations for AI decisions
    - PostgreSQLManager: Async database connection management

Typical Usage:
    Basic analysis:
        analyzer = MultiModelAnalyzer()
        await analyzer.initialize()
        result = analyzer.analyze_song("Kendrick Lamar", "HUMBLE.", lyrics)
        await analyzer.close()

    Analysis with safety validation:
        result = analyzer.analyze_song_with_safety("Drake", "Hotline Bling", lyrics)
        if result['is_safe']:
            print(f"Reliable: {result['summary']}")

    Explainable analysis:
        explainable = analyzer.analyze_with_explanations("Artist", "Title", lyrics)
        print(f"Confidence: {explainable.confidence:.2f}")
        print(f"Explanations: {explainable.explanation}")

    Batch processing:
        await analyzer.batch_analyze_from_db(limit=100)

Dependencies:
    - asyncpg, psycopg2-binary: PostgreSQL connectivity
    - ollama: Local model inference (optional)
    - google-generativeai: Gemma API access (optional)
    - pydantic: Data validation and models
    - requests: HTTP requests for Ollama API

Environment Variables:
    - POSTGRES_HOST: PostgreSQL server (default: localhost)
    - POSTGRES_PORT: PostgreSQL port (default: 5432)
    - POSTGRES_DATABASE: Database name (default: rap_lyrics)
    - POSTGRES_USERNAME: Database user (default: rap_user)
    - POSTGRES_PASSWORD: Database password (required)
    - GOOGLE_API_KEY: Google Gemma API key (optional)

Author:
    AI Assistant

Version:
    2.0.0 - Multi-model with safety validation
"""

# TODO(code_review): [HIGH] Move imports to follow Google Python Style Guide order:
# 1. Standard library imports
# 2. Third-party imports
# 3. Local application imports
# Currently mixing all import types without clear separation
import asyncio
import json
import logging
import os
import re
from datetime import datetime

import asyncpg
import psycopg2
import requests
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel, Field

# TODO(code_review): [CRITICAL] Avoid module-level side effects (load_dotenv())
# Move to main() or create explicit initialization function
# This breaks testability and causes issues with import order
# Загрузка переменных окружения
load_dotenv()

# TODO(code_review): [HIGH] Avoid module-level logging configuration
# This affects global logging state and breaks when imported as library
# Move to main() or use __name__ == "__main__" guard
# Consider using logging.getLogger(__name__).setLevel() instead
# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ai_analysis.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ===== PostgreSQL Configuration =====
# TODO(code_review): [HIGH] Convert to dataclass or use __init__ for proper instance attributes
# Current implementation uses class attributes which are shared across all instances
# This can lead to unexpected behavior and testing issues
# TODO(code_review): [CRITICAL] NEVER hardcode credentials, even as defaults
# Remove "securepassword123" default - fail fast if password not provided
# Use required environment variables or raise ConfigurationError
class DatabaseConfig:
    """PostgreSQL database connection configuration.

    Configuration class for PostgreSQL connection parameters loaded from
    environment variables with sensible defaults.

    Attributes:
        host: PostgreSQL server hostname (default: localhost).
        port: PostgreSQL server port (default: 5432).
        database: Target database name (default: rap_lyrics).
        username: Database username for authentication (default: rap_user).
        password: Database password for authentication (default: securepassword123).
        max_connections: Maximum connection pool size (default: 20).
        min_connections: Minimum connection pool size (default: 5).

    Note:
        All attributes are loaded from environment variables with POSTGRES_ prefix.
        Connection pooling parameters should be tuned based on expected load.
    """
    # TODO(code_review): [MEDIUM] Extract magic numbers to named constants at module level
    # DEFAULT_POSTGRES_HOST = "localhost"
    # DEFAULT_POSTGRES_PORT = 5432
    # etc.

    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    database: str = os.getenv("POSTGRES_DATABASE", "rap_lyrics")
    username: str = os.getenv("POSTGRES_USERNAME", "rap_user")
    password: str = os.getenv("POSTGRES_PASSWORD", "securepassword123")  # TODO(code_review): [CRITICAL] SECURITY: Remove hardcoded password!
    max_connections: int = int(os.getenv("POSTGRES_MAX_CONNECTIONS", "20"))
    min_connections: int = int(os.getenv("POSTGRES_MIN_CONNECTIONS", "5"))


# ===== Data Models =====
class SongMetadata(BaseModel):
    """Song metadata and high-level characteristics.

    Pydantic model for storing basic song metadata including genre classification,
    emotional characteristics, and content warnings.

    Attributes:
        genre: Music genre classification (e.g., "rap", "trap", "drill", "old_school").
            Default: "rap".
        mood: Emotional mood/tone (e.g., "aggressive", "melancholic", "energetic", "neutral").
            Default: "neutral".
        energy_level: Energy intensity level ("low", "medium", "high").
            Default: "medium".
        explicit_content: Whether song contains explicit language or mature themes.
            Default: False.
    """

    genre: str = Field(default="rap")
    mood: str = Field(default="neutral")
    energy_level: str = Field(default="medium")
    explicit_content: bool = Field(default=False)


class LyricsAnalysis(BaseModel):
    """Detailed lyrics structure and literary analysis.

    Pydantic model for in-depth analysis of lyrical content, structure,
    themes, and artistic techniques.

    Attributes:
        structure: Song structure pattern (e.g., "verse-chorus-verse", "freestyle", "hook").
            Default: "verse".
        rhyme_scheme: Rhyme pattern (e.g., "AABB", "ABAB", "complex", "simple").
            Default: "unknown".
        complexity_level: Lyrical complexity rating ("beginner", "intermediate", "advanced").
            Default: "intermediate".
        main_themes: List of identified thematic elements (e.g., ["money", "street_life"]).
            Default: empty list.
        emotional_tone: Overall emotional tone ("positive", "negative", "neutral", "mixed").
            Default: "neutral".
        storytelling_type: Narrative style ("narrative", "abstract", "conversational").
            Default: "conversational".
        wordplay_quality: Quality of wordplay and linguistic creativity ("basic", "good", "excellent").
            Default: "basic".
    """

    structure: str = Field(default="verse")
    rhyme_scheme: str = Field(default="unknown")
    complexity_level: str = Field(default="intermediate")
    main_themes: list[str] = Field(default_factory=list)
    emotional_tone: str = Field(default="neutral")
    storytelling_type: str = Field(default="conversational")
    wordplay_quality: str = Field(default="basic")


class QualityMetrics(BaseModel):
    """Quality and authenticity metrics for song analysis.

    Pydantic model for quantitative quality assessments across multiple dimensions
    including authenticity, creativity, commercial viability, and AI detection.

    Attributes:
        authenticity_score: Perceived authenticity and genuineness (0.0-1.0).
            Higher values indicate more authentic street/real expression.
            Default: 0.5.
        lyrical_creativity: Creative and linguistic innovation level (0.0-1.0).
            Measures wordplay, metaphors, and unique expression.
            Default: 0.5.
        commercial_appeal: Mainstream commercial potential (0.0-1.0).
            Likelihood of broad audience appeal and radio play.
            Default: 0.5.
        uniqueness: Originality and distinctiveness (0.0-1.0).
            How unique the style and content are.
            Default: 0.5.
        overall_quality: Aggregate quality rating ("poor", "fair", "good", "excellent").
            Default: "fair".
        ai_likelihood: Probability lyrics are AI-generated (0.0-1.0).
            Higher values suggest potential AI authorship.
            Default: 0.5.

    Note:
        All float metrics are constrained to [0.0, 1.0] range via Pydantic validation.
    """

    authenticity_score: float = Field(default=0.5, ge=0.0, le=1.0)
    lyrical_creativity: float = Field(default=0.5, ge=0.0, le=1.0)
    commercial_appeal: float = Field(default=0.5, ge=0.0, le=1.0)
    uniqueness: float = Field(default=0.5, ge=0.0, le=1.0)
    overall_quality: str = Field(default="fair")
    ai_likelihood: float = Field(default=0.5, ge=0.0, le=1.0)


class EnhancedSongData(BaseModel):
    """Complete AI analysis results for a song.

    Comprehensive analysis result combining metadata, lyrical analysis,
    quality metrics, and analysis metadata.

    Attributes:
        artist: Artist/performer name.
        title: Song title.
        metadata: High-level metadata (genre, mood, energy, explicit).
        lyrics_analysis: Detailed lyrical analysis (structure, themes, complexity).
        quality_metrics: Quality scores (authenticity, creativity, commercial appeal).
        model_used: Name of AI model/provider used (e.g., "ollama", "gemma-2-27b-it").
        analysis_date: ISO 8601 timestamp of analysis completion.

    Example:
        >>> data = EnhancedSongData(
        ...     artist="Kendrick Lamar",
        ...     title="HUMBLE.",
        ...     metadata=SongMetadata(genre="trap", mood="aggressive"),
        ...     lyrics_analysis=LyricsAnalysis(complexity_level="advanced"),
        ...     quality_metrics=QualityMetrics(authenticity_score=0.92),
        ...     model_used="ollama-llama3.2",
        ...     analysis_date="2025-11-02T10:30:00"
        ... )
    """

    artist: str
    title: str
    metadata: SongMetadata
    lyrics_analysis: LyricsAnalysis
    quality_metrics: QualityMetrics
    model_used: str
    analysis_date: str


class ExplainableAnalysisResult(BaseModel):
    """Analysis result with AI decision explanations and interpretability.

    Extended analysis result that includes base analysis plus interpretability
    features: explanations, confidence scores, decision factors, and influential phrases.

    Attributes:
        analysis: Base EnhancedSongData analysis result.
        explanation: Category-keyed explanations for AI decisions.
            Keys: "genre_indicators", "mood_triggers", "authenticity_markers", "quality_indicators".
            Values: List of human-readable explanation strings.
        confidence: Overall confidence score in analysis (0.0-1.0).
            Based on text length, genre evidence, metric consistency, and detail presence.
        decision_factors: Dictionary of factor names to importance scores (0.0-1.0).
            E.g., {"trap_keywords": 0.85, "authenticity": 0.73, "word_diversity": 0.67}.
        influential_phrases: Category-keyed lists of influential lyrics phrases.
            Keys: "genre_phrases", "mood_phrases", "authenticity_phrases", "quality_phrases".
            Values: Lists of actual lyrics lines that influenced the decision.

    Example:
        >>> result = ExplainableAnalysisResult(
        ...     analysis=enhanced_data,
        ...     explanation={"genre_indicators": ["Genre 'trap' detected: молли, lean, скрр"]},
        ...     confidence=0.87,
        ...     decision_factors={"trap_keywords": 0.92, "authenticity": 0.78},
        ...     influential_phrases={"genre_phrases": ["Молли в моей чашке, я lean пью"]}
        ... )
    """

    analysis: EnhancedSongData
    explanation: dict[str, list[str]]
    confidence: float
    decision_factors: dict[str, float]
    influential_phrases: dict[str, list[str]]


# ===== PostgreSQL Database Manager =====
class PostgreSQLManager:
    """PostgreSQL connection manager with async connection pooling.

    Manages asyncpg connection pool for efficient async database operations.
    Supports both async and synchronous connection modes with automatic
    connection lifecycle management.

    Attributes:
        config: DatabaseConfig instance with connection parameters.
        pool: asyncpg connection pool (None until initialized).
        logger: Logger instance for database operations.

    Example:
        >>> db = PostgreSQLManager()
        >>> await db.initialize()
        >>> async with db.get_connection() as conn:
        ...     result = await conn.fetch("SELECT * FROM tracks LIMIT 10")
        >>> await db.close()
    """

    def __init__(self, config: DatabaseConfig = None):
        """Initialize PostgreSQL manager with configuration.

        Args:
            config: DatabaseConfig instance. If None, creates default config
                from environment variables.

        Note:
            Connection pool is not created until initialize() is called.
        """
        self.config = config or DatabaseConfig()
        self.pool = None
        self.logger = logging.getLogger(f"{__name__}.PostgreSQLManager")

    # TODO(code_review): [MEDIUM] Add type hints for async context manager protocol
    # Consider implementing __aenter__ and __aexit__ for proper async context manager
    async def initialize(self) -> bool:
        """Initialize asyncpg connection pool and test connectivity.

        Creates connection pool with configured min/max size and tests
        database connectivity by executing a simple query.

        Returns:
            True if pool initialized and test query succeeds, False otherwise.

        Side Effects:
            - Creates self.pool asyncpg connection pool
            - Logs initialization status

        Note:
            This method is idempotent - calling multiple times recreates the pool.
            Connection timeout is set to 60 seconds for all queries.
        """
        try:
            self.logger.info("Initializing PostgreSQL connection pool")
            # TODO(code_review): [HIGH] Avoid string interpolation for DSN with credentials
            # Use asyncpg.create_pool() parameters directly for better security
            # Current approach logs credentials if dsn variable is printed
            dsn = f"postgresql://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"
            # TODO(code_review): [MEDIUM] Extract magic number 60 to named constant
            # COMMAND_TIMEOUT_SECONDS = 60
            self.pool = await asyncpg.create_pool(
                dsn,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                command_timeout=60,  # TODO(code_review): [MEDIUM] Magic number - extract to constant
                server_settings={
                    "application_name": "multi_model_analyzer",
                    "timezone": "UTC",
                },
            )

            # Test connection
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT 1")
            # TODO(code_review): [LOW] Avoid emoji in production logs - breaks parsing/monitoring tools
            # Use structured logging with severity levels instead
            self.logger.info("✅ PostgreSQL connection pool initialized successfully")
            return True

        except Exception as e:
            # TODO(code_review): [HIGH] Catch specific exceptions (asyncpg.PostgresError, etc.)
            # Generic Exception catching hides bugs and makes debugging harder
            # TODO(code_review): [MEDIUM] Add exception context: logger.error(..., exc_info=True)
            self.logger.error(f"❌ Failed to initialize PostgreSQL: {e}")
            return False

    async def get_connection(self):
        """Get connection from pool, initializing if necessary.

        Returns:
            asyncpg.pool.PoolAcquireContext: Connection context manager.
                Use with async context manager pattern.

        Side Effects:
            If pool not initialized, calls initialize() automatically.

        Example:
            >>> async with db.get_connection() as conn:
            ...     rows = await conn.fetch("SELECT * FROM tracks")
        """
        if not self.pool:
            await self.initialize()
        return self.pool.acquire()

    async def close(self):
        """Close connection pool and release all connections.

        Gracefully closes all pooled connections and resets pool to None.
        Safe to call multiple times (idempotent).

        Side Effects:
            - Closes all active connections in pool
            - Sets self.pool to None
        """
        if self.pool:
            await self.pool.close()
            self.pool = None

    def get_sync_connection(self):
        """Get synchronous psycopg2 connection for non-async operations.

        Creates a new synchronous connection using psycopg2 with RealDictCursor
        for dict-style row access. Connection is NOT pooled.

        Returns:
            psycopg2.extensions.connection: Synchronous database connection
                with RealDictCursor factory.

        Warning:
            Caller is responsible for closing the connection.
            Prefer async methods when possible for better performance.

        Example:
            >>> conn = db.get_sync_connection()
            >>> try:
            ...     cursor = conn.cursor()
            ...     cursor.execute("SELECT * FROM tracks LIMIT 1")
            ...     row = cursor.fetchone()
            ... finally:
            ...     conn.close()
        """
        return psycopg2.connect(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.username,
            password=self.config.password,
            cursor_factory=RealDictCursor,
        )


# TODO(code_review): [HIGH] Large class (>600 lines) - violates Single Responsibility Principle
# Split into smaller focused classes:
# - ThemeValidator, MoodValidator, ConsistencyChecker, HallucinationDetector
class SafetyValidator:
    """Validator for AI analysis reliability and hallucination detection.

    Comprehensive validation system that checks AI-generated analysis results
    for internal consistency, factual accuracy, hallucinations, and text-analysis
    alignment. Uses keyword-based validation with English and Russian support.

    Key Validation Checks:
        - Internal consistency: Logical coherence of predictions
        - Factual accuracy: Claims match actual lyrics content
        - Hallucination detection: Identifies fabricated themes/attributes
        - Text alignment: Analysis matches lyrics characteristics
        - Warning flags: Identifies suspicious patterns

    Attributes:
        theme_keywords: Dict mapping themes to English keyword lists.
        mood_indicators: Dict mapping moods to English keyword lists.
        consistency_threshold: Minimum score for consistency (default: 0.6).
        hallucination_threshold: Maximum acceptable hallucination risk (default: 0.4).

    Example:
        >>> validator = SafetyValidator()
        >>> result = validator.validate_analysis(lyrics, analysis_dict)
        >>> if result['is_reliable']:
        ...     print(f"✅ {result['validation_summary']}")
        ... else:
        ...     print(f"⚠️ Warnings: {result['warning_flags']}")
    """
    # TODO(code_review): [MEDIUM] Extract keyword dictionaries to external config file (JSON/YAML)
    # Hardcoded dictionaries make internationalization and updates difficult
    # Consider using external keyword database or ML-based theme detection

    def __init__(self):
        """Initialize SafetyValidator with keyword dictionaries and thresholds.

        Sets up theme and mood keyword dictionaries for validation, primarily
        focused on English keywords with some Russian support.

        Note:
            Thresholds can be adjusted after initialization if needed:
            - consistency_threshold: Lower = more permissive (default 0.6)
            - hallucination_threshold: Higher = stricter (default 0.4)
        """
        # TODO(code_review): [LOW] Mixed language comments (Russian) - use English for consistency
        # Follow Google Style Guide: use English for all code/comments
        # Словари для проверки тематик (English-focused)
        # TODO(code_review): [MEDIUM] Large hardcoded data structure - extract to constants or config
        # This makes the __init__ method hard to read and test
        self.theme_keywords = {
            "money": [
                "cash",
                "money",
                "dollars",
                "bands",
                "racks",
                "bread",
                "paper",
                "coins",
                "wealth",
                "riches",
                "bank",
                "rich",
                "fortune",
            ],
            "relationships": [
                "love",
                "girl",
                "boy",
                "girlfriend",
                "boyfriend",
                "wife",
                "husband",
                "family",
                "bae",
                "baby",
                "relationship",
                "romance",
            ],
            "street_life": [
                "street",
                "block",
                "neighborhood",
                "ghetto",
                "projects",
                "corners",
                "trap",
                "streets",
                "hood",
                "city",
                "urban",
            ],
            "success": [
                "success",
                "famous",
                "star",
                "career",
                "achievement",
                "winning",
                "made it",
                "top",
                "win",
                "champion",
                "glory",
            ],
            "struggle": [
                "struggle",
                "pain",
                "problems",
                "hardship",
                "suffering",
                "tough",
                "hard times",
                "grind",
                "difficult",
                "rough",
            ],
            "drugs": [
                "drugs",
                "molly",
                "xanax",
                "percs",
                "pills",
                "cocaine",
                "heroin",
                "marijuana",
                "cannabis",
                "weed",
                "lean",
                "high",
            ],
            "violence": [
                "war",
                "fight",
                "murder",
                "blood",
                "gun",
                "knife",
                "shoot",
                "kill",
                "weapon",
                "violence",
                "battle",
                "beef",
            ],
            "party": [
                "party",
                "club",
                "dance",
                "fun",
                "alcohol",
                "beer",
                "drunk",
                "drinking",
                "turn up",
                "lit",
                "celebration",
            ],
            "depression": [
                "depression",
                "sad",
                "suicide",
                "death",
                "lonely",
                "sorrow",
                "depressed",
                "dark",
                "pain",
                "hurt",
                "broken",
            ],
            "social_issues": [
                "politics",
                "society",
                "system",
                "power",
                "protest",
                "revolution",
                "government",
                "social",
                "justice",
                "change",
            ],
        }

        # Словари для проверки настроений (English-focused)
        self.mood_indicators = {
            "aggressive": [
                "hate",
                "angry",
                "mad",
                "kill",
                "war",
                "blood",
                "fight",
                "rage",
                "fury",
                "violence",
                "beef",
                "pissed",
            ],
            "melancholic": [
                "sad",
                "sadness",
                "tears",
                "depression",
                "lonely",
                "pain",
                "hurt",
                "broken",
                "crying",
                "sorrow",
            ],
            "energetic": [
                "party",
                "club",
                "dance",
                "hype",
                "lit",
                "turn up",
                "wild",
                "crazy",
                "bounce",
                "jump",
                "energy",
            ],
            "neutral": [
                "talking",
                "telling",
                "thinking",
                "know",
                "remember",
                "see",
                "saying",
                "speaking",
                "telling",
            ],
        }

        # TODO(code_review): [HIGH] Extract magic numbers to module-level constants
        # CONSISTENCY_THRESHOLD = 0.6
        # HALLUCINATION_THRESHOLD = 0.4
        # Consider making these configurable via constructor parameters
        # Пороги для различных проверок
        self.consistency_threshold = 0.6  # Понижен для более гибкой оценки  # TODO(code_review): [MEDIUM] Magic number
        self.hallucination_threshold = 0.4  # Повышен для строгого контроля галлюцинаций  # TODO(code_review): [MEDIUM] Magic number

    # TODO(code_review): [HIGH] Return TypedDict instead of plain dict for type safety
    # Define ValidationResult(TypedDict) with all expected fields
    # Current return type is untyped dict which breaks IDE autocomplete and type checking
    def validate_analysis(self, lyrics: str, ai_analysis: dict) -> dict:  # TODO(code_review): [HIGH] Add proper return type hint
        """Perform comprehensive reliability validation of AI analysis results.

        Validates AI-generated analysis through multiple checks including internal
        consistency, factual accuracy, hallucination detection, and text alignment.
        Returns detailed validation metrics and overall reliability verdict.

        Args:
            lyrics: Original song lyrics text (any language).
            ai_analysis: Dictionary containing AI analysis results with expected keys:
                - metadata: dict with genre, mood, energy_level, explicit_content
                - lyrics_analysis: dict with structure, main_themes, complexity_level
                - quality_metrics: dict with authenticity_score, commercial_appeal, etc.
                (Keys may vary; missing keys are handled gracefully)

        Returns:
            Dictionary with validation results:
                - is_reliable (bool): Overall reliability verdict based on all checks
                - reliability_score (float): Aggregate reliability 0.0-1.0
                - consistency_score (float): Internal consistency 0.0-1.0
                - factual_accuracy (float): Factual claims accuracy 0.0-1.0
                - hallucination_risk (float): Risk of hallucinations 0.0-1.0
                - text_alignment (float): Text-analysis alignment 0.0-1.0
                - warning_flags (list[str]): List of warning flag identifiers
                - validation_summary (str): Human-readable summary message

        Example:
            >>> validator = SafetyValidator()
            >>> analysis = {
            ...     "genre": "trap", "mood": "aggressive",
            ...     "main_themes": ["money", "street_life"],
            ...     "authenticity_score": 0.85
            ... }
            >>> result = validator.validate_analysis(lyrics, analysis)
            >>> print(f"Reliable: {result['is_reliable']}")
            >>> print(f"Hallucination risk: {result['hallucination_risk']:.2f}")

        Note:
            Result is considered reliable if:
            - hallucination_risk < 0.4
            - consistency_score > 0.6
            - factual_accuracy > 0.5
            - text_alignment > 0.4
            - No critical warning flags
        """

        # 1. Проверка внутренней консистентности
        consistency_score = self.check_internal_consistency(ai_analysis)

        # 2. Валидация фактических утверждений
        factual_accuracy = self.validate_factual_claims(lyrics, ai_analysis)

        # 3. Детекция галлюцинаций
        hallucination_risk = self.detect_hallucinations(lyrics, ai_analysis)

        # 4. Проверка соответствия текста и анализа
        text_alignment = self.check_text_analysis_alignment(lyrics, ai_analysis)

        # 5. Получение предупреждающих флагов
        warning_flags = self.get_warning_flags(ai_analysis, lyrics)

        # Итоговая оценка надежности
        is_reliable = (
            hallucination_risk < self.hallucination_threshold
            and consistency_score > self.consistency_threshold
            and factual_accuracy > 0.5  # Понижен порог
            and text_alignment > 0.4  # Понижен порог
            and len(warning_flags) == 0  # Никаких критических предупреждений
        )

        return {
            "is_reliable": is_reliable,
            "reliability_score": (consistency_score + factual_accuracy + text_alignment)
            / 3,
            "consistency_score": consistency_score,
            "factual_accuracy": factual_accuracy,
            "hallucination_risk": hallucination_risk,
            "text_alignment": text_alignment,
            "warning_flags": warning_flags,
            "validation_summary": self._generate_validation_summary(
                is_reliable, hallucination_risk, consistency_score, warning_flags
            ),
        }

    # TODO(code_review): [MEDIUM] Method too long (70+ lines) - violates SRP
    # Split into smaller focused methods: check_theme_hallucinations(), check_mood_hallucinations(), etc.
    def detect_hallucinations(self, lyrics: str, analysis: dict) -> float:
        """Detect potential hallucinations in AI analysis results.

        Checks if AI-claimed themes, moods, genre, and explicit content are actually
        supported by evidence in the lyrics. Accumulates penalty scores for
        unsupported claims.

        Args:
            lyrics: Original song lyrics text.
            analysis: Dict with analysis results (genre, mood, main_themes, etc.).

        Returns:
            Hallucination risk score 0.0-1.0, where:
                - 0.0 = No hallucinations detected
                - 0.4+ = High risk (threshold for unreliable)
                - 1.0 = Maximum risk (capped)

        Note:
            Penalties are accumulated:
            - Theme not found: +0.15 per theme
            - Mood unsupported: +0.2
            - Inappropriate genre: +0.3
            - Explicit content mismatch: +0.1
            - Unrealistic quality scores: +0.1
        """
        # TODO(code_review): [MEDIUM] Extract penalty values to named constants
        # THEME_PENALTY = 0.15
        # MOOD_PENALTY = 0.2
        # GENRE_PENALTY = 0.3, etc.
        hallucination_score = 0.0
        lyrics_lower = lyrics.lower()

        # Проверяем заявленные темы
        if "main_themes" in analysis:
            claimed_themes = analysis["main_themes"]
            if isinstance(claimed_themes, list):
                for theme in claimed_themes:
                    if not self.theme_present_in_lyrics(theme, lyrics_lower):
                        hallucination_score += 0.15  # TODO(code_review): [MEDIUM] Magic number - extract to constant
                        logger.warning(
                            f"🚨 Possible hallucination: theme '{theme}' not found in lyrics"
                        )

        # Проверяем настроение
        if "mood" in analysis:
            claimed_mood = analysis["mood"].lower()
            if not self.mood_supported_by_lyrics(claimed_mood, lyrics_lower):
                hallucination_score += 0.2
                logger.warning(
                    f"🚨 Possible hallucination: mood '{claimed_mood}' not supported by lyrics"
                )

        # Проверяем жанр (менее строго, так как жанр может быть музыкальным)
        if "genre" in analysis:
            claimed_genre = analysis["genre"].lower()
            if (
                claimed_genre in ["classical", "jazz", "country"]
                and "rap" not in lyrics_lower
            ):
                hallucination_score += 0.3  # Явно неподходящий жанр

        # Проверяем explicit content
        if "explicit_content" in analysis:
            claimed_explicit = analysis["explicit_content"]
            actual_explicit = self.detect_explicit_content(lyrics_lower)
            if claimed_explicit != actual_explicit:
                hallucination_score += 0.1

        # Проверяем качественные метрики на разумность
        if "authenticity_score" in analysis:
            auth_score = analysis["authenticity_score"]
            if isinstance(auth_score, (int, float)):
                if (
                    auth_score > 0.9 and len(lyrics.split()) < 50
                ):  # Высокая аутентичность при коротком тексте
                    hallucination_score += 0.1

        return min(hallucination_score, 1.0)

    def theme_present_in_lyrics(self, theme: str, lyrics_lower: str) -> bool:
        """Проверяет, присутствует ли тема в тексте песни"""
        theme_lower = theme.lower().replace("_", " ")

        # Прямое совпадение
        if theme_lower in lyrics_lower:
            return True

        # Проверка по ключевым словам
        if theme_lower in self.theme_keywords:
            keywords = self.theme_keywords[theme_lower]
            found_keywords = sum(1 for keyword in keywords if keyword in lyrics_lower)
            return found_keywords >= 1  # Достаточно одного ключевого слова

        # Частичные совпадения для составных тем (English-focused)
        if "street" in theme_lower and any(
            word in lyrics_lower
            for word in ["street", "block", "hood", "neighborhood", "ghetto"]
        ):
            return True
        if "money" in theme_lower and any(
            word in lyrics_lower
            for word in ["cash", "money", "dollars", "bands", "racks", "bread"]
        ):
            return True
        if "love" in theme_lower and any(
            word in lyrics_lower
            for word in ["love", "girl", "relationship", "girlfriend", "romance"]
        ):
            return True

        return False

    def mood_supported_by_lyrics(self, mood: str, lyrics_lower: str) -> bool:
        """Проверяет, соответствует ли заявленное настроение тексту"""
        if mood in self.mood_indicators:
            indicators = self.mood_indicators[mood]
            found_indicators = sum(
                1 for indicator in indicators if indicator in lyrics_lower
            )
            return found_indicators >= 1

        # Для неизвестных настроений возвращаем True (не можем проверить)
        return True

    # TODO(code_review): [HIGH] Hardcoded profanity list is incomplete and unmaintainable
    # Use external profanity filter library (e.g., better-profanity, profanity-check)
    # Current approach:
    # 1. Misses common profanity variations (f**k, sh!t, etc.)
    # 2. Doesn't support multiple languages properly
    # 3. No context awareness (Scunthorpe problem)
    # 4. Hardcoded list is difficult to update/customize
    def detect_explicit_content(self, lyrics_lower: str) -> bool:
        """Детектирует explicit контент в тексте (English-focused)"""
        # TODO(code_review): [MEDIUM] Extract to module-level constant or config file
        # TODO(code_review): [HIGH] Consider using set instead of list for O(1) lookup
        explicit_words = [
            "fuck",
            "shit",
            "bitch",
            "asshole",
            "damn",
            "hell",
            "pussy",
            "dick",
            "cock",
            "motherfucker",
            "nigga",
            "nigger",
            "whore",
            "slut",
            "cunt",
            "bastard",
            "piss",
        ]
        # TODO(code_review): [MEDIUM] Inefficient O(n*m) algorithm
        # Convert explicit_words to set for O(n) performance
        # Or use regex compilation for better performance
        return any(word in lyrics_lower for word in explicit_words)

    def check_internal_consistency(self, analysis: dict) -> float:
        """Check internal logical consistency of analysis results.

        Validates that different analysis dimensions are logically coherent
        (e.g., aggressive mood with low energy is suspicious).

        Args:
            analysis: Dict with analysis results (mood, energy_level, quality metrics).

        Returns:
            Consistency score 0.0-1.0, where:
                - 1.0 = Perfectly consistent
                - 0.6+ = Acceptable consistency (threshold)
                - 0.0 = Highly inconsistent

        Note:
            Penalties for logical contradictions:
            - Melancholic mood + high energy: -0.2
            - Aggressive mood + low energy: -0.3
            - Very high authenticity + very high commercial: -0.2
            - Advanced complexity + poor quality: -0.2
            - Beginner complexity + excellent quality: -0.1
        """
        consistency_score = 1.0

        # Проверяем соответствие настроения и энергии
        mood = analysis.get("mood", "").lower()
        energy = analysis.get("energy_level", "").lower()

        # Логические противоречия
        if mood == "melancholic" and energy == "high":
            consistency_score -= 0.2  # Грустная, но энергичная - возможно
        if mood == "aggressive" and energy == "low":
            consistency_score -= 0.3  # Агрессивная, но низкая энергия - странно

        # Проверяем качественные метрики
        if "authenticity_score" in analysis and "commercial_appeal" in analysis:
            auth = analysis["authenticity_score"]
            commercial = analysis["commercial_appeal"]
            if isinstance(auth, (int, float)) and isinstance(commercial, (int, float)):
                # Очень высокая аутентичность И очень высокий коммерческий аппеал - редко
                if auth > 0.9 and commercial > 0.9:
                    consistency_score -= 0.2

        # Проверяем соответствие сложности и качества
        complexity = analysis.get("complexity_level", "").lower()
        overall_quality = analysis.get("overall_quality", "").lower()

        if complexity == "advanced" and overall_quality == "poor":
            consistency_score -= 0.2
        if complexity == "beginner" and overall_quality == "excellent":
            consistency_score -= 0.1

        return max(consistency_score, 0.0)

    def validate_factual_claims(self, lyrics: str, analysis: dict) -> float:
        """Validate factual claims in analysis against actual lyrics.

        Checks if structural and complexity claims are reasonable given
        the actual lyrics length, structure, and characteristics.

        Args:
            lyrics: Original song lyrics text.
            analysis: Dict with structure, rhyme_scheme, complexity_level claims.

        Returns:
            Factual accuracy score 0.0-1.0, where:
                - 1.0 = All claims validated
                - 0.5+ = Acceptable accuracy (threshold)
                - 0.0 = Multiple invalid claims

        Note:
            Penalties for unrealistic claims:
            - Complex structure claimed but too few lines: -0.2
            - Hook structure but too many lines: -0.1
            - Complex rhyme scheme but simple repetition: -0.1
            - Advanced complexity but < 100 words: -0.2
            - Beginner complexity but > 500 words: -0.1
        """
        factual_score = 1.0
        lyrics_lower = lyrics.lower()

        # Проверяем структуру
        claimed_structure = analysis.get("structure", "").lower()
        if claimed_structure:
            # Подсчет строк для валидации структуры
            lines = [line for line in lyrics.split("\n") if line.strip()]

            if "verse-chorus-verse" in claimed_structure and len(lines) < 8:
                factual_score -= 0.2  # Слишком короткий для такой структуры
            if "hook" in claimed_structure and len(lines) > 20:
                factual_score -= 0.1  # Слишком длинный для hook

        # Проверяем схему рифм
        rhyme_scheme = analysis.get("rhyme_scheme", "").lower()
        if rhyme_scheme and rhyme_scheme != "unknown":
            # Упрощенная проверка рифм
            lines = [line.strip() for line in lyrics.split("\n") if line.strip()]
            if len(lines) >= 4:
                # Если заявлена сложная схема, но текст простой
                if (
                    "complex" in rhyme_scheme
                    and len(set(line.split()[-1] for line in lines[:4] if line.split()))
                    == 1
                ):
                    factual_score -= 0.1

        # Проверяем количество слов vs сложность
        word_count = len(lyrics.split())
        complexity = analysis.get("complexity_level", "").lower()

        if complexity == "advanced" and word_count < 100:
            factual_score -= 0.2
        if complexity == "beginner" and word_count > 500:
            factual_score -= 0.1

        return max(factual_score, 0.0)

    def check_text_analysis_alignment(self, lyrics: str, analysis: dict) -> float:
        """Check alignment between lyrics characteristics and analysis.

        Validates that analysis matches observable text characteristics like
        length, explicit content, energy indicators (punctuation, caps).

        Args:
            lyrics: Original song lyrics text.
            analysis: Dict with analysis results.

        Returns:
            Alignment score 0.0-1.0, where:
                - 1.0 = Perfect alignment
                - 0.4+ = Acceptable alignment (threshold)
                - 0.0 = Poor alignment

        Note:
            Penalties for misalignment:
            - Short text but detailed analysis: -0.2
            - Explicit content mismatch: -0.3
            - High energy but no indicators: -0.2
            - Low energy but many indicators: -0.2
        """
        alignment_score = 1.0
        lyrics_lower = lyrics.lower()

        # Проверяем соответствие длины текста и детальности анализа
        word_count = len(lyrics.split())

        # Если текст короткий, но анализ очень детальный - подозрительно
        if word_count < 50:
            detailed_fields = sum(
                1
                for key in ["main_themes", "structure", "rhyme_scheme"]
                if analysis.get(key)
            )
            if detailed_fields > 2:
                alignment_score -= 0.2

        # Проверяем explicit content alignment
        actual_explicit = self.detect_explicit_content(lyrics_lower)
        claimed_explicit = analysis.get("explicit_content", False)

        if actual_explicit != claimed_explicit:
            alignment_score -= 0.3

        # Проверяем energy level alignment
        energy = analysis.get("energy_level", "").lower()
        exclamation_count = lyrics.count("!")
        caps_ratio = sum(1 for c in lyrics if c.isupper()) / max(len(lyrics), 1)

        if energy == "high" and exclamation_count == 0 and caps_ratio < 0.05:
            alignment_score -= 0.2
        if energy == "low" and exclamation_count > 5:
            alignment_score -= 0.2

        return max(alignment_score, 0.0)

    def get_warning_flags(self, analysis: dict, lyrics: str) -> list:
        """Получает список предупреждающих флагов"""
        flags = []

        # Проверка на подозрительно высокие оценки
        if analysis.get("authenticity_score", 0) > 0.95:
            flags.append("SUSPICIOUSLY_HIGH_AUTHENTICITY")

        if analysis.get("uniqueness", 0) > 0.95:
            flags.append("SUSPICIOUSLY_HIGH_UNIQUENESS")

        # Проверка на несоответствие длины и сложности
        word_count = len(lyrics.split())
        complexity = analysis.get("complexity_level", "").lower()

        if word_count < 50 and complexity == "advanced":
            flags.append("SHORT_TEXT_HIGH_COMPLEXITY")

        # Проверка на отсутствие тем в коротком тексте
        themes = analysis.get("main_themes", [])
        if word_count < 100 and len(themes) > 4:
            flags.append("SHORT_TEXT_MANY_THEMES")

        # Проверка на противоречивые метрики
        mood = analysis.get("mood", "").lower()
        commercial = analysis.get("commercial_appeal", 0)

        if mood == "melancholic" and commercial > 0.8:
            flags.append("SAD_MOOD_HIGH_COMMERCIAL")

        return flags

    def _generate_validation_summary(
        self,
        is_reliable: bool,
        hallucination_risk: float,
        consistency_score: float,
        warning_flags: list,
    ) -> str:
        """Генерирует текстовое резюме валидации"""
        if is_reliable:
            return f"✅ Анализ надежен (риск галлюцинаций: {hallucination_risk:.2f})"
        issues = []
        if hallucination_risk > 0.4:  # Обновленный порог
            issues.append(f"высокий риск галлюцинаций ({hallucination_risk:.2f})")
        if consistency_score < 0.6:  # Обновленный порог
            issues.append(f"низкая консистентность ({consistency_score:.2f})")
        if warning_flags:
            issues.append(f"предупреждения: {len(warning_flags)}")

        return f"⚠️ Анализ ненадежен: {', '.join(issues)}"


# TODO(code_review): [HIGH] Large class (375 lines) - violates SRP
# Split into smaller classes: ExplanationGenerator, ConfidenceCalculator, FactorExtractor
# TODO(code_review): [MEDIUM] Duplicate code with SafetyValidator keyword dictionaries
# Extract shared keyword dictionaries to separate KeywordRegistry class
class InterpretableAnalyzer:
    """Analyzer with AI decision explanations and interpretability features.

    Wraps base analyzer to provide interpretability by explaining classification
    decisions, calculating confidence scores, identifying key decision factors,
    and extracting influential phrases from lyrics.

    Uses keyword-based feature extraction with Russian and English support
    to explain genre, mood, and authenticity classifications.

    Attributes:
        base_analyzer: Base analyzer instance (e.g., MultiModelAnalyzer).
        genre_keywords: Dict mapping genres to keyword lists.
        mood_keywords: Dict mapping moods to keyword lists.
        authenticity_keywords: Dict mapping authenticity types to keywords.

    Example:
        >>> base = MultiModelAnalyzer()
        >>> interpreter = InterpretableAnalyzer(base)
        >>> result = interpreter.analyze_with_explanation("Artist", "Title", lyrics)
        >>> print(f"Confidence: {result.confidence:.2f}")
        >>> for category, explanations in result.explanation.items():
        ...     print(f"{category}: {explanations}")
    """

    def __init__(self, base_analyzer):
        """Initialize InterpretableAnalyzer with base analyzer.

        Args:
            base_analyzer: Base analyzer instance that provides analyze_song() method.
                Typically MultiModelAnalyzer.

        Note:
            Initializes genre, mood, and authenticity keyword dictionaries
            for decision explanation generation.
        """
        self.base_analyzer = base_analyzer

        # Словари ключевых слов для разных категорий
        self.genre_keywords = {
            "trap": ["trap", "молли", "lean", "xanax", "скрр", "йа", "bando", "plug"],
            "drill": ["drill", "smoke", "opps", "block", "gang", "sliding", "packed"],
            "old_school": [
                "boom bap",
                "real hip hop",
                "90s",
                "golden era",
                "conscious",
            ],
            "gangsta": ["glock", "ak", "blood", "crip", "hood", "street", "thug"],
            "emo_rap": [
                "депрессия",
                "суицид",
                "боль",
                "грусть",
                "одиночество",
                "слезы",
            ],
        }

        self.mood_keywords = {
            "aggressive": ["убью", "война", "кровь", "драка", "hate", "angry", "mad"],
            "melancholic": ["грусть", "печаль", "слезы", "депрессия", "одиночество"],
            "energetic": ["party", "club", "dance", "energy", "вперед", "движение"],
            "chill": ["расслабон", "спокойно", "медленно", "vibe", "атмосфера"],
        }

        self.authenticity_keywords = {
            "real": ["правда", "реально", "честно", "без фальши", "по-настоящему"],
            "fake": ["понт", "фейк", "пижон", "показуха", "притворство"],
            "street": ["улица", "район", "двор", "подъезд", "квартал", "гетто"],
            "commercial": ["money", "brand", "коммерция", "продажи", "mainstream"],
        }

    def analyze_with_explanation(
        self, artist: str, title: str, lyrics: str
    ) -> ExplainableAnalysisResult | None:
        """Analyze song with AI decision explanations and confidence scores.

        Performs base analysis and augments it with interpretability features:
        explanations of classification decisions, confidence score, key decision
        factors, and influential lyrics phrases.

        Args:
            artist: Artist/performer name.
            title: Song title.
            lyrics: Complete song lyrics text.

        Returns:
            ExplainableAnalysisResult containing:
                - analysis: Base EnhancedSongData with full analysis
                - explanation: Dict of category to list of explanation strings
                - confidence: Overall confidence score 0.0-1.0
                - decision_factors: Dict of factor names to importance scores
                - influential_phrases: Dict of category to influential lyrics
            Returns None if base analysis fails.

        Example:
            >>> result = analyzer.analyze_with_explanation("Kendrick", "DNA.", lyrics)
            >>> if result:
            ...     print(f"Genre: {result.analysis.metadata.genre}")
            ...     print(f"Confidence: {result.confidence:.2f}")
            ...     for expl in result.explanation['genre_indicators']:
            ...         print(f"  - {expl}")

        Note:
            Confidence is based on text length, genre evidence strength,
            quality metric consistency, and detail presence.
        """
        try:
            # Базовый анализ
            base_result = self.base_analyzer.analyze_song(artist, title, lyrics)
            if not base_result:
                return None

            # Генерируем объяснения
            explanation = self.explain_decision(lyrics, base_result)
            confidence = self.calculate_confidence(base_result, lyrics)
            decision_factors = self.extract_key_factors(lyrics, base_result)
            influential_phrases = self.find_influential_phrases(lyrics, base_result)

            return ExplainableAnalysisResult(
                analysis=base_result,
                explanation=explanation,
                confidence=confidence,
                decision_factors=decision_factors,
                influential_phrases=influential_phrases,
            )

        except Exception as e:
            logging.error(f"⌛ Ошибка интерпретируемого анализатора: {e}")
            return None

    def explain_decision(
        self, lyrics: str, result: EnhancedSongData
    ) -> dict[str, list[str]]:
        """Объясняет, на основе чего модель приняла решение"""
        lyrics_lower = lyrics.lower()
        explanations = {
            "genre_indicators": [],
            "mood_triggers": [],
            "authenticity_markers": [],
            "quality_indicators": [],
        }

        # Анализ жанровых индикаторов
        detected_genre = result.metadata.genre.lower()
        for genre, keywords in self.genre_keywords.items():
            if genre in detected_genre:
                found_keywords = [kw for kw in keywords if kw in lyrics_lower]
                if found_keywords:
                    explanations["genre_indicators"].extend(
                        [
                            f"Жанр '{genre}' определен по словам: {', '.join(found_keywords[:3])}"
                        ]
                    )

        # Анализ настроения
        detected_mood = result.metadata.mood.lower()
        for mood, keywords in self.mood_keywords.items():
            if mood in detected_mood:
                found_keywords = [kw for kw in keywords if kw in lyrics_lower]
                if found_keywords:
                    explanations["mood_triggers"].extend(
                        [
                            f"Настроение '{mood}' определено по словам: {', '.join(found_keywords[:3])}"
                        ]
                    )

        # Анализ аутентичности
        auth_score = result.quality_metrics.authenticity_score
        if auth_score > 0.7:
            real_words = [
                kw for kw in self.authenticity_keywords["real"] if kw in lyrics_lower
            ]
            street_words = [
                kw for kw in self.authenticity_keywords["street"] if kw in lyrics_lower
            ]
            if real_words or street_words:
                explanations["authenticity_markers"].append(
                    f"Высокая аутентичность ({auth_score:.2f}) благодаря: {', '.join((real_words + street_words)[:3])}"
                )
        elif auth_score < 0.4:
            fake_words = [
                kw for kw in self.authenticity_keywords["fake"] if kw in lyrics_lower
            ]
            commercial_words = [
                kw
                for kw in self.authenticity_keywords["commercial"]
                if kw in lyrics_lower
            ]
            if fake_words or commercial_words:
                explanations["authenticity_markers"].append(
                    f"Низкая аутентичность ({auth_score:.2f}) из-за: {', '.join((fake_words + commercial_words)[:3])}"
                )

        # Анализ качества
        creativity = result.quality_metrics.lyrical_creativity
        wordplay = result.lyrics_analysis.wordplay_quality
        explanations["quality_indicators"].append(
            f"Креативность: {creativity:.2f}, Wordplay: {wordplay}"
        )

        return explanations

    def calculate_confidence(self, result: EnhancedSongData, lyrics: str) -> float:
        """Рассчитывает уверенность в анализе"""
        confidence_factors = []

        # Фактор 1: Длина текста (больше текста = больше уверенности)
        text_length_factor = min(len(lyrics) / 1000, 1.0)  # Нормализуем к 1.0
        confidence_factors.append(text_length_factor * 0.2)

        # Фактор 2: Наличие явных индикаторов жанра
        genre_confidence = self._calculate_genre_confidence(
            lyrics, result.metadata.genre
        )
        confidence_factors.append(genre_confidence * 0.3)

        # Фактор 3: Консистентность метрик качества
        quality_consistency = self._calculate_quality_consistency(
            result.quality_metrics
        )
        confidence_factors.append(quality_consistency * 0.3)

        # Фактор 4: Наличие конкретных деталей (имена, места, события)
        detail_factor = self._calculate_detail_factor(lyrics)
        confidence_factors.append(detail_factor * 0.2)

        return min(sum(confidence_factors), 1.0)

    def _calculate_genre_confidence(self, lyrics: str, genre: str) -> float:
        """Рассчитывает уверенность в определении жанра"""
        lyrics_lower = lyrics.lower()
        genre_lower = genre.lower()

        matching_keywords = 0
        total_keywords = 0

        for g, keywords in self.genre_keywords.items():
            if g in genre_lower:
                total_keywords = len(keywords)
                matching_keywords = sum(1 for kw in keywords if kw in lyrics_lower)
                break

        if total_keywords == 0:
            return 0.5  # Средняя уверенность для неизвестных жанров

        return matching_keywords / total_keywords

    def _calculate_quality_consistency(self, metrics: QualityMetrics) -> float:
        """Проверяет консистентность метрик качества"""
        scores = [
            metrics.authenticity_score,
            metrics.lyrical_creativity,
            metrics.commercial_appeal,
            metrics.uniqueness,
        ]

        # Рассчитываем стандартное отклонение
        mean_score = sum(scores) / len(scores)
        variance = sum((x - mean_score) ** 2 for x in scores) / len(scores)
        std_dev = variance**0.5

        # Низкое стандартное отклонение = высокая консистентность
        consistency = max(0, 1 - (std_dev * 2))  # Нормализуем
        return consistency

    def _calculate_detail_factor(self, lyrics: str) -> float:
        """Рассчитывает наличие конкретных деталей"""
        detail_indicators = [
            r"\b[A-Z][a-z]+\b",  # Имена собственные
            r"\b\d{4}\b",  # Годы
            r"\b\d+[км]\b",  # Расстояния
            r"\$\d+",  # Деньги
            r"\b[А-ЯЁ][а-яё]+\b",  # Русские имена собственные
        ]

        total_details = 0
        for pattern in detail_indicators:
            matches = re.findall(pattern, lyrics)
            total_details += len(matches)

        # Нормализуем к длине текста
        detail_density = total_details / max(len(lyrics.split()), 1)
        return min(detail_density * 10, 1.0)  # Масштабируем

    def extract_key_factors(
        self, lyrics: str, result: EnhancedSongData
    ) -> dict[str, float]:
        """Извлекает ключевые факторы, влияющие на анализ"""
        factors = {}
        lyrics_lower = lyrics.lower()

        # Частота ключевых слов по категориям
        for category, keywords in {**self.genre_keywords, **self.mood_keywords}.items():
            keyword_count = sum(1 for kw in keywords if kw in lyrics_lower)
            factors[f"{category}_keywords"] = keyword_count / len(keywords)

        # Структурные факторы
        factors["text_length"] = min(len(lyrics) / 2000, 1.0)
        factors["line_count"] = min(len(lyrics.split("\n")) / 50, 1.0)
        factors["word_diversity"] = len(set(lyrics.lower().split())) / max(
            len(lyrics.split()), 1
        )

        # Метрики качества как факторы
        factors["authenticity"] = result.quality_metrics.authenticity_score
        factors["creativity"] = result.quality_metrics.lyrical_creativity
        factors["commercial_appeal"] = result.quality_metrics.commercial_appeal
        factors["uniqueness"] = result.quality_metrics.uniqueness

        return factors

    def find_influential_phrases(
        self, lyrics: str, result: EnhancedSongData
    ) -> dict[str, list[str]]:
        """Находит конкретные фразы, которые повлияли на оценку"""
        influential = {
            "genre_phrases": [],
            "mood_phrases": [],
            "authenticity_phrases": [],
            "quality_phrases": [],
        }

        lines = lyrics.split("\n")

        # Поиск влиятельных фраз для жанра
        genre_lower = result.metadata.genre.lower()
        for genre, keywords in self.genre_keywords.items():
            if genre in genre_lower:
                for line in lines:
                    if any(kw in line.lower() for kw in keywords):
                        influential["genre_phrases"].append(line.strip())
                        if len(influential["genre_phrases"]) >= 3:
                            break

        # Поиск фраз для настроения
        mood_lower = result.metadata.mood.lower()
        for mood, keywords in self.mood_keywords.items():
            if mood in mood_lower:
                for line in lines:
                    if any(kw in line.lower() for kw in keywords):
                        influential["mood_phrases"].append(line.strip())
                        if len(influential["mood_phrases"]) >= 3:
                            break

        # Поиск фраз для аутентичности
        auth_score = result.quality_metrics.authenticity_score
        auth_keywords = (
            self.authenticity_keywords["real"] + self.authenticity_keywords["street"]
        )
        if auth_score > 0.7:
            for line in lines:
                if any(kw in line.lower() for kw in auth_keywords):
                    influential["authenticity_phrases"].append(line.strip())
                    if len(influential["authenticity_phrases"]) >= 2:
                        break

        # Поиск качественных wordplay фраз
        if result.lyrics_analysis.wordplay_quality == "excellent":
            # Ищем строки с рифмами или аллитерацией
            for line in lines:
                words = line.lower().split()
                if len(words) >= 4:
                    # Простая проверка на рифму (одинаковые окончания)
                    endings = [word[-2:] for word in words if len(word) > 3]
                    if (
                        len(set(endings)) < len(endings) * 0.8
                    ):  # Много повторяющихся окончаний
                        influential["quality_phrases"].append(line.strip())
                        if len(influential["quality_phrases"]) >= 2:
                            break

        return influential


class ModelProvider:
    """Base class for AI provider implementations.

    Abstract base class defining interface for AI model providers.
    Concrete implementations must provide availability checking and
    song analysis functionality.

    Attributes:
        name: Provider name (e.g., "Ollama", "Gemma", "Mock").
        available: Whether provider is available/initialized (bool).
        cost_per_1k_tokens: Cost per 1000 tokens in USD (float).

    Note:
        Subclasses must implement check_availability() and analyze_song().
    """

    def __init__(self, name: str):
        """Initialize provider with name.

        Args:
            name: Provider identifier string.
        """
        self.name = name
        self.available = False
        self.cost_per_1k_tokens = 0.0

    def check_availability(self) -> bool:
        """Check if provider is available and operational.

        Returns:
            True if provider can be used, False otherwise.

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        raise NotImplementedError

    def analyze_song(
        self, artist: str, title: str, lyrics: str
    ) -> EnhancedSongData | None:
        """Analyze song lyrics and return structured results.

        Args:
            artist: Artist/performer name.
            title: Song title.
            lyrics: Complete song lyrics text.

        Returns:
            EnhancedSongData with analysis results, or None on failure.

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        raise NotImplementedError


class OllamaProvider(ModelProvider):
    """Provider for local Ollama models.

    Connects to locally-running Ollama server for free, offline AI inference.
    Automatically checks availability and attempts to pull missing models.

    Attributes:
        model_name: Ollama model identifier (e.g., "llama3.2:3b").
        base_url: Ollama API base URL (default: http://localhost:11434).
        cost_per_1k_tokens: Always 0.0 (free local inference).

    Example:
        >>> provider = OllamaProvider(model_name="llama3.2:3b")
        >>> if provider.available:
        ...     result = provider.analyze_song("Artist", "Title", lyrics)

    Note:
        Requires Ollama server running: `ollama serve`
        Timeout is 60 seconds for analysis requests.
    """

    def __init__(
        self, model_name: str = "llama3.2:3b", base_url: str = "http://localhost:11434"
    ):
        """Initialize Ollama provider with model and URL.

        Args:
            model_name: Ollama model to use (default: "llama3.2:3b").
            base_url: Ollama API endpoint (default: "http://localhost:11434").

        Note:
            Automatically calls check_availability() during initialization.
            If model not found, attempts to pull it automatically.
        """
        super().__init__("Ollama")
        self.model_name = model_name
        self.base_url = base_url
        self.cost_per_1k_tokens = 0.0  # Бесплатно!
        self.available = self.check_availability()

    def check_availability(self) -> bool:
        """Check if Ollama server is running and model is available.

        Makes HTTP request to Ollama API /api/tags to verify:
        1. Server is running and responsive
        2. Configured model exists locally
        3. If model missing, attempts automatic pull

        Returns:
            True if Ollama accessible and model available/downloaded,
            False if server unreachable or model pull fails.

        Side Effects:
            - Logs availability status and model list
            - May trigger model download via _pull_model()

        Note:
            Uses 5 second timeout for API request.
            Disables proxies for local connection.
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
                proxies={"http": "", "https": ""},
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                logger.info(f"🦙 Ollama доступен. Модели: {available_models}")

                # Проверяем наличие нужной модели
                if any(self.model_name in model for model in available_models):
                    logger.info(f"✅ Модель {self.model_name} найдена")
                    return True
                logger.warning(
                    f"⚠️ Модель {self.model_name} не найдена. Попытка загрузки..."
                )
                return self._pull_model()
            return False
        except requests.exceptions.RequestException as e:
            logger.warning(f"⌛ Ollama недоступен: {e}")
            return False

    def _pull_model(self) -> bool:
        """Загрузка модели если её нет"""
        try:
            logger.info(f"🔥 Загружаем модель {self.model_name}...")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model_name},
                timeout=300,  # 5 минут на загрузку
                proxies={"http": "", "https": ""},
            )
            if response.status_code == 200:
                logger.info(f"✅ Модель {self.model_name} загружена")
                return True
            logger.error(f"⌛ Не удалось загрузить модель: {response.text}")
            return False
        except Exception as e:
            logger.error(f"⌛ Ошибка загрузки модели: {e}")
            return False

    def analyze_song(
        self, artist: str, title: str, lyrics: str
    ) -> EnhancedSongData | None:
        """Analyze song using local Ollama model.

        Sends lyrics to Ollama with structured prompt requesting JSON analysis.
        Parses response and constructs EnhancedSongData.

        Args:
            artist: Artist/performer name.
            title: Song title.
            lyrics: Complete song lyrics (truncated to 2000 chars in prompt).

        Returns:
            EnhancedSongData with analysis results, or None if:
                - Provider not available
                - API request fails
                - JSON parsing fails

        Note:
            Uses temperature=0.1 for consistent results.
            60 second timeout for analysis.
        """
        if not self.available:
            return None

        try:
            prompt = self._create_analysis_prompt(artist, title, lyrics)

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Низкая температура для консистентности
                        "top_p": 0.9,
                        "max_tokens": 1500,
                    },
                },
                timeout=60,
                proxies={"http": "", "https": ""},
            )

            if response.status_code == 200:
                result = response.json()
                analysis_text = result.get("response", "")
                return self._parse_analysis(analysis_text, artist, title)
            logger.error(f"⌛ Ollama ошибка: {response.status_code} - {response.text}")
            return None

        except Exception as e:
            logger.error(f"⌛ Ошибка анализа Ollama: {e}")
            return None

    # TODO(code_review): [HIGH] Method returns large multiline string - extract to template file
    # Use jinja2 or similar template engine for better maintainability
    # Current approach makes prompt versioning and A/B testing difficult
    def _create_analysis_prompt(self, artist: str, title: str, lyrics: str) -> str:
        """Создание промпта для анализа"""
        # TODO(code_review): [MEDIUM] Magic number 2000 - extract to constant
        # LYRICS_MAX_LENGTH = 2000
        # TODO(code_review): [HIGH] Truncating lyrics at 2000 chars may cut mid-word/sentence
        # Use proper text truncation that respects word boundaries
        return f"""
Проанализируй рэп-песню и верни результат СТРОГО в JSON формате.

Исполнитель: {artist}
Название: {title}
Текст: {lyrics[:2000]}...

Верни ТОЛЬКО валидный JSON без дополнительного текста:

{{
    "metadata": {{
        "genre": "rap",
        "mood": "aggressive",
        "energy_level": "high",
        "explicit_content": true
    }},
    "lyrics_analysis": {{
        "structure": "verse-chorus-verse",
        "rhyme_scheme": "ABAB",
        "complexity_level": "advanced",
        "main_themes": ["street_life", "success", "relationships"],
        "emotional_tone": "mixed",
        "storytelling_type": "narrative",
        "wordplay_quality": "excellent"
    }},
    "quality_metrics": {{
        "authenticity_score": 0.8,
        "lyrical_creativity": 0.9,
        "commercial_appeal": 0.7,
        "uniqueness": 0.6,
        "overall_quality": "excellent",
        "ai_likelihood": 0.1
    }}
}}

ОБЯЗАТЕЛЬНЫЕ ПОЛЯ:
- emotional_tone: positive/negative/neutral/mixed
- storytelling_type: narrative/abstract/conversational
- wordplay_quality: basic/good/excellent

Верни ТОЛЬКО JSON без комментариев!
"""

    # TODO(code_review): [CRITICAL] Code duplication - identical method in GemmaProvider
    # Extract to shared utility function or base class method
    # DRY principle violation - same logic duplicated 80+ lines
    def _parse_analysis(
        self, analysis_text: str, artist: str, title: str
    ) -> EnhancedSongData | None:
        """Парсинг результата анализа"""
        try:
            # TODO(code_review): [MEDIUM] Naive JSON extraction - fragile parsing logic
            # Use regex or proper parsing library to handle edge cases
            # Current approach fails if JSON contains nested braces
            # Извлекаем JSON из ответа
            json_start = analysis_text.find("{")
            json_end = analysis_text.rfind("}") + 1

            if json_start == -1 or json_end <= json_start:
                logger.error("⌛ JSON не найден в ответе")
                return None

            json_str = analysis_text[json_start:json_end]
            data = json.loads(json_str)  # TODO(code_review): [MEDIUM] Add JSONDecodeError handling separately

            # Проверяем и дополняем отсутствующие поля
            metadata_data = data.get("metadata", {})
            lyrics_data = data.get("lyrics_analysis", {})
            quality_data = data.get("quality_metrics", {})

            # Дополняем отсутствующие поля в lyrics_analysis
            if "emotional_tone" not in lyrics_data:
                lyrics_data["emotional_tone"] = "neutral"
                logger.warning("⚠️ Добавлено значение по умолчанию для emotional_tone")

            if "storytelling_type" not in lyrics_data:
                lyrics_data["storytelling_type"] = "conversational"
                logger.warning(
                    "⚠️ Добавлено значение по умолчанию для storytelling_type"
                )

            if "wordplay_quality" not in lyrics_data:
                lyrics_data["wordplay_quality"] = "basic"
                logger.warning("⚠️ Добавлено значение по умолчанию для wordplay_quality")

            # Создаем структурированный анализ
            metadata = SongMetadata(**metadata_data)
            lyrics_analysis = LyricsAnalysis(**lyrics_data)
            quality_metrics = QualityMetrics(**quality_data)

            return EnhancedSongData(
                artist=artist,
                title=title,
                metadata=metadata,
                lyrics_analysis=lyrics_analysis,
                quality_metrics=quality_metrics,
                model_used="gemma-2-27b-it",
                analysis_date=datetime.now().isoformat(),
            )

        except json.JSONDecodeError as e:
            logger.error(f"⌛ Ошибка парсинга JSON Gemma: {e}")
            logger.debug(f"Ответ модели: {analysis_text[:500]}")
            return None
        except Exception as e:
            logger.error(f"⌛ Ошибка создания анализа Gemma: {e}")
            return None


# TODO(code_review): [HIGH] MockProvider.analyze_song() is 187 lines - violates SRP
# Extract rule-based analysis logic to separate analyzers:
# - GenreClassifier, MoodDetector, QualityEstimator
# Use strategy pattern or composition instead of single monolithic method
class MockProvider(ModelProvider):
    """Mock provider for testing and demonstration.

    Provides rule-based analysis without external AI models. Always available
    and free, serves as fallback when other providers fail. Uses keyword matching
    and heuristics for genre, mood, and quality estimation.

    Attributes:
        cost_per_1k_tokens: Always 0.0 (no cost for mock analysis).
        available: Always True (no dependencies).

    Example:
        >>> provider = MockProvider()
        >>> result = provider.analyze_song("Test", "Song", lyrics)
        >>> print(f"Genre: {result.metadata.genre}")

    Note:
        Provides reasonable estimates but not true AI analysis.
        Useful for testing, demos, and fallback scenarios.
    """

    def __init__(self):
        """Initialize MockProvider (always available).

        No external dependencies required. Sets available=True immediately.
        """
        super().__init__("Mock")
        self.cost_per_1k_tokens = 0.0  # Бесплатно для тестов
        self.available = True  # Всегда доступен

    def check_availability(self) -> bool:
        """Check availability (always returns True).

        Returns:
            True (MockProvider has no dependencies and is always available).
        """
        logger.info("✅ Mock провайдер готов для демонстрации")
        return True

    def analyze_song(
        self, artist: str, title: str, lyrics: str
    ) -> EnhancedSongData | None:
        """Analyze song using rule-based heuristics.

        Performs keyword-based analysis for genre, mood, themes, and quality
        without external AI. Uses pattern matching and statistical features.

        Args:
            artist: Artist/performer name.
            title: Song title.
            lyrics: Complete song lyrics text.

        Returns:
            EnhancedSongData with heuristic analysis, or None on error.

        Note:
            Analysis logic:
            - Genre: Keyword matching (trap, drill, emo_rap, etc.)
            - Mood: Sentiment keywords (aggressive, sad, energetic)
            - Energy: Punctuation and caps ratio
            - Explicit: Profanity detection
            - Quality: Word diversity and length heuristics
        """
        try:
            lyrics_lower = lyrics.lower()

            # Умное определение жанра на основе ключевых слов
            genre = "rap"
            if any(word in lyrics_lower for word in ["trap", "молли", "lean", "скрр"]):
                genre = "trap"
            elif any(
                word in lyrics_lower for word in ["drill", "smoke", "opps", "gang"]
            ):
                genre = "drill"
            elif any(
                word in lyrics_lower for word in ["улица", "район", "двор", "подъезд"]
            ):
                genre = "gangsta_rap"
            elif any(word in lyrics_lower for word in ["депрессия", "грусть", "слезы"]):
                genre = "emo_rap"

            # Умное определение настроения
            mood = "neutral"
            aggressive_words = ["убью", "война", "драка", "hate", "angry"]
            sad_words = ["грусть", "печаль", "слезы", "депрессия", "одиночество"]
            positive_words = ["party", "счастье", "радость", "love", "успех"]

            if any(word in lyrics_lower for word in aggressive_words):
                mood = "aggressive"
            elif any(word in lyrics_lower for word in sad_words):
                mood = "melancholic"
            elif any(word in lyrics_lower for word in positive_words):
                mood = "energetic"

            # Анализ энергии
            energy = "medium"
            if (
                len(lyrics.split("!")) > 3
                or "йа" in lyrics_lower
                or "скрр" in lyrics_lower
            ):
                energy = "high"
            elif any(word in lyrics_lower for word in ["медленно", "спокойно", "тихо"]):
                energy = "low"

            # Определение explicit content
            explicit_words = [
                "сука",
                "блять",
                "хуй",
                "пизда",
                "ебать",
                "fuck",
                "shit",
                "bitch",
            ]
            explicit_content = any(word in lyrics_lower for word in explicit_words)

            # Анализ структуры
            lines = lyrics.strip().split("\n")
            non_empty_lines = [line for line in lines if line.strip()]

            structure = "verse"
            if len(non_empty_lines) > 16:
                structure = "verse-chorus-verse"
            elif len(non_empty_lines) < 8:
                structure = "hook"

            # Анализ рифмы (упрощенный)
            rhyme_scheme = "ABAB"
            if len(non_empty_lines) >= 4:
                # Проверяем последние слова строк
                last_words = [
                    line.strip().split()[-1].lower()
                    for line in non_empty_lines[:4]
                    if line.strip().split()
                ]
                if len(set(last_words)) == 1:
                    rhyme_scheme = "AAAA"
                elif len(set(last_words)) == 2:
                    rhyme_scheme = "AABB"

            # Определение сложности
            complexity = "intermediate"
            word_count = len(lyrics.split())
            unique_words = len(set(lyrics.lower().split()))
            diversity = unique_words / max(word_count, 1)

            if diversity > 0.7 and word_count > 200:
                complexity = "advanced"
            elif diversity < 0.5 or word_count < 100:
                complexity = "beginner"

            # Основные темы
            themes = []
            theme_keywords = {
                "street_life": ["улица", "район", "двор", "подъезд"],
                "money": ["деньги", "cash", "money", "бабки", "лавэ"],
                "relationships": ["любовь", "девочка", "отношения", "семья"],
                "success": ["успех", "fame", "слава", "топ"],
                "struggle": ["борьба", "struggle", "проблемы", "трудности"],
            }

            for theme, keywords in theme_keywords.items():
                if any(keyword in lyrics_lower for keyword in keywords):
                    themes.append(theme)

            if not themes:
                themes = ["life"]

            # Качественные метрики
            authenticity_score = 0.5
            street_words = ["улица", "район", "двор", "подъезд", "правда", "реально"]
            fake_words = ["понт", "фэйк", "показуха"]

            street_count = sum(1 for word in street_words if word in lyrics_lower)
            fake_count = sum(1 for word in fake_words if word in lyrics_lower)

            authenticity_score = min(
                0.3 + (street_count * 0.15) - (fake_count * 0.1), 1.0
            )

            creativity = min(0.4 + (diversity * 0.6), 1.0)
            commercial_appeal = (
                0.5
                + (0.1 if explicit_content else 0.2)
                + (0.1 if energy == "high" else 0)
            )
            uniqueness = diversity * 0.8 + 0.2

            # Общее качество
            avg_quality = (
                authenticity_score + creativity + commercial_appeal + uniqueness
            ) / 4
            if avg_quality > 0.8:
                overall_quality = "excellent"
            elif avg_quality > 0.6:
                overall_quality = "good"
            elif avg_quality > 0.4:
                overall_quality = "fair"
            else:
                overall_quality = "poor"

            # AI likelihood (обратная зависимость от аутентичности)
            ai_likelihood = max(0.1, 1.0 - authenticity_score)

            # Создание результата
            metadata = SongMetadata(
                genre=genre,
                mood=mood,
                energy_level=energy,
                explicit_content=explicit_content,
            )

            lyrics_analysis = LyricsAnalysis(
                structure=structure,
                rhyme_scheme=rhyme_scheme,
                complexity_level=complexity,
                main_themes=themes,
                emotional_tone=mood,
                storytelling_type="narrative"
                if "история" in lyrics_lower or len(non_empty_lines) > 12
                else "conversational",
                wordplay_quality="excellent"
                if creativity > 0.8
                else ("good" if creativity > 0.6 else "basic"),
            )

            quality_metrics = QualityMetrics(
                authenticity_score=authenticity_score,
                lyrical_creativity=creativity,
                commercial_appeal=commercial_appeal,
                uniqueness=uniqueness,
                overall_quality=overall_quality,
                ai_likelihood=ai_likelihood,
            )

            return EnhancedSongData(
                artist=artist,
                title=title,
                metadata=metadata,
                lyrics_analysis=lyrics_analysis,
                quality_metrics=quality_metrics,
                model_used="mock_analyzer_v1",
                analysis_date=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"⌛ Ошибка Mock анализа: {e}")
            return None


# TODO(code_review): [HIGH] GemmaProvider duplicates 100+ lines from OllamaProvider
# Extract shared logic to base class or mixin:
# - _create_analysis_prompt() is identical
# - _parse_analysis() is identical
# Use template method pattern or composition to eliminate duplication
class GemmaProvider(ModelProvider):
    """Provider for Google Gemma API.

    Connects to Google's Gemma model API for cloud-based AI analysis.
    Requires GOOGLE_API_KEY environment variable.

    Attributes:
        api_key: Google API key from environment (str or None).
        cost_per_1k_tokens: 0.0 within free tier limits.

    Example:
        >>> os.environ['GOOGLE_API_KEY'] = 'your_key_here'
        >>> provider = GemmaProvider()
        >>> if provider.available:
        ...     result = provider.analyze_song("Artist", "Title", lyrics)

    Note:
        Uses gemma-2-27b-it model.
        Requires google-generativeai package installed.
    """

    def __init__(self):
        """Initialize GemmaProvider with API key from environment.

        Reads GOOGLE_API_KEY from environment and checks availability.
        """
        super().__init__("Gemma")
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.available = self.check_availability()
        self.cost_per_1k_tokens = 0.0  # Free tier в пределах лимитов

    def check_availability(self) -> bool:
        """Check if Google API key is valid and library is installed.

        Returns:
            True if API key present and google-generativeai importable,
            False otherwise.

        Note:
            Configures API with key if available.
            Logs warnings if key missing or import fails.
        """
        if not self.api_key:
            logger.warning("⌛ GOOGLE_API_KEY не найден в .env")
            return False

        try:
            import google.generativeai as genai
            from google.generativeai.client import configure
            from google.generativeai.generative_models import GenerativeModel

            configure(api_key=self.api_key)
            logger.info("✅ Google Gemma API готов к использованию")
            return True
        except ImportError:
            logger.warning("⌛ google-generativeai не установлен")
            return False
        except Exception as e:
            logger.error(f"⌛ Ошибка проверки Gemma API: {e}")
            return False

    def analyze_song(
        self, artist: str, title: str, lyrics: str
    ) -> EnhancedSongData | None:
        """Analyze song using Google Gemma API.

        Sends structured prompt to Gemma requesting JSON analysis.
        Parses response and constructs EnhancedSongData.

        Args:
            artist: Artist/performer name.
            title: Song title.
            lyrics: Complete song lyrics (truncated to 2000 chars in prompt).

        Returns:
            EnhancedSongData with analysis results, or None if:
                - Provider not available
                - API request fails
                - JSON parsing fails

        Note:
            Uses temperature=0.1 and max 1500 output tokens.
            Model: gemma-2-27b-it
        """
        if not self.available:
            return None

        try:
            from google.generativeai.generative_models import GenerativeModel

            model = GenerativeModel("gemma-2-27b-it")
            prompt = self._create_analysis_prompt(artist, title, lyrics)

            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 1500,
                },
            )

            if response.text:
                return self._parse_analysis(response.text, artist, title)
            logger.error("⌛ Gemma: пустой ответ")
            return None

        except Exception as e:
            logger.error(f"⌛ Ошибка анализа Gemma: {e}")
            return None

    def _create_analysis_prompt(self, artist: str, title: str, lyrics: str) -> str:
        """Создание промпта для Gemma"""
        return f"""
Analyze this rap song and return results in STRICT JSON format:

Artist: {artist}
Title: {title}
Lyrics: {lyrics[:2000]}...

Return ONLY valid JSON with these exact fields:
{{
    "metadata": {{
        "genre": "rap/trap/drill/old-school/gangsta/emo-rap",
        "mood": "aggressive/melancholic/energetic/neutral",
        "energy_level": "low/medium/high",
        "explicit_content": true/false
    }},
    "lyrics_analysis": {{
        "structure": "verse-chorus-verse/freestyle/storytelling",
        "rhyme_scheme": "AABA/ABAB/complex/simple",
        "complexity_level": "beginner/intermediate/advanced",
        "main_themes": ["money", "relationships", "street_life", "success"],
        "emotional_tone": "positive/negative/neutral/mixed",
        "storytelling_type": "narrative/abstract/conversational",
        "wordplay_quality": "basic/good/excellent"
    }},
    "quality_metrics": {{
        "authenticity_score": 0.0-1.0,
        "lyrical_creativity": 0.0-1.0,
        "commercial_appeal": 0.0-1.0,
        "uniqueness": 0.0-1.0,
        "overall_quality": "poor/fair/good/excellent",
        "ai_likelihood": 0.0-1.0
    }}
}}

Return ONLY JSON, no additional text!
"""

    def _parse_analysis(
        self, analysis_text: str, artist: str, title: str
    ) -> EnhancedSongData | None:
        """Парсинг результата анализа"""
        try:
            # Извлекаем JSON из ответа
            json_start = analysis_text.find("{")
            json_end = analysis_text.rfind("}") + 1

            if json_start == -1 or json_end <= json_start:
                logger.error("⌛ JSON не найден в ответе Gemma")
                return None

            json_str = analysis_text[json_start:json_end]
            data = json.loads(json_str)

            # Проверяем и дополняем отсутствующие поля
            metadata_data = data.get("metadata", {})
            lyrics_data = data.get("lyrics_analysis", {})
            quality_data = data.get("quality_metrics", {})

            # Дополняем отсутствующие поля в lyrics_analysis
            if "emotional_tone" not in lyrics_data:
                lyrics_data["emotional_tone"] = "neutral"
                logger.warning("⚠️ Добавлено значение по умолчанию для emotional_tone")

            if "storytelling_type" not in lyrics_data:
                lyrics_data["storytelling_type"] = "conversational"
                logger.warning(
                    "⚠️ Добавлено значение по умолчанию для storytelling_type"
                )

            if "wordplay_quality" not in lyrics_data:
                lyrics_data["wordplay_quality"] = "basic"
                logger.warning("⚠️ Добавлено значение по умолчанию для wordplay_quality")

            # Создаем структурированный анализ
            metadata = SongMetadata(**metadata_data)
            lyrics_analysis = LyricsAnalysis(**lyrics_data)
            quality_metrics = QualityMetrics(**quality_data)

            return EnhancedSongData(
                artist=artist,
                title=title,
                metadata=metadata,
                lyrics_analysis=lyrics_analysis,
                quality_metrics=quality_metrics,
                model_used="gemma-2-27b-it",
                analysis_date=datetime.now().isoformat(),
            )

        except json.JSONDecodeError as e:
            logger.error(f"⌛ Ошибка парсинга JSON Gemma: {e}")
            logger.debug(f"Ответ модели: {analysis_text[:500]}")
            return None
        except Exception as e:
            logger.error(f"⌛ Ошибка создания анализа Gemma: {e}")
            return None


# TODO(code_review): [HIGH] God class - 562 lines with too many responsibilities
# Violates SRP - handles:
# 1. Provider management
# 2. Database operations
# 3. Batch processing
# 4. Statistics tracking
# 5. Safety validation orchestration
# Split into: ProviderManager, AnalysisOrchestrator, BatchProcessor, StatsCollector
# TODO(code_review): [MEDIUM] No unit tests - only integration test in main()
# Add proper unit tests with mocked dependencies
class MultiModelAnalyzer:
    """Multi-provider AI analyzer with fallback, safety validation, and interpretability.

    Main analyzer class that coordinates multiple AI providers with automatic fallback,
    provides safety validation, hallucination detection, and interpretable analysis
    with explanations.

    Architecture:
        - Provider priority: Ollama (free local) -> Gemma (cloud) -> Mock (fallback)
        - Safety validation via SafetyValidator
        - Interpretability via InterpretableAnalyzer
        - PostgreSQL persistence via PostgreSQLManager

    Attributes:
        providers: List of ModelProvider instances in priority order.
        current_provider: Active provider (first available).
        db_manager: PostgreSQLManager for database operations.
        stats: Dict tracking usage statistics (analyzed count, costs).
        interpretable_analyzer: InterpretableAnalyzer for explanations.
        safety_validator: SafetyValidator for reliability checks.

    Example:
        >>> analyzer = MultiModelAnalyzer()
        >>> await analyzer.initialize()
        >>> result = analyzer.analyze_song("Kendrick", "HUMBLE.", lyrics)
        >>> safe_result = analyzer.analyze_song_with_safety("Drake", "God's Plan", lyrics)
        >>> await analyzer.batch_analyze_from_db(limit=100)
        >>> await analyzer.close()
    """

    def __init__(self):
        """Initialize MultiModelAnalyzer with all providers and validators.

        Sets up provider chain (Ollama -> Gemma -> Mock), database manager,
        interpretable analyzer, and safety validator. Initializes usage statistics.

        Note:
            Database connection not established until initialize() is called.
            Providers check their own availability during initialization.
        """
        self.providers = []
        self.current_provider = None
        self.db_manager = PostgreSQLManager()
        self.stats = {
            "total_analyzed": 0,
            "ollama_used": 0,
            "gemma_used": 0,
            "mock_used": 0,
            "total_cost": 0.0,
        }

        # Инициализация провайдеров в порядке приоритета
        self._init_providers()

        # Инициализация интерпретируемого анализатора
        self.interpretable_analyzer = InterpretableAnalyzer(self)

        # Инициализация валидатора безопасности
        self.safety_validator = SafetyValidator()

    async def initialize(self) -> bool:
        """Initialize database connection pool.

        Returns:
            True if database initialized successfully, False otherwise.

        Note:
            Must be called before any database operations (e.g., batch_analyze_from_db).
        """
        return await self.db_manager.initialize()

    async def close(self):
        """Close database connections and cleanup resources.

        Gracefully closes PostgreSQL connection pool.
        """
        await self.db_manager.close()

    def analyze_with_explanations(
        self, artist: str, title: str, lyrics: str
    ) -> ExplainableAnalysisResult | None:
        """Analyze song with AI decision explanations and interpretability.

        Delegates to InterpretableAnalyzer for explainable analysis.

        Args:
            artist: Artist/performer name.
            title: Song title.
            lyrics: Complete song lyrics.

        Returns:
            ExplainableAnalysisResult with analysis, explanations, confidence,
            decision factors, and influential phrases. None on failure.

        Example:
            >>> result = analyzer.analyze_with_explanations("Artist", "Title", lyrics)
            >>> print(f"Confidence: {result.confidence:.2f}")
            >>> print(result.explanation['genre_indicators'])
        """
        return self.interpretable_analyzer.analyze_with_explanation(
            artist, title, lyrics
        )

    async def explain_existing_analysis(self, track_id: int) -> dict | None:
        """Объясняет существующий анализ из базы данных"""
        try:
            async with self.db_manager.get_connection() as conn:
                # Получаем данные песни и анализа
                query = """
                    SELECT t.artist, t.title, t.lyrics, ar.*
                    FROM tracks t
                    JOIN analysis_results ar ON t.id = ar.track_id
                    WHERE t.id = $1 AND ar.analyzer_type = 'multi_model_ai'
                """

                row = await conn.fetchrow(query, track_id)
                if not row:
                    logger.warning(
                        f"Песня с ID {track_id} не найдена или не проанализирована"
                    )
                    return None

                # Парсим analysis_data
                analysis_data = json.loads(row["analysis_data"])

                # Создаем объект EnhancedSongData из данных БД
                metadata = SongMetadata(
                    genre=analysis_data.get("metadata", {}).get("genre", "rap"),
                    mood=analysis_data.get("metadata", {}).get("mood", "neutral"),
                    energy_level=analysis_data.get("metadata", {}).get(
                        "energy_level", "medium"
                    ),
                    explicit_content=analysis_data.get("metadata", {}).get(
                        "explicit_content", False
                    ),
                )

                lyrics_analysis = LyricsAnalysis(
                    structure=analysis_data.get("lyrics_analysis", {}).get(
                        "structure", "verse"
                    ),
                    rhyme_scheme=analysis_data.get("lyrics_analysis", {}).get(
                        "rhyme_scheme", "unknown"
                    ),
                    complexity_level=analysis_data.get("lyrics_analysis", {}).get(
                        "complexity_level", "intermediate"
                    ),
                    main_themes=analysis_data.get("lyrics_analysis", {}).get(
                        "main_themes", []
                    ),
                    emotional_tone=analysis_data.get("lyrics_analysis", {}).get(
                        "emotional_tone", "neutral"
                    ),
                    storytelling_type=analysis_data.get("lyrics_analysis", {}).get(
                        "storytelling_type", "conversational"
                    ),
                    wordplay_quality=analysis_data.get("lyrics_analysis", {}).get(
                        "wordplay_quality", "basic"
                    ),
                )

                quality_metrics = QualityMetrics(
                    authenticity_score=analysis_data.get("quality_metrics", {}).get(
                        "authenticity_score", 0.5
                    ),
                    lyrical_creativity=analysis_data.get("quality_metrics", {}).get(
                        "lyrical_creativity", 0.5
                    ),
                    commercial_appeal=analysis_data.get("quality_metrics", {}).get(
                        "commercial_appeal", 0.5
                    ),
                    uniqueness=analysis_data.get("quality_metrics", {}).get(
                        "uniqueness", 0.5
                    ),
                    overall_quality=analysis_data.get("quality_metrics", {}).get(
                        "overall_quality", "fair"
                    ),
                    ai_likelihood=analysis_data.get("quality_metrics", {}).get(
                        "ai_likelihood", 0.5
                    ),
                )

                enhanced_data = EnhancedSongData(
                    artist=row["artist"],
                    title=row["title"],
                    metadata=metadata,
                    lyrics_analysis=lyrics_analysis,
                    quality_metrics=quality_metrics,
                    model_used=row["model_version"],
                    analysis_date=row["created_at"].isoformat(),
                )

                # Генерируем объяснения
                explanation = self.interpretable_analyzer.explain_decision(
                    row["lyrics"], enhanced_data
                )
                confidence = self.interpretable_analyzer.calculate_confidence(
                    enhanced_data, row["lyrics"]
                )
                decision_factors = self.interpretable_analyzer.extract_key_factors(
                    row["lyrics"], enhanced_data
                )
                influential_phrases = (
                    self.interpretable_analyzer.find_influential_phrases(
                        row["lyrics"], enhanced_data
                    )
                )

                return {
                    "song_info": {
                        "id": track_id,
                        "artist": row["artist"],
                        "title": row["title"],
                    },
                    "analysis": enhanced_data.model_dump(),
                    "explanation": explanation,
                    "confidence": confidence,
                    "decision_factors": decision_factors,
                    "influential_phrases": influential_phrases,
                }

        except Exception as e:
            logger.error(f"⌛ Ошибка объяснения анализа: {e}")
            return None

    def _init_providers(self):
        """Инициализация провайдеров в порядке приоритета"""
        logger.info("🔍 Инициализация AI провайдеров...")

        # 1. Ollama (приоритет - бесплатно)
        ollama = OllamaProvider()
        if ollama.available:
            self.providers.append(ollama)
            logger.info("✅ Ollama готов к использованию")

        # 2. Google Gemma (cloud fallback)
        gemma = GemmaProvider()
        if gemma.available:
            self.providers.append(gemma)
            logger.info("✅ Google Gemma готов к использованию")

        # 3. Mock Provider (всегда добавляем для надежности)
        mock = MockProvider()
        self.providers.append(mock)
        logger.info("✅ Mock провайдер добавлен как fallback")

        if not self.providers:
            logger.error("⌛ Ни один AI провайдер недоступен!")
            raise Exception("No AI providers available")

        self.current_provider = self.providers[0]
        logger.info(f"🎯 Активный провайдер: {self.current_provider.name}")

    def analyze_song(
        self, artist: str, title: str, lyrics: str
    ) -> EnhancedSongData | None:
        """Analyze song using multi-provider fallback strategy.

        Attempts analysis with providers in priority order (Ollama -> Gemma -> Mock).
        Returns first successful result. Updates usage statistics.

        Args:
            artist: Artist/performer name.
            title: Song title.
            lyrics: Complete song lyrics text.

        Returns:
            EnhancedSongData with analysis results from first successful provider,
            or None if all providers fail.

        Side Effects:
            - Updates self.stats with usage counts
            - Logs provider attempts and results

        Example:
            >>> result = analyzer.analyze_song("Kendrick", "HUMBLE.", lyrics)
            >>> if result:
            ...     print(f"Analyzed by: {result.model_used}")
            ...     print(f"Genre: {result.metadata.genre}")
        """

        for provider in self.providers:
            try:
                logger.info(f"🤖 Анализируем через {provider.name}: {artist} - {title}")

                result = provider.analyze_song(artist, title, lyrics)

                if result:
                    # Обновляем статистику
                    self.stats["total_analyzed"] += 1
                    if provider.name == "Ollama":
                        self.stats["ollama_used"] += 1
                    elif provider.name == "Gemma":
                        self.stats["gemma_used"] += 1
                    elif provider.name == "Mock":
                        self.stats["mock_used"] += 1

                    logger.info(f"✅ Анализ завершен через {provider.name}")
                    return result
                logger.warning(f"⚠️ {provider.name} не смог проанализировать")

            except Exception as e:
                logger.error(f"⌛ Ошибка {provider.name}: {e}")
                continue

        logger.error(
            f"⌛ Все провайдеры не смогли проанализировать: {artist} - {title}"
        )
        return None

    def get_stats(self) -> dict:
        """Получение статистики использования"""
        return {
            **self.stats,
            "available_providers": [p.name for p in self.providers],
            "current_provider": self.current_provider.name
            if self.current_provider
            else None,
        }

    # TODO(code_review): [HIGH] Method too long (60+ lines) - extract helper methods
    # Split into: fetch_unanalyzed_tracks(), analyze_single_track(), save_results()
    # TODO(code_review): [MEDIUM] Hardcoded 2 second sleep - make configurable
    # Add rate_limit_delay parameter with default value
    async def batch_analyze_from_db(self, limit: int = 100, offset: int = 0):  # TODO(code_review): [MEDIUM] Add return type hint -> None
        """Batch analyze unanalyzed songs from database.

        Fetches songs without multi_model_ai analysis from database,
        analyzes them using multi-provider strategy, and saves results.
        Includes progress tracking and error handling.

        Args:
            limit: Maximum number of songs to analyze (default: 100).
            offset: Number of songs to skip (default: 0).

        Returns:
            None (logs progress and summary).

        Side Effects:
            - Fetches songs from tracks table
            - Saves analysis results to analysis_results table
            - Updates self.stats with usage counts
            - 2 second delay between analyses to avoid rate limits

        Example:
            >>> analyzer = MultiModelAnalyzer()
            >>> await analyzer.initialize()
            >>> await analyzer.batch_analyze_from_db(limit=50)
            # Logs: "✅ Успешно: 45, ⌛ Ошибок: 5"

        Note:
            Requires initialize() to be called first.
            Only analyzes songs with lyrics longer than 50 characters.
        """

        logger.info(f"🎵 Начинаем batch анализ: {limit} песен с offset {offset}")

        try:
            async with self.db_manager.get_connection() as conn:
                # Получаем песни для анализа
                query = """
                    SELECT t.id, t.artist, t.title, t.lyrics 
                    FROM tracks t
                    LEFT JOIN analysis_results ar ON t.id = ar.track_id 
                        AND ar.analyzer_type = 'multi_model_ai'
                    WHERE t.lyrics IS NOT NULL 
                        AND LENGTH(TRIM(t.lyrics)) > 50
                        AND ar.id IS NULL  -- Только неанализированные
                    ORDER BY t.id
                    LIMIT $1 OFFSET $2
                """

                rows = await conn.fetch(query, limit, offset)
                logger.info(f"📊 Найдено {len(rows)} песен для анализа")

                successful = 0
                failed = 0

                for i, row in enumerate(rows, 1):
                    try:
                        logger.info(
                            f"📈 Прогресс: {i}/{len(rows)} - {row['artist']} - {row['title']}"
                        )

                        analysis = self.analyze_song(
                            row["artist"], row["title"], row["lyrics"]
                        )

                        if analysis:
                            # Сохраняем в БД
                            await self._save_analysis_to_db(conn, row["id"], analysis)
                            successful += 1
                            logger.info(f"✅ Сохранен анализ #{successful}")
                        else:
                            failed += 1
                            logger.warning("⌛ Не удалось проанализировать")

                        # Пауза между запросами
                        if i < len(rows):  # Не делаем паузу после последней песни
                            await asyncio.sleep(2)  # 2 секунды между анализами  # TODO(code_review): [MEDIUM] Magic number - extract to constant or parameter

                    except Exception as e:
                        failed += 1
                        logger.error(f"⌛ Ошибка анализа песни {row['id']}: {e}")
                        continue

                logger.info(f"""
                🎉 Batch анализ завершен!
                ✅ Успешно: {successful}
                ⌛ Ошибок: {failed}
                📊 Статистика: {self.get_stats()}
                """)

        except Exception as e:
            logger.error(f"⌛ Ошибка batch анализа: {e}")

    # TODO(code_review): [MEDIUM] Add return type hint -> None
    async def _save_analysis_to_db(
        self, conn: asyncpg.Connection, track_id: int, analysis: EnhancedSongData
    ):  # TODO(code_review): [MEDIUM] Missing return type
        """Сохранение анализа в базу данных"""
        try:
            analysis_data = {
                "metadata": analysis.metadata.model_dump(),
                "lyrics_analysis": analysis.lyrics_analysis.model_dump(),
                "quality_metrics": analysis.quality_metrics.model_dump(),
                "analysis_info": {
                    "analyzer_version": "multi_model_v2",  # TODO(code_review): [MEDIUM] Hardcoded version - use __version__ from module
                    "analysis_timestamp": analysis.analysis_date,
                    "model_used": analysis.model_used,
                },
            }
            # TODO(code_review): [MEDIUM] SQL query embedded in code - extract to constants or SQL file
            # Makes query optimization and testing difficult
            await conn.execute(
                """
                INSERT INTO analysis_results (
                    track_id, analyzer_type, sentiment, confidence,
                    complexity_score, themes, analysis_data,
                    processing_time_ms, model_version, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
                track_id,
                "multi_model_ai",  # TODO(code_review): [MEDIUM] Magic string - extract to constant
                analysis.metadata.mood,
                analysis.quality_metrics.authenticity_score,
                analysis.quality_metrics.lyrical_creativity,
                json.dumps(analysis.lyrics_analysis.main_themes),
                json.dumps(analysis_data),
                1000.0,  # placeholder processing time  # TODO(code_review): [HIGH] Fake value 1000.0 - implement actual timing or remove
                analysis.model_used,
                datetime.now(),
            )

        except Exception as e:
            logger.error(f"⌛ Ошибка сохранения в БД: {e}")
            raise

    def analyze_song_with_safety(
        self, artist: str, title: str, lyrics: str
    ) -> dict | None:
        """Analyze song with AI safety validation and hallucination detection.

        Performs standard multi-provider analysis followed by comprehensive
        safety validation using SafetyValidator to detect hallucinations,
        check consistency, and verify factual accuracy.

        Args:
            artist: Artist/performer name.
            title: Song title.
            lyrics: Complete song lyrics text.

        Returns:
            Dictionary containing:
                - analysis (EnhancedSongData): Full AI analysis result
                - validation (dict): Detailed validation metrics
                - is_safe (bool): Whether analysis passed validation
                - confidence (float): Overall reliability score 0.0-1.0
                - warnings (list): List of warning flag strings
                - summary (str): Human-readable validation summary
            Returns None if initial analysis fails.

        Example:
            >>> result = analyzer.analyze_song_with_safety("Drake", "God's Plan", lyrics)
            >>> if result and result['is_safe']:
            ...     print(f"✅ Reliable: {result['summary']}")
            ...     print(f"Confidence: {result['confidence']:.2f}")
            >>> else:
            ...     print(f"⚠️ Warnings: {result['warnings']}")
            ...     print(f"Risk: {result['validation']['hallucination_risk']:.2f}")

        Note:
            Analysis considered reliable if:
            - hallucination_risk < 0.4
            - consistency_score > 0.6
            - factual_accuracy > 0.5
            - text_alignment > 0.4
            - No critical warning flags
        """

        logger.info(f"🛡️ Безопасный анализ: {artist} - {title}")

        # 1. Выполняем стандартный анализ
        analysis_result = self.analyze_song(artist, title, lyrics)

        if not analysis_result:
            logger.error("⌛ Не удалось получить анализ для валидации")
            return None

        # 2. Конвертируем результат в словарь для валидации
        analysis_dict = {
            "genre": analysis_result.metadata.genre,
            "mood": analysis_result.metadata.mood,
            "energy_level": analysis_result.metadata.energy_level,
            "explicit_content": analysis_result.metadata.explicit_content,
            "structure": analysis_result.lyrics_analysis.structure,
            "rhyme_scheme": analysis_result.lyrics_analysis.rhyme_scheme,
            "complexity_level": analysis_result.lyrics_analysis.complexity_level,
            "main_themes": analysis_result.lyrics_analysis.main_themes,
            "authenticity_score": analysis_result.quality_metrics.authenticity_score,
            "lyrical_creativity": analysis_result.quality_metrics.lyrical_creativity,
            "commercial_appeal": analysis_result.quality_metrics.commercial_appeal,
            "uniqueness": analysis_result.quality_metrics.uniqueness,
            "overall_quality": analysis_result.quality_metrics.overall_quality,
            "ai_likelihood": analysis_result.quality_metrics.ai_likelihood,
        }

        # 3. Валидация через SafetyValidator
        validation_result = self.safety_validator.validate_analysis(
            lyrics, analysis_dict
        )

        # 4. Логирование результатов валидации
        logger.info(
            f"🔍 Результат валидации: {validation_result['validation_summary']}"
        )

        if not validation_result["is_reliable"]:
            logger.warning("⚠️ ВНИМАНИЕ: Анализ признан ненадежным!")
            logger.warning(
                f"   • Риск галлюцинаций: {validation_result['hallucination_risk']:.3f}"
            )
            logger.warning(
                f"   • Консистентность: {validation_result['consistency_score']:.3f}"
            )
            logger.warning(
                f"   • Точность фактов: {validation_result['factual_accuracy']:.3f}"
            )

            if validation_result["warning_flags"]:
                logger.warning(
                    f"   • Предупреждения: {', '.join(validation_result['warning_flags'])}"
                )
        else:
            logger.info("✅ Анализ прошел валидацию безопасности")
            logger.info(
                f"   • Надежность: {validation_result['reliability_score']:.3f}"
            )

        # 5. Возвращаем расширенный результат
        return {
            "analysis": analysis_result,
            "validation": validation_result,
            "is_safe": validation_result["is_reliable"],
            "confidence": validation_result["reliability_score"],
            "warnings": validation_result["warning_flags"],
            "summary": validation_result["validation_summary"],
        }


# TODO(code_review): [HIGH] main() is 171 lines - too long for a test function
# Split into separate test functions: test_explainable_analysis(), test_safety_validation(), etc.
# TODO(code_review): [CRITICAL] Integration tests in main() instead of proper test suite
# Move to tests/ directory using pytest framework with fixtures and mocks
# Current approach:
# 1. Can't run individual tests
# 2. No test isolation
# 3. Requires live database
# 4. No assertions - just prints
# TODO(code_review): [HIGH] Test data hardcoded in main() - extract to fixtures
async def main():
    """Test multi-model analyzer with interpretability and safety features.

    Comprehensive test suite demonstrating:
        - Multi-provider initialization and fallback
        - Explainable analysis with decision explanations
        - Safety validation and hallucination detection
        - Statistics tracking and cost optimization

    Returns:
        None. Prints test results to stdout and logs to file.

    Raises:
        Exception: Any unhandled errors are logged with traceback.

    Example:
        >>> asyncio.run(main())
        # Outputs test results with analysis examples and validation demos

    Note:
        Uses test lyrics in Russian for demonstration.
        Requires database connection (continues if fails).
    """

    print("🤖 Многомодельный AI анализатор с объяснениями решений")
    print("=" * 70)  # TODO(code_review): [LOW] Magic number 70 - extract to constant

    try:
        analyzer = MultiModelAnalyzer()

        # Инициализация базы данных
        if not await analyzer.initialize():
            print("⌛ Не удалось инициализировать базу данных")
            return

        print(f"📊 Доступные провайдеры: {[p.name for p in analyzer.providers]}")
        print(
            f"🎯 Активный провайдер: {analyzer.current_provider.name if analyzer.current_provider else 'None'}"
        )

        # Демонстрация анализа с объяснениями
        print("\n🧪 Тестирование анализа с объяснениями...")

        # Тестовый текст песни
        test_lyrics = """
        Я с улицы, район меня воспитал
        В подъездах темных правду познавал
        Молодость прошла в дыму и драках
        Теперь читаю правду в этих строках
        
        Деньги, слава - все это пустота
        Главное остаться собой до конца
        Семья и верные друзья рядом
        Это богатство, а не фальшивый яд
        """

        # Анализ с объяснениями
        explainable_result = analyzer.analyze_with_explanations(
            "Тестовый артист", "Тестовый трек", test_lyrics
        )

        if explainable_result:
            print("\n🎯 РЕЗУЛЬТАТ АНАЛИЗА С ОБЪЯСНЕНИЯМИ:")
            print("-" * 50)

            # Основной анализ
            analysis = explainable_result.analysis
            print(f"🎵 Жанр: {analysis.metadata.genre}")
            print(f"😊 Настроение: {analysis.metadata.mood}")
            print(f"⚡ Энергия: {analysis.metadata.energy_level}")
            print(f"🏆 Качество: {analysis.quality_metrics.overall_quality}")
            print(f"📝 Уверенность: {explainable_result.confidence:.2f}")

            # Объяснения
            print("\n💡 ОБЪЯСНЕНИЯ РЕШЕНИЙ:")
            for category, explanations in explainable_result.explanation.items():
                if explanations:
                    print(f"  {category.replace('_', ' ').title()}:")
                    for exp in explanations:
                        print(f"    • {exp}")

            # Влиятельные фразы
            print("\n🔍 ВЛИЯТЕЛЬНЫЕ ФРАЗЫ:")
            for category, phrases in explainable_result.influential_phrases.items():
                if phrases:
                    print(f"  {category.replace('_', ' ').title()}:")
                    for phrase in phrases[:2]:  # Показываем только первые 2
                        print(f"    • '{phrase}'")

            # Ключевые факторы
            print("\n📊 КЛЮЧЕВЫЕ ФАКТОРЫ (топ-5):")
            top_factors = sorted(
                explainable_result.decision_factors.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]
            for factor, value in top_factors:
                print(f"  • {factor.replace('_', ' ').title()}: {value:.3f}")

        # Демонстрация SafetyValidator
        print("\n🛡️ Тестирование AI Safety & Hallucination Detection...")

        # Тест с потенциально проблемным текстом
        problematic_lyrics = """
        Короткий текст
        """

        safe_result = analyzer.analyze_song_with_safety(
            "Test Artist", "Problematic Track", problematic_lyrics
        )

        if safe_result:
            print("\n🛡️ РЕЗУЛЬТАТ БЕЗОПАСНОГО АНАЛИЗА:")
            print("-" * 50)
            print(
                f"✅ Безопасность: {'НАДЕЖЕН' if safe_result['is_safe'] else 'НЕНАДЕЖЕН'}"
            )
            print(f"📝 Уверенность: {safe_result['confidence']:.3f}")
            print(f"📄 Резюме: {safe_result['summary']}")

            if safe_result["warnings"]:
                print("⚠️ Предупреждения:")
                for warning in safe_result["warnings"]:
                    print(f"   • {warning}")

            # Детали валидации
            validation = safe_result["validation"]
            print("\n📊 ДЕТАЛИ ВАЛИДАЦИИ:")
            print(f"   • Риск галлюцинаций: {validation['hallucination_risk']:.3f}")
            print(f"   • Консистентность: {validation['consistency_score']:.3f}")
            print(f"   • Точность фактов: {validation['factual_accuracy']:.3f}")
            print(f"   • Соответствие тексту: {validation['text_alignment']:.3f}")

        # Тест с нормальным текстом
        print("\n📄 Тест с качественным текстом...")
        normal_safe_result = analyzer.analyze_song_with_safety(
            "Тестовый артист", "Качественный трек", test_lyrics
        )

        if normal_safe_result:
            print(
                f"✅ Нормальный текст: {'НАДЕЖЕН' if normal_safe_result['is_safe'] else 'НЕНАДЕЖЕН'}"
            )
            print(f"📝 Уверенность: {normal_safe_result['confidence']:.3f}")
            print(f"📄 Резюме: {normal_safe_result['summary']}")

        # Показываем статистику
        stats = analyzer.get_stats()
        print("\n📈 СТАТИСТИКА:")
        print(f"  • Всего проанализировано: {stats['total_analyzed']}")
        print(f"  • Ollama использован: {stats['ollama_used']} раз")
        print(f"  • Gemma использован: {stats['gemma_used']} раз")
        print(f"  • Mock использован: {stats['mock_used']} раз")
        print(f"  • Общая стоимость: ${stats['total_cost']:.4f}")

        print("\n✅ AI Safety & Hallucination Detection - ГОТОВО!")
        print("🛡️ Теперь AI анализ включает:")
        print("   • Interpretability & Model Understanding")
        print("   • Safety & Hallucination Detection")
        print("   • Consistency Validation")
        print("   • Factual Accuracy Checking")
        print("🎯 Продукционная система с валидацией надежности!")

        # Закрываем соединения
        await analyzer.close()

    except Exception as e:
        logger.error(f"⌛ Ошибка: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
