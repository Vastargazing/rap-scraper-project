#!/usr/bin/env python3
# ==============================================================================
# COMPREHENSIVE CODE REVIEW FINDINGS - GOOGLE PYTHON STYLE GUIDE
# ==============================================================================
#
# This file has been reviewed against Google Python Style Guide standards.
# Below is a summary of issues found, categorized by severity and type.
#
# CRITICAL ISSUES (Security & Data Loss):
# - Security: Empty password defaults (line 151)
# - Security: DSN contains password in plain text (line 1029)
# - Security: Password could be exposed in logs
# - Error Handling: sys.exit(1) at module level (line 87) - makes module untestable
# - Error Handling: Broad exception catching without specificity
#
# MAJOR ISSUES (Code Quality & Maintainability):
# - Code Organization: 1736 lines in single file - violates SRP
# - i18n: Russian comments throughout - should be English only (lines 123, 143, 151)
# - Documentation: Missing/incomplete Google-style docstrings (Args/Returns/Raises)
# - Type Hints: Missing or incomplete type annotations throughout
# - Testing: Code structure makes unit testing extremely difficult
# - Thread Safety: Signal handler modifies state without synchronization (line 1214)
# - Side Effects: setup_logging() modifies global state (line 323)
#
# STYLE ISSUES (Google Style Guide):
# - Imports: Not properly grouped per PEP 8 (standard/third-party/local)
# - Naming: Some constants not in UPPER_CASE
# - Line Length: Many lines exceed 80 characters (Google recommends 80, max 100)
# - Magic Numbers: Hardcoded values throughout (2, 50, 100, etc.)
# - Code Quality: Emojis in output (lines 1718+) - unprofessional for production
#
# PERFORMANCE ISSUES:
# - O(n²) complexity in rhyme analysis
# - Regex patterns compiled on each use instead of once
# - F-strings in logging (inefficient)
# - Large hardcoded lists could be lazy-loaded
#
# ARCHITECTURAL ISSUES:
# - Single Responsibility Principle violations
# - Tight coupling between components
# - No dependency injection
# - No interface/protocol definitions
# - Global state management
#
# ==============================================================================

# TODO(Google Style): Add proper module-level docstring with sections:
#   - Typical usage example
#   - License information
#   - More detailed description of the module's purpose
# TODO(Security): Document security considerations for database credentials
# TODO(Testing): Add reference to test file location
"""
Professional PostgreSQL Lyrics Analyzer
========================================

A production-ready lyrics analysis system with comprehensive progress tracking,
robust error handling, and professional logging capabilities.

Features:
- Intelligent progress persistence with atomic operations
- Comprehensive error recovery and retry mechanisms
- Production-grade logging with structured output
- Performance monitoring and statistics tracking
- Graceful shutdown handling and resource cleanup
- Configurable analysis parameters and batch processing
- Database integrity verification and health checks

Usage:
    python analyzer.py                    # Analyze all unprocessed tracks
    python analyzer.py --demo             # Run demonstration
    python analyzer.py --batch-size 25    # Custom batch size
    python analyzer.py --max-tracks 1000  # Limit analysis count
    python analyzer.py --resume           # Force resume from checkpoint
    python analyzer.py --verify           # Verify database integrity

Author: Professional Development Team
Version: 2.0.0
"""

# TODO(Code Quality): Consider splitting this 1736-line file into multiple
#   modules following Single Responsibility Principle:
#   - config.py (DatabaseConfig, constants)
#   - logging_config.py (StructuredFormatter, setup_logging)
#   - models.py (LyricsFeatures, AnalysisProgress)
#   - analyzer.py (LyricsAnalyzer)
#   - database.py (PostgreSQLManager)
#   - engine.py (AnalysisEngine)
#   - cli.py (main, argument parser)

# TODO(Google Style): Group imports according to PEP 8 and Google Style Guide:
#   1. Standard library imports (alphabetically)
#   2. Related third party imports (alphabetically)
#   3. Local application/library specific imports (alphabetically)
#   Each group should be separated by a blank line
import argparse
import asyncio
import json
import logging
import os

# Core analysis imports
import re
import signal
import sys
import time
import traceback
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

# TODO(Type Hints): Add TYPE_CHECKING import for forward references
# from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator

# PostgreSQL imports with fallback
# TODO(Error Handling): Don't use sys.exit(1) at module level - this makes
#   the module untestable and unusable as a library. Raise ImportError instead
#   and handle it in main()
# TODO(Logging): Use logging instead of print() for error messages
try:
    import asyncpg

    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("ERROR: PostgreSQL dependencies not installed")
    print("Install with: pip install asyncpg")
    sys.exit(1)

# ===== Configuration and Constants =====

# TODO(Code Quality): Define module-level constants for magic numbers
# MIN_LINES_FOR_RHYME_ANALYSIS = 2
# MIN_LYRICS_LENGTH = 50
# DEFAULT_BATCH_SIZE = 50
# MAX_TRACK_ID_CACHE_SIZE = 1000
# SYLLABLE_ESTIMATION_MIN_COUNT = 1
# TTR_ADJUSTMENT_THRESHOLD = 100
# COMPLEX_WORD_MIN_LENGTH = 6
# COMPLEX_WORD_MIN_SYLLABLES = 3

# TODO(Code Quality): Define SQL query constants to avoid duplication
# TODO(Security): Use parameterized queries - never string interpolation

# TODO(Naming): Following Google Style Guide, use UPPER_CASE for module constants


@dataclass
class DatabaseConfig:
    """PostgreSQL connection configuration with validation"""

    host: str = "localhost"
    port: int = 5432
    database: str = "rap_lyrics"
    username: str = "rap_user"
    password: str = ""
    max_connections: int = 20
    min_connections: int = 5
    connection_timeout: int = 30
    command_timeout: int = 60

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Load configuration from environment variables or config_loader.

        TODO(Google Style): Add proper Google-style docstring with Args/Returns/Raises sections.
        TODO(i18n): Remove Russian comments - use English only per Google Style Guide.
        TODO(Error Handling): Don't catch broad (ImportError, AttributeError) - be specific.
        TODO(Logging): Add logging for which config source is being used.
        TODO(Type Hints): Add return type annotation for better IDE support.
        TODO(Security): Document that password must come from secure source (env var or config).

        Returns:
            DatabaseConfig instance populated from environment or config file.

        Raises:
            ValueError: If required configuration values are missing.
        """
        # TODO(i18n): Translate comment to English
        # Попытка использовать новую систему config_loader
        try:
            from config.config_loader import get_config

            config_obj = get_config()
            db_config = config_obj.database
            return cls(
                host=db_config.host,
                port=db_config.port,
                database=db_config.database,
                username=db_config.username,
                password=db_config.password,
                max_connections=db_config.pool_size or 20,
                min_connections=db_config.min_pool_size or 5,
                connection_timeout=db_config.timeout or 30,
                command_timeout=int(os.getenv("POSTGRES_COMMAND_TIMEOUT", "60")),
            )
        except (ImportError, AttributeError):
            # TODO(i18n): Translate comment to English
            # Fallback на environment variables - БЕЗ hardcoded секретов!
            # TODO(Security): Empty password default is unsafe - should raise error if not set
            # TODO(Magic Numbers): Extract default values to class constants
            return cls(
                host=os.getenv("POSTGRES_HOST") or "localhost",
                port=int(os.getenv("POSTGRES_PORT", "5432")),
                database=os.getenv("POSTGRES_DATABASE", "rap_lyrics"),
                username=os.getenv("POSTGRES_USERNAME", "rap_user"),
                password=os.getenv("POSTGRES_PASSWORD") or "",  # ОБЯЗАТЕЛЬНО из ENV!
                max_connections=int(os.getenv("POSTGRES_MAX_CONNECTIONS", "20")),
                min_connections=int(os.getenv("POSTGRES_MIN_CONNECTIONS", "5")),
                connection_timeout=int(os.getenv("POSTGRES_CONNECTION_TIMEOUT", "30")),
                command_timeout=int(os.getenv("POSTGRES_COMMAND_TIMEOUT", "60")),
            )

    def validate(self) -> list[str]:
        """Validate configuration parameters.

        Args:
            None

        Returns:
            list[str]: List of validation error messages. Empty if valid.

        Example:
            >>> config = DatabaseConfig()
            >>> errors = config.validate()
            >>> if errors:
            ...     print(errors)
        """
        errors = []
        if not self.host:
            errors.append("Database host is required")
        if not 1 <= self.port <= 65535:
            errors.append("Port must be between 1-65535")
        if not self.database:
            errors.append("Database name is required")
        if not self.username:
            errors.append("Username is required")
        if not self.password:
            errors.append("Database password is required")
        if self.max_connections <= self.min_connections:
            errors.append("max_connections must be greater than min_connections")
        return errors


@dataclass
class AnalysisProgress:
    """Progress tracking with comprehensive state management"""

    session_id: str
    last_processed_id: int = 0
    total_processed: int = 0
    total_errors: int = 0
    session_start: str = ""
    last_update: str = ""
    processing_rate: float = 0.0  # tracks per second
    estimated_completion: str | None = None
    error_details: list[dict] = None
    batch_statistics: dict = None

    def __post_init__(self):
        if self.error_details is None:
            self.error_details = []
        if self.batch_statistics is None:
            self.batch_statistics = {"successful_batches": 0, "failed_batches": 0}

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisProgress":
        """Create from dictionary"""
        return cls(**data)

    def calculate_eta(self, remaining_tracks: int) -> datetime | None:
        """Calculate estimated time of completion"""
        if self.processing_rate <= 0 or remaining_tracks <= 0:
            return None

        eta_seconds = remaining_tracks / self.processing_rate
        return datetime.now() + timedelta(seconds=eta_seconds)


# ===== Logging Configuration =====


class StructuredFormatter(logging.Formatter):
    """Structured logging formatter for production environments.

    TODO(Google Style): Add comprehensive class docstring with:
      - Detailed description
      - Attributes section
      - Example usage
    TODO(Performance): Consider caching the field exclusion set
    TODO(Type Hints): Add type hints for all methods
    """

    # TODO(Code Quality): Extract field exclusion list to class constant
    _EXCLUDED_FIELDS = frozenset([
        "name", "msg", "args", "levelname", "levelno", "pathname",
        "filename", "module", "lineno", "funcName", "created", "msecs",
        "relativeCreated", "thread", "threadName", "processName",
        "process", "getMessage", "exc_info", "exc_text", "stack_info",
    ])

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        TODO(Google Style): Add Args/Returns/Raises sections
        TODO(Time): Use UTC explicitly - datetime.now() uses local time
        TODO(Security): Sanitize message content to prevent log injection
        TODO(Performance): Consider using orjson for faster JSON serialization
        """
        # TODO(Time): This appends 'Z' but uses local time - should use datetime.utcnow()
        #   or datetime.now(timezone.utc) for true UTC timestamps
        log_entry = {
            "timestamp": datetime.now().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # TODO(Performance): Use set lookup instead of list for O(1) instead of O(n)
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in self._EXCLUDED_FIELDS:
                # TODO(Naming): Consider using a separator other than underscore
                #   to avoid conflicts with actual field names
                log_entry["extra_" + key] = value

        return json.dumps(log_entry, ensure_ascii=False, default=str)


def setup_logging(
    log_level: str = "INFO", log_file: str | None = None
) -> logging.Logger:
    """Configure production-grade logging.

    TODO(Google Style): Add comprehensive docstring with Args/Returns/Raises
    TODO(Side Effects): Function modifies global state (root logger) - consider
      returning configured logger instead or document side effects clearly
    TODO(Type Hints): Use typing.Optional or pathlib.Path for log_file
    TODO(Error Handling): Add try/except for mkdir and file operations
    TODO(Testing): Current design makes unit testing difficult due to global state
    TODO(Magic Strings): Extract "logs" directory name to constant
    TODO(Validation): Validate log_level is valid before using getattr

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to log file. Auto-generated if not provided.

    Returns:
        Configured logger instance for the module.

    Raises:
        ValueError: If log_level is invalid.
        OSError: If log directory cannot be created or file cannot be opened.
    """
    # TODO(Validation): Validate log_level before using getattr to prevent AttributeError
    # TODO(Magic Strings): Extract to constant: LOG_DIR = Path("logs")
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # TODO(Side Effects): Modifying root logger affects all other code using logging
    #   Consider using a named logger instead: logging.getLogger(__name__)
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # TODO(Side Effects): Clearing handlers may affect other parts of the application
    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler with structured format
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = StructuredFormatter()
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # TODO(Type Hints): log_file type should be str | Path | None
    # File handler
    if not log_file:
        # TODO(Time): Use UTC timestamp for consistency with structured logs
        log_file = (
            log_dir / f"lyrics_analyzer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

    # TODO(Error Handling): Add try/except for FileHandler creation
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_formatter = StructuredFormatter()
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    return logging.getLogger(__name__)


# ===== Analysis Models =====


class LyricsFeatures(BaseModel):
    """Comprehensive lyrics features with validation"""

    # Rhyme analysis
    rhyme_density: float = Field(ge=0.0, le=1.0)
    perfect_rhymes: int = Field(ge=0)
    internal_rhymes: int = Field(ge=0)
    alliteration_score: float = Field(ge=0.0, le=1.0)
    rhyme_scheme: str = Field(max_length=50)

    # Vocabulary analysis
    ttr_score: float = Field(ge=0.0, le=1.0)
    unique_words: int = Field(ge=0)
    total_words: int = Field(ge=0)
    average_word_length: float = Field(ge=0.0)
    complex_words_ratio: float = Field(ge=0.0, le=1.0)

    # Metaphor and creativity
    metaphor_count: int = Field(ge=0)
    wordplay_instances: int = Field(ge=0)
    creativity_score: float = Field(ge=0.0, le=1.0)

    # Flow analysis
    syllable_count: int = Field(ge=0)
    average_syllables_per_line: float = Field(ge=0.0)
    flow_consistency: float = Field(ge=0.0, le=1.0)
    flow_breaks: int = Field(ge=0)

    # Composite scores
    overall_complexity: float = Field(ge=0.0, le=1.0)
    artistic_sophistication: float = Field(ge=0.0, le=1.0)
    technical_skill: float = Field(ge=0.0, le=1.0)
    innovation_score: float = Field(ge=0.0, le=1.0)

    # Analysis metadata
    confidence_score: float = Field(ge=0.0, le=1.0)
    processing_time_ms: float = Field(ge=0.0)
    analyzer_version: str = Field(default="2.0.0")

    @field_validator("rhyme_scheme")
    @classmethod
    def validate_rhyme_scheme(cls, v):
        """Validate rhyme scheme format"""
        if not re.match(r"^[A-Z]*$|^insufficient$|^unknown$", v):
            raise ValueError("Invalid rhyme scheme format")
        return v


# ===== Core Analyzer Engine =====


class LyricsAnalyzer:
    """Professional lyrics analyzer with comprehensive feature extraction.

    TODO(Google Style): Add comprehensive class docstring with:
      - Detailed description of analysis capabilities
      - Attributes section documenting all instance variables
      - Example usage section
    TODO(Code Quality): Class has too many responsibilities (SRP violation):
      - Rhyme analysis
      - Vocabulary analysis
      - Creativity analysis
      - Flow analysis
      Consider splitting into separate analyzer classes
    TODO(Performance): Compile regex patterns once in __init__ instead of on each use
    TODO(Testability): Large hardcoded data makes testing difficult - consider
      injecting these as dependencies or loading from config
    """

    # TODO(Code Quality): Extract these to module-level constants or config file
    #   for better maintainability and testing
    # TODO(i18n): Consider supporting multiple languages
    # TODO(Performance): Use frozenset for O(1) lookups instead of list
    _METAPHOR_KEYWORDS = frozenset([
        "like", "as", "such as", "similar to", "compared to", "akin to",
        "resembles", "appears as", "seems like", "looks like", "feels like",
        "sounds like", "flows like", "hits like", "cuts like", "burns like",
        "explodes like", "crashes like", "soars like", "moves like",
    ])

    # TODO(Performance): Compile regex patterns as class constants
    _WORDPLAY_PATTERNS = [
        r"\b(\w+)\s+\w*\1\w*\b",  # Word repetitions
        r"\b\w*([aeiou]{2})\w*\s+\w*\1\w*\b",  # Assonance patterns
        r"\b(\w)\w+\s+\1\w+\b",  # Alliteration
        r"\b(\w{3,})\w*\s+\w*\1\b",  # Root word repetitions
    ]

    # TODO(Code Quality): Use NLTK's stopwords instead of hardcoded list
    # TODO(Maintainability): Load from external resource file
    _STOP_WORDS = frozenset([
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "up", "about", "into",
        "through", "during", "before", "after", "above", "below",
        "between", "among", "this", "that", "these", "those", "i", "me",
        "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
        "yours", "he", "him", "his", "himself", "she", "her", "hers",
        "herself", "it", "its", "itself", "they", "them", "their",
        "theirs", "themselves", "what", "which", "who", "whom", "am",
        "is", "are", "was", "were", "be", "been", "being", "have",
        "has", "had", "having", "do", "does", "did", "doing", "will",
        "would", "could", "should", "yeah", "uh", "got", "get", "gotta",
        "wanna", "gonna", "ain", "yall", "em", "ya",
    ])

    def __init__(self):
        # TODO(Type Hints): Add type hints for all instance variables
        # TODO(Dependency Injection): Consider injecting logger for better testing
        self.logger = logging.getLogger(f"{__name__}.LyricsAnalyzer")

        # TODO(Performance): Compile regex patterns here for reuse
        # TODO(Code Quality): Move these to class constants
        # Analysis configuration
        self.metaphor_keywords = list(self._METAPHOR_KEYWORDS)
        self.wordplay_patterns = self._WORDPLAY_PATTERNS
        self.stop_words = self._STOP_WORDS

    def analyze(self, lyrics: str, track_id: int | None = None) -> LyricsFeatures:
        """Perform comprehensive lyrics analysis with timing.

        TODO(Google Style): Add full docstring with Args/Returns/Raises sections
        TODO(Validation): Validate lyrics parameter (not None, not empty, reasonable length)
        TODO(Performance): Consider caching analysis results for identical lyrics
        TODO(Error Handling): Catch specific exceptions instead of broad Exception
        TODO(Metrics): Add metrics/instrumentation for monitoring
        TODO(Type Hints): track_id should be Optional[int] explicitly

        Args:
            lyrics: Raw lyrics text to analyze.
            track_id: Optional track identifier for logging purposes.

        Returns:
            LyricsFeatures object containing all analyzed features.

        Raises:
            ValueError: If lyrics is empty or invalid.
            AnalysisError: If feature extraction fails.
        """
        # TODO(Performance): Use time.perf_counter() for more accurate timing
        start_time = time.time()

        # TODO(Error Handling): This catches ALL exceptions - be more specific
        #   Catch ValueError, AttributeError, etc. separately
        try:
            # TODO(Validation): Add input validation here
            self.logger.debug(
                "Starting analysis",
                extra={"track_id": track_id, "lyrics_length": len(lyrics)},
            )

            # TODO(Code Organization): Consider extracting to separate method
            #   _extract_features(lyrics) for better testability
            # Preprocessing
            lines = self._preprocess_lyrics(lyrics)
            words = self._tokenize_lyrics(lyrics)

            # TODO(Performance): These could potentially run in parallel
            # Core analysis components
            rhyme_features = self._analyze_rhymes(lines, words)
            vocabulary_features = self._analyze_vocabulary(words)
            creativity_features = self._analyze_creativity(lyrics, words)
            flow_features = self._analyze_flow(lines, words)

            # Calculate composite scores
            composite_scores = self._calculate_composite_scores(
                rhyme_features, vocabulary_features, creativity_features, flow_features
            )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            # TODO(Code Quality): Consider using builder pattern for complex object construction
            # Build comprehensive feature set
            features = LyricsFeatures(
                **rhyme_features,
                **vocabulary_features,
                **creativity_features,
                **flow_features,
                **composite_scores,
                processing_time_ms=processing_time,
                confidence_score=self._calculate_confidence(lines, words),
            )

            self.logger.debug(
                "Analysis completed",
                extra={
                    "track_id": track_id,
                    "processing_time_ms": processing_time,
                    "confidence_score": features.confidence_score,
                },
            )

            return features

        # TODO(Error Handling): Don't catch bare Exception - be specific
        #   Use custom exceptions like AnalysisError, ValidationError
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.logger.error(
                "Analysis failed",
                extra={
                    "track_id": track_id,
                    "error": str(e),
                    "processing_time_ms": processing_time,
                },
                exc_info=True,
            )
            raise

    def _preprocess_lyrics(self, lyrics: str) -> list[str]:
        """Clean and prepare lyrics for analysis"""
        # Remove excessive whitespace and empty lines
        lines = [line.strip() for line in lyrics.split("\n") if line.strip()]

        # Filter out metadata lines (common in lyrics)
        cleaned_lines = []
        for line in lines:
            # Skip lines that look like metadata
            if not re.match(
                r"^\[.*\]$|^\(.*\)$|^Verse \d+|^Chorus|^Bridge|^Outro",
                line,
                re.IGNORECASE,
            ):
                cleaned_lines.append(line)

        return cleaned_lines

    def _tokenize_lyrics(self, lyrics: str) -> list[str]:
        """Advanced tokenization with stop word removal"""
        # Extract words using improved regex
        words = re.findall(r"\b[a-zA-Z]{2,}\b", lyrics.lower())

        # Remove stop words and very short words
        meaningful_words = [
            word for word in words if word not in self.stop_words and len(word) >= 3
        ]

        return meaningful_words

    def _analyze_rhymes(self, lines: list[str], words: list[str]) -> dict:
        """Comprehensive rhyme analysis.

        TODO(Google Style): Add complete Args/Returns docstring
        TODO(Type Hints): Return type should be TypedDict or dataclass, not dict
        TODO(Magic Numbers): Extract 2 to constant MIN_LINES_FOR_RHYME_ANALYSIS
        TODO(Code Quality): Too many responsibilities - split into smaller methods
        TODO(Performance): O(n²) complexity in rhyme counting - consider optimization
        """
        # TODO(Magic Numbers): Extract to constant
        if len(lines) < 2:
            # TODO(Code Quality): Use a factory method or named constant for default values
            return {
                "rhyme_density": 0.0,
                "perfect_rhymes": 0,
                "internal_rhymes": 0,
                "alliteration_score": 0.0,
                "rhyme_scheme": "insufficient",
            }

        # Extract line endings
        line_endings = []
        for line in lines:
            words_in_line = line.strip().split()
            if words_in_line:
                # Clean the ending word
                ending = re.sub(r"[^\w]", "", words_in_line[-1].lower())
                if ending:
                    line_endings.append(ending)

        # Analyze rhyme scheme
        rhyme_scheme = self._detect_rhyme_scheme(line_endings)
        perfect_rhymes = self._count_perfect_rhymes(line_endings)
        internal_rhymes = self._count_internal_rhymes(lines)
        alliteration_score = self._calculate_alliteration(lines)

        # Calculate rhyme density
        total_lines = len(lines)
        rhyme_density = min(perfect_rhymes / max(total_lines / 2, 1), 1.0)

        return {
            "rhyme_density": rhyme_density,
            "perfect_rhymes": perfect_rhymes,
            "internal_rhymes": internal_rhymes,
            "alliteration_score": alliteration_score,
            "rhyme_scheme": rhyme_scheme,
        }

    def _detect_rhyme_scheme(self, endings: list[str]) -> str:
        """Detect rhyme scheme pattern with improved accuracy"""
        if len(endings) < 4:
            return "insufficient"

        # Limit analysis to first 12 lines for pattern detection
        sample_endings = endings[:12]

        scheme = []
        rhyme_groups = {}
        current_letter = "A"

        for ending in sample_endings:
            found_rhyme = False

            # Check against existing groups
            for group_ending, letter in rhyme_groups.items():
                if self._endings_rhyme(ending, group_ending):
                    scheme.append(letter)
                    found_rhyme = True
                    break

            # Create new rhyme group if no match found
            if not found_rhyme:
                rhyme_groups[ending] = current_letter
                scheme.append(current_letter)
                current_letter = chr(ord(current_letter) + 1)

        return "".join(scheme)

    def _endings_rhyme(self, word1: str, word2: str) -> bool:
        """Improved rhyme detection"""
        if len(word1) < 2 or len(word2) < 2:
            return False

        # Perfect rhyme: same ending sounds
        if word1[-2:] == word2[-2:] or word1[-3:] == word2[-3:]:
            return True

        # Near rhyme: similar vowel sounds
        vowels = "aeiou"
        word1_vowels = [c for c in word1[-3:] if c in vowels]
        word2_vowels = [c for c in word2[-3:] if c in vowels]

        if word1_vowels and word2_vowels and word1_vowels[-1] == word2_vowels[-1]:
            return True

        return False

    def _count_perfect_rhymes(self, endings: list[str]) -> int:
        """Count perfect rhyming pairs"""
        rhyme_count = 0
        for i in range(len(endings)):
            for j in range(i + 1, len(endings)):
                if self._endings_rhyme(endings[i], endings[j]):
                    rhyme_count += 1
        return rhyme_count

    def _count_internal_rhymes(self, lines: list[str]) -> int:
        """Count internal rhymes within lines"""
        internal_count = 0

        for line in lines:
            words = [re.sub(r"[^\w]", "", word.lower()) for word in line.split()]
            words = [word for word in words if len(word) >= 3]

            if len(words) < 2:
                continue

            for i in range(len(words) - 1):
                for j in range(i + 1, len(words)):
                    if self._endings_rhyme(words[i], words[j]):
                        internal_count += 1

        return internal_count

    def _calculate_alliteration(self, lines: list[str]) -> float:
        """Calculate alliteration score"""
        alliteration_count = 0
        total_word_pairs = 0

        for line in lines:
            words = [
                word.lower()
                for word in line.split()
                if len(word) >= 3 and word.isalpha()
            ]

            if len(words) < 2:
                continue

            total_word_pairs += len(words) - 1

            for i in range(len(words) - 1):
                if words[i][0] == words[i + 1][0]:
                    alliteration_count += 1

        return alliteration_count / max(total_word_pairs, 1)

    def _analyze_vocabulary(self, words: list[str]) -> dict:
        """Comprehensive vocabulary analysis"""
        if not words:
            return {
                "ttr_score": 0.0,
                "unique_words": 0,
                "total_words": 0,
                "average_word_length": 0.0,
                "complex_words_ratio": 0.0,
            }

        unique_words = len(set(words))
        total_words = len(words)

        # Type-Token Ratio with adjustment for text length
        ttr_score = unique_words / total_words

        # Adjust TTR for text length (longer texts naturally have lower TTR)
        if total_words > 100:
            ttr_score = min(ttr_score * 1.2, 1.0)

        # Calculate average word length
        average_word_length = sum(len(word) for word in words) / len(words)

        # Complex words (length > 6 or contain specific patterns)
        complex_words = 0
        for word in words:
            if len(word) > 6 or self._is_complex_word(word):
                complex_words += 1

        complex_words_ratio = complex_words / total_words

        return {
            "ttr_score": ttr_score,
            "unique_words": unique_words,
            "total_words": total_words,
            "average_word_length": average_word_length,
            "complex_words_ratio": complex_words_ratio,
        }

    def _is_complex_word(self, word: str) -> bool:
        """Identify complex words beyond just length"""
        # Words with multiple syllables or sophisticated patterns
        syllable_count = self._count_syllables(word)
        return syllable_count >= 3 or len(word) > 7

    def _analyze_creativity(self, lyrics: str, words: list[str]) -> dict:
        """Analyze creative elements like metaphors and wordplay"""
        if not words:
            return {
                "metaphor_count": 0,
                "wordplay_instances": 0,
                "creativity_score": 0.0,
            }

        # Count metaphors
        metaphor_count = 0
        lyrics_lower = lyrics.lower()
        for keyword in self.metaphor_keywords:
            metaphor_count += lyrics_lower.count(keyword)

        # Count wordplay instances
        wordplay_count = 0
        for pattern in self.wordplay_patterns:
            matches = re.findall(pattern, lyrics, re.IGNORECASE)
            wordplay_count += len(matches)

        # Calculate creativity score
        text_length_factor = len(words)
        metaphor_density = metaphor_count / max(text_length_factor / 20, 1)
        wordplay_density = wordplay_count / max(text_length_factor / 30, 1)

        creativity_score = min((metaphor_density + wordplay_density) / 2, 1.0)

        return {
            "metaphor_count": metaphor_count,
            "wordplay_instances": wordplay_count,
            "creativity_score": creativity_score,
        }

    def _analyze_flow(self, lines: list[str], words: list[str]) -> dict:
        """Analyze flow and rhythm characteristics"""
        if not lines:
            return {
                "syllable_count": 0,
                "average_syllables_per_line": 0.0,
                "flow_consistency": 0.0,
                "flow_breaks": 0,
            }

        # Count total syllables
        syllable_count = sum(self._count_syllables(word) for word in words)

        # Calculate average syllables per line
        average_syllables_per_line = syllable_count / len(lines)

        # Analyze flow consistency by looking at line length patterns
        line_syllable_counts = []
        for line in lines:
            line_words = [word for word in re.findall(r"\b\w+\b", line.lower())]
            line_syllables = sum(self._count_syllables(word) for word in line_words)
            line_syllable_counts.append(line_syllables)

        # Calculate flow consistency based on variance in line lengths
        if len(line_syllable_counts) > 1:
            mean_syllables = sum(line_syllable_counts) / len(line_syllable_counts)
            variance = sum(
                (x - mean_syllables) ** 2 for x in line_syllable_counts
            ) / len(line_syllable_counts)
            flow_consistency = max(0, 1 - (variance / (mean_syllables + 1)))
        else:
            flow_consistency = 1.0

        # Count flow breaks (punctuation that interrupts flow)
        flow_breaks = sum(
            line.count(",")
            + line.count(".")
            + line.count("!")
            + line.count("?")
            + line.count(";")
            for line in lines
        )

        return {
            "syllable_count": syllable_count,
            "average_syllables_per_line": average_syllables_per_line,
            "flow_consistency": flow_consistency,
            "flow_breaks": flow_breaks,
        }

    def _count_syllables(self, word: str) -> int:
        """Improved syllable counting for English.

        TODO(Google Style): Add Args/Returns docstring sections
        TODO(Accuracy): This is a simplified heuristic - consider using a library
          like pyphen or nltk.corpus.cmudict for accurate syllable counting
        TODO(Magic Numbers): Extract 2 to constant MIN_WORD_LENGTH_FOR_SYLLABLE_CHECK
        TODO(Performance): Consider caching results for common words
        TODO(i18n): Only works for English - document this limitation
        TODO(Testing): Add comprehensive unit tests with edge cases

        Note:
            This is a heuristic approximation and may not be accurate for all words.
            For production use, consider using CMU Pronouncing Dictionary.

        Args:
            word: English word to count syllables for.

        Returns:
            Estimated syllable count (minimum 1).
        """
        # TODO(Validation): Add input validation for empty string
        word = word.lower()

        # TODO(Magic Numbers): Extract to constant
        if len(word) <= 2:
            return 1

        # TODO(Accuracy): Silent 'e' rule has many exceptions (e.g., 'cafe', 'lee')
        # Remove silent e at end
        if word.endswith("e") and len(word) > 2:
            word = word[:-1]

        # TODO(Code Quality): Extract vowels to class constant
        # Count vowel groups
        vowels = "aeiouy"
        syllables = 0
        prev_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllables += 1
            prev_was_vowel = is_vowel

        # Minimum of 1 syllable
        return max(syllables, 1)

    def _calculate_composite_scores(
        self,
        rhyme_features: dict,
        vocabulary_features: dict,
        creativity_features: dict,
        flow_features: dict,
    ) -> dict:
        """Calculate composite sophistication scores.

        TODO(Google Style): Add complete Args/Returns docstring
        TODO(Type Hints): Use TypedDict for dict parameters and return type
        TODO(Magic Numbers): Extract all weights (0.25, 0.4, etc.) to named constants
        TODO(Configuration): Move weights to config file for tuning without code changes
        TODO(Validation): Validate that all input dicts have required keys
        TODO(Testing): Add unit tests with known input/output pairs
        TODO(Documentation): Document the rationale behind these specific weights

        Note:
            Weights for composite scores are empirically derived and may need
            tuning based on domain-specific requirements.
        """
        # TODO(Magic Numbers): Extract weights to constants:
        #   OVERALL_COMPLEXITY_WEIGHTS = {"rhyme": 0.25, "vocabulary": 0.25, ...}
        # Overall complexity: balanced mix of all factors
        overall_complexity = (
            rhyme_features["rhyme_density"] * 0.25
            + vocabulary_features["ttr_score"] * 0.25
            + creativity_features["creativity_score"] * 0.25
            + flow_features["flow_consistency"] * 0.25
        )

        # TODO(Magic Numbers): Extract 0.4, 0.3 to named constants
        # Artistic sophistication: emphasis on creativity and vocabulary
        artistic_sophistication = (
            creativity_features["creativity_score"] * 0.4
            + vocabulary_features["ttr_score"] * 0.3
            + rhyme_features["alliteration_score"] * 0.3
        )

        # TODO(Magic Numbers): Extract 0.4, 0.35, 0.25, 2 to named constants
        # Technical skill: emphasis on rhyme and flow
        technical_skill = (
            rhyme_features["rhyme_density"] * 0.4
            + flow_features["flow_consistency"] * 0.35
            + min(vocabulary_features["complex_words_ratio"] * 2, 1.0) * 0.25
        )

        # TODO(Magic Numbers): Extract 0.5, 10, 0.3, 1.5, 0.2 to named constants
        # Innovation score: unique elements and creative patterns
        innovation_score = (
            creativity_features["creativity_score"] * 0.5
            + min(rhyme_features["internal_rhymes"] / 10, 1.0) * 0.3
            + min(vocabulary_features["complex_words_ratio"] * 1.5, 1.0) * 0.2
        )

        return {
            "overall_complexity": overall_complexity,
            "artistic_sophistication": artistic_sophistication,
            "technical_skill": technical_skill,
            "innovation_score": innovation_score,
        }

    def _calculate_confidence(self, lines: list[str], words: list[str]) -> float:
        """Calculate confidence in analysis results"""
        confidence_factors = []

        # Text length factor
        text_length_factor = min(len(words) / 50, 1.0) if words else 0
        confidence_factors.append(text_length_factor)

        # Line count factor
        line_count_factor = min(len(lines) / 8, 1.0) if lines else 0
        confidence_factors.append(line_count_factor)

        # Content quality factor (based on word variety)
        if words:
            unique_ratio = len(set(words)) / len(words)
            content_quality_factor = min(unique_ratio * 1.5, 1.0)
        else:
            content_quality_factor = 0
        confidence_factors.append(content_quality_factor)

        return sum(confidence_factors) / len(confidence_factors)


# ===== Database Manager =====


class PostgreSQLManager:
    """Professional PostgreSQL connection manager with comprehensive error handling.

    TODO(Google Style): Add complete class docstring with Attributes/Example sections
    TODO(Thread Safety): Document thread-safety guarantees of this class
    TODO(Resource Management): Implement __enter__/__exit__ for context manager support
    TODO(Testing): Add integration tests with test database
    """

    def __init__(self, config: DatabaseConfig | None = None):
        """Initialize PostgreSQL manager.

        TODO(Google Style): Add complete Args/Raises docstring
        TODO(Dependency Injection): Consider requiring config parameter instead of
          defaulting to from_env() for better testability
        TODO(Type Hints): Add explicit type hints for all attributes
        """
        # TODO(Code Quality): This side effect in __init__ makes testing difficult
        self.config = config or DatabaseConfig.from_env()
        self.logger = logging.getLogger(f"{__name__}.PostgreSQLManager")
        self.connection_pool: asyncpg.Pool | None = None
        # TODO(Documentation): Document purpose of _shutdown_event
        self._shutdown_event = asyncio.Event()

        # TODO(Error Handling): Validation in __init__ can make object construction fail
        #   Consider lazy validation or separate validate() method
        # Validate configuration
        config_errors = self.config.validate()
        if config_errors:
            raise ValueError(
                f"Database configuration errors: {', '.join(config_errors)}"
            )

    async def initialize(self) -> bool:
        """Initialize connection pool with comprehensive error handling.

        TODO(Google Style): Add complete docstring with Returns/Raises sections
        TODO(Error Handling): Return bool is antipattern - raise exception on failure
        TODO(Retry Logic): Add connection retry logic with exponential backoff
        TODO(Metrics): Add connection pool metrics/monitoring

        Returns:
            True if initialization successful, False otherwise.

        Raises:
            asyncpg.PostgresError: If database connection fails.
            ValueError: If configuration is invalid.
        """
        # TODO(Error Handling): Catching broad Exception hides specific errors
        try:
            # TODO(Security): Don't log password even in extra fields
            self.logger.info(
                "Initializing PostgreSQL connection pool",
                extra={
                    "host": self.config.host,
                    "port": self.config.port,
                    "database": self.config.database,
                    "max_connections": self.config.max_connections,
                },
            )

            # TODO(Security): DSN contains password in plain text - risk of logging
            #   Use separate connection parameters instead of DSN string
            # TODO(Security): If this DSN is ever logged, password will be exposed
            # Create connection pool
            dsn = f"postgresql://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"

            # TODO(Configuration): Extract application_name to config constant
            # TODO(Configuration): Consider making timezone configurable
            self.connection_pool = await asyncpg.create_pool(
                dsn,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                command_timeout=self.config.command_timeout,
                server_settings={
                    "application_name": "lyrics_analyzer_v2",
                    "timezone": "UTC",
                },
            )

            # Test connection
            async with self.connection_pool.acquire() as conn:
                await conn.execute("SELECT 1")

            self.logger.info(
                "PostgreSQL connection pool initialized successfully",
                extra={
                    "pool_min_size": self.config.min_connections,
                    "pool_max_size": self.config.max_connections,
                },
            )

            return True

        except Exception as e:
            self.logger.error(
                "Failed to initialize PostgreSQL connection pool",
                extra={
                    "error": str(e),
                    "host": self.config.host,
                    "port": self.config.port,
                    "database": self.config.database,
                },
                exc_info=True,
            )
            return False

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Get database connection with automatic cleanup"""
        if not self.connection_pool:
            raise RuntimeError("Database connection pool not initialized")

        connection = None
        try:
            connection = await self.connection_pool.acquire()
            yield connection
        finally:
            if connection:
                await self.connection_pool.release(connection)

    async def execute_query(
        self, query: str, *args, fetch_one: bool = False, fetch_all: bool = False
    ):
        """Execute query with error handling and logging"""
        start_time = time.time()

        try:
            async with self.get_connection() as conn:
                if fetch_one:
                    result = await conn.fetchrow(query, *args)
                elif fetch_all:
                    result = await conn.fetch(query, *args)
                else:
                    result = await conn.execute(query, *args)

                execution_time = (time.time() - start_time) * 1000
                self.logger.debug(
                    "Query executed successfully",
                    extra={
                        "query": query[:100] + "..." if len(query) > 100 else query,
                        "execution_time_ms": execution_time,
                        "args_count": len(args),
                    },
                )

                return result

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.logger.error(
                "Query execution failed",
                extra={
                    "query": query[:100] + "..." if len(query) > 100 else query,
                    "error": str(e),
                    "execution_time_ms": execution_time,
                    "args_count": len(args),
                },
                exc_info=True,
            )
            raise

    async def get_database_stats(self) -> dict:
        """Get comprehensive database statistics"""
        try:
            stats_query = """
                SELECT 
                    (SELECT COUNT(*) FROM tracks WHERE lyrics IS NOT NULL AND LENGTH(TRIM(lyrics)) > 50) as total_tracks,
                    (SELECT COUNT(*) FROM analysis_results WHERE analyzer_type = 'simplified_features_v2') as analyzed_tracks,
                    (SELECT AVG(processing_time_ms) FROM analysis_results WHERE analyzer_type = 'simplified_features_v2') as avg_processing_time,
                    (SELECT MAX(id) FROM tracks) as max_track_id,
                    (SELECT pg_size_pretty(pg_database_size(current_database()))) as database_size
            """

            result = await self.execute_query(stats_query, fetch_one=True)

            if result:
                stats = dict(result)
                # Calculate analysis percentage
                if stats["total_tracks"] and stats["analyzed_tracks"]:
                    stats["analysis_percentage"] = (
                        stats["analyzed_tracks"] / stats["total_tracks"]
                    ) * 100
                else:
                    stats["analysis_percentage"] = 0.0

                return stats

            return {}

        except Exception:
            self.logger.error("Failed to get database statistics", exc_info=True)
            return {}

    async def close(self):
        """Close connection pool gracefully"""
        if self.connection_pool:
            self.logger.info("Closing PostgreSQL connection pool")
            await self.connection_pool.close()
            self.connection_pool = None
            self.logger.info("PostgreSQL connection pool closed")


# ===== Analysis Engine =====


class AnalysisEngine:
    """Main analysis engine with progress tracking and error recovery"""

    def __init__(self, config: DatabaseConfig | None = None):
        self.db_manager = PostgreSQLManager(config)
        self.analyzer = LyricsAnalyzer()
        self.logger = logging.getLogger(f"{__name__}.AnalysisEngine")

        # Progress tracking
        self.progress_file = Path("results/analysis_progress_v2.json")
        self.current_progress: AnalysisProgress | None = None

        # Statistics tracking
        self.session_stats = {
            "start_time": datetime.now(),
            "tracks_processed": 0,
            "tracks_failed": 0,
            "total_processing_time": 0.0,
            "batches_completed": 0,
            "errors": [],
        }

        # Shutdown handling
        self._shutdown_requested = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers.

        TODO(Google Style): Add complete docstring
        TODO(Testing): Signal handlers are difficult to test - consider dependency injection
        TODO(Thread Safety): Modifying _shutdown_requested from signal handler without lock
        TODO(Code Quality): Nested function makes testing impossible - extract to method
        TODO(Portability): SIGTERM not available on Windows - add platform check
        TODO(Best Practice): Use asyncio signal handlers instead of signal.signal
        """
        # TODO(Testing): This nested function cannot be tested in isolation
        # TODO(Thread Safety): Need threading.Lock or asyncio.Lock for thread-safe access
        def signal_handler(signum, frame):
            # TODO(Logging): f-string in logging is inefficient - use % or .format()
            self.logger.info(
                f"Received shutdown signal {signum}, initiating graceful shutdown"
            )
            # TODO(Thread Safety): This write is not thread-safe
            self._shutdown_requested = True

        # TODO(Portability): Windows doesn't support SIGTERM
        # TODO(Best Practice): Use loop.add_signal_handler() for async code
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def initialize(self) -> bool:
        """Initialize the analysis engine"""
        self.logger.info("Initializing analysis engine")

        # Create results directory if it doesn't exist
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        # Initialize database connection
        if not await self.db_manager.initialize():
            self.logger.error("Failed to initialize database connection")
            return False

        # Load existing progress
        self.current_progress = self._load_progress()

        # Log initialization success
        self.logger.info(
            "Analysis engine initialized successfully",
            extra={
                "session_id": self.current_progress.session_id,
                "last_processed_id": self.current_progress.last_processed_id,
            },
        )

        return True

    def _load_progress(self) -> AnalysisProgress:
        """Load progress from file or create new session"""
        try:
            if self.progress_file.exists():
                with open(self.progress_file, encoding="utf-8") as f:
                    data = json.load(f)

                progress = AnalysisProgress.from_dict(data)

                # Check if we should start a new session
                last_update = datetime.fromisoformat(progress.last_update)
                if datetime.now() - last_update > timedelta(hours=24):
                    self.logger.info(
                        "Starting new session (previous session > 24h old)"
                    )
                    return self._create_new_progress()

                self.logger.info(
                    "Loaded existing progress",
                    extra={
                        "session_id": progress.session_id,
                        "last_processed_id": progress.last_processed_id,
                        "total_processed": progress.total_processed,
                    },
                )

                return progress

        except Exception as e:
            self.logger.warning(f"Could not load progress file: {e}")

        return self._create_new_progress()

    def _create_new_progress(self) -> AnalysisProgress:
        """Create new progress session"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        progress = AnalysisProgress(
            session_id=session_id,
            session_start=datetime.now().isoformat(),
            last_update=datetime.now().isoformat(),
        )

        self.logger.info(
            "Created new analysis session", extra={"session_id": session_id}
        )
        return progress

    def _save_progress(self):
        """Save progress to file with atomic write"""
        try:
            self.current_progress.last_update = datetime.now().isoformat()

            # Calculate processing rate
            if self.current_progress.total_processed > 0:
                session_start = datetime.fromisoformat(
                    self.current_progress.session_start
                )
                elapsed_seconds = (datetime.now() - session_start).total_seconds()
                self.current_progress.processing_rate = (
                    self.current_progress.total_processed / max(elapsed_seconds, 1)
                )

            # Atomic write using temporary file
            temp_file = self.progress_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(
                    self.current_progress.to_dict(), f, indent=2, ensure_ascii=False
                )

            # Atomic rename
            temp_file.replace(self.progress_file)

            self.logger.debug(
                "Progress saved successfully",
                extra={
                    "last_processed_id": self.current_progress.last_processed_id,
                    "total_processed": self.current_progress.total_processed,
                    "processing_rate": self.current_progress.processing_rate,
                },
            )

        except Exception as e:
            self.logger.error(f"Failed to save progress: {e}", exc_info=True)

    async def analyze_all_tracks(
        self, batch_size: int = 50, max_tracks: int | None = None
    ) -> dict:
        """Analyze all unprocessed tracks with comprehensive progress tracking"""

        self.logger.info(
            "Starting full database analysis",
            extra={
                "batch_size": batch_size,
                "max_tracks": max_tracks,
                "resume_from_id": self.current_progress.last_processed_id,
            },
        )

        # Get initial statistics
        initial_stats = await self.db_manager.get_database_stats()
        self.logger.info("Initial database statistics", extra=initial_stats)

        total_processed_this_session = 0
        total_errors_this_session = 0
        batch_number = 0

        try:
            while not self._shutdown_requested:
                batch_number += 1

                # Get next batch of tracks
                tracks = await self._get_next_batch(batch_size)
                if not tracks:
                    self.logger.info("No more tracks to process")
                    break

                # Check max_tracks limit
                if max_tracks and total_processed_this_session >= max_tracks:
                    self.logger.info(f"Reached max_tracks limit: {max_tracks}")
                    break

                self.logger.info(
                    f"Processing batch {batch_number}",
                    extra={
                        "batch_size": len(tracks),
                        "first_track_id": tracks[0]["id"],
                        "last_track_id": tracks[-1]["id"],
                    },
                )

                # Process batch
                batch_results = await self._process_batch(tracks)

                # Update progress
                total_processed_this_session += batch_results["processed"]
                total_errors_this_session += batch_results["errors"]

                self.current_progress.last_processed_id = tracks[-1]["id"]
                self.current_progress.total_processed += batch_results["processed"]
                self.current_progress.total_errors += batch_results["errors"]
                self.current_progress.batch_statistics["successful_batches"] += (
                    1 if batch_results["processed"] > 0 else 0
                )
                self.current_progress.batch_statistics["failed_batches"] += (
                    1 if batch_results["errors"] > 0 else 0
                )

                # Save progress
                self._save_progress()

                # Log progress
                self.logger.info(
                    "Batch completed",
                    extra={
                        "batch_number": batch_number,
                        "batch_processed": batch_results["processed"],
                        "batch_errors": batch_results["errors"],
                        "session_total_processed": total_processed_this_session,
                        "session_total_errors": total_errors_this_session,
                        "processing_rate": self.current_progress.processing_rate,
                    },
                )

                # Small delay between batches
                if not self._shutdown_requested:
                    await asyncio.sleep(0.1)

        except Exception:
            self.logger.error("Analysis loop failed", exc_info=True)
            raise

        # Final statistics
        final_stats = await self.db_manager.get_database_stats()

        results = {
            "session_processed": total_processed_this_session,
            "session_errors": total_errors_this_session,
            "total_batches": batch_number,
            "initial_stats": initial_stats,
            "final_stats": final_stats,
            "session_id": self.current_progress.session_id,
            "processing_rate": self.current_progress.processing_rate,
            "shutdown_requested": self._shutdown_requested,
        }

        self.logger.info("Analysis completed", extra=results)
        return results

    async def _get_next_batch(self, batch_size: int) -> list[dict]:
        """Get next batch of unprocessed tracks"""
        query = """
            SELECT t.id, t.title, t.artist, t.lyrics
            FROM tracks t
            LEFT JOIN analysis_results ar ON t.id = ar.track_id 
                AND ar.analyzer_type = 'simplified_features_v2'
            WHERE t.id > $1
                AND t.lyrics IS NOT NULL
                AND LENGTH(TRIM(t.lyrics)) > 50
                AND ar.id IS NULL
            ORDER BY t.id
            LIMIT $2
        """

        try:
            result = await self.db_manager.execute_query(
                query,
                self.current_progress.last_processed_id,
                batch_size,
                fetch_all=True,
            )

            return [dict(row) for row in result] if result else []

        except Exception:
            self.logger.error(
                "Failed to get next batch",
                extra={
                    "last_processed_id": self.current_progress.last_processed_id,
                    "batch_size": batch_size,
                },
                exc_info=True,
            )
            return []

    async def _process_batch(self, tracks: list[dict]) -> dict[str, int]:
        """Process a batch of tracks with comprehensive error handling"""
        processed = 0
        errors = 0
        batch_start_time = time.time()

        try:
            async with self.db_manager.get_connection() as conn:
                async with conn.transaction():
                    for track in tracks:
                        if self._shutdown_requested:
                            self.logger.info(
                                "Shutdown requested, stopping batch processing"
                            )
                            break

                        try:
                            # Analyze track
                            features = self.analyzer.analyze(
                                track["lyrics"], track["id"]
                            )

                            # Save results
                            await self._save_analysis_result(conn, track, features)
                            processed += 1

                            # Update session statistics
                            self.session_stats["tracks_processed"] += 1
                            self.session_stats["total_processing_time"] += (
                                features.processing_time_ms
                            )

                        except Exception as e:
                            errors += 1
                            error_info = {
                                "track_id": track["id"],
                                "artist": track.get("artist", "Unknown"),
                                "title": track.get("title", "Unknown"),
                                "error": str(e),
                                "timestamp": datetime.now().isoformat(),
                            }

                            self.session_stats["errors"].append(error_info)
                            self.session_stats["tracks_failed"] += 1

                            self.logger.warning(
                                "Track processing failed", extra=error_info
                            )

        except Exception:
            self.logger.error(
                "Batch processing failed",
                extra={
                    "batch_size": len(tracks),
                    "processed": processed,
                    "errors": errors,
                },
                exc_info=True,
            )
            errors += len(tracks) - processed

        batch_time = (time.time() - batch_start_time) * 1000
        self.session_stats["batches_completed"] += 1

        self.logger.debug(
            "Batch processing completed",
            extra={
                "processed": processed,
                "errors": errors,
                "batch_time_ms": batch_time,
                "avg_time_per_track": batch_time / len(tracks) if tracks else 0,
            },
        )

        return {"processed": processed, "errors": errors}

    async def _save_analysis_result(
        self, conn: asyncpg.Connection, track: dict, features: LyricsFeatures
    ):
        """Save analysis results to database"""
        query = """
            INSERT INTO analysis_results (
                track_id, analyzer_type, sentiment, confidence,
                complexity_score, themes, analysis_data,
                processing_time_ms, model_version, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """

        analysis_data = {
            "features": features.model_dump(),
            "metadata": {
                "analyzer_version": features.analyzer_version,
                "analysis_timestamp": datetime.now().isoformat(),
                "track_info": {
                    "artist": track.get("artist"),
                    "title": track.get("title"),
                },
            },
        }

        await conn.execute(
            query,
            track["id"],  # track_id
            "simplified_features_v2",  # analyzer_type
            None,  # sentiment (not used in this analyzer)
            features.confidence_score,  # confidence
            features.overall_complexity,  # complexity_score
            json.dumps(["rap", "hip-hop", "lyrics"]),  # themes
            json.dumps(analysis_data),  # analysis_data
            features.processing_time_ms,  # processing_time_ms
            features.analyzer_version,  # model_version
            datetime.now(),  # created_at
        )

    async def close(self):
        """Clean shutdown of analysis engine"""
        self.logger.info("Shutting down analysis engine")

        # Save final progress
        if self.current_progress:
            self._save_progress()

        # Close database connections
        await self.db_manager.close()

        # Log session summary
        session_duration = datetime.now() - self.session_stats["start_time"]
        self.logger.info(
            "Analysis engine shutdown complete",
            extra={
                "session_duration": str(session_duration),
                "tracks_processed": self.session_stats["tracks_processed"],
                "tracks_failed": self.session_stats["tracks_failed"],
                "batches_completed": self.session_stats["batches_completed"],
                "total_errors": len(self.session_stats["errors"]),
            },
        )


# ===== CLI Interface =====


def create_argument_parser() -> argparse.ArgumentParser:
    """Create comprehensive argument parser"""
    parser = argparse.ArgumentParser(
        description="Professional PostgreSQL Lyrics Analyzer v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analyzer.py                     # Analyze all unprocessed tracks
    python analyzer.py --demo              # Run demonstration analysis
    python analyzer.py --batch-size 25     # Use smaller batch size
    python analyzer.py --max-tracks 1000   # Limit analysis to 1000 tracks
    python analyzer.py --resume            # Force resume from last checkpoint
    python analyzer.py --verify            # Verify database integrity
        """,
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration analysis on sample lyrics",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        metavar="N",
        help="Number of tracks to process per batch (default: 50)",
    )

    parser.add_argument(
        "--max-tracks",
        type=int,
        metavar="N",
        help="Maximum number of tracks to analyze (default: unlimited)",
    )

    parser.add_argument(
        "--resume", action="store_true", help="Force resume from last checkpoint"
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify database integrity and show statistics",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )

    parser.add_argument(
        "--log-file", metavar="FILE", help="Log file path (default: auto-generated)"
    )

    parser.add_argument(
        "--config", metavar="FILE", help="Configuration file path (JSON format)"
    )

    return parser


async def run_demo_analysis():
    """Run demonstration analysis.

    TODO(Google Style): Add complete docstring with description
    TODO(Logging): Use logger instead of print() for output
    TODO(Code Quality): Hardcoded sample lyrics - load from file or config
    TODO(Testing): Add this as a proper test case instead of demo function
    TODO(User Experience): Consider adding demo mode parameter to control output format
    TODO(i18n): Add emojis conditionally or remove for professional code
    """
    # TODO(Logging): Replace print() with logger.info() throughout this function
    # TODO(User Experience): Magic number 60 - extract to constant SEPARATOR_WIDTH
    print("\n" + "=" * 60)
    print("PROFESSIONAL LYRICS ANALYZER v2.0 - DEMO")
    print("=" * 60)

    sample_lyrics = """
    Rising through the city like smoke in the night
    My words cut deeper than a samurai's blade
    In this maze of concrete dreams, I found my light
    Rhymes cascade like waterfalls, never to fade

    Money talks but wisdom whispers in the dark
    In this chess game of life, I'm moving my pieces
    My flow ignites souls, leaves a permanent mark
    While lesser rappers crumble like autumn leaves

    Time is currency, I invest in every line
    Building empires from metaphors and beats
    My voice echoes through dimensions, truly divine
    Where poetry meets rhythm, that's where genius meets
    """

    try:
        analyzer = LyricsAnalyzer()
        features = analyzer.analyze(sample_lyrics)

        # TODO(Code Quality): Remove emojis from professional code or make conditional
        # TODO(Logging): Use logger instead of print
        print("\n🎵 ANALYSIS RESULTS:")
        print(f"  Rhyme Density: {features.rhyme_density:.3f}")
        print(f"  Rhyme Scheme: {features.rhyme_scheme}")
        print(f"  Perfect Rhymes: {features.perfect_rhymes}")
        print(f"  Internal Rhymes: {features.internal_rhymes}")
        print(f"  Alliteration Score: {features.alliteration_score:.3f}")

        # TODO(Code Quality): Remove emojis - not professional for FAANG standards
        print("\n📚 VOCABULARY:")
        print(f"  TTR Score: {features.ttr_score:.3f}")
        print(f"  Unique Words: {features.unique_words}")
        print(f"  Total Words: {features.total_words}")
        print(f"  Avg Word Length: {features.average_word_length:.1f}")
        print(f"  Complex Words: {features.complex_words_ratio:.3f}")

        # TODO(Code Quality): Remove emojis
        print("\n🎨 CREATIVITY:")
        print(f"  Metaphor Count: {features.metaphor_count}")
        print(f"  Wordplay Instances: {features.wordplay_instances}")
        print(f"  Creativity Score: {features.creativity_score:.3f}")

        # TODO(Code Quality): Remove emojis
        print("\n🎯 FLOW:")
        print(f"  Syllable Count: {features.syllable_count}")
        print(f"  Avg Syllables/Line: {features.average_syllables_per_line:.1f}")
        print(f"  Flow Consistency: {features.flow_consistency:.3f}")
        print(f"  Flow Breaks: {features.flow_breaks}")

        # TODO(Code Quality): Remove emojis
        print("\n📊 COMPOSITE SCORES:")
        print(f"  Overall Complexity: {features.overall_complexity:.3f}")
        print(f"  Artistic Sophistication: {features.artistic_sophistication:.3f}")
        print(f"  Technical Skill: {features.technical_skill:.3f}")
        print(f"  Innovation Score: {features.innovation_score:.3f}")

        # TODO(Code Quality): Remove emojis
        print("\n⚡ PERFORMANCE:")
        print(f"  Processing Time: {features.processing_time_ms:.1f}ms")
        print(f"  Confidence Score: {features.confidence_score:.3f}")
        print(f"  Analyzer Version: {features.analyzer_version}")

        print("\n" + "=" * 60)
        print("Demo completed successfully!")

    except Exception as e:
        print(f"Demo failed: {e}")
        traceback.print_exc()


async def main():
    """Main application entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)

    logger.info(
        "Starting Professional Lyrics Analyzer v2.0",
        extra={
            "program_args": vars(args),
            "python_version": sys.version,
            "postgres_available": POSTGRES_AVAILABLE,
        },
    )

    try:
        # Run demo if requested
        if args.demo:
            await run_demo_analysis()
            return

        # Initialize analysis engine
        engine = AnalysisEngine()

        if not await engine.initialize():
            logger.error("Failed to initialize analysis engine")
            sys.exit(1)

        try:
            # Verify database if requested
            if args.verify:
                stats = await engine.db_manager.get_database_stats()
                print("\n" + "=" * 60)
                print("DATABASE VERIFICATION RESULTS")
                print("=" * 60)
                for key, value in stats.items():
                    print(f"  {key.replace('_', ' ').title()}: {value}")
                print("=" * 60)
                return

            # Run main analysis
            logger.info("Starting main analysis process")
            results = await engine.analyze_all_tracks(
                batch_size=args.batch_size, max_tracks=args.max_tracks
            )

            # Print final results
            print("\n" + "=" * 60)
            print("ANALYSIS COMPLETED")
            print("=" * 60)
            print(f"  Session ID: {results['session_id']}")
            print(f"  Tracks Processed: {results['session_processed']}")
            print(f"  Processing Errors: {results['session_errors']}")
            print(f"  Total Batches: {results['total_batches']}")
            print(f"  Processing Rate: {results['processing_rate']:.2f} tracks/sec")

            if results.get("shutdown_requested"):
                print("  Status: Gracefully shutdown (SIGINT/SIGTERM)")
            else:
                print("  Status: Completed successfully")

            print("=" * 60)

        finally:
            await engine.close()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down")
        print("\nShutdown requested by user")

    except Exception as e:
        logger.error("Application failed", exc_info=True)
        print(f"Application failed: {e}")
        sys.exit(1)

    logger.info("Application shutdown complete")


if __name__ == "__main__":
    # Ensure proper asyncio event loop handling
    if sys.platform == "win32":
        # Use ProactorEventLoop on Windows for better performance
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    asyncio.run(main())
