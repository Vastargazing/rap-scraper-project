"""Quick ML Dataset Preparation.

TODO(code-review): Convert module docstring to English and follow Google style guide.
TODO(code-review): Remove emoji from docstrings - not professional for production code.

A simplified version for quick ML dataset creation.

This module provides utilities for rapid dataset preparation from PostgreSQL
database for machine learning model training and testing.
"""

# TODO(code-review): Group imports according to Google style guide:
# 1. Standard library imports
# 2. Related third party imports
# 3. Local application/library specific imports
import asyncio
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any  # TODO(code-review): Add type hints throughout

import pandas as pd

# Add project root to path
# TODO(code-review): Replace sys.path manipulation with proper package installation
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.database.postgres_adapter import PostgreSQLManager

# Setup logging
# TODO(code-review): Move logging configuration to a separate config module
# TODO(code-review): Use structured logging (e.g., structlog) for better observability
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QuickDatasetPreparator:
    """Quick ML dataset preparation for testing.

    TODO(code-review): Add comprehensive class docstring following Google style:
    - Class description
    - Attributes section
    - Example usage section

    TODO(code-review): Consider splitting this class - it violates Single Responsibility
    Principle by handling DB connection, data extraction, feature engineering, and saving.
    """

    def __init__(self) -> None:
        """Initialize the dataset preparator.

        TODO(code-review): Add Args and Attributes sections to docstring.
        TODO(code-review): Consider dependency injection for PostgreSQLManager.
        """
        # TODO(code-review): Type hint should be Optional[PostgreSQLManager]
        self.db: Optional[Any] = None

    async def initialize(self) -> None:
        """Initialize database connection.

        TODO(code-review): Add proper docstring with Raises section.
        TODO(code-review): Remove emoji from log messages.

        Raises:
            Exception: If database initialization fails.
        """
        try:
            self.db = PostgreSQLManager()
            await self.db.initialize()
            logger.info("PostgreSQL connection established")  # TODO(code-review): Removed emoji
        except Exception as e:
            # TODO(code-review): Catch specific exceptions (e.g., ConnectionError, DatabaseError)
            # TODO(code-review): Add more context to error message (connection string, etc.)
            logger.error(f"Initialization failed: {e}")  # TODO(code-review): Removed emoji
            raise

    async def extract_sample_data(self, limit: int = 1000) -> pd.DataFrame:
        """Extract sample data for quick testing.

        TODO(code-review): Add comprehensive docstring with Args, Returns, Raises sections.
        TODO(code-review): Remove emoji from log messages.

        Args:
            limit: Maximum number of samples to extract.

        Returns:
            DataFrame containing extracted track data with features.

        Raises:
            ValueError: If limit is invalid.
            DatabaseError: If database query fails.
        """
        # TODO(code-review): Validate input parameter
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")

        logger.info(f"Extracting sample data (limit: {limit})...")  # TODO(code-review): Removed emoji

        # TODO(security): CRITICAL SQL INJECTION RISK - Using f-string with user input
        # Replace with parameterized query to prevent SQL injection attacks
        # Bad: query = f"... LIMIT {limit}"
        # Good: Use query parameters: conn.fetch(query, limit)
        query = f"""
        SELECT
            t.id, t.artist, t.title, t.lyrics,
            t.word_count, t.explicit,

            -- Get latest Qwen analysis
            ar.sentiment as qwen_sentiment,
            ar.confidence as qwen_confidence,
            ar.themes as qwen_themes,
            ar.complexity_score as qwen_complexity,

            -- Parse Spotify data if available
            t.spotify_data

        FROM tracks t
        LEFT JOIN analysis_results ar ON t.id = ar.track_id
            AND ar.analyzer_type = 'qwen-3-4b-fp8'
        WHERE t.lyrics IS NOT NULL
          AND CHAR_LENGTH(t.lyrics) > 100
          AND ar.id IS NOT NULL
        LIMIT {limit}
        """
        # TODO(code-review): Extract SQL query to separate module/file for better maintainability
        # TODO(code-review): Consider using SQLAlchemy for query building

        try:
            # TODO(code-review): Add timeout to prevent hanging connections
            async with self.db.get_connection() as conn:
                result = await conn.fetch(query)

            df = pd.DataFrame([dict(row) for row in result])
            logger.info(f"Extracted {len(df)} sample tracks")  # TODO(code-review): Removed emoji
            # TODO(code-review): Add validation - what if result is empty?
            return df

        except Exception as e:
            # TODO(code-review): Catch specific exceptions instead of bare Exception
            # TODO(code-review): Add retry logic with exponential backoff
            logger.error(f"Data extraction failed: {e}")  # TODO(code-review): Removed emoji
            raise

    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic features for ML models.

        TODO(code-review): Add comprehensive docstring with Args, Returns.
        TODO(code-review): Method too long (>40 lines) - split into smaller functions.
        TODO(code-review): Extract magic numbers to named constants (5, 0.5, etc.).
        TODO(code-review): Remove emoji from log messages.

        Args:
            df: Input DataFrame with raw track data.

        Returns:
            DataFrame with engineered features added.
        """
        logger.info("Creating basic features...")  # TODO(code-review): Removed emoji

        # TODO(code-review): Add input validation - check required columns exist
        # TODO(code-review): Handle potential division by zero in avg_words_per_line

        # Fix word_count if it's None
        df["word_count"] = df["word_count"].fillna(0)
        # Recalculate word_count from lyrics if needed
        # TODO(code-review): This modifies df in place - consider making a copy first
        df.loc[df["word_count"] == 0, "word_count"] = (
            df.loc[df["word_count"] == 0, "lyrics"].str.split().str.len()
        )

        # Basic text features
        df["lyrics_length"] = df["lyrics"].str.len()
        df["lines_count"] = df["lyrics"].str.count("\n") + 1
        # TODO(code-review): Potential division by zero if lines_count is 0
        df["avg_words_per_line"] = df["word_count"] / df["lines_count"]

        # Create style labels
        # TODO(code-review): Extract magic number 5 to constant MIN_ARTIST_TRACKS
        MIN_ARTIST_TRACKS = 5  # TODO(code-review): Define at module level
        artist_counts = df["artist"].value_counts()
        top_artists = artist_counts[artist_counts >= MIN_ARTIST_TRACKS].index.tolist()

        # TODO(code-review): Lambda function is hard to test - extract to named function
        df["artist_style"] = df["artist"].apply(
            lambda x: x.lower().replace(" ", "_") if x in top_artists else "other"
        )

        # Mood labels
        # TODO(code-review): Extract mapping to module-level constant
        mood_mapping = {"positive": "upbeat", "negative": "dark", "neutral": "chill"}
        df["mood_label"] = df["qwen_sentiment"].map(mood_mapping).fillna("chill")

        # Theme categories
        # TODO(code-review): Extract nested function to class method for better testability
        # TODO(code-review): Use constants for theme keywords instead of hardcoded strings
        def categorize_themes(themes_str: Optional[str]) -> str:
            """Categorize themes from theme string.

            TODO(code-review): Add proper docstring.
            TODO(code-review): Extract theme keywords to configuration.
            """
            if pd.isna(themes_str):
                return "general"
            themes_lower = str(themes_str).lower()
            if "love" in themes_lower or "romance" in themes_lower:
                return "love"
            if "money" in themes_lower or "success" in themes_lower:
                return "success"
            if "struggle" in themes_lower or "pain" in themes_lower:
                return "struggle"
            if "party" in themes_lower or "club" in themes_lower:
                return "party"
            return "general"

        df["theme_category"] = df["qwen_themes"].apply(categorize_themes)

        # Quality score (simplified)
        # TODO(code-review): Extract magic number 0.5 to constant DEFAULT_QUALITY_SCORE
        df["quality_score"] = df["qwen_complexity"].fillna(0.5)

        logger.info("Basic features created")  # TODO(code-review): Removed emoji
        return df

    async def create_quick_dataset(
        self, limit: int = 1000, output_path: str = "data/ml/quick_dataset.pkl"
    ) -> Dict[str, Any]:
        """Create quick ML dataset.

        TODO(code-review): Add comprehensive docstring.
        TODO(code-review): Remove hardcoded path - use Path object and config.
        TODO(code-review): Add progress indicators for long operations.
        TODO(code-review): Remove emoji from log messages.

        Args:
            limit: Maximum number of samples to include.
            output_path: Path where to save the dataset.

        Returns:
            Dictionary containing the ML dataset and metadata.

        Raises:
            IOError: If dataset cannot be saved.
            ValueError: If dataset creation fails.
        """
        logger.info("Creating quick ML dataset...")  # TODO(code-review): Removed emoji

        try:
            # Extract sample data
            df = await self.extract_sample_data(limit)

            # Create features
            df = self.create_basic_features(df)

            # Create simple dataset structure
            # TODO(code-review): Use dataclass or TypedDict for better type safety
            ml_dataset = {
                "raw_data": df,
                "metadata": {
                    "creation_date": datetime.now().isoformat(),
                    "total_tracks": len(df),
                    "sample_limit": limit,
                },
            }

            # Save dataset
            # TODO(code-review): Use Path object instead of os.path for better cross-platform support
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # TODO(code-review): Add error handling for file write operations
            # TODO(security): Validate output_path to prevent path traversal attacks
            with open(output_path, "wb") as f:
                pickle.dump(ml_dataset, f)

            # Save CSV for inspection
            csv_path = output_path.replace(".pkl", ".csv")
            # TODO(code-review): Handle potential errors when saving CSV
            df.to_csv(csv_path, index=False)

            logger.info(f"Quick dataset created: {output_path}")  # TODO(code-review): Removed emoji
            logger.info(f"   Tracks: {len(df)}")
            logger.info(f"   Artists: {df['artist'].nunique()}")
            # TODO(performance): value_counts() can be expensive - consider limiting output
            logger.info(f"   Styles: {df['artist_style'].value_counts().to_dict()}")
            logger.info(f"   Moods: {df['mood_label'].value_counts().to_dict()}")
            logger.info(f"   Themes: {df['theme_category'].value_counts().to_dict()}")

            return ml_dataset

        except Exception as e:
            # TODO(code-review): Catch specific exceptions
            logger.error(f"Quick dataset creation failed: {e}")  # TODO(code-review): Removed emoji
            raise
        finally:
            if self.db:
                # TODO(code-review): Log database connection closure
                await self.db.close()


async def main() -> bool:
    """Main function.

    TODO(code-review): Add comprehensive docstring following Google style.
    TODO(code-review): Remove emoji from print statements.

    Returns:
        True if successful, False otherwise.
    """
    try:
        preparator = QuickDatasetPreparator()
        await preparator.initialize()

        # Create quick dataset with 1000 samples
        # TODO(code-review): Make limit configurable via CLI args or config file
        dataset = await preparator.create_quick_dataset(limit=1000)

        # TODO(code-review): Remove emoji from output messages
        print("\n" + "=" * 50)
        print("QUICK ML DATASET READY!")  # TODO(code-review): Removed emoji
        print("=" * 50)
        print(f"Samples: {dataset['metadata']['total_tracks']}")  # TODO(code-review): Removed emoji
        print(f"Created: {dataset['metadata']['creation_date']}")  # TODO(code-review): Removed emoji
        print("\nNext step: Train conditional generation model")
        print("Command: python models/conditional_generation.py --mode train")

        return True

    except Exception as e:
        # TODO(code-review): Add stack trace logging for debugging
        logger.error(f"Pipeline failed: {e}")  # TODO(code-review): Removed emoji
        logger.exception("Full traceback:")  # TODO(code-review): Added exception logging
        return False


if __name__ == "__main__":
    # TODO(code-review): Add argparse for CLI arguments (limit, output_path, etc.)
    # TODO(code-review): Add version info and help text
    # TODO(code-review): Consider using a proper CLI framework like Click or Typer
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
