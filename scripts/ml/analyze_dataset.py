"""Dataset Analysis Script.

TODO(code-review): Add module docstring following Google style.
TODO(code-review): No argument parser - hardcoded file path.
TODO(code-review): No error handling - script will crash on missing file.
TODO(code-review): Extract to functions for better organization and testability.
TODO(code-review): Add type hints.
TODO(code-review): Add logging instead of print statements.
TODO(code-review): No tests for this analysis script.

This script analyzes the quick ML dataset and prints statistics.
"""

import pickle
from pathlib import Path
from typing import Dict, Any

# TODO(code-review): Add argument parser for file path
# TODO(code-review): Add validation for file existence
# TODO(security): Validate file path to prevent path traversal
# TODO(code-review): Extract to constant at module level
DATASET_PATH = "data/ml/quick_dataset.pkl"

# TODO(code-review): Wrap in main() function
# TODO(code-review): Add error handling for file operations
# TODO(security): pickle.load() is unsafe - validate source or use safer serialization
try:
    # Load dataset
    with open(DATASET_PATH, "rb") as f:
        data: Dict[str, Any] = pickle.load(f)
except FileNotFoundError:
    # TODO(code-review): Use proper logging instead of print
    print(f"Error: Dataset file not found at {DATASET_PATH}")
    exit(1)
except Exception as e:
    # TODO(code-review): Catch specific exceptions
    print(f"Error loading dataset: {e}")
    exit(1)

# TODO(code-review): Add validation that 'raw_data' key exists
df = data["raw_data"]

# TODO(code-review): Extract all analysis to separate functions for testability
print("=== QUICK DATASET ANALYSIS ===")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

print("\n=== WORD COUNT STATS ===")
# TODO(code-review): Add error handling if column doesn't exist
print(df["word_count"].describe())

print("\n=== LYRICS LENGTH STATS ===")
# TODO(code-review): Add error handling if column doesn't exist
print(df["lyrics"].str.len().describe())

print("\n=== SENTIMENT VALUES ===")
# TODO(code-review): Add error handling if column doesn't exist
print(df["qwen_sentiment"].value_counts())

print("\n=== NULL VALUES CHECK ===")
# TODO(code-review): Add error handling if columns don't exist
print(f"qwen_sentiment nulls: {df['qwen_sentiment'].isnull().sum()}")
print(f"lyrics nulls: {df['lyrics'].isnull().sum()}")

print("\n=== SAMPLE DATA ===")
# TODO(code-review): Add error handling if columns don't exist
print(
    df[["artist", "word_count", "qwen_sentiment", "artist_style", "mood_label"]].head()
)

# Check filtering criteria
print("\n=== FILTERING ANALYSIS ===")
# TODO(code-review): Extract magic numbers 50, 200 to constants
# TODO(code-review): Add error handling for missing columns
criteria1 = df["word_count"] >= 50
criteria2 = df["lyrics"].str.len() >= 200
criteria3 = df["qwen_sentiment"].notna()

print(f"Word count >= 50: {criteria1.sum()}/{len(df)} ({criteria1.mean() * 100:.1f}%)")
print(
    f"Lyrics length >= 200: {criteria2.sum()}/{len(df)} ({criteria2.mean() * 100:.1f}%)"
)
print(f"Has sentiment: {criteria3.sum()}/{len(df)} ({criteria3.mean() * 100:.1f}%)")

all_criteria = criteria1 & criteria2 & criteria3
print(
    f"All criteria: {all_criteria.sum()}/{len(df)} ({all_criteria.mean() * 100:.1f}%)"
)

# TODO(code-review): Add if __name__ == "__main__" guard
# TODO(code-review): Return exit code based on success/failure
