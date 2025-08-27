#!/usr/bin/env python3
"""
Run full dataset analysis in batches with resume support.

Usage examples (PowerShell):
    python run_full_analysis.py --db rap_lyrics.db --batch-size 50
    python run_full_analysis.py --db rap_lyrics.db --batch-size 5 --sleep 1 --max-batches 100
    python run_full_analysis.py --db rap_lyrics.db --batch-size 1 --dry-run

This script repeatedly calls the MultiModelAnalyzer.batch_analyze_from_db in batches until
no unanalysed songs remain. It supports a dry-run mode and graceful interruption.
"""

import argparse
import sqlite3
import time
import logging
import sys
from typing import Optional

# ...existing code...

# Import analyzer from project
from multi_model_analyzer import MultiModelAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def count_remaining(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*) as cnt
        FROM songs s
        LEFT JOIN ai_analysis a ON s.id = a.song_id
        WHERE a.id IS NULL
    """)
    row = cur.fetchone()
    conn.close()
    return int(row[0]) if row else 0


def main(db_path: str, batch_size: int, sleep_sec: float, max_batches: Optional[int], dry_run: bool):
    logger.info(f"Starting full analysis runner: db={db_path} batch_size={batch_size} sleep={sleep_sec} dry_run={dry_run}")

    remaining = count_remaining(db_path)
    logger.info(f"Songs remaining to analyze: {remaining}")

    if dry_run:
        logger.info("Dry-run mode - exiting without invoking analyzer")
        return

    analyzer = MultiModelAnalyzer()

    batches = 0
    try:
        while True:
            remaining = count_remaining(db_path)
            if remaining <= 0:
                logger.info("No remaining songs to analyze. Exiting.")
                break

            if max_batches is not None and batches >= max_batches:
                logger.info(f"Reached max_batches ({max_batches}). Exiting.")
                break

            current_batch = min(batch_size, remaining)
            logger.info(f"Batch {batches+1}: analyzing up to {current_batch} songs (remaining: {remaining})")

            # Use analyzer's existing batch method which itself handles DB saving
            analyzer.batch_analyze_from_db(db_path=db_path, limit=current_batch, offset=0)

            batches += 1

            # small sleep to be polite to local/remote services
            if remaining > 0:
                logger.info(f"Sleeping for {sleep_sec}s before next batch")
                time.sleep(sleep_sec)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
    finally:
        remaining = count_remaining(db_path)
        logger.info(f"Finished. Batches run: {batches}. Songs remaining: {remaining}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run full dataset analysis in batches with resume support')
    parser.add_argument('--db', type=str, default='rap_lyrics.db', help='Path to SQLite DB')
    parser.add_argument('--batch-size', type=int, default=50, help='Number of songs per batch')
    parser.add_argument('--sleep', type=float, default=2.0, help='Seconds to sleep between batches')
    parser.add_argument('--max-batches', type=int, default=None, help='Maximum number of batches to run (optional)')
    parser.add_argument('--dry-run', action='store_true', help='Do not invoke analyzer, just show counts')

    args = parser.parse_args()
    main(db_path=args.db, batch_size=args.batch_size, sleep_sec=args.sleep, max_batches=args.max_batches, dry_run=args.dry_run)
