#!/usr/bin/env python3
"""
🤖 Batch AI Analysis Tool - Production-grade dataset processor

Processes large rap lyrics datasets in configurable batches with resume support.
Ideal for running AI analysis on thousands of songs without overwhelming system resources.

✨ Features:
- 📊 Batch processing with configurable size
- 🔄 Resume support (continues from where it left off)
- ⏸️ Graceful interruption handling
- 🧪 Dry-run mode for testing
- 📈 Progress tracking and logging
- 💾 Resource-conscious processing

Usage examples:
    python scripts/tools/run_full_analysis.py --batch-size 50
    python scripts/tools/run_full_analysis.py --batch-size 10 --sleep 3 --max-batches 50
    python scripts/tools/run_full_analysis.py --dry-run

⚠️  Production tool - processes entire database in AI analysis pipeline.
"""

import argparse
import sqlite3
import time
import logging
import sys
from typing import Optional

# Add project root to path for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import analyzer from project
from src.analyzers.multi_model_analyzer import MultiModelAnalyzer

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
    """Main batch processing function with enhanced logging"""
    logger.info("🤖 Starting Batch AI Analysis Tool")
    logger.info("=" * 60)
    logger.info(f"📊 Config: db={db_path} | batch_size={batch_size} | sleep={sleep_sec}s | max_batches={max_batches}")
    
    remaining = count_remaining(db_path)
    logger.info(f"🎵 Songs remaining to analyze: {remaining:,}")

    if dry_run:
        estimated_batches = (remaining + batch_size - 1) // batch_size  # Ceiling division
        estimated_time = estimated_batches * sleep_sec / 60  # Minutes
        logger.info("🧪 DRY-RUN MODE - Analysis preview:")
        logger.info(f"   📊 Estimated batches: {estimated_batches}")
        logger.info(f"   ⏱️  Estimated time: {estimated_time:.1f} minutes")
        logger.info("   🔄 No actual processing will occur")
        return

    logger.info("🚀 Initializing MultiModel Analyzer...")
    analyzer = MultiModelAnalyzer()
    logger.info("✅ Analyzer ready")

    batches = 0
    start_time = time.time()
    total_processed = 0
    
    try:
        logger.info("\n🔄 Starting batch processing...")
        while True:
            remaining = count_remaining(db_path)
            if remaining <= 0:
                logger.info("🎉 All songs analyzed! No remaining songs to process.")
                break

            if max_batches is not None and batches >= max_batches:
                logger.info(f"🛑 Reached max_batches limit ({max_batches})")
                break

            current_batch = min(batch_size, remaining)
            progress = f"({batches+1}" + (f"/{max_batches}" if max_batches else "") + ")"
            
            logger.info(f"\n📦 Batch {progress}: Processing {current_batch} songs | {remaining:,} remaining")

            # Process batch using analyzer's existing method
            batch_start = time.time()
            analyzer.batch_analyze_from_db(db_path=db_path, limit=current_batch, offset=0)
            batch_time = time.time() - batch_start
            
            batches += 1
            total_processed += current_batch
            
            # Performance metrics
            avg_time_per_song = batch_time / current_batch if current_batch > 0 else 0
            logger.info(f"✅ Batch completed in {batch_time:.1f}s | {avg_time_per_song:.2f}s per song")

            # Sleep between batches (except for last one)
            if remaining > current_batch and sleep_sec > 0:
                logger.info(f"⏸️  Sleeping {sleep_sec}s before next batch...")
                time.sleep(sleep_sec)

    except KeyboardInterrupt:
        logger.info("\n⚠️  Processing interrupted by user (Ctrl+C)")
    except Exception as e:
        logger.exception(f"💥 Unexpected error during processing: {e}")
    finally:
        total_time = time.time() - start_time
        remaining_final = count_remaining(db_path)
        
        logger.info("\n" + "=" * 60)
        logger.info("🏁 BATCH PROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"📦 Batches completed: {batches}")
        logger.info(f"🎵 Songs processed: {total_processed:,}")
        logger.info(f"🎵 Songs remaining: {remaining_final:,}")
        logger.info(f"⏱️  Total time: {total_time/60:.1f} minutes")
        
        if total_processed > 0:
            avg_time = total_time / total_processed
            logger.info(f"📊 Average time per song: {avg_time:.2f}s")
        
        if remaining_final == 0:
            logger.info("🎉 SUCCESS: All songs analyzed!")
        elif batches > 0:
            logger.info("✅ Partial success - resume with same command to continue")
        
        logger.info("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='🤖 Batch AI Analysis Tool - Process large datasets in configurable batches',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --batch-size 50                    # Process 50 songs per batch
  %(prog)s --batch-size 10 --sleep 3          # Smaller batches with 3s pause
  %(prog)s --max-batches 20 --dry-run         # Test run for first 20 batches
  %(prog)s --db custom.db --batch-size 100    # Custom database path

Note: Tool automatically resumes from where it left off.
        """
    )
    
    parser.add_argument('--db', type=str, default='data/rap_lyrics.db', 
                       help='Path to SQLite database (default: data/rap_lyrics.db)')
    parser.add_argument('--batch-size', type=int, default=25,
                       help='Number of songs per batch (default: 25)')
    parser.add_argument('--sleep', type=float, default=1.0,
                       help='Seconds to sleep between batches (default: 1.0)')
    parser.add_argument('--max-batches', type=int, default=None,
                       help='Maximum batches to run (optional, for testing)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show processing plan without running analysis')

    args = parser.parse_args()
    main(db_path=args.db, batch_size=args.batch_size, sleep_sec=args.sleep, 
         max_batches=args.max_batches, dry_run=args.dry_run)
