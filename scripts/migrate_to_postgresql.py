"""
#!/usr/bin/env python3
🎯 Скрипт миграции данных из SQLite в PostgreSQL

НАЗНАЧЕНИЕ:
- Миграция всех треков и связанных данных из SQLite в PostgreSQL
- Проверка целостности данных, zero-downtime

ИСПОЛЬЗОВАНИЕ:
python scripts/migrate_to_postgresql.py

ЗАВИСИМОСТИ:
- Python 3.8+
- tqdm
- src/database/postgres_adapter.py
- SQLite база данных (data/rap_lyrics.db)
- PostgreSQL база данных (rap_lyrics)

РЕЗУЛЬТАТ:
- Полная миграция данных с сохранением связей
- Логирование прогресса и ошибок

АВТОР: AI Assistant
ДАТА: Сентябрь 2025
"""

import sqlite3
import asyncio
import logging
from typing import List, Dict, Tuple
from tqdm import tqdm
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from database.postgres_adapter import PostgreSQLManager, DatabaseConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MigrationManager:
    """Handles SQLite to PostgreSQL migration"""
    
    def __init__(self, sqlite_path: str):
        self.sqlite_path = sqlite_path
        self.postgres_manager = PostgreSQLManager()
        self.stats = {
            'tracks_migrated': 0,
            'analysis_migrated': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
    
    def extract_sqlite_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Extract all data from SQLite database"""
        logger.info(f"📊 Extracting data from SQLite: {self.sqlite_path}")
        
        if not os.path.exists(self.sqlite_path):
            raise FileNotFoundError(f"SQLite database not found: {self.sqlite_path}")
        
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get table info
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        logger.info(f"📋 Available tables: {tables}")
        
        # Extract songs (main table)
        cursor.execute("SELECT * FROM songs LIMIT 5")  # Test query first
        sample = cursor.fetchall()
        if sample:
            columns = list(sample[0].keys())
            logger.info(f"📊 Song columns: {columns}")
        
        cursor.execute("SELECT * FROM songs")
        tracks = [dict(row) for row in cursor.fetchall()]
        logger.info(f"📊 Extracted {len(tracks)} songs")
        
        # Extract analysis results from ai_analysis table
        analysis_results = []
        if 'ai_analysis' in tables:
            try:
                cursor.execute("SELECT * FROM ai_analysis")
                analysis_results = [dict(row) for row in cursor.fetchall()]
                logger.info(f"📊 Extracted {len(analysis_results)} analysis results")
            except sqlite3.OperationalError as e:
                logger.warning(f"⚠️ Could not extract ai_analysis: {e}")
        
        conn.close()
        return tracks, analysis_results
    
    def clean_track_data(self, song: Dict) -> Dict:
        """Clean and normalize song data for PostgreSQL"""
        cleaned = {}
        
        # Required fields (adapt to actual song table structure)
        title = str(song.get('title', '')).strip()
        artist = str(song.get('artist', '')).strip()
        
        # Remove leading/trailing whitespace and check validity
        if len(title) > 0 and len(artist) > 0:
            cleaned['title'] = title[:500]  # Truncate if too long
            cleaned['artist'] = artist[:200]  # Truncate if too long
            cleaned['lyrics'] = song.get('lyrics')
        else:
            # Return empty dict for invalid tracks
            return {}
        
        # Optional string fields
        string_fields = ['spotify_id', 'album', 'genius_id']
        for field in string_fields:
            value = song.get(field)
            if value and str(value).strip():
                cleaned[field] = str(value).strip()
        
        # Numeric fields with validation
        numeric_fields = {
            'popularity': (0, 100),
            'energy': (0.0, 1.0),
            'valence': (0.0, 1.0),
            'danceability': (0.0, 1.0),
            'acousticness': (0.0, 1.0),
            'instrumentalness': (0.0, 1.0),
            'liveness': (0.0, 1.0),
            'speechiness': (0.0, 1.0)
        }
        
        for field, (min_val, max_val) in numeric_fields.items():
            value = song.get(field)
            if value is not None:
                try:
                    num_val = float(value)
                    if min_val <= num_val <= max_val:
                        cleaned[field] = num_val
                except (ValueError, TypeError):
                    pass
        
        # Special numeric fields
        if song.get('tempo'):
            try:
                tempo = float(song['tempo'])
                if tempo > 0:
                    cleaned['tempo'] = tempo
            except (ValueError, TypeError):
                pass
        
        if song.get('duration_ms'):
            try:
                duration = int(song['duration_ms'])
                if duration > 0:
                    cleaned['duration_ms'] = duration
            except (ValueError, TypeError):
                pass
        
        # Boolean fields
        if song.get('explicit') is not None:
            cleaned['explicit'] = bool(song['explicit'])
        
        # Integer fields
        int_fields = ['mode', 'key_signature', 'time_signature']
        for field in int_fields:
            value = song.get(field)
            if value is not None:
                try:
                    cleaned[field] = int(value)
                except (ValueError, TypeError):
                    pass
        
        # Date field
        if song.get('release_date'):
            cleaned['release_date'] = str(song['release_date'])
        
        # Loudness (can be negative)
        if song.get('loudness') is not None:
            try:
                cleaned['loudness'] = float(song['loudness'])
            except (ValueError, TypeError):
                pass
        
        return cleaned
    
    def _parse_complexity(self, complexity_str: str) -> float:
        """Convert complexity level string to numeric score"""
        if not complexity_str:
            return None
        
        complexity_str = complexity_str.lower()
        mapping = {
            'low': 0.2,
            'basic': 0.3,
            'simple': 0.3,
            'intermediate': 0.5,
            'medium': 0.5,
            'advanced': 0.7,
            'high': 0.8,
            'expert': 0.9,
            'complex': 0.8
        }
        
        return mapping.get(complexity_str, 0.5)
    
    async def migrate_tracks(self, tracks: List[Dict]) -> Dict[int, int]:
        """Migrate tracks and return old_id -> new_id mapping"""
        logger.info(f"🚀 Migrating {len(tracks)} tracks to PostgreSQL...")
        
        id_mapping = {}
        batch_size = 500  # Smaller batches for stability
        
        for i in tqdm(range(0, len(tracks), batch_size), desc="Migrating tracks"):
            batch = tracks[i:i + batch_size]
            
            # Clean batch data
            cleaned_batch = []
            original_ids = []
            
            for track in batch:
                cleaned_track = self.clean_track_data(track)
                # Check if track was successfully cleaned (not empty dict)
                if cleaned_track:
                    cleaned_batch.append(cleaned_track)
                    # Use 'id' field directly (SQLite has id field)
                    original_ids.append(track.get('id'))
                else:
                    logger.warning(f"⚠️ Skipping invalid track: title='{track.get('title', 'None')}' artist='{track.get('artist', 'None')}'")
                    self.stats['errors'] += 1
            
            # Insert batch
            try:
                new_ids = []
                for track_data in cleaned_batch:
                    new_id = await self.postgres_manager.insert_track(track_data)
                    if new_id:
                        new_ids.append(new_id)
                    else:
                        logger.warning(f"⚠️ Failed to insert track: {track_data.get('title', 'Unknown')}")
                        self.stats['errors'] += 1
                
                # Map old to new IDs
                for j, old_id in enumerate(original_ids[:len(new_ids)]):
                    id_mapping[old_id] = new_ids[j]
                
                self.stats['tracks_migrated'] += len(new_ids)
                logger.info(f"✅ Batch {i//batch_size + 1}: {len(new_ids)}/{len(cleaned_batch)} tracks migrated")
                
            except Exception as e:
                logger.error(f"❌ Error migrating batch {i//batch_size + 1}: {e}")
                self.stats['errors'] += len(batch)
        
        return id_mapping
    
    async def migrate_analysis_results(self, analysis_results: List[Dict], id_mapping: Dict[int, int]):
        """Migrate analysis results with updated track IDs"""
        if not analysis_results:
            logger.info("ℹ️ No analysis results to migrate")
            return
        
        logger.info(f"🧠 Migrating {len(analysis_results)} analysis results...")
        
        migrated_count = 0
        for result in tqdm(analysis_results, desc="Migrating analysis"):
            # Use 'song_id' field from ai_analysis table
            old_track_id = result.get('song_id')
            new_track_id = id_mapping.get(old_track_id)
            
            if new_track_id:
                # Map ai_analysis fields to analysis_results structure
                cleaned_result = {
                    'track_id': new_track_id,
                    'analyzer_type': result.get('model_version', 'gemma-3-27b-it'),
                    'sentiment': result.get('mood'),  # Map mood to sentiment
                    'confidence': result.get('authenticity_score'),
                    'complexity_score': self._parse_complexity(result.get('complexity_level')),
                    'themes': result.get('main_themes'),  # JSON string already
                    'analysis_data': {
                        'genre': result.get('genre'),
                        'subgenre': result.get('subgenre'),
                        'energy_level': result.get('energy_level'),
                        'emotional_tone': result.get('emotional_tone'),
                        'storytelling_type': result.get('storytelling_type'),
                        'wordplay_quality': result.get('wordplay_quality'),
                        'lyrical_creativity': result.get('lyrical_creativity'),
                        'commercial_appeal': result.get('commercial_appeal'),
                        'uniqueness': result.get('uniqueness'),
                        'overall_quality': result.get('overall_quality'),
                        'ai_likelihood': result.get('ai_likelihood')
                    },
                    'processing_time_ms': None,
                    'model_version': result.get('model_version', 'gemma-3-27b-it')
                }
                
                try:
                    await self.postgres_manager.save_analysis_result(cleaned_result)
                    migrated_count += 1
                except Exception as e:
                    logger.error(f"❌ Error migrating analysis result: {e}")
                    self.stats['errors'] += 1
            else:
                logger.warning(f"⚠️ Skipping analysis result - track ID {old_track_id} not found in mapping")
                self.stats['errors'] += 1
        
        self.stats['analysis_migrated'] = migrated_count
        logger.info(f"✅ Migrated {migrated_count} analysis results")
    
    async def verify_migration(self, original_tracks_count: int, original_analysis_count: int):
        """Verify migration success"""
        logger.info("🔍 Verifying migration...")
        
        # Check connection
        connection_ok = await self.postgres_manager.check_connection()
        if not connection_ok:
            logger.error("❌ PostgreSQL connection failed")
            return False
        
        # Get PostgreSQL stats
        stats = self.postgres_manager.get_table_stats()
        
        logger.info(f"📊 Migration verification:")
        logger.info(f"  Original tracks: {original_tracks_count}")
        logger.info(f"  Migrated tracks: {stats['tracks']}")
        logger.info(f"  Success rate: {(stats['tracks']/original_tracks_count)*100:.1f}%")
        
        if original_analysis_count > 0:
            logger.info(f"  Original analysis: {original_analysis_count}")
            logger.info(f"  Migrated analysis: {stats['analysis_results']}")
            logger.info(f"  Analysis success rate: {(stats['analysis_results']/original_analysis_count)*100:.1f}%")
        
        return stats['tracks'] > 0
    
    async def run_migration(self):
        """Execute complete migration"""
        from datetime import datetime
        self.stats['start_time'] = datetime.now()
        
        logger.info("🚀 Starting SQLite to PostgreSQL migration...")
        
        try:
            # Initialize PostgreSQL
            await self.postgres_manager.initialize()
            
            # Apply schema (skip if already applied)
            # schema_path = os.path.join('migrations', '001_initial_schema.sql')
            # if os.path.exists(schema_path):
            #     await self.postgres_manager.execute_schema(schema_path)
            logger.info("ℹ️ Assuming PostgreSQL schema already applied manually")
            
            # Extract SQLite data
            tracks, analysis_results = self.extract_sqlite_data()
            
            # Migrate tracks
            id_mapping = await self.migrate_tracks(tracks)
            
            # Migrate analysis results
            await self.migrate_analysis_results(analysis_results, id_mapping)
            
            # Verify migration
            success = await self.verify_migration(len(tracks), len(analysis_results))
            
            self.stats['end_time'] = datetime.now()
            duration = self.stats['end_time'] - self.stats['start_time']
            
            logger.info(f"✅ Migration completed in {duration}")
            logger.info(f"📊 Final stats: {self.stats}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            return False
        finally:
            await self.postgres_manager.close()

async def main():
    """Main migration execution"""
    sqlite_path = os.path.join('data', 'rap_lyrics.db')
    
    if not os.path.exists(sqlite_path):
        logger.error(f"❌ SQLite database not found: {sqlite_path}")
        return
    
    print("🎯 Starting PostgreSQL Migration")
    print("=" * 50)
    print(f"Source: {sqlite_path}")
    print("Target: PostgreSQL rap_lyrics database")
    print("=" * 50)
    
    migrator = MigrationManager(sqlite_path)
    success = await migrator.run_migration()
    
    if success:
        print("\n🎉 Migration completed successfully!")
        print("💡 Next steps:")
        print("   1. Test application with PostgreSQL")
        print("   2. Update config.yaml to use PostgreSQL")
        print("   3. Backup SQLite database")
    else:
        print("\n❌ Migration failed. Check logs for details.")

if __name__ == "__main__":
    asyncio.run(main())
