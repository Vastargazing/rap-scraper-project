"""
🚀 Quick ML Dataset Preparation
Упрощенная версия для быстрого создания ML dataset
"""

import pandas as pd
import numpy as np
import asyncio
import json
import pickle
import sys
import os
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.database.postgres_adapter import PostgreSQLManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickDatasetPreparator:
    """Quick ML dataset preparation for testing"""
    
    def __init__(self):
        self.db = None
        
    async def initialize(self):
        """Initialize database connection"""
        try:
            self.db = PostgreSQLManager()
            await self.db.initialize()
            logger.info("✅ PostgreSQL connection established")
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            raise
    
    async def extract_sample_data(self, limit: int = 1000) -> pd.DataFrame:
        """Extract sample data for quick testing"""
        logger.info(f"📊 Extracting sample data (limit: {limit})...")
        
        # Simplified query focusing on essential data
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
        
        try:
            async with self.db.get_connection() as conn:
                result = await conn.fetch(query)
            
            df = pd.DataFrame([dict(row) for row in result])
            logger.info(f"📊 Extracted {len(df)} sample tracks")
            return df
            
        except Exception as e:
            logger.error(f"❌ Data extraction failed: {e}")
            raise
    
    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic features for ML models"""
        logger.info("🔧 Creating basic features...")
        
        # Fix word_count if it's None
        df['word_count'] = df['word_count'].fillna(0)
        # Recalculate word_count from lyrics if needed
        df.loc[df['word_count'] == 0, 'word_count'] = df.loc[df['word_count'] == 0, 'lyrics'].str.split().str.len()
        
        # Basic text features
        df['lyrics_length'] = df['lyrics'].str.len()
        df['lines_count'] = df['lyrics'].str.count('\n') + 1
        df['avg_words_per_line'] = df['word_count'] / df['lines_count']
        
        # Create style labels
        artist_counts = df['artist'].value_counts()
        top_artists = artist_counts[artist_counts >= 5].index.tolist()
        
        df['artist_style'] = df['artist'].apply(
            lambda x: x.lower().replace(' ', '_') if x in top_artists else 'other'
        )
        
        # Mood labels
        mood_mapping = {
            'positive': 'upbeat',
            'negative': 'dark', 
            'neutral': 'chill'
        }
        df['mood_label'] = df['qwen_sentiment'].map(mood_mapping).fillna('chill')
        
        # Theme categories
        def categorize_themes(themes_str):
            if pd.isna(themes_str):
                return 'general'
            themes_lower = str(themes_str).lower()
            if 'love' in themes_lower or 'romance' in themes_lower:
                return 'love'
            elif 'money' in themes_lower or 'success' in themes_lower:
                return 'success'
            elif 'struggle' in themes_lower or 'pain' in themes_lower:
                return 'struggle'
            elif 'party' in themes_lower or 'club' in themes_lower:
                return 'party'
            else:
                return 'general'
        
        df['theme_category'] = df['qwen_themes'].apply(categorize_themes)
        
        # Quality score (simplified)
        df['quality_score'] = df['qwen_complexity'].fillna(0.5)
        
        logger.info("✅ Basic features created")
        return df
    
    async def create_quick_dataset(self, limit: int = 1000, output_path: str = 'data/ml/quick_dataset.pkl'):
        """Create quick ML dataset"""
        logger.info("🚀 Creating quick ML dataset...")
        
        try:
            # Extract sample data
            df = await self.extract_sample_data(limit)
            
            # Create features
            df = self.create_basic_features(df)
            
            # Create simple dataset structure
            ml_dataset = {
                'raw_data': df,
                'metadata': {
                    'creation_date': datetime.now().isoformat(),
                    'total_tracks': len(df),
                    'sample_limit': limit
                }
            }
            
            # Save dataset
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump(ml_dataset, f)
            
            # Save CSV for inspection
            csv_path = output_path.replace('.pkl', '.csv')
            df.to_csv(csv_path, index=False)
            
            logger.info(f"✅ Quick dataset created: {output_path}")
            logger.info(f"   Tracks: {len(df)}")
            logger.info(f"   Artists: {df['artist'].nunique()}")
            logger.info(f"   Styles: {df['artist_style'].value_counts().to_dict()}")
            logger.info(f"   Moods: {df['mood_label'].value_counts().to_dict()}")
            logger.info(f"   Themes: {df['theme_category'].value_counts().to_dict()}")
            
            return ml_dataset
            
        except Exception as e:
            logger.error(f"❌ Quick dataset creation failed: {e}")
            raise
        finally:
            if self.db:
                await self.db.close()

async def main():
    """Main function"""
    try:
        preparator = QuickDatasetPreparator()
        await preparator.initialize()
        
        # Create quick dataset with 1000 samples
        dataset = await preparator.create_quick_dataset(limit=1000)
        
        print("\n" + "="*50)
        print("🎯 QUICK ML DATASET READY!")
        print("="*50)
        print(f"📊 Samples: {dataset['metadata']['total_tracks']}")
        print(f"📅 Created: {dataset['metadata']['creation_date']}")
        print("\nNext step: Train conditional generation model")
        print("Command: python models/conditional_generation.py --mode train")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)