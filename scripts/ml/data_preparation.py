"""
🚀 ML Data Preparation Pipeline
Подготовка dataset из PostgreSQL для custom ML models

Dataset: 57,718 треков с 269,646 анализами
Features: Text embeddings, Spotify features, AI analysis results
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
from typing import Dict, List, Optional, Tuple
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.database.postgres_adapter import PostgreSQLManager
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RapDatasetPreparator:
    """
    Подготовка comprehensive ML dataset из PostgreSQL
    
    Features extracted:
    - Text features: embeddings, complexity, length, vocabulary
    - AI analysis: sentiment, themes, quality scores
    - Spotify audio features: danceability, energy, valence, etc.
    - Artist features: popularity, followers, genres
    """
    
    def __init__(self):
        self.db = None
        self.encoder = None  # Will be loaded when needed
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    async def initialize(self):
        """Initialize database connection and text encoder"""
        try:
            self.db = PostgreSQLManager()
            await self.db.initialize()
            logger.info("✅ PostgreSQL connection established")
            
            # Initialize sentence transformer for embeddings
            logger.info("🤖 Loading sentence transformer model...")
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Text encoder loaded")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            raise
    
    async def extract_comprehensive_data(self) -> pd.DataFrame:
        """
        Extract comprehensive dataset combining all data sources
        """
        logger.info("📊 Extracting comprehensive dataset from PostgreSQL...")
        
        # Main query to join all data sources
        query = """
        SELECT 
            -- Track information
            t.id, t.artist, t.title, t.lyrics,
            t.genre, t.explicit, t.word_count,
            t.scraped_date, t.album, t.language,
            t.popularity_score, t.lyrics_quality_score,
            COALESCE(CHAR_LENGTH(t.lyrics), 0) as lyrics_length,
            
            -- AI Analysis aggregated results (latest per analyzer)
            MAX(CASE WHEN ar.analyzer_type = 'qwen-3-4b-fp8' THEN ar.sentiment END) as qwen_sentiment,
            MAX(CASE WHEN ar.analyzer_type = 'qwen-3-4b-fp8' THEN ar.confidence END) as qwen_confidence,
            MAX(CASE WHEN ar.analyzer_type = 'qwen-3-4b-fp8' THEN ar.themes END) as qwen_themes,
            MAX(CASE WHEN ar.analyzer_type = 'qwen-3-4b-fp8' THEN ar.complexity_score END) as qwen_complexity,
            MAX(CASE WHEN ar.analyzer_type = 'qwen-3-4b-fp8' THEN ar.analysis_data END) as qwen_analysis,
            
            MAX(CASE WHEN ar.analyzer_type = 'gemma-3-27b-it' THEN ar.sentiment END) as gemma_sentiment,
            MAX(CASE WHEN ar.analyzer_type = 'gemma-3-27b-it' THEN ar.confidence END) as gemma_confidence,
            MAX(CASE WHEN ar.analyzer_type = 'gemma-3-27b-it' THEN ar.complexity_score END) as gemma_complexity,
            
            MAX(CASE WHEN ar.analyzer_type = 'simplified_features_v2' THEN ar.analysis_data END) as algo_features,
            
            -- Spotify data (JSON parsing)
            t.spotify_data,
            
            -- Analysis counts and coverage
            COUNT(ar.id) as total_analyses,
            COUNT(DISTINCT ar.analyzer_type) as analyzer_count
            
        FROM tracks t
        LEFT JOIN analysis_results ar ON t.id = ar.track_id
        WHERE t.lyrics IS NOT NULL 
          AND CHAR_LENGTH(t.lyrics) > 100  -- Minimum lyrics length
        GROUP BY t.id, t.artist, t.title, t.lyrics, t.genre, t.explicit, 
                 t.word_count, t.scraped_date, t.album, t.language,
                 t.popularity_score, t.lyrics_quality_score, t.spotify_data
        HAVING COUNT(ar.id) > 0  -- Only tracks with analysis
        ORDER BY t.id
        """
        
        try:
            # Execute query
            async with self.db.get_connection() as conn:
                result = await conn.fetch(query)
            
            # Convert to DataFrame
            df = pd.DataFrame([dict(row) for row in result])
            logger.info(f"📊 Extracted {len(df)} tracks with analysis")
            
            if len(df) == 0:
                raise ValueError("No data extracted from database")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Data extraction failed: {e}")
            raise
    
    def parse_spotify_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse Spotify audio features from JSON"""
        logger.info("🎵 Parsing Spotify audio features...")
        
        # Initialize feature columns
        spotify_features = [
            'danceability', 'energy', 'valence', 'tempo',
            'acousticness', 'instrumentalness', 'speechiness',
            'liveness', 'loudness', 'key', 'mode', 'time_signature'
        ]
        
        for feature in spotify_features:
            df[f'spotify_{feature}'] = np.nan
        
        # Parse JSON data
        for idx, row in df.iterrows():
            if pd.notna(row['spotify_data']):
                try:
                    spotify_json = json.loads(row['spotify_data']) if isinstance(row['spotify_data'], str) else row['spotify_data']
                    
                    # Extract audio features
                    if 'audio_features' in spotify_json:
                        audio_features = spotify_json['audio_features']
                        for feature in spotify_features:
                            if feature in audio_features:
                                df.at[idx, f'spotify_{feature}'] = audio_features[feature]
                    
                    # Extract artist popularity and followers
                    if 'artist_info' in spotify_json:
                        artist_info = spotify_json['artist_info']
                        df.at[idx, 'artist_popularity'] = artist_info.get('popularity', np.nan)
                        df.at[idx, 'artist_followers'] = artist_info.get('followers', {}).get('total', np.nan)
                        df.at[idx, 'artist_genres'] = ', '.join(artist_info.get('genres', []))
                
                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    continue
        
        # Fill missing values with median/mode
        for feature in spotify_features:
            col_name = f'spotify_{feature}'
            if col_name in df.columns:
                if feature in ['key', 'mode', 'time_signature']:
                    # Categorical features - use mode
                    mode_val = df[col_name].mode().iloc[0] if not df[col_name].mode().empty else 0
                    df[col_name] = df[col_name].fillna(mode_val)
                else:
                    # Continuous features - use median
                    median_val = df[col_name].median()
                    df[col_name] = df[col_name].fillna(median_val)
        
        # Fill artist features
        df['artist_popularity'] = df['artist_popularity'].fillna(df['artist_popularity'].median())
        df['artist_followers'] = df['artist_followers'].fillna(df['artist_followers'].median())
        df['artist_genres'] = df['artist_genres'].fillna('unknown')
        
        spotify_count = df['spotify_data'].notna().sum()
        logger.info(f"✅ Parsed Spotify features for {spotify_count} tracks ({spotify_count/len(df)*100:.1f}%)")
        
        return df
    
    def parse_ai_analysis_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse and engineer features from AI analysis results"""
        logger.info("🤖 Parsing AI analysis features...")
        
        # Parse Qwen analysis JSON
        for idx, row in df.iterrows():
            if pd.notna(row['qwen_analysis']):
                try:
                    qwen_data = json.loads(row['qwen_analysis']) if isinstance(row['qwen_analysis'], str) else row['qwen_analysis']
                    
                    # Extract additional metrics
                    if 'quality_rating' in qwen_data:
                        df.at[idx, 'qwen_quality'] = qwen_data['quality_rating']
                    if 'emotion_intensity' in qwen_data:
                        df.at[idx, 'qwen_emotion_intensity'] = qwen_data['emotion_intensity']
                    if 'wordplay_score' in qwen_data:
                        df.at[idx, 'qwen_wordplay'] = qwen_data['wordplay_score']
                        
                except (json.JSONDecodeError, TypeError, KeyError):
                    continue
            
            # Parse algorithmic features
            if pd.notna(row['algo_features']):
                try:
                    algo_data = json.loads(row['algo_features']) if isinstance(row['algo_features'], str) else row['algo_features']
                    
                    # Extract algorithmic metrics
                    if 'rhyme_density' in algo_data:
                        df.at[idx, 'rhyme_density'] = algo_data['rhyme_density']
                    if 'vocabulary_diversity' in algo_data:
                        df.at[idx, 'vocab_diversity'] = algo_data['vocabulary_diversity']
                    if 'sentiment_score' in algo_data:
                        df.at[idx, 'algo_sentiment_score'] = algo_data['sentiment_score']
                        
                except (json.JSONDecodeError, TypeError, KeyError):
                    continue
        
        # Create consensus features from multiple analyzers
        df['consensus_sentiment'] = df['qwen_sentiment'].fillna(df['gemma_sentiment'])
        df['consensus_complexity'] = df[['qwen_complexity', 'gemma_complexity']].mean(axis=1, skipna=True)
        
        # Theme processing
        df['theme_count'] = df['qwen_themes'].str.count(',') + 1
        df['theme_count'] = df['theme_count'].fillna(1)
        
        # Fill missing AI features
        ai_features = ['qwen_quality', 'qwen_emotion_intensity', 'qwen_wordplay', 
                      'rhyme_density', 'vocab_diversity', 'algo_sentiment_score']
        
        for feature in ai_features:
            if feature in df.columns:
                df[feature] = df[feature].fillna(df[feature].median())
        
        logger.info("✅ AI analysis features parsed and engineered")
        return df
    
    def engineer_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer comprehensive text features"""
        logger.info("📝 Engineering text features...")
        
        # Basic text metrics
        df['lines_count'] = df['lyrics'].str.count('\n') + 1
        df['avg_words_per_line'] = df['word_count'] / df['lines_count']
        df['chars_per_word'] = df['lyrics_length'] / df['word_count']
        
        # Advanced text features
        def calculate_vocabulary_diversity(text):
            if pd.isna(text) or len(text) == 0:
                return 0
            words = str(text).lower().split()
            if len(words) == 0:
                return 0
            unique_words = len(set(words))
            return unique_words / len(words)
        
        def count_profanity(text):
            if pd.isna(text):
                return 0
            # Simple profanity detection
            profane_words = ['shit', 'fuck', 'bitch', 'damn', 'ass', 'hell']
            text_lower = str(text).lower()
            return sum(1 for word in profane_words if word in text_lower)
        
        def count_repetitive_patterns(text):
            if pd.isna(text):
                return 0
            lines = str(text).split('\n')
            if len(lines) <= 1:
                return 0
            
            # Count repeated lines (hooks/choruses)
            line_counts = {}
            for line in lines:
                line = line.strip().lower()
                if len(line) > 10:  # Ignore short lines
                    line_counts[line] = line_counts.get(line, 0) + 1
            
            repeated_lines = sum(1 for count in line_counts.values() if count > 1)
            return repeated_lines / len(lines)
        
        logger.info("🔄 Calculating vocabulary diversity...")
        df['vocabulary_diversity'] = df['lyrics'].apply(calculate_vocabulary_diversity)
        
        logger.info("🔄 Counting profanity...")
        df['profanity_count'] = df['lyrics'].apply(count_profanity)
        
        logger.info("🔄 Analyzing repetitive patterns...")
        df['repetitive_ratio'] = df['lyrics'].apply(count_repetitive_patterns)
        
        # Normalize text features
        df['words_per_line_norm'] = df['avg_words_per_line'] / df['avg_words_per_line'].max()
        df['length_category'] = pd.cut(df['word_count'], 
                                     bins=[0, 100, 300, 500, np.inf],
                                     labels=['short', 'medium', 'long', 'very_long'])
        
        logger.info("✅ Text features engineered")
        return df
    
    def create_text_embeddings(self, df: pd.DataFrame, sample_size: int = 5000) -> pd.DataFrame:
        """Create text embeddings for semantic analysis"""
        logger.info(f"🔗 Creating text embeddings (sampling {sample_size} tracks)...")
        
        # Sample lyrics for embedding (to manage memory)
        if len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
            embedding_indices = df_sample.index
        else:
            df_sample = df
            embedding_indices = df.index
        
        # Create embeddings
        lyrics_texts = df_sample['lyrics'].fillna('').astype(str).tolist()
        
        logger.info("🤖 Generating sentence embeddings...")
        embeddings = self.encoder.encode(lyrics_texts, show_progress_bar=True)
        
        # Reduce dimensionality with PCA
        logger.info("📊 Reducing embedding dimensions with PCA...")
        pca = PCA(n_components=50)  # Reduce to 50 dimensions
        embeddings_reduced = pca.fit_transform(embeddings)
        
        # Add embeddings to dataframe
        for i in range(embeddings_reduced.shape[1]):
            df.loc[embedding_indices, f'embedding_{i}'] = embeddings_reduced[:, i]
        
        # Fill missing embeddings with zeros
        embedding_cols = [f'embedding_{i}' for i in range(50)]
        for col in embedding_cols:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = df[col].fillna(0.0)
        
        # Save PCA model for later use
        with open('data/ml/text_embedding_pca.pkl', 'wb') as f:
            pickle.dump(pca, f)
        
        logger.info(f"✅ Text embeddings created ({embeddings_reduced.shape[1]} dimensions)")
        return df
    
    def create_style_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create style and target labels for ML models"""
        logger.info("🏷️ Creating style and target labels...")
        
        # Artist style labels (top artists)
        artist_counts = df['artist'].value_counts()
        top_artists = artist_counts[artist_counts >= 10].index.tolist()  # Min 10 songs
        
        df['artist_style'] = df['artist'].apply(
            lambda x: x.lower().replace(' ', '_') if x in top_artists else 'other'
        )
        
        logger.info(f"📊 Identified {len(top_artists)} major artists for style classification")
        
        # Mood labels from sentiment
        mood_mapping = {
            'positive': 'upbeat',
            'negative': 'dark',
            'neutral': 'chill'
        }
        df['mood_label'] = df['consensus_sentiment'].map(mood_mapping).fillna('chill')
        
        # Quality tiers based on combined metrics
        def calculate_quality_score(row):
            score = 0
            
            # AI quality rating (40%)
            if pd.notna(row.get('qwen_quality')):
                score += row['qwen_quality'] * 0.4
            elif pd.notna(row.get('lyrics_quality_score')):
                score += row['lyrics_quality_score'] * 0.4
            else:
                score += 0.5 * 0.4
            
            # Complexity score (30%)
            if pd.notna(row.get('consensus_complexity')):
                score += row['consensus_complexity'] * 0.3
            else:
                score += 0.5 * 0.3
            
            # Spotify popularity (20%)
            if pd.notna(row.get('artist_popularity')):
                score += (row['artist_popularity'] / 100) * 0.2
            else:
                score += 0.3 * 0.2
            
            # Text quality metrics (10%)
            vocab_score = row.get('vocabulary_diversity', 0.5)
            score += vocab_score * 0.1
            
            return min(max(score, 0), 1)  # Clamp to [0, 1]
        
        df['quality_score'] = df.apply(calculate_quality_score, axis=1)
        df['quality_tier'] = pd.cut(
            df['quality_score'],
            bins=[0, 0.4, 0.7, 1.0],
            labels=['low', 'medium', 'high']
        )
        
        # Theme categories (simplified)
        def categorize_themes(themes_str):
            if pd.isna(themes_str):
                return 'general'
            
            themes_lower = str(themes_str).lower()
            
            if any(word in themes_lower for word in ['love', 'romance', 'relationship']):
                return 'love'
            elif any(word in themes_lower for word in ['money', 'success', 'wealth', 'fame']):
                return 'success'
            elif any(word in themes_lower for word in ['struggle', 'pain', 'poverty', 'hardship']):
                return 'struggle'
            elif any(word in themes_lower for word in ['party', 'club', 'dance', 'celebration']):
                return 'party'
            elif any(word in themes_lower for word in ['social', 'society', 'justice', 'politics']):
                return 'social_issues'
            else:
                return 'general'
        
        df['theme_category'] = df['qwen_themes'].apply(categorize_themes)
        
        # Commercial potential score
        def calculate_commercial_potential(row):
            score = 0
            
            # Artist popularity (30%)
            if pd.notna(row.get('artist_popularity')):
                score += (row['artist_popularity'] / 100) * 0.3
            
            # Spotify audio features (40%)
            danceability = row.get('spotify_danceability', 0.5)
            energy = row.get('spotify_energy', 0.5)
            valence = row.get('spotify_valence', 0.5)
            score += (danceability * 0.15 + energy * 0.15 + valence * 0.1)
            
            # Quality score (20%)
            score += row['quality_score'] * 0.2
            
            # Length optimization (10%) - not too short, not too long
            word_count = row.get('word_count', 200)
            length_score = 1 - abs(word_count - 250) / 500  # Optimal around 250 words
            length_score = max(0, min(1, length_score))
            score += length_score * 0.1
            
            return min(max(score, 0), 1)
        
        df['commercial_potential'] = df.apply(calculate_commercial_potential, axis=1)
        
        logger.info("✅ Style and target labels created")
        
        # Print label distributions
        logger.info("\n📊 Label Distributions:")
        logger.info(f"Artist Styles: {df['artist_style'].value_counts().head(10).to_dict()}")
        logger.info(f"Mood Labels: {df['mood_label'].value_counts().to_dict()}")
        logger.info(f"Quality Tiers: {df['quality_tier'].value_counts().to_dict()}")
        logger.info(f"Theme Categories: {df['theme_category'].value_counts().to_dict()}")
        logger.info(f"Quality Score - Mean: {df['quality_score'].mean():.3f}, Std: {df['quality_score'].std():.3f}")
        logger.info(f"Commercial Potential - Mean: {df['commercial_potential'].mean():.3f}, Std: {df['commercial_potential'].std():.3f}")
        
        return df
    
    def prepare_final_dataset(self, df: pd.DataFrame) -> Dict:
        """Prepare final ML-ready dataset with feature selection"""
        logger.info("🎯 Preparing final ML dataset...")
        
        # Select features for ML models
        feature_columns = []
        
        # Text features
        text_features = [
            'word_count', 'lyrics_length', 'lines_count', 'avg_words_per_line',
            'vocabulary_diversity', 'profanity_count', 'repetitive_ratio',
            'theme_count', 'chars_per_word'
        ]
        feature_columns.extend(text_features)
        
        # AI analysis features
        ai_features = [
            'qwen_confidence', 'consensus_complexity', 'qwen_quality',
            'qwen_emotion_intensity', 'qwen_wordplay', 'rhyme_density',
            'vocab_diversity', 'algo_sentiment_score'
        ]
        feature_columns.extend([f for f in ai_features if f in df.columns])
        
        # Spotify features
        spotify_features = [
            'spotify_danceability', 'spotify_energy', 'spotify_valence',
            'spotify_tempo', 'spotify_acousticness', 'spotify_instrumentalness',
            'spotify_speechiness', 'spotify_liveness', 'spotify_loudness'
        ]
        feature_columns.extend([f for f in spotify_features if f in df.columns])
        
        # Artist features
        artist_features = ['artist_popularity', 'artist_followers']
        feature_columns.extend([f for f in artist_features if f in df.columns])
        
        # Embeddings
        embedding_features = [f'embedding_{i}' for i in range(50) if f'embedding_{i}' in df.columns]
        feature_columns.extend(embedding_features)
        
        # Categorical features (encoded)
        categorical_features = ['artist_style', 'mood_label', 'theme_category', 'quality_tier']
        
        # Encode categorical features
        encoded_features = []
        for cat_feature in categorical_features:
            if cat_feature in df.columns:
                if cat_feature not in self.label_encoders:
                    self.label_encoders[cat_feature] = LabelEncoder()
                
                encoded_col = f'{cat_feature}_encoded'
                df[encoded_col] = self.label_encoders[cat_feature].fit_transform(df[cat_feature].fillna('unknown'))
                encoded_features.append(encoded_col)
        
        feature_columns.extend(encoded_features)
        
        # Target variables
        target_columns = ['quality_score', 'commercial_potential']
        
        # Create final feature matrix
        X = df[feature_columns].fillna(0)  # Fill any remaining NaNs
        y = df[target_columns]
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Prepare metadata
        metadata = {
            'creation_date': datetime.now().isoformat(),
            'total_tracks': len(df),
            'feature_count': len(feature_columns),
            'target_variables': target_columns,
            'feature_columns': feature_columns,
            'categorical_mappings': {
                name: {int(k): v for k, v in enumerate(encoder.classes_)}
                for name, encoder in self.label_encoders.items()
            },
            'data_statistics': {
                'spotify_coverage': df['spotify_data'].notna().sum() / len(df),
                'qwen_coverage': df['qwen_sentiment'].notna().sum() / len(df),
                'gemma_coverage': df['gemma_sentiment'].notna().sum() / len(df),
                'embedding_coverage': len(embedding_features) / 50
            }
        }
        
        logger.info(f"✅ Final dataset prepared:")
        logger.info(f"   Tracks: {len(df)}")
        logger.info(f"   Features: {len(feature_columns)}")
        logger.info(f"   Targets: {len(target_columns)}")
        logger.info(f"   Spotify Coverage: {metadata['data_statistics']['spotify_coverage']:.1%}")
        logger.info(f"   Qwen Coverage: {metadata['data_statistics']['qwen_coverage']:.1%}")
        
        return {
            'X': X_scaled,
            'y': y,
            'raw_data': df,
            'metadata': metadata,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }
    
    async def create_ml_dataset(self, output_path: str = 'data/ml/rap_ml_dataset.pkl'):
        """Create complete ML dataset pipeline"""
        logger.info("🚀 Starting ML dataset creation pipeline...")
        
        try:
            # Extract data
            df = await self.extract_comprehensive_data()
            
            # Parse and engineer features
            df = self.parse_spotify_features(df)
            df = self.parse_ai_analysis_features(df)
            df = self.engineer_text_features(df)
            df = self.create_text_embeddings(df)
            df = self.create_style_labels(df)
            
            # Prepare final dataset
            ml_dataset = self.prepare_final_dataset(df)
            
            # Save dataset
            logger.info(f"💾 Saving ML dataset to {output_path}...")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'wb') as f:
                pickle.dump(ml_dataset, f)
            
            # Save metadata separately
            metadata_path = output_path.replace('.pkl', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(ml_dataset['metadata'], f, indent=2)
            
            # Save raw CSV for analysis
            csv_path = output_path.replace('.pkl', '.csv')
            ml_dataset['raw_data'].to_csv(csv_path, index=False)
            
            logger.info("✅ ML dataset creation completed successfully!")
            logger.info(f"   Dataset: {output_path}")
            logger.info(f"   Metadata: {metadata_path}")
            logger.info(f"   CSV Export: {csv_path}")
            
            return ml_dataset
            
        except Exception as e:
            logger.error(f"❌ ML dataset creation failed: {e}")
            raise
        finally:
            if self.db:
                await self.db.close()

async def main():
    """Main execution function"""
    try:
        # Initialize preparator
        preparator = RapDatasetPreparator()
        await preparator.initialize()
        
        # Create ML dataset
        ml_dataset = await preparator.create_ml_dataset()
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 ML DATASET CREATION SUMMARY")
        print("="*60)
        print(f"📊 Total Tracks: {ml_dataset['metadata']['total_tracks']:,}")
        print(f"🔢 Features: {ml_dataset['metadata']['feature_count']}")
        print(f"🎯 Targets: {len(ml_dataset['metadata']['target_variables'])}")
        print(f"🎵 Spotify Coverage: {ml_dataset['metadata']['data_statistics']['spotify_coverage']:.1%}")
        print(f"🤖 Qwen Coverage: {ml_dataset['metadata']['data_statistics']['qwen_coverage']:.1%}")
        print(f"💎 Average Quality Score: {ml_dataset['y']['quality_score'].mean():.3f}")
        print(f"💰 Average Commercial Potential: {ml_dataset['y']['commercial_potential'].mean():.3f}")
        
        print("\n🔥 Dataset ready for ML model training!")
        print("Next steps:")
        print("  1. Train Conditional Generation Model")
        print("  2. Train Style Transfer Model")
        print("  3. Train Quality Prediction Model")
        print("  4. Train Trend Analysis Model")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    import sys
    sys.exit(0 if asyncio.run(main()) else 1)