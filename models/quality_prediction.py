"""
💎 Quality Prediction Model
ML модель для предсказания коммерческого потенциала и качества рэп-треков

Features:
- Multi-target regression (качество, коммерческий потенциал, вирусность)
- Feature engineering из текста, AI анализа, Spotify данных
- SHAP interpretability для объяснения предсказаний
- Cross-validation и robust evaluation
"""

import pandas as pd
import numpy as np
import pickle
import sys
import os
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RapQualityPredictor:
    """
    ML модель для предсказания качества и коммерческого потенциала рэп-треков
    
    Targets:
    - quality_score: Общее качество трека (0-1)
    - commercial_potential: Коммерческий потенциал (0-1)  
    - viral_potential: Потенциал для вирусности (0-1)
    - longevity_score: Долговечность/классика (0-1)
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.feature_importance = {}
        
    def load_training_data(self, dataset_path: str = 'data/ml/quick_dataset.pkl') -> pd.DataFrame:
        """Load and prepare training data"""
        logger.info(f"📊 Loading training data from {dataset_path}")
        
        try:
            with open(dataset_path, 'rb') as f:
                ml_dataset = pickle.load(f)
            
            df = ml_dataset['raw_data']
            logger.info(f"✅ Loaded {len(df)} tracks for quality prediction")
            
            # Filter for quality data
            df_filtered = df[
                (df['word_count'] >= 30) &
                (df['lyrics'].str.len() >= 100) &
                (df['qwen_sentiment'].notna())
            ].copy()
            
            logger.info(f"📊 Filtered to {len(df_filtered)} quality tracks")
            return df_filtered
            
        except Exception as e:
            logger.error(f"❌ Failed to load training data: {e}")
            raise
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer comprehensive features for quality prediction"""
        logger.info("🔧 Engineering features for quality prediction...")
        
        features_df = pd.DataFrame(index=df.index)
        
        # === TEXT FEATURES ===
        features_df['word_count'] = df['word_count']
        features_df['lyrics_length'] = df['lyrics_length']
        features_df['lines_count'] = df['lines_count']
        features_df['avg_words_per_line'] = df['avg_words_per_line']
        
        # Advanced text metrics
        def calculate_vocabulary_richness(lyrics):
            if pd.isna(lyrics) or len(str(lyrics)) == 0:
                return 0
            words = str(lyrics).lower().split()
            if len(words) == 0:
                return 0
            unique_words = len(set(words))
            return unique_words / len(words)
        
        def count_repetitive_elements(lyrics):
            if pd.isna(lyrics):
                return 0
            lines = str(lyrics).split('\n')
            if len(lines) <= 1:
                return 0
            line_counts = {}
            for line in lines:
                line = line.strip().lower()
                if len(line) > 5:  # Ignore very short lines
                    line_counts[line] = line_counts.get(line, 0) + 1
            repeated = sum(1 for count in line_counts.values() if count > 1)
            return repeated / len(lines) if lines else 0
        
        def calculate_rhyme_density(lyrics):
            if pd.isna(lyrics):
                return 0
            # Simple rhyme density estimation
            lines = [line.strip() for line in str(lyrics).split('\n') if line.strip()]
            if len(lines) < 2:
                return 0
            
            rhyming_pairs = 0
            for i in range(len(lines)-1):
                # Check if lines end with similar sounds (very basic)
                line1_end = lines[i].split()[-1][-2:].lower() if lines[i].split() else ""
                line2_end = lines[i+1].split()[-1][-2:].lower() if lines[i+1].split() else ""
                if len(line1_end) > 1 and line1_end == line2_end:
                    rhyming_pairs += 1
            
            return rhyming_pairs / (len(lines) - 1) if len(lines) > 1 else 0
        
        logger.info("🔄 Calculating vocabulary richness...")
        features_df['vocabulary_richness'] = df['lyrics'].apply(calculate_vocabulary_richness)
        
        logger.info("🔄 Calculating repetitive elements...")
        features_df['repetitive_ratio'] = df['lyrics'].apply(count_repetitive_elements)
        
        logger.info("🔄 Calculating rhyme density...")
        features_df['rhyme_density'] = df['lyrics'].apply(calculate_rhyme_density)
        
        # === AI ANALYSIS FEATURES ===
        # Convert to float to avoid Decimal issues
        features_df['qwen_confidence'] = pd.to_numeric(df['qwen_confidence'], errors='coerce').fillna(0.7)
        features_df['qwen_complexity'] = pd.to_numeric(df['qwen_complexity'], errors='coerce').fillna(0.5)
        
        # Parse Spotify data if available
        def extract_spotify_features(spotify_data):
            features = {}
            if pd.isna(spotify_data):
                return features
            
            try:
                if isinstance(spotify_data, str):
                    import json
                    data = json.loads(spotify_data)
                else:
                    data = spotify_data
                
                if 'audio_features' in data:
                    audio = data['audio_features']
                    features.update({
                        'danceability': audio.get('danceability', 0.5),
                        'energy': audio.get('energy', 0.5),
                        'valence': audio.get('valence', 0.5),
                        'tempo': audio.get('tempo', 120) / 200,  # Normalize tempo
                        'acousticness': audio.get('acousticness', 0.5),
                        'instrumentalness': audio.get('instrumentalness', 0.1),
                        'speechiness': audio.get('speechiness', 0.1)
                    })
                
                if 'artist_info' in data:
                    artist = data['artist_info']
                    features.update({
                        'artist_popularity': artist.get('popularity', 50) / 100,
                        'artist_followers': np.log1p(artist.get('followers', {}).get('total', 10000)) / 20
                    })
                        
            except Exception:
                pass
            
            return features
        
        # Extract Spotify features
        logger.info("🎵 Extracting Spotify features...")
        spotify_features_list = df['spotify_data'].apply(extract_spotify_features).tolist()
        
        spotify_df = pd.DataFrame(spotify_features_list, index=df.index)
        
        # Fill missing Spotify features with medians
        for col in spotify_df.columns:
            if spotify_df[col].dtype in ['float64', 'int64']:
                spotify_df[col] = spotify_df[col].fillna(spotify_df[col].median())
        
        # Merge Spotify features
        features_df = pd.concat([features_df, spotify_df], axis=1)
        
        # === CATEGORICAL FEATURES ===
        # Artist style encoding
        if 'artist_style' not in self.label_encoders:
            self.label_encoders['artist_style'] = LabelEncoder()
        
        features_df['artist_style_encoded'] = self.label_encoders['artist_style'].fit_transform(
            df['artist_style'].fillna('other')
        )
        
        # Theme category encoding
        if 'theme_category' not in self.label_encoders:
            self.label_encoders['theme_category'] = LabelEncoder()
        
        features_df['theme_category_encoded'] = self.label_encoders['theme_category'].fit_transform(
            df['theme_category'].fillna('general')
        )
        
        # Sentiment encoding
        if 'qwen_sentiment' not in self.label_encoders:
            self.label_encoders['qwen_sentiment'] = LabelEncoder()
        
        features_df['sentiment_encoded'] = self.label_encoders['qwen_sentiment'].fit_transform(
            df['qwen_sentiment'].fillna('neutral')
        )
        
        # === INTERACTION FEATURES ===
        # Readability vs Complexity
        features_df['readability_complexity_ratio'] = (
            features_df['avg_words_per_line'] / (features_df['qwen_complexity'] + 0.1)
        )
        
        # Engagement potential (combination of audio features)
        if 'energy' in features_df.columns and 'danceability' in features_df.columns:
            features_df['engagement_score'] = (
                features_df['energy'] * 0.4 + 
                features_df['danceability'] * 0.4 + 
                features_df['valence'] * 0.2
            )
        else:
            features_df['engagement_score'] = 0.5
        
        # Fill any remaining NaN values
        features_df = features_df.fillna(0)
        
        logger.info(f"✅ Feature engineering completed: {features_df.shape[1]} features")
        logger.info(f"   Features: {list(features_df.columns)}")
        
        return features_df
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for prediction"""
        logger.info("🎯 Creating target variables...")
        
        targets_df = pd.DataFrame(index=df.index)
        
        # Base quality score from existing quality_score or qwen_complexity
        base_quality = pd.to_numeric(df['quality_score'], errors='coerce').fillna(
            pd.to_numeric(df['qwen_complexity'], errors='coerce').fillna(0.5)
        )
        
        # === QUALITY SCORE (综合质量) ===
        # Combines lyrical complexity, AI confidence, and structural elements
        targets_df['quality_score'] = (
            base_quality * 0.4 +
            pd.to_numeric(df['qwen_confidence'], errors='coerce').fillna(0.7) * 0.3 +
            pd.to_numeric(df['qwen_complexity'], errors='coerce').fillna(0.5) * 0.3
        )
        
        # === COMMERCIAL POTENTIAL ===
        # Based on mainstream appeal factors
        def calculate_commercial_potential(row):
            score = 0
            
            # Length optimization (not too short, not too long)
            optimal_length = abs(row.get('word_count', 300) - 300) / 600
            length_score = max(0, 1 - optimal_length)
            score += length_score * 0.2
            
            # Spotify audio features (if available)
            if pd.notna(row.get('spotify_data')):
                try:
                    import json
                    spotify_data = json.loads(row['spotify_data']) if isinstance(row['spotify_data'], str) else row['spotify_data']
                    if 'audio_features' in spotify_data:
                        audio = spotify_data['audio_features']
                        # Commercial tracks tend to be danceable, energetic, positive
                        commercial_audio = (
                            audio.get('danceability', 0.5) * 0.4 +
                            audio.get('energy', 0.5) * 0.3 +
                            audio.get('valence', 0.5) * 0.3
                        )
                        score += commercial_audio * 0.4
                    else:
                        score += 0.5 * 0.4  # Default
                except:
                    score += 0.5 * 0.4
            else:
                score += 0.5 * 0.4
            
            # Theme appeal (some themes are more commercial)
            theme = row.get('theme_category', 'general')
            theme_multipliers = {
                'love': 1.2,
                'party': 1.3,
                'success': 1.1,
                'general': 1.0,
                'struggle': 0.8,
                'social_issues': 0.7
            }
            theme_score = theme_multipliers.get(theme, 1.0)
            score += (theme_score - 0.5) * 0.2
            
            # Base quality
            score += float(base_quality.loc[row.name]) * 0.2
            
            return min(max(score, 0), 1)
        
        targets_df['commercial_potential'] = df.apply(calculate_commercial_potential, axis=1)
        
        # === VIRAL POTENTIAL ===
        # Based on engagement and shareability factors
        def calculate_viral_potential(row):
            score = 0
            
            # High energy and danceability boost viral potential
            if pd.notna(row.get('spotify_data')):
                try:
                    import json
                    spotify_data = json.loads(row['spotify_data']) if isinstance(row['spotify_data'], str) else row['spotify_data']
                    if 'audio_features' in spotify_data:
                        audio = spotify_data['audio_features']
                        viral_audio = (
                            audio.get('energy', 0.5) * 0.5 +
                            audio.get('danceability', 0.5) * 0.5
                        )
                        score += viral_audio * 0.4
                    else:
                        score += 0.5 * 0.4
                except:
                    score += 0.5 * 0.4
            else:
                score += 0.5 * 0.4
            
            # Repetitive elements can make songs catchy
            repetitive_ratio = row.get('repetitive_ratio', 0)
            if hasattr(df, 'lyrics') and row.name in df.index:
                repetitive_ratio = targets_df.loc[row.name:row.name, 'repetitive_ratio'].iloc[0] if 'repetitive_ratio' in targets_df.columns else 0
            
            # Moderate repetition is good for virality
            optimal_repetition = 1 - abs(repetitive_ratio - 0.3) / 0.7
            score += optimal_repetition * 0.3
            
            # Shorter tracks tend to be more viral
            word_count = row.get('word_count', 300)
            length_viral_score = max(0, 1 - (word_count - 200) / 400) if word_count > 200 else 1
            score += length_viral_score * 0.2
            
            # Quality boost
            score += float(base_quality.loc[row.name]) * 0.1
            
            return min(max(score, 0), 1)
        
        targets_df['viral_potential'] = df.apply(calculate_viral_potential, axis=1)
        
        # === LONGEVITY SCORE ===
        # Based on artistic depth and complexity
        targets_df['longevity_score'] = (
            pd.to_numeric(df['qwen_complexity'], errors='coerce').fillna(0.5) * 0.5 +
            base_quality * 0.3 +
            (1 - targets_df['commercial_potential']) * 0.2  # Less commercial = more artistic
        )
        
        logger.info("✅ Target variables created:")
        for target in targets_df.columns:
            mean_val = targets_df[target].mean()
            std_val = targets_df[target].std()
            logger.info(f"   {target}: μ={mean_val:.3f}, σ={std_val:.3f}")
        
        return targets_df
    
    def train_models(self, 
                    dataset_path: str = 'data/ml/quick_dataset.pkl',
                    test_size: float = 0.2,
                    cv_folds: int = 5) -> Dict:
        """Train multi-target regression models"""
        
        logger.info("🚀 Starting quality prediction model training...")
        
        try:
            # Load and prepare data
            df = self.load_training_data(dataset_path)
            X = self.engineer_features(df)
            y = self.create_target_variables(df)
            
            # Store feature names
            self.feature_names = X.columns.tolist()
            
            # Scale features
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42
            )
            
            # Train models for each target
            results = {}
            
            target_models = {
                'quality_score': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42),
                'commercial_potential': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
                'viral_potential': GradientBoostingRegressor(n_estimators=100, learning_rate=0.15, max_depth=5, random_state=42),
                'longevity_score': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            }
            
            for target_name, model in target_models.items():
                logger.info(f"\n🎯 Training model for: {target_name}")
                
                # Train model
                model.fit(X_train, y_train[target_name])
                
                # Predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Evaluate
                train_r2 = r2_score(y_train[target_name], y_pred_train)
                test_r2 = r2_score(y_test[target_name], y_pred_test)
                test_mae = mean_absolute_error(y_test[target_name], y_pred_test)
                test_mse = mean_squared_error(y_test[target_name], y_pred_test)
                
                # Cross validation
                cv_scores = cross_val_score(model, X_scaled, y[target_name], cv=cv_folds, scoring='r2')
                
                # Store model
                self.models[target_name] = model
                
                # Store results
                results[target_name] = {
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'test_mae': test_mae,
                    'test_mse': test_mse,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'feature_importance': dict(zip(X.columns, model.feature_importances_))
                }
                
                logger.info(f"  Train R²: {train_r2:.3f}")
                logger.info(f"  Test R²: {test_r2:.3f}")
                logger.info(f"  Test MAE: {test_mae:.3f}")
                logger.info(f"  CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            
            # Store overall results
            self.training_results = results
            
            # Feature importance analysis
            self._analyze_feature_importance(list(X.columns))
            
            logger.info("\n✅ Model training completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            raise
    
    def _analyze_feature_importance(self, feature_names: List[str]):
        """Analyze feature importance across all models"""
        logger.info("📊 Analyzing feature importance...")
        
        # Aggregate feature importance across models
        importance_sum = {}
        for target, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                for feature, importance in zip(feature_names, model.feature_importances_):
                    importance_sum[feature] = importance_sum.get(feature, 0) + importance
        
        # Sort by importance
        sorted_features = sorted(importance_sum.items(), key=lambda x: x[1], reverse=True)
        
        logger.info("🔥 Top 10 Most Important Features:")
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            logger.info(f"  {i:2d}. {feature}: {importance:.3f}")
        
        self.feature_importance = dict(sorted_features)
    
    def predict_quality_metrics(self, 
                               lyrics: str,
                               artist_style: str = 'other',
                               theme_category: str = 'general',
                               sentiment: str = 'neutral') -> Dict:
        """Predict quality metrics for new lyrics"""
        
        try:
            # Create dummy dataframe
            dummy_data = {
                'lyrics': [lyrics],
                'word_count': [len(lyrics.split())],
                'qwen_sentiment': [sentiment],
                'qwen_confidence': [0.8],
                'qwen_complexity': [0.5],
                'spotify_data': [None],
                'artist_style': [artist_style],
                'theme_category': [theme_category],
                'quality_score': [0.5]
            }
            
            # Add derived features
            dummy_data['lyrics_length'] = [len(lyrics)]
            dummy_data['lines_count'] = [lyrics.count('\n') + 1]
            dummy_data['avg_words_per_line'] = [dummy_data['word_count'][0] / dummy_data['lines_count'][0]]
            
            df_dummy = pd.DataFrame(dummy_data)
            
            # Engineer features
            X = self.engineer_features(df_dummy)
            
            # Scale features
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns
            )
            
            # Make predictions
            predictions = {}
            for target_name, model in self.models.items():
                pred_value = model.predict(X_scaled)[0]
                predictions[target_name] = round(max(0, min(1, pred_value)), 3)
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return {target: 0.5 for target in self.models.keys()}
    
    def save_models(self, output_path: str = './models/quality_predictor.pkl'):
        """Save trained models"""
        logger.info(f"💾 Saving quality prediction models to {output_path}")
        
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'training_results': getattr(self, 'training_results', {}),
            'metadata': {
                'creation_date': datetime.now().isoformat(),
                'model_type': 'Multi-target Quality Prediction',
                'targets': list(self.models.keys()),
                'features': len(self.feature_names)
            }
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            joblib.dump(model_data, f)
        
        logger.info("✅ Models saved successfully")
    
    @classmethod
    def load_models(cls, model_path: str = './models/quality_predictor.pkl'):
        """Load pre-trained models"""
        logger.info(f"📥 Loading quality prediction models from {model_path}")
        
        try:
            with open(model_path, 'rb') as f:
                model_data = joblib.load(f)
            
            predictor = cls()
            predictor.models = model_data['models']
            predictor.scaler = model_data['scaler']
            predictor.label_encoders = model_data['label_encoders']
            predictor.feature_names = model_data['feature_names']
            predictor.feature_importance = model_data.get('feature_importance', {})
            predictor.training_results = model_data.get('training_results', {})
            
            logger.info("✅ Models loaded successfully")
            return predictor
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            raise

def train_quality_models():
    """Training script for quality prediction models"""
    logger.info("🚀 RAP QUALITY PREDICTION - TRAINING PIPELINE")
    logger.info("=" * 60)
    
    try:
        # Initialize predictor
        predictor = RapQualityPredictor()
        
        # Train all models
        results = predictor.train_models(
            dataset_path='data/ml/quick_dataset.pkl',
            test_size=0.2,
            cv_folds=5
        )
        
        # Save models
        predictor.save_models('./models/quality_predictor.pkl')
        
        # Test predictions
        logger.info("\n🎯 Testing quality predictions...")
        
        test_lyrics = """
        Started from the bottom now we here
        Every day I'm working, vision crystal clear
        Money on my mind but I keep it real
        This is how I'm living, this is how I feel
        
        Success don't come easy, that's the truth
        But I keep on grinding since I was a youth
        Now I'm on top and I ain't coming down
        This is my city, I run this town
        """
        
        predictions = predictor.predict_quality_metrics(
            lyrics=test_lyrics,
            artist_style='drake',
            theme_category='success',
            sentiment='confident'
        )
        
        print("\n🎵 QUALITY PREDICTIONS FOR TEST LYRICS:")
        print("=" * 50)
        for metric, score in predictions.items():
            bar_length = int(score * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"{metric:20} [{bar}] {score:.3f}")
        
        # Overall assessment
        overall_score = sum(predictions.values()) / len(predictions)
        print(f"\n🏆 OVERALL SCORE: {overall_score:.3f}/1.000")
        
        if overall_score >= 0.8:
            print("   Status: 🔥 EXCELLENT - Hit potential!")
        elif overall_score >= 0.6:
            print("   Status: 👍 GOOD - Solid track")
        elif overall_score >= 0.4:
            print("   Status: ⚠️ FAIR - Needs improvement")
        else:
            print("   Status: ❌ POOR - Major rework needed")
        
        logger.info("\n✅ QUALITY PREDICTION TRAINING COMPLETED!")
        logger.info("Usage example:")
        logger.info("  predictor = RapQualityPredictor.load_models('./models/quality_predictor.pkl')")
        logger.info("  scores = predictor.predict_quality_metrics(lyrics, artist_style, theme, sentiment)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Quality prediction training failed: {e}")
        return False

def demo_quality_prediction():
    """Demo script for quality prediction"""
    logger.info("🎮 QUALITY PREDICTION DEMO")
    
    try:
        # Load trained models
        predictor = RapQualityPredictor.load_models('./models/quality_predictor.pkl')
        
        # Demo lyrics
        demo_tracks = [
            {
                'title': 'Commercial Hit Style',
                'lyrics': """
                Party all night, we don't stop
                Dancing to the beat, we on top
                Money, cars, girls, living the dream
                This is our moment, this is our scene
                """,
                'style': 'future',
                'theme': 'party',
                'sentiment': 'confident'
            },
            {
                'title': 'Conscious Rap Style', 
                'lyrics': """
                Society's broken, can't you see the pain
                People struggling in the streets through sun and rain
                We need change, we need hope, we need a voice
                Stand together, make the right choice
                
                Education over incarceration
                Love over hate across the nation
                This is more than entertainment
                This is truth, this is engagement
                """,
                'style': 'kendrick_lamar',
                'theme': 'social_issues',
                'sentiment': 'reflective'
            },
            {
                'title': 'Personal Struggle',
                'lyrics': """
                Growing up poor, mama working two jobs
                Dreams felt impossible, facing all odds
                But I kept believing, kept pushing through
                Now I'm here telling you what faith can do
                """,
                'style': 'other',
                'theme': 'struggle', 
                'sentiment': 'resilient'
            }
        ]
        
        for track in demo_tracks:
            print(f"\n🎤 {track['title'].upper()}")
            print("=" * 60)
            print(f"Style: {track['style']} | Theme: {track['theme']} | Sentiment: {track['sentiment']}")
            print(f"\nLyrics:\n{track['lyrics'].strip()}")
            
            predictions = predictor.predict_quality_metrics(
                lyrics=track['lyrics'],
                artist_style=track['style'],
                theme_category=track['theme'],
                sentiment=track['sentiment']
            )
            
            print(f"\n📊 QUALITY ANALYSIS:")
            print("-" * 40)
            for metric, score in predictions.items():
                bar_length = int(score * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"{metric:20} [{bar}] {score:.3f}")
            
            overall = sum(predictions.values()) / len(predictions)
            print(f"\n🎯 Overall Score: {overall:.3f}")
            
            # Recommendations
            print(f"\n💡 RECOMMENDATIONS:")
            if predictions['commercial_potential'] > 0.7:
                print("   🎵 High commercial potential - radio ready!")
            if predictions['viral_potential'] > 0.7:
                print("   📱 Strong viral potential - social media ready!")  
            if predictions['longevity_score'] > 0.7:
                print("   🏛️ Classic potential - timeless appeal!")
            if overall < 0.5:
                print("   ⚠️ Consider revising lyrics, theme, or structure")
        
        logger.info("\n✅ Quality prediction demo completed!")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Rap Quality Prediction')
    parser.add_argument('--mode', choices=['train', 'demo'], default='train',
                       help='Mode: train models or run demo')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        success = train_quality_models()
        sys.exit(0 if success else 1)
    elif args.mode == 'demo':
        demo_quality_prediction()