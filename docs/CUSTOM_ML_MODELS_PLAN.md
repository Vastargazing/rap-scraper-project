# 🚀 Custom ML Models для Rap Generation - Full Implementation Guide

> **From Production Dataset to Custom AI Models: Complete MLOps Pipeline**

---

## 🎯 Overview

Твоя база из **57,718 треков + 269,646 анализов** готова для создания custom ML models. Этот гайд покрывает полный цикл: от data preparation до production deployment.

**Current State**: Production-ready dataset с PostgreSQL + pgvector + AI analysis  
**Goal**: Custom ML models для rap generation, style transfer, quality prediction

---

## 📊 Phase 0: Data Preparation & Feature Engineering

### 1. **Consolidate Your Dataset**

```python
# scripts/ml/data_preparation.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sentence_transformers import SentenceTransformer
import pickle

class RapDatasetPreparator:
    """Подготовка dataset для ML моделей"""
    
    def __init__(self, db_path="data/rap_lyrics.db"):
        self.db_path = db_path
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def extract_features(self):
        """Извлечение и инженерия признаков"""
        
        # SQL запрос для объединения всех данных
        query = """
        SELECT 
            t.id, t.artist, t.title, t.lyrics,
            t.genre, t.explicit, t.word_count,
            
            -- AI Analysis Features
            ar.sentiment, ar.confidence, ar.themes,
            ar.complexity_score, ar.quality_rating,
            
            -- Spotify Audio Features  
            sf.danceability, sf.energy, sf.valence,
            sf.tempo, sf.acousticness, sf.instrumentalness,
            
            -- Spotify Artist Features
            sa.popularity as artist_popularity,
            sa.followers, sa.genres as artist_genres
            
        FROM tracks t
        LEFT JOIN analysis_results ar ON t.id = ar.track_id
        LEFT JOIN spotify_audio_features sf ON t.spotify_id = sf.track_spotify_id  
        LEFT JOIN spotify_artists sa ON t.artist = sa.artist_name
        WHERE t.lyrics IS NOT NULL AND ar.sentiment IS NOT NULL
        """
        
        df = pd.read_sql_query(query, sqlite3.connect(self.db_path))
        
        # Feature Engineering
        df['lyrics_length'] = df['lyrics'].str.len()
        df['words_per_line'] = df['word_count'] / df['lyrics'].str.count('\n')
        df['unique_words_ratio'] = df.apply(self._calculate_vocabulary_diversity, axis=1)
        
        # Text Embeddings для semantic features
        df['lyrics_embedding'] = list(self.encoder.encode(df['lyrics'].tolist()))
        
        return df
    
    def create_style_labels(self, df):
        """Создание labels для style transfer"""
        
        # Artist Style Labels (top 20 artists)
        top_artists = df['artist'].value_counts().head(20).index.tolist()
        df['artist_style'] = df['artist'].apply(
            lambda x: x if x in top_artists else 'other'
        )
        
        # Mood Labels (из AI analysis)
        mood_mapping = {
            'positive': 'upbeat',
            'negative': 'dark', 
            'neutral': 'chill'
        }
        df['mood_label'] = df['sentiment'].map(mood_mapping)
        
        # Quality Tiers
        df['quality_tier'] = pd.cut(
            df['quality_rating'], 
            bins=[0, 0.6, 0.8, 1.0], 
            labels=['low', 'medium', 'high']
        )
        
        return df
    
    def save_dataset(self, df, path="data/ml_dataset.pkl"):
        """Сохранение подготовленного dataset"""
        with open(path, 'wb') as f:
            pickle.dump(df, f)
        print(f"✅ Dataset saved: {len(df)} samples, {df.shape[1]} features")
```

---

## 🎨 Phase 1: Conditional Generation Model

### **Goal**: Генерация треков по условиям (стиль + настроение + тема)

```python
# models/conditional_generation.py
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TrainingArguments, Trainer
import wandb

class ConditionalRapGenerator:
    """Fine-tuned GPT-2 для conditional rap generation"""
    
    def __init__(self, model_name='gpt2-medium'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({
            'pad_token': '<PAD>',
            'bos_token': '<BOS>',
            'eos_token': '<EOS>'
        })
        
        # Custom tokens для conditioning
        special_tokens = [
            '<STYLE:drake>', '<STYLE:eminem>', '<STYLE:kendrick>',
            '<MOOD:dark>', '<MOOD:upbeat>', '<MOOD:chill>',  
            '<THEME:love>', '<THEME:struggle>', '<THEME:success>'
        ]
        self.tokenizer.add_tokens(special_tokens)
        
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
    def prepare_training_data(self, df):
        """Подготовка данных с conditioning tokens"""
        
        training_texts = []
        for _, row in df.iterrows():
            # Conditioning prefix
            style_token = f"<STYLE:{row['artist'].lower().replace(' ', '_')}>"
            mood_token = f"<MOOD:{row['mood_label']}>"  
            theme_token = f"<THEME:{self._extract_theme(row['themes'])}>"
            
            # Formatted training text
            text = f"{style_token} {mood_token} {theme_token} <BOS> {row['lyrics']} <EOS>"
            training_texts.append(text)
            
        return training_texts
    
    def fine_tune(self, training_texts, epochs=3, batch_size=4):
        """Fine-tuning на rap dataset"""
        
        # Tokenization
        encodings = self.tokenizer(
            training_texts, 
            truncation=True, 
            padding=True, 
            max_length=512,
            return_tensors='pt'
        )
        
        # Dataset class
        class RapDataset(torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.encodings = encodings
            
            def __getitem__(self, idx):
                return {key: val[idx] for key, val in self.encodings.items()}
            
            def __len__(self):
                return len(self.encodings['input_ids'])
        
        dataset = RapDataset(encodings)
        
        # Training configuration
        training_args = TrainingArguments(
            output_dir='./models/conditional_rap',
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            logging_dir='./logs',
            logging_steps=100,
            warmup_steps=100,
            learning_rate=5e-5,
        )
        
        # W&B tracking
        wandb.init(project="rap-generation")
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )
        
        trainer.train()
        trainer.save_model()
        
    def generate_conditional(self, style, mood, theme, max_length=200):
        """Генерация с условиями"""
        
        prompt = f"<STYLE:{style}> <MOOD:{mood}> <THEME:{theme}> <BOS>"
        
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=3,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        results = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            # Remove conditioning prefix
            text = text.split('<BOS>')[-1].strip()
            results.append(text)
            
        return results

# Training Script
def train_conditional_model():
    """Training pipeline для conditional generation"""
    
    # Load prepared dataset
    with open('data/ml_dataset.pkl', 'rb') as f:
        df = pickle.load(f)
    
    # Initialize model
    generator = ConditionalRapGenerator()
    
    # Prepare training data
    training_texts = generator.prepare_training_data(df)
    print(f"📚 Training samples: {len(training_texts)}")
    
    # Fine-tune
    generator.fine_tune(training_texts, epochs=5, batch_size=2)
    
    # Test generation
    samples = generator.generate_conditional('drake', 'upbeat', 'success')
    for i, sample in enumerate(samples):
        print(f"\n🎤 Sample {i+1}:\n{sample}")

if __name__ == "__main__":
    train_conditional_model()
```

### **Usage Example**:

```python
# Generate Drake-style upbeat song about success
generator = ConditionalRapGenerator.from_pretrained('./models/conditional_rap')
lyrics = generator.generate_conditional('drake', 'upbeat', 'success')
print(lyrics[0])
```

---

## 🎭 Phase 2: Style Transfer Model

### **Goal**: "Сделай как у Drake, но на тему любви"

```python
# models/style_transfer.py
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import DataLoader, Dataset

class RapStyleTransfer:
    """Style transfer для rap lyrics using T5 architecture"""
    
    def __init__(self, model_name='t5-base'):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Add task prefix для T5
        self.task_prefix = "transfer rap style: "
        
    def prepare_style_transfer_data(self, df):
        """Создание пар (source_style → target_style) для одной темы"""
        
        transfer_pairs = []
        
        # Group by theme для cross-style pairs
        for theme in df['themes'].unique():
            theme_songs = df[df['themes'] == theme]
            
            # Create pairs: (artist_A_song, artist_B_style) → artist_B_version
            artists = theme_songs['artist'].unique()
            
            for source_artist in artists:
                for target_artist in artists:
                    if source_artist != target_artist:
                        source_songs = theme_songs[theme_songs['artist'] == source_artist]
                        target_songs = theme_songs[theme_songs['artist'] == target_artist]
                        
                        if len(source_songs) > 0 and len(target_songs) > 0:
                            source_song = source_songs.sample(1).iloc[0]
                            target_song = target_songs.sample(1).iloc[0]
                            
                            # Input: original + target style
                            input_text = f"source: {source_song['lyrics']} target_style: {target_artist}"
                            target_text = target_song['lyrics']
                            
                            transfer_pairs.append((input_text, target_text))
        
        return transfer_pairs
    
    def fine_tune_style_transfer(self, transfer_pairs, epochs=3):
        """Fine-tuning T5 для style transfer"""
        
        class StyleTransferDataset(Dataset):
            def __init__(self, pairs, tokenizer, max_length=512):
                self.pairs = pairs
                self.tokenizer = tokenizer
                self.max_length = max_length
                
            def __len__(self):
                return len(self.pairs)
            
            def __getitem__(self, idx):
                source, target = self.pairs[idx]
                
                # Tokenize source
                source_encoding = self.tokenizer(
                    self.task_prefix + source,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                # Tokenize target
                target_encoding = self.tokenizer(
                    target,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                return {
                    'input_ids': source_encoding['input_ids'].flatten(),
                    'attention_mask': source_encoding['attention_mask'].flatten(),
                    'labels': target_encoding['input_ids'].flatten()
                }
        
        # Create dataset
        dataset = StyleTransferDataset(transfer_pairs, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Training loop
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-5)
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}, Average Loss: {total_loss/len(dataloader):.4f}")
        
        # Save model
        self.model.save_pretrained('./models/style_transfer')
        self.tokenizer.save_pretrained('./models/style_transfer')
    
    def transfer_style(self, original_lyrics, target_artist_style):
        """Perform style transfer"""
        
        input_text = f"{self.task_prefix}source: {original_lyrics} target_style: {target_artist_style}"
        
        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            max_length=512,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=200,
                num_beams=4,
                early_stopping=True,
                temperature=0.8
            )
        
        transferred_lyrics = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return transferred_lyrics

# Usage Example
def demo_style_transfer():
    """Демо style transfer"""
    
    transfer_model = RapStyleTransfer.from_pretrained('./models/style_transfer')
    
    original = """
    I wake up every morning, hustle on my mind
    Making moves in silence, leaving doubters behind
    """
    
    # Transfer to Drake's style
    drake_version = transfer_model.transfer_style(original, "drake")
    print(f"🎤 Drake Style:\n{drake_version}")
    
    # Transfer to Kendrick's style  
    kendrick_version = transfer_model.transfer_style(original, "kendrick_lamar")
    print(f"🎤 Kendrick Style:\n{kendrick_version}")
```

---

## 💎 Phase 3: Quality Prediction Model

### **Goal**: ML модель для оценки коммерческого потенциала

```python
# models/quality_prediction.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import shap

class RapQualityPredictor:
    """ML модель для предсказания качества и коммерческого потенциала"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        self.feature_importance = None
        
    def engineer_features(self, df):
        """Feature engineering для quality prediction"""
        
        features = pd.DataFrame()
        
        # Текстовые признаки
        features['lyrics_length'] = df['lyrics'].str.len()
        features['word_count'] = df['word_count']
        features['unique_words_ratio'] = df['unique_words_ratio']
        features['lines_count'] = df['lyrics'].str.count('\n')
        features['avg_words_per_line'] = features['word_count'] / features['lines_count']
        
        # AI анализ признаки
        features['complexity_score'] = df['complexity_score']
        features['ai_quality_rating'] = df['quality_rating'] 
        features['sentiment_confidence'] = df['confidence']
        
        # Spotify audio features
        audio_features = ['danceability', 'energy', 'valence', 'tempo', 
                         'acousticness', 'instrumentalness']
        for feature in audio_features:
            features[feature] = df[feature]
        
        # Artist features
        features['artist_popularity'] = df['artist_popularity']
        features['artist_followers_log'] = np.log1p(df['followers'])
        
        # Categorical encoding
        categorical_features = ['genre', 'sentiment', 'artist_style', 'mood_label']
        for cat_feature in categorical_features:
            if cat_feature not in self.label_encoders:
                self.label_encoders[cat_feature] = LabelEncoder()
                features[f'{cat_feature}_encoded'] = self.label_encoders[cat_feature].fit_transform(df[cat_feature].fillna('unknown'))
            else:
                features[f'{cat_feature}_encoded'] = self.label_encoders[cat_feature].transform(df[cat_feature].fillna('unknown'))
        
        # Advanced features
        features['theme_diversity'] = df['themes'].str.count(',') + 1  # Number of themes
        features['explicit_flag'] = df['explicit'].astype(int)
        
        return features
    
    def create_target_variables(self, df):
        """Создание target variables для prediction"""
        
        targets = {}
        
        # Commercial potential (composite score)
        targets['commercial_potential'] = (
            0.3 * df['artist_popularity'] / 100 +
            0.2 * df['danceability'] + 
            0.2 * df['energy'] +
            0.1 * df['valence'] +
            0.2 * df['quality_rating']
        )
        
        # Artistic quality
        targets['artistic_quality'] = df['quality_rating']
        
        # Viral potential (energy + danceability + popularity)
        targets['viral_potential'] = (
            0.4 * df['energy'] +
            0.4 * df['danceability'] +
            0.2 * (df['artist_popularity'] / 100)
        )
        
        # Longevity score (complexity + lyrical quality)
        targets['longevity_score'] = (
            0.6 * df['complexity_score'] +
            0.4 * df['quality_rating']
        )
        
        return targets
    
    def train_multi_target_model(self, df):
        """Обучение multi-target regression model"""
        
        # Feature engineering
        X = self.engineer_features(df)
        X_scaled = self.scaler.fit_transform(X)
        
        # Target variables
        targets = self.create_target_variables(df)
        
        models = {}
        scores = {}
        
        for target_name, y in targets.items():
            print(f"\n🎯 Training model for: {target_name}")
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Model training
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Evaluation
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            # Cross validation
            cv_scores = cross_val_score(model, X_scaled, y, cv=5)
            
            models[target_name] = model
            scores[target_name] = {
                'r2': r2,
                'mse': mse,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"  R² Score: {r2:.3f}")
            print(f"  MSE: {mse:.3f}")
            print(f"  CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        self.models = models
        self.feature_names = X.columns.tolist()
        
        # Feature importance analysis
        self.analyze_feature_importance(X, targets)
        
        # Save models
        self.save_models()
        
        return scores
    
    def analyze_feature_importance(self, X, targets):
        """SHAP analysis для feature importance"""
        
        import matplotlib.pyplot as plt
        
        for target_name, model in self.models.items():
            print(f"\n📊 Feature Importance Analysis: {target_name}")
            
            # SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X.sample(1000))  # Sample for speed
            
            # Top 10 features
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': np.abs(shap_values).mean(axis=0)
            }).sort_values('importance', ascending=False).head(10)
            
            print(feature_importance)
            
            # Save plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X.sample(1000), show=False)
            plt.savefig(f'./models/feature_importance_{target_name}.png', 
                       bbox_inches='tight', dpi=300)
            plt.close()
    
    def predict_quality_metrics(self, lyrics, artist_features=None):
        """Предсказание всех quality metrics для новых lyrics"""
        
        # Create dummy dataframe для feature engineering
        dummy_data = {
            'lyrics': [lyrics],
            'word_count': [len(lyrics.split())],
            # Default values - в production брать из API/анализа
            'complexity_score': [0.5],
            'quality_rating': [0.5],
            'confidence': [0.8],
            'danceability': [0.5],
            'energy': [0.5],
            'valence': [0.5],
            'tempo': [120],
            'acousticness': [0.5],
            'instrumentalness': [0.1],
            'artist_popularity': [50],
            'followers': [1000000],
            'genre': ['rap'],
            'sentiment': ['neutral'],
            'artist_style': ['other'],
            'mood_label': ['chill'],
            'themes': ['life'],
            'explicit': [False]
        }
        
        if artist_features:
            dummy_data.update(artist_features)
        
        df_dummy = pd.DataFrame(dummy_data)
        
        # Feature engineering
        X = self.engineer_features(df_dummy)
        X_scaled = self.scaler.transform(X)
        
        # Predictions
        predictions = {}
        for target_name, model in self.models.items():
            pred_value = model.predict(X_scaled)[0]
            predictions[target_name] = round(pred_value, 3)
        
        return predictions
    
    def save_models(self):
        """Сохранение обученных моделей"""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, './models/quality_predictor.pkl')
        print("✅ Quality prediction models saved")

# Training Pipeline
def train_quality_models():
    """Training pipeline для quality prediction"""
    
    # Load dataset
    with open('data/ml_dataset.pkl', 'rb') as f:
        df = pickle.load(f)
    
    # Initialize predictor
    predictor = RapQualityPredictor()
    
    # Train models
    scores = predictor.train_multi_target_model(df)
    
    # Demo predictions
    test_lyrics = """
    Started from the bottom now we here
    All my people with me, crystal clear
    Money talks but wisdom speaks louder  
    Every day I'm getting stronger, prouder
    """
    
    predictions = predictor.predict_quality_metrics(test_lyrics)
    print(f"\n🎤 Quality Predictions for Test Lyrics:")
    for metric, value in predictions.items():
        print(f"  {metric}: {value}")

if __name__ == "__main__":
    train_quality_models()
```

---

## 📈 Phase 4: Trend Analysis Model

### **Goal**: Предсказание популярных тем и музыкальных трендов

```python
# models/trend_analysis.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class RapTrendAnalyzer:
    """Анализ трендов в rap music и предсказание популярных тем"""
    
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.topic_clusters = None
        self.trend_model = None
        
    def temporal_analysis(self, df):
        """Временной анализ изменения тем и настроений"""
        
        # Convert scraped_date to datetime
        df['date'] = pd.to_datetime(df['scraped_date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.to_period('M')
        
        # Тренды по годам
        yearly_trends = df.groupby(['year', 'sentiment']).size().unstack(fill_value=0)
        yearly_trends_pct = yearly_trends.div(yearly_trends.sum(axis=1), axis=0) * 100
        
        # Популярные темы по времени
        theme_evolution = []
        for period in df['month'].unique():
            period_data = df[df['month'] == period]
            
            # Extract top themes
            all_themes = []
            for themes_str in period_data['themes'].dropna():
                all_themes.extend(themes_str.split(','))
            
            theme_counts = pd.Series(all_themes).str.strip().value_counts().head(10)
            
            theme_evolution.append({
                'period': period,
                'top_themes': theme_counts.to_dict()
            })
        
        return yearly_trends_pct, theme_evolution
    
    def cluster_music_styles(self, df):
        """Кластеризация музыкальных стилей на основе audio features"""
        
        # Audio features для кластеризации
        audio_features = ['danceability', 'energy', 'valence', 'tempo',
                         'acousticness', 'instrumentalness', 'complexity_score']
        
        # Prepare data
        cluster_data = df[audio_features].dropna()
        
        # Scaling
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(cluster_data)
        
        # KMeans clustering
        kmeans = KMeans(n_clusters=8, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        
        # Add cluster labels
        df_clusters = df[df.index.isin(cluster_data.index)].copy()
        df_clusters['style_cluster'] = clusters
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(8):
            cluster_songs = df_clusters[df_clusters['style_cluster'] == cluster_id]
            
            cluster_analysis[cluster_id] = {
                'count': len(cluster_songs),
                'top_artists': cluster_songs['artist'].value_counts().head(5).to_dict(),
                'avg_features': cluster_songs[audio_features].mean().to_dict(),
                'sentiment_dist': cluster_songs['sentiment'].value_counts().to_dict(),
                'popularity': cluster_songs['artist_popularity'].mean()
            }
        
        self.cluster_analysis = cluster_analysis
        return df_clusters, cluster_analysis
    
    def predict_emerging_trends(self, df, forecast_months=6):
        """Предсказание emerging trends на основе роста популярности"""
        
        # Prepare temporal data
        df['date'] = pd.to_datetime(df['scraped_date'])
        df = df.sort_values('date')
        
        # Extract themes over time
        monthly_themes = {}
        for month in pd.date_range(df['date'].min(), df['date'].max(), freq='M'):
            month_data = df[df['date'].dt.to_period('M') == month.to_period('M')]
            
            all_themes = []
            for themes_str in month_data['themes'].dropna():
                all_themes.extend([t.strip() for t in themes_str.split(',')])
            
            theme_counts = pd.Series(all_themes).value_counts()
            monthly_themes[month] = theme_counts.to_dict()
        
        # Calculate growth rates
        trend_predictions = {}
        all_themes = set()
        for themes_dict in monthly_themes.values():
            all_themes.update(themes_dict.keys())
        
        for theme in all_themes:
            if theme and len(theme) > 2:  # Skip empty themes
                monthly_counts = []
                months = []
                
                for month, themes_dict in monthly_themes.items():
                    count = themes_dict.get(theme, 0)
                    monthly_counts.append(count)
                    months.append(month)
                
                if len(monthly_counts) >= 3 and sum(monthly_counts) >= 10:  # Minimum threshold
                    # Calculate trend using linear regression
                    x = np.arange(len(monthly_counts))
                    if len(monthly_counts) > 1 and np.std(monthly_counts) > 0:
                        trend_slope = np.polyfit(x, monthly_counts, 1)[0]
                        
                        # Predict future popularity
                        future_months = len(monthly_counts) + forecast_months
                        predicted_count = trend_slope * future_months + np.mean(monthly_counts)
                        
                        trend_predictions[theme] = {
                            'current_popularity': monthly_counts[-1] if monthly_counts else 0,
                            'trend_slope': trend_slope,
                            'predicted_popularity': max(0, predicted_count),
                            'growth_rate': trend_slope / (np.mean(monthly_counts) + 1) * 100
                        }
        
        # Sort by predicted growth
        emerging_trends = sorted(
            trend_predictions.items(),
            key=lambda x: x[1]['growth_rate'],
            reverse=True
        )[:20]
        
        return emerging_trends, trend_predictions
    
    def analyze_viral_patterns(self, df):
        """Анализ паттернов viral треков"""
        
        # Define viral threshold (top 20% popularity + high energy/danceability)
        popularity_threshold = df['artist_popularity'].quantile(0.8)
        energy_threshold = df['energy'].quantile(0.7)
        dance_threshold = df['danceability'].quantile(0.7)
        
        viral_tracks = df[
            (df['artist_popularity'] >= popularity_threshold) &
            (df['energy'] >= energy_threshold) &
            (df['danceability'] >= dance_threshold)
        ]
        
        # Analyze viral patterns
        viral_analysis = {
            'count': len(viral_tracks),
            'avg_features': {
                'energy': viral_tracks['energy'].mean(),
                'danceability': viral_tracks['danceability'].mean(),
                'valence': viral_tracks['valence'].mean(),
                'tempo': viral_tracks['tempo'].mean(),
                'lyrics_length': viral_tracks['lyrics'].str.len().mean(),
                'word_count': viral_tracks['word_count'].mean(),
                'complexity': viral_tracks['complexity_score'].mean()
            },
            'common_themes': self._extract_common_patterns(viral_tracks['themes'].dropna()),
            'sentiment_distribution': viral_tracks['sentiment'].value_counts().to_dict(),
            'top_artists': viral_tracks['artist'].value_counts().head(10).to_dict()
        }
        
        return viral_tracks, viral_analysis
    
    def create_trend_dashboard(self, df):
        """Создание интерактивного dashboard для трендов"""
        
        # Temporal analysis
        yearly_trends, theme_evolution = self.temporal_analysis(df)
        
        # Clustering
        df_clustered, cluster_analysis = self.cluster_music_styles(df)
        
        # Emerging trends
        emerging_trends, _ = self.predict_emerging_trends(df)
        
        # Viral patterns
        viral_tracks, viral_analysis = self.analyze_viral_patterns(df)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sentiment Trends Over Time', 'Music Style Clusters', 
                          'Emerging Themes', 'Viral Track Features'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # Plot 1: Sentiment trends
        for sentiment in yearly_trends.columns:
            fig.add_trace(
                go.Scatter(
                    x=yearly_trends.index,
                    y=yearly_trends[sentiment],
                    mode='lines+markers',
                    name=f'{sentiment} sentiment',
                    line=dict(width=3)
                ),
                row=1, col=1
            )
        
        # Plot 2: Style clusters (PCA visualization)
        if len(df_clustered) > 0:
            audio_features = ['danceability', 'energy', 'valence', 'tempo',
                             'acousticness', 'instrumentalness', 'complexity_score']
            
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(df_clustered[audio_features].fillna(0))
            
            for cluster_id in df_clustered['style_cluster'].unique():
                cluster_data = df_clustered[df_clustered['style_cluster'] == cluster_id]
                cluster_indices = cluster_data.index
                
                fig.add_trace(
                    go.Scatter(
                        x=pca_result[df_clustered.index.get_indexer(cluster_indices), 0],
                        y=pca_result[df_clustered.index.get_indexer(cluster_indices), 1],
                        mode='markers',
                        name=f'Cluster {cluster_id}',
                        text=cluster_data['artist'].tolist(),
                        hovertemplate='Artist: %{text}<extra></extra>'
                    ),
                    row=1, col=2
                )
        
        # Plot 3: Emerging trends
        if emerging_trends:
            trend_themes = [trend[0] for trend in emerging_trends[:15]]
            growth_rates = [trend[1]['growth_rate'] for trend in emerging_trends[:15]]
            
            fig.add_trace(
                go.Bar(
                    x=growth_rates,
                    y=trend_themes,
                    orientation='h',
                    name='Growth Rate %',
                    marker=dict(color=growth_rates, colorscale='Viridis')
                ),
                row=2, col=1
            )
        
        # Plot 4: Viral track features
        viral_features = list(viral_analysis['avg_features'].keys())
        viral_values = list(viral_analysis['avg_features'].values())
        
        fig.add_trace(
            go.Bar(
                x=viral_features,
                y=viral_values,
                name='Viral Features',
                marker=dict(color='red', opacity=0.7)
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Rap Music Trend Analysis Dashboard",
            showlegend=True
        )
        
        # Save dashboard
        fig.write_html('./models/trend_dashboard.html')
        fig.show()
        
        return fig
    
    def generate_trend_report(self, df):
        """Генерация comprehensive trend report"""
        
        report = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'dataset_size': len(df),
            'date_range': f"{df['scraped_date'].min()} to {df['scraped_date'].max()}"
        }
        
        # Temporal analysis
        yearly_trends, theme_evolution = self.temporal_analysis(df)
        report['sentiment_trends'] = yearly_trends.to_dict()
        
        # Clustering analysis
        df_clustered, cluster_analysis = self.cluster_music_styles(df)
        report['style_clusters'] = cluster_analysis
        
        # Emerging trends
        emerging_trends, _ = self.predict_emerging_trends(df)
        report['emerging_trends'] = dict(emerging_trends[:10])
        
        # Viral analysis
        viral_tracks, viral_analysis = self.analyze_viral_patterns(df)
        report['viral_patterns'] = viral_analysis
        
        # Key insights
        report['key_insights'] = self._generate_insights(report)
        
        # Save report
        import json
        with open('./models/trend_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("📊 Trend Analysis Report Generated")
        return report
    
    def _extract_common_patterns(self, themes_series):
        """Extract common theme patterns"""
        all_themes = []
        for themes_str in themes_series:
            if pd.notna(themes_str):
                all_themes.extend([t.strip() for t in themes_str.split(',')])
        return pd.Series(all_themes).value_counts().head(15).to_dict()
    
    def _generate_insights(self, report):
        """Generate key insights from analysis"""
        insights = []
        
        # Top emerging trend
        if report['emerging_trends']:
            top_trend = list(report['emerging_trends'].keys())[0]
            insights.append(f"🔥 Fastest growing theme: '{top_trend}'")
        
        # Dominant sentiment
        latest_year = max(report['sentiment_trends'].keys())
        dominant_sentiment = max(report['sentiment_trends'][latest_year], 
                               key=report['sentiment_trends'][latest_year].get)
        insights.append(f"😊 Dominant sentiment in {latest_year}: {dominant_sentiment}")
        
        # Viral pattern insight
        viral_energy = report['viral_patterns']['avg_features']['energy']
        viral_dance = report['viral_patterns']['avg_features']['danceability']
        insights.append(f"🎵 Viral tracks average: {viral_energy:.2f} energy, {viral_dance:.2f} danceability")
        
        # Cluster insight
        largest_cluster = max(report['style_clusters'], 
                            key=lambda x: report['style_clusters'][x]['count'])
        insights.append(f"🎯 Largest music style cluster: {largest_cluster} with {report['style_clusters'][largest_cluster]['count']} tracks")
        
        return insights

# Training and Analysis Pipeline
def run_trend_analysis():
    """Complete trend analysis pipeline"""
    
    # Load dataset
    with open('data/ml_dataset.pkl', 'rb') as f:
        df = pickle.load(f)
    
    # Initialize analyzer
    analyzer = RapTrendAnalyzer()
    
    # Generate comprehensive report
    report = analyzer.generate_trend_report(df)
    
    # Create interactive dashboard
    dashboard = analyzer.create_trend_dashboard(df)
    
    # Print key insights
    print("\n🔍 KEY INSIGHTS:")
    for insight in report['key_insights']:
        print(f"  {insight}")
    
    # Predict next 6 months trends
    emerging_trends, _ = analyzer.predict_emerging_trends(df, forecast_months=6)
    
    print(f"\n📈 TOP 10 EMERGING TRENDS (Next 6 months):")
    for i, (theme, data) in enumerate(emerging_trends[:10], 1):
        print(f"  {i}. {theme}: {data['growth_rate']:.1f}% growth rate")
    
    return report, dashboard

if __name__ == "__main__":
    run_trend_analysis()
```

---

## 🚀 Phase 5: Production ML Pipeline Integration

### **Integration с существующей Kubernetes архитектурой**

```python
# services/ml_api_service.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
import torch
import joblib
import pickle

# Load models at startup
class MLModelService:
    def __init__(self):
        self.models = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all trained models"""
        try:
            # Conditional Generation
            from models.conditional_generation import ConditionalRapGenerator
            self.models['generator'] = ConditionalRapGenerator.from_pretrained('./models/conditional_rap')
            
            # Style Transfer
            from models.style_transfer import RapStyleTransfer  
            self.models['style_transfer'] = RapStyleTransfer.from_pretrained('./models/style_transfer')
            
            # Quality Prediction
            with open('./models/quality_predictor.pkl', 'rb') as f:
                quality_data = joblib.load(f)
                self.models['quality'] = quality_data
            
            # Trend Analysis
            with open('./models/trend_analysis_report.json', 'r') as f:
                self.models['trends'] = json.load(f)
                
            print("✅ All ML models loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")

app = FastAPI(title="Rap ML API", version="2.0.0")
ml_service = MLModelService()

# Request/Response models
class GenerationRequest(BaseModel):
    style: str  # drake, eminem, kendrick
    mood: str   # upbeat, dark, chill
    theme: str  # love, success, struggle
    max_length: Optional[int] = 200

class StyleTransferRequest(BaseModel):
    original_lyrics: str
    target_style: str

class QualityPredictionRequest(BaseModel):
    lyrics: str
    artist_features: Optional[Dict] = None

class GenerationResponse(BaseModel):
    generated_lyrics: List[str]
    generation_time: float
    model_confidence: float

# API Endpoints
@app.post("/generate", response_model=GenerationResponse)
async def generate_lyrics(request: GenerationRequest):
    """Generate rap lyrics with conditions"""
    try:
        start_time = time.time()
        
        lyrics = ml_service.models['generator'].generate_conditional(
            style=request.style,
            mood=request.mood, 
            theme=request.theme,
            max_length=request.max_length
        )
        
        generation_time = time.time() - start_time
        
        return GenerationResponse(
            generated_lyrics=lyrics,
            generation_time=generation_time,
            model_confidence=0.85  # TODO: implement confidence calculation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/style-transfer")
async def transfer_style(request: StyleTransferRequest):
    """Transfer lyrical style"""
    try:
        transferred = ml_service.models['style_transfer'].transfer_style(
            request.original_lyrics,
            request.target_style
        )
        
        return {"transferred_lyrics": transferred}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-quality")
async def predict_quality(request: QualityPredictionRequest):
    """Predict quality metrics for lyrics"""
    try:
        # Reconstruct predictor
        quality_data = ml_service.models['quality']
        predictor = RapQualityPredictor()
        predictor.models = quality_data['models']
        predictor.scaler = quality_data['scaler']
        predictor.label_encoders = quality_data['label_encoders']
        
        predictions = predictor.predict_quality_metrics(
            request.lyrics,
            request.artist_features
        )
        
        return {"quality_predictions": predictions}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trends")
async def get_current_trends():
    """Get current trend analysis"""
    return ml_service.models['trends']

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = {name: "loaded" for name in ml_service.models.keys()}
    return {"status": "healthy", "models": model_status}
```

### **Docker Configuration для ML Service**

```dockerfile
# Dockerfile.ml
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt

# Copy models and code
COPY models/ ./models/
COPY services/ ./services/
COPY src/ ./src/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

EXPOSE 8001
CMD ["uvicorn", "services.ml_api_service:app", "--host", "0.0.0.0", "--port", "8001"]
```

### **Kubernetes ML Service Deployment**

```yaml
# k8s/ml/ml-service-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rap-ml-service
  namespace: rap-analyzer
spec:
  replicas: 2
  selector:
    matchLabels:
      app: rap-ml-service
  template:
    metadata:
      labels:
        app: rap-ml-service
    spec:
      containers:
      - name: ml-service
        image: rap-analyzer:ml-latest
        ports:
        - containerPort: 8001
        env:
        - name: MODEL_PATH
          value: "/app/models"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health  
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: ml-models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: rap-ml-service
  namespace: rap-analyzer
spec:
  selector:
    app: rap-ml-service
  ports:
  - port: 8001
    targetPort: 8001
  type: ClusterIP
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ml-models-pvc
  namespace: rap-analyzer
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

---

## 📋 Phase 6: MLOps Pipeline & Monitoring

### **Model Training Pipeline**

```python
# scripts/ml/training_pipeline.py
import mlflow
import mlflow.pytorch
import wandb
from datetime import datetime
import argparse

class MLTrainingPipeline:
    """Automated training pipeline с MLOps best practices"""
    
    def __init__(self, config_path="ml_config.yaml"):
        self.config = self.load_config(config_path)
        self.setup_tracking()
        
    def setup_tracking(self):
        """Setup MLflow и W&B tracking"""
        mlflow.set_tracking_uri("http://mlflow-service:5000")
        mlflow.set_experiment("rap-generation")
        
        wandb.init(
            project="rap-ml-models",
            config=self.config
        )
    
    def train_all_models(self):
        """Train all models in sequence"""
        
        models_to_train = [
            ("conditional_generation", self.train_conditional_model),
            ("style_transfer", self.train_style_transfer),
            ("quality_prediction", self.train_quality_models),
            ("trend_analysis", self.run_trend_analysis)
        ]
        
        results = {}
        
        for model_name, train_func in models_to_train:
            print(f"\n🚀 Training {model_name}...")
            
            with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                try:
                    # Training
                    metrics = train_func()
                    
                    # Log metrics
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(metric_name, value)
                        wandb.log({f"{model_name}_{metric_name}": value})
                    
                    # Log model artifacts
                    mlflow.log_artifacts(f"./models/{model_name}")
                    
                    results[model_name] = {"status": "success", "metrics": metrics}
                    
                except Exception as e:
                    print(f"❌ Failed to train {model_name}: {e}")
                    results[model_name] = {"status": "failed", "error": str(e)}
        
        return results
    
    def validate_models(self):
        """Validate all trained models"""
        validation_results = {}
        
        # Test conditional generation
        try:
            from models.conditional_generation import ConditionalRapGenerator
            generator = ConditionalRapGenerator.from_pretrained('./models/conditional_rap')
            
            test_lyrics = generator.generate_conditional('drake', 'upbeat', 'success')
            validation_results['generation'] = {
                'status': 'pass' if len(test_lyrics) > 0 else 'fail',
                'sample_output': test_lyrics[0][:100] if test_lyrics else None
            }
        except Exception as e:
            validation_results['generation'] = {'status': 'fail', 'error': str(e)}
        
        # Test quality prediction
        try:
            with open('./models/quality_predictor.pkl', 'rb') as f:
                quality_data = joblib.load(f)
            
            validation_results['quality'] = {'status': 'pass', 'models_count': len(quality_data['models'])}
        except Exception as e:
            validation_results['quality'] = {'status': 'fail', 'error': str(e)}
        
        return validation_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['all', 'generation', 'style', 'quality', 'trends'], 
                       default='all', help='Model to train')
    parser.add_argument('--config', default='ml_config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    pipeline = MLTrainingPipeline(args.config)
    
    if args.model == 'all':
        results = pipeline.train_all_models()
    else:
        # Train specific model
        results = {args.model: pipeline.train_specific_model(args.model)}
    
    # Validation
    validation_results = pipeline.validate_models()
    
    print("\n📊 TRAINING RESULTS:")
    for model, result in results.items():
        print(f"  {model}: {result['status']}")
    
    print("\n✅ VALIDATION RESULTS:")
    for model, result in validation_results.items():
        print(f"  {model}: {result['status']}")

if __name__ == "__main__":
    main()
```

---

## 🎯 Usage Examples & Demo

### **Complete Workflow Demo**

```python
# demo/ml_models_demo.py
"""
🎯 Complete ML Models Demo
Демонстрация всех custom ML capabilities
"""

async def demo_complete_pipeline():
    """Демо полного ML pipeline"""
    
    print("🚀 RAP ML MODELS DEMO")
    print("=" * 50)
    
    # 1. Conditional Generation
    print("\n1️⃣ CONDITIONAL GENERATION:")
    generator = ConditionalRapGenerator.from_pretrained('./models/conditional_rap')
    
    conditions = [
        ('drake', 'upbeat', 'success'),
        ('eminem', 'dark', 'struggle'),
        ('kendrick', 'chill', 'social_issues')
    ]
    
    for style, mood, theme in conditions:
        lyrics = generator.generate_conditional(style, mood, theme)
        print(f"\n🎤 {style.title()} style - {mood} - {theme}:")
        print(f"   {lyrics[0][:200]}...")
    
    # 2. Style Transfer
    print("\n2️⃣ STYLE TRANSFER:")
    transfer_model = RapStyleTransfer.from_pretrained('./models/style_transfer')
    
    original = """
    Started from the bottom, now I'm here
    Working every day to make it clear
    Success don't come easy, that's for sure
    But I keep grinding, staying pure
    """
    
    styles = ['drake', 'eminem', 'kendrick_lamar']
    
    for style in styles:
        transferred = transfer_model.transfer_style(original, style)
        print(f"\n🎭 {style.title()} version:")
        print(f"   {transferred[:150]}...")
    
    # 3. Quality Prediction
    print("\n3️⃣ QUALITY PREDICTION:")
    with open('./models/quality_predictor.pkl', 'rb') as f:
        quality_data = joblib.load(f)
    
    predictor = RapQualityPredictor()
    predictor.models = quality_data['models']
    predictor.scaler = quality_data['scaler']
    predictor.label_encoders = quality_data['label_encoders']
    
    test_lyrics = """
    Life's a game of chess, I'm thinking three moves ahead
    Every bar I drop is like bread for the soul that's underfed
    Poetry in motion, every word's like a magic potion
    Creating waves in this ocean of sound and emotion
    """
    
    quality_scores = predictor.predict_quality_metrics(test_lyrics)
    
    print("📊 Quality Metrics:")
    for metric, score in quality_scores.items():
        bar_length = int(score * 20)  # Scale to 20 chars
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"   {metric:20} [{bar}] {score:.3f}")
    
    # 4. Trend Analysis
    print("\n4️⃣ TREND ANALYSIS:")
    with open('./models/trend_analysis_report.json', 'r') as f:
        trends = json.load(f)
    
    print("📈 Top Emerging Trends:")
    for i, (theme, data) in enumerate(list(trends['emerging_trends'].items())[:5], 1):
        print(f"   {i}. {theme}: {data['growth_rate']:.1f}% growth")
    
    print(f"\n🔍 Key Insights:")
    for insight in trends['key_insights']:
        print(f"   • {insight}")
    
    # 5. Interactive Generation
    print("\n5️⃣ INTERACTIVE GENERATION:")
    await interactive_generation_demo()

async def interactive_generation_demo():
    """Interactive demo с пользовательским input"""
    
    print("\n🎮 Interactive Rap Generation:")
    print("Available styles: drake, eminem, kendrick, j_cole, travis_scott")
    print("Available moods: upbeat, dark, chill, aggressive, romantic")  
    print("Available themes: love, success, struggle, party, social_issues")
    
    # Example generation
    style = "drake"
    mood = "romantic" 
    theme = "love"
    
    print(f"\n🎯 Generating: {style} x {mood} x {theme}")
    
    generator = ConditionalRapGenerator.from_pretrained('./models/conditional_rap')
    lyrics = generator.generate_conditional(style, mood, theme, max_length=300)
    
    print(f"\n🎤 Generated Lyrics:")
    print("=" * 40)
    print(lyrics[0])
    print("=" * 40)
    
    # Quality assessment
    with open('./models/quality_predictor.pkl', 'rb') as f:
        quality_data = joblib.load(f)
    
    predictor = RapQualityPredictor()
    predictor.models = quality_data['models']
    predictor.scaler = quality_data['scaler']
    predictor.label_encoders = quality_data['label_encoders']
    
    quality_scores = predictor.predict_quality_metrics(lyrics[0])
    
    print("\n📊 AI Quality Assessment:")
    total_score = sum(quality_scores.values()) / len(quality_scores)
    print(f"Overall Quality: {total_score:.2f}/1.0")
    
    if total_score >= 0.7:
        print("🔥 HIGH QUALITY - Ready for release!")
    elif total_score >= 0.5:
        print("👍 GOOD QUALITY - Minor improvements recommended")
    else:
        print("⚠️ NEEDS WORK - Significant improvements needed")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_complete_pipeline())
```

---

## 📈 Performance Metrics & Evaluation

### **Model Performance Benchmarks**

```python
# evaluation/model_evaluation.py
"""
📊 Comprehensive Model Evaluation Suite
Оценка производительности всех ML моделей
"""

class ModelEvaluationSuite:
    """Comprehensive evaluation для всех ML моделей"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_generation_model(self):
        """Evaluate conditional generation model"""
        
        # Load model
        generator = ConditionalRapGenerator.from_pretrained('./models/conditional_rap')
        
        # Test conditions
        test_conditions = [
            ('drake', 'upbeat', 'success'),
            ('eminem', 'dark', 'struggle'),
            ('kendrick', 'chill', 'social_issues')
        ]
        
        metrics = {
            'average_length': 0,
            'unique_outputs': 0,
            'generation_time': 0,
            'diversity_score': 0
        }
        
        all_outputs = []
        start_time = time.time()
        
        for style, mood, theme in test_conditions:
            outputs = generator.generate_conditional(style, mood, theme)
            all_outputs.extend(outputs)
            
            for output in outputs:
                metrics['average_length'] += len(output.split())
        
        # Calculate metrics
        metrics['generation_time'] = time.time() - start_time
        metrics['average_length'] /= len(all_outputs)
        metrics['unique_outputs'] = len(set(all_outputs))
        metrics['diversity_score'] = self._calculate_diversity(all_outputs)
        
        self.results['generation'] = metrics
        return metrics
    
    def evaluate_style_transfer(self):
        """Evaluate style transfer model"""
        
        transfer_model = RapStyleTransfer.from_pretrained('./models/style_transfer')
        
        # Test samples
        test_samples = [
            "Money on my mind, success in my veins, working every day through the joy and the pains",
            "Walking through the city lights, feeling so alive, dreams are getting bigger, watch me thrive",
            "Started from nothing, built my empire strong, every single step forward proves them wrong"
        ]
        
        target_styles = ['drake', 'eminem', 'kendrick_lamar']
        
        metrics = {
            'transfer_time': 0,
            'style_consistency': 0,
            'content_preservation': 0,
            'output_quality': 0
        }
        
        start_time = time.time()
        transfers = 0
        
        for sample in test_samples:
            for style in target_styles:
                transferred = transfer_model.transfer_style(sample, style)
                transfers += 1
                
                # Evaluate style consistency (простая метрика)
                style_score = self._evaluate_style_consistency(transferred, style)
                metrics['style_consistency'] += style_score
                
                # Content preservation (BLEU-like score)
                content_score = self._calculate_content_similarity(sample, transferred)
                metrics['content_preservation'] += content_score
        
        # Average metrics
        metrics['transfer_time'] = time.time() - start_time
        metrics['style_consistency'] /= transfers
        metrics['content_preservation'] /= transfers
        metrics['output_quality'] = (metrics['style_consistency'] + metrics['content_preservation']) / 2
        
        self.results['style_transfer'] = metrics
        return metrics
    
    def evaluate_quality_prediction(self):
        """Evaluate quality prediction models"""
        
        # Load test dataset
        with open('data/ml_dataset.pkl', 'rb') as f:
            df = pickle.load(f)
        
        # Load trained models
        with open('./models/quality_predictor.pkl', 'rb') as f:
            quality_data = joblib.load(f)
        
        predictor = RapQualityPredictor()
        predictor.models = quality_data['models']
        predictor.scaler = quality_data['scaler']
        predictor.label_encoders = quality_data['label_encoders']
        
        # Test set (последние 20% данных)
        test_size = len(df) // 5
        test_df = df.tail(test_size)
        
        metrics = {}
        
        for target_name, model in predictor.models.items():
            # Predictions
            predictions = []
            actuals = []
            
            for _, row in test_df.iterrows():
                try:
                    pred = predictor.predict_quality_metrics(row['lyrics'])
                    if target_name in pred:
                        predictions.append(pred[target_name])
                        
                        # Actual value (synthetic для demo)
                        if target_name == 'commercial_potential':
                            actual = (row['artist_popularity'] / 100 * 0.3 + 
                                    row.get('danceability', 0.5) * 0.2 + 
                                    row.get('energy', 0.5) * 0.2 +
                                    row.get('valence', 0.5) * 0.1 +
                                    row.get('quality_rating', 0.5) * 0.2)
                        else:
                            actual = row.get('quality_rating', 0.5)
                        
                        actuals.append(actual)
                        
                except Exception:
                    continue
            
            if len(predictions) > 0:
                from sklearn.metrics import mean_absolute_error, r2_score
                mae = mean_absolute_error(actuals, predictions)
                r2 = r2_score(actuals, predictions) if len(set(actuals)) > 1 else 0
                
                metrics[target_name] = {
                    'mae': mae,
                    'r2': r2,
                    'sample_size': len(predictions)
                }
        
        self.results['quality_prediction'] = metrics
        return metrics
    
    def evaluate_trend_analysis(self):
        """Evaluate trend analysis accuracy"""
        
        # Load trend report
        with open('./models/trend_analysis_report.json', 'r') as f:
            trends = json.load(f)
        
        metrics = {
            'trends_identified': len(trends.get('emerging_trends', {})),
            'clusters_found': len(trends.get('style_clusters', {})),
            'viral_patterns_detected': 1 if trends.get('viral_patterns') else 0,
            'analysis_completeness': 0
        }
        
        # Completeness score
        required_sections = ['emerging_trends', 'style_clusters', 'viral_patterns', 'key_insights']
        completed_sections = sum(1 for section in required_sections if section in trends)
        metrics['analysis_completeness'] = completed_sections / len(required_sections)
        
        self.results['trend_analysis'] = metrics
        return metrics
    
    def _calculate_diversity(self, texts):
        """Calculate diversity score для generated texts"""
        if len(texts) < 2:
            return 0.0
        
        # Unique words ratio
        all_words = []
        for text in texts:
            all_words.extend(text.lower().split())
        
        unique_words = len(set(all_words))
        total_words = len(all_words)
        
        return unique_words / total_words if total_words > 0 else 0.0
    
    def _evaluate_style_consistency(self, text, target_style):
        """Evaluate if text matches target style"""
        
        # Style keywords (simplified)
        style_keywords = {
            'drake': ['started', 'bottom', 'views', 'toronto', '6ix'],
            'eminem': ['rap', 'god', 'slim', 'shady', 'detroit'],
            'kendrick_lamar': ['compton', 'damn', 'humble', 'loyalty', 'pride']
        }
        
        keywords = style_keywords.get(target_style, [])
        text_lower = text.lower()
        
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        return matches / len(keywords) if keywords else 0.5
    
    def _calculate_content_similarity(self, original, transferred):
        """Calculate content similarity using simple word overlap"""
        
        orig_words = set(original.lower().split())
        trans_words = set(transferred.lower().split())
        
        intersection = orig_words.intersection(trans_words)
        union = orig_words.union(trans_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report"""
        
        print("\n🔍 RUNNING COMPREHENSIVE MODEL EVALUATION...")
        
        # Evaluate all models
        generation_metrics = self.evaluate_generation_model()
        style_metrics = self.evaluate_style_transfer()
        quality_metrics = self.evaluate_quality_prediction()
        trend_metrics = self.evaluate_trend_analysis()
        
        # Generate report
        report = {
            'evaluation_date': datetime.now().isoformat(),
            'models_evaluated': ['generation', 'style_transfer', 'quality_prediction', 'trend_analysis'],
            'results': self.results
        }
        
        # Save report
        with open('./evaluation/model_evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        self._print_evaluation_summary()
        
        return report
    
    def _print_evaluation_summary(self):
        """Print evaluation summary"""
        
        print("\n📊 MODEL EVALUATION SUMMARY")
        print("=" * 50)
        
        # Generation Model
        if 'generation' in self.results:
            gen = self.results['generation']
            print(f"\n🎤 CONDITIONAL GENERATION:")
            print(f"  Average Length: {gen['average_length']:.1f} words")
            print(f"  Generation Time: {gen['generation_time']:.2f}s")
            print(f"  Diversity Score: {gen['diversity_score']:.3f}")
            print(f"  Unique Outputs: {gen['unique_outputs']}")
        
        # Style Transfer
        if 'style_transfer' in self.results:
            style = self.results['style_transfer']
            print(f"\n🎭 STYLE TRANSFER:")
            print(f"  Transfer Time: {style['transfer_time']:.2f}s")
            print(f"  Style Consistency: {style['style_consistency']:.3f}")
            print(f"  Content Preservation: {style['content_preservation']:.3f}")
            print(f"  Overall Quality: {style['output_quality']:.3f}")
        
        # Quality Prediction
        if 'quality_prediction' in self.results:
            quality = self.results['quality_prediction']
            print(f"\n💎 QUALITY PREDICTION:")
            for target, metrics in quality.items():
                print(f"  {target}:")
                print(f"    MAE: {metrics['mae']:.3f}")
                print(f"    R²: {metrics['r2']:.3f}")
                print(f"    Samples: {metrics['sample_size']}")
        
        # Trend Analysis
        if 'trend_analysis' in self.results:
            trend = self.results['trend_analysis']
            print(f"\n📈 TREND ANALYSIS:")
            print(f"  Trends Identified: {trend['trends_identified']}")
            print(f"  Style Clusters: {trend['clusters_found']}")
            print(f"  Analysis Completeness: {trend['analysis_completeness']:.1%}")
        
        # Overall Assessment
        self._print_overall_assessment()
    
    def _print_overall_assessment(self):
        """Print overall model performance assessment"""
        
        print(f"\n🎯 OVERALL ASSESSMENT:")
        
        scores = []
        
        if 'generation' in self.results:
            gen_score = min(self.results['generation']['diversity_score'] * 2, 1.0)
            scores.append(('Generation', gen_score))
        
        if 'style_transfer' in self.results:
            style_score = self.results['style_transfer']['output_quality']
            scores.append(('Style Transfer', style_score))
        
        if 'quality_prediction' in self.results:
            qual_scores = [m['r2'] for m in self.results['quality_prediction'].values() if m['r2'] > 0]
            qual_score = sum(qual_scores) / len(qual_scores) if qual_scores else 0
            scores.append(('Quality Prediction', qual_score))
        
        if 'trend_analysis' in self.results:
            trend_score = self.results['trend_analysis']['analysis_completeness']
            scores.append(('Trend Analysis', trend_score))
        
        # Print scores
        for model_name, score in scores:
            bar_length = int(score * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            status = "🔥" if score >= 0.8 else "👍" if score >= 0.6 else "⚠️"
            print(f"  {model_name:20} [{bar}] {score:.2f} {status}")
        
        # Overall score
        overall_score = sum(score for _, score in scores) / len(scores)
        print(f"\n🏆 OVERALL SCORE: {overall_score:.2f}/1.00")
        
        if overall_score >= 0.8:
            print("   Status: 🔥 EXCELLENT - Production ready!")
        elif overall_score >= 0.6:
            print("   Status: 👍 GOOD - Minor improvements recommended")
        elif overall_score >= 0.4:
            print("   Status: ⚠️ FAIR - Significant improvements needed")
        else:
            print("   Status: ❌ POOR - Major rework required")

# Run evaluation
def run_model_evaluation():
    """Run complete model evaluation suite"""
    
    evaluator = ModelEvaluationSuite()
    report = evaluator.generate_evaluation_report()
    
    print(f"\n📄 Detailed report saved to: ./evaluation/model_evaluation_report.json")
    return report

if __name__ == "__main__":
    run_model_evaluation()
```

---

## 🚀 Deployment & Production Pipeline

### **Complete CI/CD Pipeline для ML Models**

```yaml
# .github/workflows/ml-models-cicd.yml
name: ML Models CI/CD Pipeline

on:
  push:
    branches: [main, develop]
    paths: 
      - 'models/**'
      - 'scripts/ml/**'
      - 'requirements-ml.txt'
  pull_request:
    paths:
      - 'models/**' 
      - 'scripts/ml/**'

jobs:
  test-models:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements-ml.txt
        pip install pytest pytest-cov
    
    - name: Run model tests
      run: |
        pytest tests/test_ml_models.py -v --cov=models/
    
    - name: Model validation
      run: |
        python evaluation/model_evaluation.py
    
  build-and-deploy:
    needs: test-models
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Build ML service image
      run: |
        docker build -f Dockerfile.ml -t rap-analyzer:ml-${{ github.sha }} .
        docker tag rap-analyzer:ml-${{ github.sha }} rap-analyzer:ml-latest
    
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/rap-ml-service ml-service=rap-analyzer:ml-${{ github.sha }}
        kubectl rollout status deployment/rap-ml-service
    
    - name: Run integration tests
      run: |
        python tests/integration/test_ml_api.py
```

### **Production Monitoring & Alerting**

```python
# monitoring/ml_model_monitor.py
"""
📊 ML Model Performance Monitoring
Production-ready monitoring для ML моделей
"""

import time
import psutil
import requests
from dataclasses import dataclass
from typing import Dict, List
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge

# Prometheus metrics
model_requests = Counter('ml_model_requests_total', 'Total ML model requests', ['model_type', 'status'])
model_latency = Histogram('ml_model_request_duration_seconds', 'ML model request latency', ['model_type'])
model_memory_usage = Gauge('ml_model_memory_usage_bytes', 'ML model memory usage', ['model_type'])
model_accuracy = Gauge('ml_model_accuracy_score', 'ML model accuracy', ['model_type'])

@dataclass
class ModelHealthMetrics:
    model_name: str
    response_time: float
    memory_usage: float
    accuracy_score: float
    error_rate: float
    request_count: int
    uptime: float

class MLModelMonitor:
    """Production monitoring для ML моделей"""
    
    def __init__(self, ml_service_url="http://rap-ml-service:8001"):
        self.ml_service_url = ml_service_url
        self.start_time = time.time()
        self.health_history = []
        
    def check_model_health(self) -> Dict[str, ModelHealthMetrics]:
        """Check health всех ML моделей"""
        
        health_metrics = {}
        
        # Test each model endpoint
        model_tests = [
            ('generation', self._test_generation_model),
            ('style_transfer', self._test_style_transfer),
            ('quality_prediction', self._test_quality_model),
            ('trend_analysis', self._test_trend_analysis)
        ]
        
        for model_name, test_func in model_tests:
            try:
                start_time = time.time()
                success, error_message = test_func()
                response_time = time.time() - start_time
                
                # Memory usage
                memory_usage = self._get_model_memory_usage(model_name)
                
                # Create metrics
                metrics = ModelHealthMetrics(
                    model_name=model_name,
                    response_time=response_time,
                    memory_usage=memory_usage,
                    accuracy_score=0.85,  # TODO: implement real accuracy tracking
                    error_rate=0.0 if success else 1.0,
                    request_count=1,
                    uptime=time.time() - self.start_time
                )
                
                health_metrics[model_name] = metrics
                
                # Update Prometheus metrics
                status = 'success' if success else 'error'
                model_requests.labels(model_type=model_name, status=status).inc()
                model_latency.labels(model_type=model_name).observe(response_time)
                model_memory_usage.labels(model_type=model_name).set(memory_usage)
                
            except Exception as e:
                print(f"❌ Health check failed for {model_name}: {e}")
                
        return health_metrics
    
    def _test_generation_model(self) -> tuple[bool, str]:
        """Test conditional generation model"""
        try:
            response = requests.post(
                f"{self.ml_service_url}/generate",
                json={
                    "style": "drake",
                    "mood": "upbeat", 
                    "theme": "success",
                    "max_length": 50
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'generated_lyrics' in data and len(data['generated_lyrics']) > 0:
                    return True, ""
                else:
                    return False, "Empty response"
            else:
                return False, f"HTTP {response.status_code}"
                
        except Exception as e:
            return False, str(e)
    
    def _test_style_transfer(self) -> tuple[bool, str]:
        """Test style transfer model"""
        try:
            response = requests.post(
                f"{self.ml_service_url}/style-transfer",
                json={
                    "original_lyrics": "Test lyrics for style transfer",
                    "target_style": "drake"
                },
                timeout=20
            )
            
            return response.status_code == 200, ""
            
        except Exception as e:
            return False, str(e)
    
    def _test_quality_model(self) -> tuple[bool, str]:
        """Test quality prediction model"""
        try:
            response = requests.post(
                f"{self.ml_service_url}/predict-quality",
                json={
                    "lyrics": "Test lyrics for quality prediction"
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return 'quality_predictions' in data, ""
            else:
                return False, f"HTTP {response.status_code}"
                
        except Exception as e:
            return False, str(e)
    
    def _test_trend_analysis(self) -> tuple[bool, str]:
        """Test trend analysis endpoint"""
        try:
            response = requests.get(f"{self.ml_service_url}/trends", timeout=5)
            return response.status_code == 200, ""
            
        except Exception as e:
            return False, str(e)
    
    def _get_model_memory_usage(self, model_name: str) -> float:
        """Get memory usage для specific model"""
        # Simplified - в production используй model-specific metrics
        process = psutil.Process()
        return process.memory_info().rss
    
    def generate_health_report(self) -> Dict:
        """Generate comprehensive health report"""
        
        health_metrics = self.check_model_health()
        
        report = {
            'timestamp': time.time(),
            'overall_status': 'healthy' if all(m.error_rate == 0 for m in health_metrics.values()) else 'degraded',
            'models': {},
            'system_metrics': {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            }
        }
        
        for model_name, metrics in health_metrics.items():
            report['models'][model_name] = {
                'status': 'healthy' if metrics.error_rate == 0 else 'error',
                'response_time': metrics.response_time,
                'memory_usage_mb': metrics.memory_usage / 1024 / 1024,
                'accuracy': metrics.accuracy_score,
                'uptime': metrics.uptime
            }
        
        return report
    
    def send_alerts(self, health_metrics: Dict[str, ModelHealthMetrics]):
        """Send alerts для unhealthy models"""
        
        for model_name, metrics in health_metrics.items():
            if metrics.error_rate > 0:
                alert_message = f"🚨 ML Model Alert: {model_name} is unhealthy (error_rate: {metrics.error_rate})"
                self._send_slack_alert(alert_message)
            
            elif metrics.response_time > 10:  # 10 second threshold
                alert_message = f"⚠️ ML Model Warning: {model_name} slow response ({metrics.response_time:.2f}s)"
                self._send_slack_alert(alert_message)
    
    def _send_slack_alert(self, message: str):
        """Send alert to Slack (placeholder)"""
        print(f"ALERT: {message}")
        # TODO: Implement Slack webhook integration

# Monitoring daemon
def run_monitoring_daemon():
    """Run continuous monitoring"""
    
    monitor = MLModelMonitor()
    
    while True:
        try:
            # Health check
            health_metrics = monitor.check_model_health()
            
            # Generate report
            report = monitor.generate_health_report()
            
            # Send alerts if needed
            monitor.send_alerts(health_metrics)
            
            # Log status
            print(f"🏥 Health Check: {report['overall_status']}")
            for model, status in report['models'].items():
                print(f"  {model}: {status['status']} ({status['response_time']:.2f}s)")
            
        except Exception as e:
            print(f"❌ Monitoring error: {e}")
        
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    run_monitoring_daemon()
```

---

## 📚 Complete Documentation & Best Practices

### **ML Models API Documentation**

```markdown
# 🎤 Rap ML Models API Documentation

## Overview
Production-ready ML API для rap generation, style transfer, quality prediction, and trend analysis.

## Base URL
```
http://rap-ml-service:8001
```

## Authentication
```bash
# Add API key header (в production)
curl -H "X-API-Key: your-api-key" \
     -H "Content-Type: application/json" \
     http://rap-ml-service:8001/generate
```

## Endpoints

### 🎵 POST /generate
Generate rap lyrics with conditions.

**Request:**
```json
{
  "style": "drake",           // drake, eminem, kendrick, j_cole
  "mood": "upbeat",          // upbeat, dark, chill, aggressive, romantic
  "theme": "success",        // love, success, struggle, party, social_issues
  "max_length": 200          // Maximum tokens (50-500)
}
```

**Response:**
```json
{
  "generated_lyrics": [
    "Started from the bottom now we here...",
    "Dreams turn to reality, crystal clear...",
    "Every step I take, getting closer to the goal..."
  ],
  "generation_time": 2.34,
  "model_confidence": 0.87
}
```

### 🎭 POST /style-transfer
Transfer lyrical style between artists.

**Request:**
```json
{
  "original_lyrics": "Money on my mind, success in my veins...",
  "target_style": "eminem"
}
```

**Response:**
```json
{
  "transferred_lyrics": "Cash flows through thoughts like poison in blood..."
}
```

### 💎 POST /predict-quality
Predict quality metrics for lyrics.

**Request:**
```json
{
  "lyrics": "Your rap lyrics here...",
  "artist_features": {        // Optional
    "popularity": 85,
    "followers": 10000000
  }
}
```

**Response:**
```json
{
  "quality_predictions": {
    "commercial_potential": 0.78,
    "artistic_quality": 0.85,
    "viral_potential": 0.72,
    "longevity_score": 0.91
  }
}
```

### 📈 GET /trends
Get current trend analysis.

**Response:**
```json
{
  "emerging_trends": {
    "mental_health": {"growth_rate": 15.2},
    "crypto": {"growth_rate": 12.8},
    "social_media": {"growth_rate": 10.5}
  },
  "viral_patterns": {
    "avg_features": {
      "energy": 0.82,
      "danceability": 0.75,
      "tempo": 125
    }
  },
  "key_insights": [
    "🔥 Fastest growing theme: 'mental_health'",
    "🎵 Viral tracks average: 0.82 energy, 0.75 danceability"
  ]
}
```

## Error Handling
```json
{
  "error": "Model temporarily unavailable",
  "code": "MODEL_ERROR",
  "timestamp": "2025-09-26T10:30:00Z"
}
```

## Rate Limits
- Generation: 60 requests/minute
- Style Transfer: 120 requests/minute  
- Quality Prediction: 180 requests/minute
- Trends: 300 requests/minute

## Performance Expectations
- Generation: ~2-5 seconds
- Style Transfer: ~1-3 seconds
- Quality Prediction: ~0.5-1 seconds
- Trends: ~0.1-0.2 seconds
```

---

## 🎯 Implementation Roadmap

### **Phase Timeline & Milestones**

```markdown
# 🗓️ ML Models Implementation Timeline

## Phase 0: Data Preparation (Week 1)
- [ ] Extract and clean 57,718 tracks dataset
- [ ] Engineer features from AI analysis + Spotify data
- [ ] Create train/validation/test splits
- [ ] Setup MLOps infrastructure (MLflow, W&B)

**Deliverables:** Clean ML dataset, feature engineering pipeline

## Phase 1: Conditional Generation (Weeks 2-3)
- [ ] Fine-tune GPT-2 на rap dataset
- [ ] Implement conditioning tokens (style, mood, theme)
- [ ] Train model на 80% данных
- [ ] Validate на test set
- [ ] Deploy model endpoint

**Success Metrics:** 
- Generation quality > 0.8 (human evaluation)
- Response time < 5 seconds
- Diversity score > 0.6

## Phase 2: Style Transfer (Weeks 4-5)
- [ ] Prepare style transfer dataset (cross-style pairs)
- [ ] Fine-tune T5 model
- [ ] Implement evaluation metrics (style consistency, content preservation)
- [ ] Deploy model endpoint
- [ ] A/B test с users

**Success Metrics:**
- Style consistency > 0.7
- Content preservation > 0.6
- User satisfaction > 75%

## Phase 3: Quality Prediction (Week 6)
- [ ] Feature engineering для quality metrics
- [ ] Train multi-target regression models
- [ ] SHAP analysis для interpretability
- [ ] Model validation и testing
- [ ] API integration

**Success Metrics:**
- R² score > 0.7 для commercial_potential
- MAE < 0.15 для all quality metrics
- Feature importance analysis complete

## Phase 4: Trend Analysis (Week 7)
- [ ] Temporal analysis implementation
- [ ] Clustering algorithm для music styles
- [ ] Emerging trends prediction model
- [ ] Interactive dashboard creation
- [ ] Report generation automation

**Success Metrics:**
- 15+ emerging trends identified
- 8+ music style clusters
- Analysis completeness > 90%

## Phase 5: Production Integration (Week 8)
- [ ] Kubernetes deployment manifests
- [ ] ML service API implementation
- [ ] Monitoring и alerting setup
- [ ] CI/CD pipeline configuration
- [ ] Load testing и performance optimization

**Success Metrics:**
- 99.5% uptime SLA
- < 5s p95 response time
- Auto-scaling working properly

## Phase 6: Evaluation & Optimization (Week 9)
- [ ] Comprehensive model evaluation
- [ ] Performance benchmarking
- [ ] User feedback collection
- [ ] Model improvements based on feedback
- [ ] Documentation completion

**Success Metrics:**
- Overall model score > 0.8
- User satisfaction > 85%
- Complete documentation
```

---

## 💡 Advanced Features & Future Enhancements

### **Next-Level ML Capabilities**

```python
# Future enhancements roadmap

class AdvancedRapML:
    """Advanced ML capabilities для future development"""
    
    def __init__(self):
        self.future_features = {
            'real_time_collaboration': self.implement_real_time_collab,
            'voice_synthesis': self.implement_voice_synthesis,
            'music_generation': self.implement_music_generation,
            'multimodal_analysis': self.implement_multimodal,
            'personalization': self.implement_personalization
        }
    
    def implement_real_time_collab(self):
        """Real-time collaborative rap writing с AI"""
        # WebSocket integration для real-time collaboration
        # Multiple users + AI working together
        # Live feedback и suggestions
        pass
    
    def implement_voice_synthesis(self):
        """AI voice synthesis для generated lyrics"""
        # Text-to-speech с artist voice cloning
        # Bark/Eleven Labs integration
        # Custom voice training
        pass
    
    def implement_music_generation(self):
        """AI-generated beats и instrumentals"""
        # MusicLM/AudioLM integration
        # Beat generation matching lyrical flow
        # Full song production pipeline
        pass
    
    def implement_multimodal(self):
        """Multimodal analysis (lyrics + audio + video)"""
        # Video analysis для music videos
        # Audio analysis для vocal patterns
        # Cross-modal recommendations
        pass
    
    def implement_personalization(self):
        """Personalized AI based на user preferences"""
        # User modeling и preference learning
        # Personalized generation styles
        # Adaptive