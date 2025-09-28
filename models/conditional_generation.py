"""
🎤 Conditional Rap Generation Model
Fine-tuned GPT-2 для генерации рэп-текстов по условиям (стиль + настроение + тема)

Features:
- Conditioning tokens для style, mood, theme
- Fine-tuning на вашем dataset (57K треков)
- Batch training с progress tracking
- Quality-aware generation
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
import json
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConditionalRapGenerator:
    """
    Fine-tuned GPT-2 для условной генерации рэп-текстов
    
    Conditioning format:
    <STYLE:drake> <MOOD:upbeat> <THEME:success> <BOS> lyrics text <EOS>
    """
    
    def __init__(self, model_name='gpt2-medium', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🔧 Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Add special tokens
        special_tokens = {
            'pad_token': '<PAD>',
            'bos_token': '<BOS>',
            'eos_token': '<EOS>',
            'unk_token': '<UNK>'
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Conditioning tokens for different categories
        self.style_tokens = [
            '<STYLE:drake>', '<STYLE:eminem>', '<STYLE:kendrick_lamar>',
            '<STYLE:j_cole>', '<STYLE:travis_scott>', '<STYLE:lil_wayne>',
            '<STYLE:jay_z>', '<STYLE:kanye_west>', '<STYLE:nicki_minaj>',
            '<STYLE:future>', '<STYLE:other>'
        ]
        
        self.mood_tokens = [
            '<MOOD:upbeat>', '<MOOD:dark>', '<MOOD:chill>',
            '<MOOD:aggressive>', '<MOOD:romantic>', '<MOOD:energetic>'
        ]
        
        self.theme_tokens = [
            '<THEME:love>', '<THEME:success>', '<THEME:struggle>',
            '<THEME:party>', '<THEME:social_issues>', '<THEME:general>'
        ]
        
        # Add all conditioning tokens
        all_conditioning_tokens = self.style_tokens + self.mood_tokens + self.theme_tokens
        self.tokenizer.add_tokens(all_conditioning_tokens)
        
        # Initialize model
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        
        logger.info(f"✅ Model initialized with {len(self.tokenizer)} tokens")
    
    def load_training_data(self, dataset_path: str = 'data/ml/quick_dataset.pkl') -> pd.DataFrame:
        """Load and prepare training data from ML dataset"""
        logger.info(f"📊 Loading training data from {dataset_path}")
        
        try:
            with open(dataset_path, 'rb') as f:
                ml_dataset = pickle.load(f)
            
            df = ml_dataset['raw_data']
            logger.info(f"✅ Loaded {len(df)} tracks for training")
            
            # Filter for quality training data
            # Only use tracks with lyrics > 50 words and some analysis
            df_filtered = df[
                (df['word_count'] >= 50) &
                (df['lyrics'].str.len() >= 200) &
                (df['qwen_sentiment'].notna())
            ].copy()
            
            logger.info(f"📊 Filtered to {len(df_filtered)} high-quality tracks")
            return df_filtered
            
        except Exception as e:
            logger.error(f"❌ Failed to load training data: {e}")
            raise
    
    def prepare_training_texts(self, df: pd.DataFrame, max_samples: int = None) -> List[str]:
        """Prepare training texts with conditioning tokens"""
        logger.info("🔄 Preparing training texts with conditioning...")
        
        training_texts = []
        
        # Sample data if specified
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)
            logger.info(f"📊 Sampled {max_samples} tracks for training")
        
        for idx, row in df.iterrows():
            try:
                # Map artist to style token
                artist = str(row.get('artist', '')).lower().replace(' ', '_')
                style_token = f'<STYLE:{artist}>' if f'<STYLE:{artist}>' in self.style_tokens else '<STYLE:other>'
                
                # Map mood
                mood = str(row.get('mood_label', 'chill')).lower()
                mood_token = f'<MOOD:{mood}>'
                
                # Map theme
                theme = str(row.get('theme_category', 'general')).lower()
                theme_token = f'<THEME:{theme}>'
                
                # Clean lyrics
                lyrics = str(row.get('lyrics', ''))
                if len(lyrics) < 50:
                    continue
                
                # Remove excessive whitespace and normalize
                lyrics = ' '.join(lyrics.split())
                
                # Create training text with conditioning
                training_text = f"{style_token} {mood_token} {theme_token} <BOS> {lyrics} <EOS>"
                training_texts.append(training_text)
                
            except Exception as e:
                logger.warning(f"⚠️ Skipped row {idx}: {e}")
                continue
        
        logger.info(f"✅ Prepared {len(training_texts)} training texts")
        return training_texts
    
    def create_dataset(self, training_texts: List[str], max_length: int = 512) -> Dataset:
        """Create PyTorch dataset for training"""
        
        class RapDataset(Dataset):
            def __init__(self, texts, tokenizer, max_length):
                self.texts = texts
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = self.texts[idx]
                
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # For language modeling, input_ids = labels
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': encoding['input_ids'].flatten()
                }
        
        dataset = RapDataset(training_texts, self.tokenizer, max_length)
        logger.info(f"✅ Created dataset with {len(dataset)} samples")
        return dataset
    
    def fine_tune(self, 
                  dataset_path: str = 'data/ml/rap_ml_dataset.pkl',
                  output_dir: str = './models/conditional_rap',
                  epochs: int = 3,
                  batch_size: int = 4,
                  learning_rate: float = 5e-5,
                  max_samples: int = None,
                  save_steps: int = 500):
        """Fine-tune model on rap dataset"""
        
        logger.info("🚀 Starting fine-tuning process...")
        
        try:
            # Load and prepare data
            df = self.load_training_data(dataset_path)
            training_texts = self.prepare_training_texts(df, max_samples)
            
            if len(training_texts) == 0:
                raise ValueError("No training texts prepared")
            
            # Create dataset
            train_dataset = self.create_dataset(training_texts)
            
            # Setup training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                overwrite_output_dir=True,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=2,  # Effective batch size = batch_size * 2
                warmup_steps=100,
                learning_rate=learning_rate,
                weight_decay=0.01,
                logging_dir=f'{output_dir}/logs',
                logging_steps=50,
                save_steps=save_steps,
                save_total_limit=3,
                eval_strategy="no",  # Updated parameter name
                fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
                dataloader_drop_last=True,
                report_to=None,  # Disable wandb for now
                remove_unused_columns=False
            )
            
            # Data collator for language modeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # We're doing causal language modeling, not masked
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
            )
            
            # Start training
            logger.info(f"🎯 Starting training:")
            logger.info(f"   Samples: {len(training_texts)}")
            logger.info(f"   Epochs: {epochs}")
            logger.info(f"   Batch size: {batch_size}")
            logger.info(f"   Learning rate: {learning_rate}")
            logger.info(f"   Device: {self.device}")
            
            # Train the model
            trainer.train()
            
            # Save final model
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            # Save training metadata
            training_metadata = {
                'training_date': datetime.now().isoformat(),
                'training_samples': len(training_texts),
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'model_size': sum(p.numel() for p in self.model.parameters()),
                'vocab_size': len(self.tokenizer),
                'conditioning_tokens': {
                    'styles': self.style_tokens,
                    'moods': self.mood_tokens,
                    'themes': self.theme_tokens
                }
            }
            
            with open(f'{output_dir}/training_metadata.json', 'w') as f:
                json.dump(training_metadata, f, indent=2)
            
            logger.info("✅ Fine-tuning completed successfully!")
            logger.info(f"   Model saved to: {output_dir}")
            
            return output_dir
            
        except Exception as e:
            logger.error(f"❌ Fine-tuning failed: {e}")
            raise
    
    def generate_conditional(self, 
                           style: str, 
                           mood: str, 
                           theme: str,
                           max_length: int = 200,
                           num_return_sequences: int = 3,
                           temperature: float = 0.8,
                           do_sample: bool = True) -> List[str]:
        """Generate rap lyrics with specified conditions"""
        
        # Create conditioning prompt
        style_token = f'<STYLE:{style.lower()}>' if f'<STYLE:{style.lower()}>' in self.style_tokens else '<STYLE:other>'
        mood_token = f'<MOOD:{mood.lower()}>'
        theme_token = f'<THEME:{theme.lower()}>'
        
        prompt = f"{style_token} {mood_token} {theme_token} <BOS>"
        
        # Tokenize prompt
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            # Generate
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                length_penalty=1.0
            )
        
        # Decode results
        results = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=False)
            
            # Extract generated lyrics (after <BOS>, before <EOS>)
            if '<BOS>' in text:
                text = text.split('<BOS>')[-1]
            if '<EOS>' in text:
                text = text.split('<EOS>')[0]
            
            # Clean up
            text = text.strip()
            if text:
                results.append(text)
        
        return results
    
    @classmethod
    def from_pretrained(cls, model_path: str, device=None):
        """Load a pre-trained conditional rap generator"""
        logger.info(f"📥 Loading pre-trained model from {model_path}")
        
        try:
            # Load tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            
            # Load model
            model = GPT2LMHeadModel.from_pretrained(model_path)
            
            # Create instance
            generator = cls.__new__(cls)  # Create without calling __init__
            generator.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            generator.tokenizer = tokenizer
            generator.model = model.to(generator.device)
            
            # Load metadata if available
            metadata_path = os.path.join(model_path, 'training_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    generator.style_tokens = metadata['conditioning_tokens']['styles']
                    generator.mood_tokens = metadata['conditioning_tokens']['moods']
                    generator.theme_tokens = metadata['conditioning_tokens']['themes']
            
            logger.info("✅ Pre-trained model loaded successfully")
            return generator
            
        except Exception as e:
            logger.error(f"❌ Failed to load pre-trained model: {e}")
            raise

def train_conditional_model():
    """Training script for conditional generation model"""
    logger.info("🚀 CONDITIONAL RAP GENERATION - TRAINING PIPELINE")
    logger.info("=" * 60)
    
    try:
        # Initialize model
        generator = ConditionalRapGenerator(model_name='gpt2-medium')
        
        # Start fine-tuning
        model_path = generator.fine_tune(
            dataset_path='data/ml/quick_dataset.pkl',
            output_dir='./models/conditional_rap',
            epochs=2,  # Reduced for faster training
            batch_size=2,  # Small batch size for memory efficiency
            learning_rate=5e-5,
            max_samples=500,  # Start with smaller subset for testing
            save_steps=50
        )
        
        logger.info("🎯 Testing trained model...")
        
        # Test generation
        test_conditions = [
            ('drake', 'upbeat', 'success'),
            ('eminem', 'dark', 'struggle'),
            ('kendrick_lamar', 'chill', 'social_issues')
        ]
        
        for style, mood, theme in test_conditions:
            logger.info(f"\n🎤 Generating: {style} x {mood} x {theme}")
            try:
                lyrics = generator.generate_conditional(style, mood, theme, max_length=150)
                if lyrics:
                    print(f"\n🎵 Sample Output:")
                    print("-" * 40)
                    print(lyrics[0][:200] + "..." if len(lyrics[0]) > 200 else lyrics[0])
                    print("-" * 40)
            except Exception as e:
                logger.warning(f"⚠️ Generation failed for {style}: {e}")
        
        logger.info("\n✅ TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"Model saved to: {model_path}")
        logger.info("Usage example:")
        logger.info("  generator = ConditionalRapGenerator.from_pretrained('./models/conditional_rap')")
        logger.info("  lyrics = generator.generate_conditional('drake', 'upbeat', 'success')")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        return False

def demo_generation():
    """Demo script to test generation capabilities"""
    logger.info("🎮 CONDITIONAL GENERATION DEMO")
    
    try:
        # Load trained model
        generator = ConditionalRapGenerator.from_pretrained('./models/conditional_rap')
        
        # Demo generations
        demos = [
            ('drake', 'romantic', 'love'),
            ('eminem', 'aggressive', 'struggle'),
            ('kendrick_lamar', 'chill', 'social_issues'),
            ('other', 'upbeat', 'party')
        ]
        
        for style, mood, theme in demos:
            print(f"\n🎤 {style.title()} - {mood} - {theme}")
            print("=" * 50)
            
            lyrics = generator.generate_conditional(style, mood, theme, num_return_sequences=2)
            
            for i, lyric in enumerate(lyrics[:2], 1):
                print(f"\nVersion {i}:")
                print(lyric)
                print("-" * 30)
                
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Conditional Rap Generation')
    parser.add_argument('--mode', choices=['train', 'demo'], default='train',
                       help='Mode: train model or run demo')
    parser.add_argument('--model-path', default='./models/conditional_rap',
                       help='Path to model directory')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        success = train_conditional_model()
        sys.exit(0 if success else 1)
    elif args.mode == 'demo':
        demo_generation()