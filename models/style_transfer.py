"""
🎭 Rap Style Transfer Model
T5-based модель для перевода рэп-текстов между стилями разных исполнителей

Features:
- Style transfer между исполнителями
- Content preservation с style adaptation
- Batch training с style pairs
- Quality-aware transfer
"""

import json
import logging
import os
import pickle
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RapStyleTransfer:
    """
    T5-based модель для style transfer рэп-текстов

    Task format: "transfer rap style: source: [lyrics] target_style: [artist]"
    """

    def __init__(self, model_name="t5-small", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🔧 Using device: {self.device}")

        # Initialize tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)

        # Task prefix for T5
        self.task_prefix = "transfer rap style: "

        logger.info(f"✅ T5 model initialized: {model_name}")

    def load_training_data(
        self, dataset_path: str = "data/ml/quick_dataset.pkl"
    ) -> pd.DataFrame:
        """Load training data for style transfer"""
        logger.info(f"📊 Loading training data from {dataset_path}")

        try:
            with open(dataset_path, "rb") as f:
                ml_dataset = pickle.load(f)

            df = ml_dataset["raw_data"]
            logger.info(f"✅ Loaded {len(df)} tracks for style transfer training")

            # Filter for quality data
            df_filtered = df[
                (df["word_count"] >= 30)
                & (df["lyrics"].str.len() >= 150)
                & (df["artist_style"] != "other")  # Only use major artists
            ].copy()

            logger.info(f"📊 Filtered to {len(df_filtered)} tracks from major artists")
            return df_filtered

        except Exception as e:
            logger.error(f"❌ Failed to load training data: {e}")
            raise

    def create_style_transfer_pairs(
        self, df: pd.DataFrame, max_pairs_per_theme: int = 50
    ) -> list[tuple[str, str]]:
        """
        Create style transfer training pairs
        Strategy: Same theme/mood, different artists
        """
        logger.info("🔄 Creating style transfer training pairs...")

        transfer_pairs = []

        # Group by theme for better content preservation
        theme_groups = df.groupby("theme_category")

        for theme, theme_df in theme_groups:
            if len(theme_df) < 2:
                continue

            # Get artists in this theme
            artist_groups = theme_df.groupby("artist_style")
            artists = list(artist_groups.groups.keys())

            if len(artists) < 2:
                continue

            pairs_created = 0
            max_attempts = max_pairs_per_theme * 5  # Prevent infinite loops
            attempts = 0

            while pairs_created < max_pairs_per_theme and attempts < max_attempts:
                attempts += 1

                # Random pair of artists
                source_artist, target_artist = random.sample(artists, 2)

                # Get random song from each artist
                source_songs = artist_groups.get_group(source_artist)
                target_songs = artist_groups.get_group(target_artist)

                if len(source_songs) == 0 or len(target_songs) == 0:
                    continue

                source_song = source_songs.sample(1).iloc[0]
                target_song = target_songs.sample(1).iloc[0]

                # Create training pair
                source_lyrics = str(source_song["lyrics"])
                target_lyrics = str(target_song["lyrics"])

                # Skip if lyrics are too short
                if len(source_lyrics) < 100 or len(target_lyrics) < 100:
                    continue

                # Input: task prefix + source lyrics + target style
                input_text = f"{self.task_prefix}source: {source_lyrics} target_style: {target_artist}"

                # Target: lyrics in target style
                target_text = target_lyrics

                transfer_pairs.append((input_text, target_text))
                pairs_created += 1

        logger.info(f"✅ Created {len(transfer_pairs)} style transfer pairs")

        # Print some statistics
        themes_count = len(set(df["theme_category"]))
        artists_count = len(set(df["artist_style"]))
        logger.info(f"   Themes covered: {themes_count}")
        logger.info(f"   Artists covered: {artists_count}")

        return transfer_pairs

    def create_dataset(
        self, transfer_pairs: list[tuple[str, str]], max_length: int = 512
    ) -> Dataset:
        """Create PyTorch dataset for style transfer training"""

        class StyleTransferDataset(Dataset):
            def __init__(self, pairs, tokenizer, max_length):
                self.pairs = pairs
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.pairs)

            def __getitem__(self, idx):
                source_text, target_text = self.pairs[idx]

                # Tokenize source (input)
                source_encoding = self.tokenizer(
                    source_text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

                # Tokenize target (labels)
                target_encoding = self.tokenizer(
                    target_text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

                # For T5, labels are input_ids of target sequence
                labels = target_encoding["input_ids"].clone()
                # Replace padding tokens with -100 (ignored in loss)
                labels[labels == self.tokenizer.pad_token_id] = -100

                return {
                    "input_ids": source_encoding["input_ids"].flatten(),
                    "attention_mask": source_encoding["attention_mask"].flatten(),
                    "labels": labels.flatten(),
                }

        dataset = StyleTransferDataset(transfer_pairs, self.tokenizer, max_length)
        logger.info(f"✅ Created dataset with {len(dataset)} training pairs")
        return dataset

    def fine_tune_style_transfer(
        self,
        dataset_path: str = "data/ml/quick_dataset.pkl",
        output_dir: str = "./models/style_transfer",
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 3e-4,
        max_pairs: int = 500,
        save_steps: int = 250,
    ):
        """Fine-tune T5 model for style transfer"""

        logger.info("🚀 Starting style transfer fine-tuning...")

        try:
            # Load and prepare data
            df = self.load_training_data(dataset_path)
            transfer_pairs = self.create_style_transfer_pairs(
                df, max_pairs_per_theme=max_pairs // 10
            )

            if len(transfer_pairs) == 0:
                raise ValueError("No transfer pairs created")

            # Limit pairs if specified
            if max_pairs and len(transfer_pairs) > max_pairs:
                transfer_pairs = random.sample(transfer_pairs, max_pairs)
                logger.info(f"📊 Limited to {max_pairs} training pairs")

            # Create dataset
            train_dataset = self.create_dataset(transfer_pairs)

            # Setup training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                overwrite_output_dir=True,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=2,
                warmup_steps=50,
                learning_rate=learning_rate,
                weight_decay=0.01,
                logging_dir=f"{output_dir}/logs",
                logging_steps=25,
                save_steps=save_steps,
                save_total_limit=2,
                evaluation_strategy="no",
                fp16=torch.cuda.is_available(),
                dataloader_drop_last=True,
                report_to=None,
                remove_unused_columns=False,
                predict_with_generate=True,  # Important for T5
            )

            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                tokenizer=self.tokenizer,
            )

            # Start training
            logger.info("🎯 Starting training:")
            logger.info(f"   Pairs: {len(transfer_pairs)}")
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
                "training_date": datetime.now().isoformat(),
                "training_pairs": len(transfer_pairs),
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "model_type": "T5 Style Transfer",
                "task_prefix": self.task_prefix,
            }

            with open(f"{output_dir}/training_metadata.json", "w") as f:
                json.dump(training_metadata, f, indent=2)

            logger.info("✅ Style transfer fine-tuning completed!")
            logger.info(f"   Model saved to: {output_dir}")

            return output_dir

        except Exception as e:
            logger.error(f"❌ Style transfer fine-tuning failed: {e}")
            raise

    def transfer_style(
        self,
        original_lyrics: str,
        target_artist_style: str,
        max_length: int = 200,
        num_beams: int = 4,
        temperature: float = 0.8,
    ) -> str:
        """Perform style transfer on lyrics"""

        # Create input text
        input_text = f"{self.task_prefix}source: {original_lyrics} target_style: {target_artist_style}"

        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        ).to(self.device)

        # Set model to evaluation mode
        self.model.eval()

        with torch.no_grad():
            # Generate transfer
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode result
        transferred_lyrics = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return transferred_lyrics.strip()

    def batch_transfer(
        self, lyrics_list: list[str], target_styles: list[str], max_length: int = 200
    ) -> list[str]:
        """Perform batch style transfer"""

        results = []

        for lyrics, style in zip(lyrics_list, target_styles):
            try:
                transferred = self.transfer_style(lyrics, style, max_length)
                results.append(transferred)
            except Exception as e:
                logger.warning(f"⚠️ Transfer failed for style {style}: {e}")
                results.append(lyrics)  # Fallback to original

        return results

    @classmethod
    def from_pretrained(cls, model_path: str, device=None):
        """Load pre-trained style transfer model"""
        logger.info(f"📥 Loading pre-trained style transfer model from {model_path}")

        try:
            # Load tokenizer and model
            tokenizer = T5Tokenizer.from_pretrained(model_path)
            model = T5ForConditionalGeneration.from_pretrained(model_path)

            # Create instance
            transfer_model = cls.__new__(cls)
            transfer_model.device = device or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            transfer_model.tokenizer = tokenizer
            transfer_model.model = model.to(transfer_model.device)

            # Load metadata if available
            metadata_path = os.path.join(model_path, "training_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    transfer_model.task_prefix = metadata.get(
                        "task_prefix", "transfer rap style: "
                    )
            else:
                transfer_model.task_prefix = "transfer rap style: "

            logger.info("✅ Pre-trained style transfer model loaded")
            return transfer_model

        except Exception as e:
            logger.error(f"❌ Failed to load pre-trained model: {e}")
            raise


def train_style_transfer_model():
    """Training script for style transfer model"""
    logger.info("🚀 RAP STYLE TRANSFER - TRAINING PIPELINE")
    logger.info("=" * 60)

    try:
        # Initialize model
        transfer_model = RapStyleTransfer(model_name="t5-small")

        # Start fine-tuning
        model_path = transfer_model.fine_tune_style_transfer(
            dataset_path="data/ml/quick_dataset.pkl",
            output_dir="./models/style_transfer",
            epochs=3,
            batch_size=2,  # Small batch size for T5
            learning_rate=3e-4,
            max_pairs=200,  # Start with smaller set
            save_steps=100,
        )

        logger.info("🎯 Testing trained model...")

        # Test style transfer
        test_original = """
        Started from the bottom now I'm here
        Making moves and the vision crystal clear
        Money on my mind, success in my heart
        Every single day I'm playing my part
        """

        test_styles = ["eminem", "kendrick_lamar", "future"]

        for style in test_styles:
            logger.info(f"\n🎭 Transferring to {style} style:")
            try:
                transferred = transfer_model.transfer_style(test_original, style)
                print(f"\n🎵 {style.title()} Style:")
                print("-" * 40)
                print(transferred)
                print("-" * 40)
            except Exception as e:
                logger.warning(f"⚠️ Transfer failed for {style}: {e}")

        logger.info("\n✅ STYLE TRANSFER TRAINING COMPLETED!")
        logger.info(f"Model saved to: {model_path}")

        return True

    except Exception as e:
        logger.error(f"❌ Style transfer training failed: {e}")
        return False


def demo_style_transfer():
    """Demo script for style transfer"""
    logger.info("🎮 STYLE TRANSFER DEMO")

    try:
        # Load trained model
        transfer_model = RapStyleTransfer.from_pretrained("./models/style_transfer")

        # Demo texts
        demo_lyrics = [
            """
            Walking through the city lights at night
            Dreams are big and the future's looking bright
            Got my crew with me, we're ready for the fight
            Success is calling and we're taking flight
            """,
            """
            Money in my pocket, problems on my mind
            Trying to stay focused, leave the past behind
            Every day's a battle but I'm staying strong
            This is my moment, been waiting too long
            """,
        ]

        demo_styles = ["drake", "eminem", "kendrick_lamar"]

        for i, lyrics in enumerate(demo_lyrics, 1):
            print(f"\n🎤 ORIGINAL LYRICS {i}:")
            print("=" * 50)
            print(lyrics.strip())

            for style in demo_styles:
                print(f"\n🎭 {style.title()} Style:")
                print("-" * 30)
                transferred = transfer_model.transfer_style(lyrics.strip(), style)
                print(transferred)

        logger.info("\n✅ Style transfer demo completed!")

    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rap Style Transfer")
    parser.add_argument(
        "--mode",
        choices=["train", "demo"],
        default="train",
        help="Mode: train model or run demo",
    )
    parser.add_argument(
        "--model-path",
        default="./models/style_transfer",
        help="Path to model directory",
    )

    args = parser.parse_args()

    if args.mode == "train":
        success = train_style_transfer_model()
        sys.exit(0 if success else 1)
    elif args.mode == "demo":
        demo_style_transfer()
