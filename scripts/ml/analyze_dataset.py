import pickle

# Load dataset
with open("data/ml/quick_dataset.pkl", "rb") as f:
    data = pickle.load(f)

df = data["raw_data"]

print("=== QUICK DATASET ANALYSIS ===")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

print("\n=== WORD COUNT STATS ===")
print(df["word_count"].describe())

print("\n=== LYRICS LENGTH STATS ===")
print(df["lyrics"].str.len().describe())

print("\n=== SENTIMENT VALUES ===")
print(df["qwen_sentiment"].value_counts())

print("\n=== NULL VALUES CHECK ===")
print(f"qwen_sentiment nulls: {df['qwen_sentiment'].isnull().sum()}")
print(f"lyrics nulls: {df['lyrics'].isnull().sum()}")

print("\n=== SAMPLE DATA ===")
print(
    df[["artist", "word_count", "qwen_sentiment", "artist_style", "mood_label"]].head()
)

# Check filtering criteria
print("\n=== FILTERING ANALYSIS ===")
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
