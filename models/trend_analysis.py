"""
📈 Trend Analysis Model
Система анализа трендов и предсказания популярных тем в рэп-музыке

Features:
- Temporal analysis изменения тем и настроений
- Clustering музыкальных стилей
- Prediction emerging trends
- Viral pattern analysis
- Interactive trend dashboard
"""

import json
import logging
import os
import pickle
import sys
import warnings
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ML libraries
# Visualization
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("📊 Note: plotly not available, using matplotlib for visualizations")

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RapTrendAnalyzer:
    """
    Анализ трендов в рэп-музыке и предсказание популярных тем

    Capabilities:
    - Temporal trend analysis
    - Music style clustering
    - Emerging themes prediction
    - Viral pattern identification
    - Trend forecasting
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.cluster_model = None
        self.trends_data = {}
        self.style_clusters = {}
        self.viral_patterns = {}

    def load_data(
        self, dataset_path: str = "data/ml/quick_dataset.pkl"
    ) -> pd.DataFrame:
        """Load dataset for trend analysis"""
        logger.info(f"📊 Loading data for trend analysis from {dataset_path}")

        try:
            with open(dataset_path, "rb") as f:
                ml_dataset = pickle.load(f)

            df = ml_dataset["raw_data"]

            # Fix Decimal type issues
            numeric_columns = [
                "word_count",
                "lyrics_length",
                "lines_count",
                "avg_words_per_line",
                "qwen_confidence",
                "qwen_complexity",
                "quality_score",
            ]

            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            logger.info(f"✅ Loaded {len(df)} tracks for trend analysis")

            # Add mock temporal data since we don't have real dates
            # In real scenario, use actual scraped_date
            np.random.seed(42)  # For reproducible results

            # Generate mock dates over the last 2 years
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)

            date_range = pd.date_range(start=start_date, end=end_date, periods=len(df))
            df["analysis_date"] = np.random.choice(
                date_range, size=len(df), replace=True
            )
            df["year"] = df["analysis_date"].dt.year
            df["month"] = df["analysis_date"].dt.to_period("M")
            df["quarter"] = df["analysis_date"].dt.to_period("Q")

            return df

        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            raise

    def analyze_temporal_trends(self, df: pd.DataFrame) -> dict:
        """Analyze temporal trends in themes and sentiments"""
        logger.info("📈 Analyzing temporal trends...")

        trends = {}

        # === SENTIMENT TRENDS OVER TIME ===
        sentiment_trends = (
            df.groupby(["quarter", "qwen_sentiment"]).size().unstack(fill_value=0)
        )
        sentiment_trends_pct = (
            sentiment_trends.div(sentiment_trends.sum(axis=1), axis=0) * 100
        )

        # Convert to serializable format
        sentiment_data = {}
        for period, row in sentiment_trends_pct.iterrows():
            sentiment_data[str(period)] = {str(k): float(v) for k, v in row.items()}

        trends["sentiment_over_time"] = {
            "data": sentiment_data,
            "periods": [str(p) for p in sentiment_trends_pct.index],
            "sentiments": list(sentiment_trends_pct.columns),
        }

        # === THEME EVOLUTION ===
        theme_evolution = {}
        for period in df["quarter"].unique():
            period_data = df[df["quarter"] == period]

            # Extract themes from qwen_themes
            all_themes = []
            for themes_str in period_data["qwen_themes"].dropna():
                if pd.notna(themes_str) and themes_str.strip():
                    # Parse themes (assume comma-separated)
                    themes = [t.strip().lower() for t in str(themes_str).split(",")]
                    all_themes.extend(themes)

            # Count theme frequency
            if all_themes:
                theme_counts = Counter(all_themes)
                # Keep top 10 themes for this period
                top_themes = dict(theme_counts.most_common(10))
                theme_evolution[str(period)] = top_themes

        trends["theme_evolution"] = theme_evolution

        # === ARTIST POPULARITY TRENDS ===
        artist_trends = {}
        for period in sorted(df["quarter"].unique()):
            period_data = df[df["quarter"] == period]
            artist_counts = period_data["artist"].value_counts().head(10).to_dict()
            artist_trends[str(period)] = artist_counts

        trends["artist_trends"] = artist_trends

        # === QUALITY TRENDS ===
        quality_trends = (
            df.groupby("quarter")["quality_score"].agg(["mean", "std"]).round(3)
        )
        trends["quality_trends"] = {
            "periods": [str(p) for p in quality_trends.index],
            "mean_quality": [float(x) for x in quality_trends["mean"].tolist()],
            "std_quality": [float(x) for x in quality_trends["std"].tolist()],
        }

        logger.info(
            f"✅ Temporal trends analyzed for {len(sentiment_trends_pct)} periods"
        )
        return trends

    def cluster_musical_styles(self, df: pd.DataFrame, n_clusters: int = 6) -> dict:
        """Cluster musical styles based on features"""
        logger.info(f"🎵 Clustering musical styles into {n_clusters} clusters...")

        # Features for clustering
        numeric_features = [
            "word_count",
            "lyrics_length",
            "lines_count",
            "avg_words_per_line",
            "qwen_confidence",
            "qwen_complexity",
        ]

        # Create feature matrix
        feature_data = df[numeric_features].fillna(df[numeric_features].median())

        # Add encoded categorical features
        categorical_features = ["qwen_sentiment", "theme_category"]
        for cat_feature in categorical_features:
            if cat_feature in df.columns:
                # Simple label encoding
                unique_vals = df[cat_feature].unique()
                val_to_num = {val: i for i, val in enumerate(unique_vals)}
                feature_data[f"{cat_feature}_encoded"] = (
                    df[cat_feature].map(val_to_num).fillna(0)
                )

        # Scale features
        scaled_features = self.scaler.fit_transform(feature_data)

        # KMeans clustering
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = self.cluster_model.fit_predict(scaled_features)

        # Add cluster labels to dataframe
        df["style_cluster"] = clusters

        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(n_clusters):
            cluster_data = df[df["style_cluster"] == cluster_id]

            # Cluster characteristics
            cluster_analysis[cluster_id] = {
                "size": len(cluster_data),
                "percentage": float(len(cluster_data) / len(df) * 100),
                "top_artists": {
                    str(k): int(v)
                    for k, v in cluster_data["artist"]
                    .value_counts()
                    .head(5)
                    .to_dict()
                    .items()
                },
                "avg_features": {
                    "word_count": float(cluster_data["word_count"].mean()),
                    "complexity": float(cluster_data["qwen_complexity"].mean()),
                    "quality": float(cluster_data["quality_score"].mean()),
                },
                "dominant_sentiment": str(cluster_data["qwen_sentiment"].mode().iloc[0])
                if not cluster_data["qwen_sentiment"].mode().empty
                else "neutral",
                "dominant_theme": str(cluster_data["theme_category"].mode().iloc[0])
                if not cluster_data["theme_category"].mode().empty
                else "general",
                "sample_songs": [
                    {str(k): str(v) for k, v in song.items()}
                    for song in cluster_data[["artist", "title"]]
                    .head(3)
                    .to_dict("records")
                ],
            }

        # PCA for visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_features)

        cluster_analysis["pca_visualization"] = {
            "x": pca_result[:, 0].tolist(),
            "y": pca_result[:, 1].tolist(),
            "clusters": clusters.tolist(),
            "explained_variance": pca.explained_variance_ratio_.tolist(),
        }

        self.style_clusters = cluster_analysis
        logger.info(f"✅ Musical styles clustered into {n_clusters} groups")

        return cluster_analysis

    def predict_emerging_trends(
        self, df: pd.DataFrame, forecast_months: int = 6
    ) -> dict:
        """Predict emerging trends based on growth patterns"""
        logger.info(
            f"🔮 Predicting emerging trends for next {forecast_months} months..."
        )

        # Analyze theme growth over time
        monthly_theme_data = {}

        for month in df["month"].unique():
            month_data = df[df["month"] == month]

            # Extract and count themes
            all_themes = []
            for themes_str in month_data["qwen_themes"].dropna():
                if pd.notna(themes_str) and themes_str.strip():
                    themes = [t.strip().lower() for t in str(themes_str).split(",")]
                    # Filter out very generic or empty themes
                    themes = [
                        t
                        for t in themes
                        if len(t) > 2 and t not in ["general", "rap", "music"]
                    ]
                    all_themes.extend(themes)

            if all_themes:
                theme_counts = Counter(all_themes)
                monthly_theme_data[str(month)] = dict(theme_counts.most_common(20))

        # Calculate growth rates
        trend_predictions = {}
        all_themes = set()
        for monthly_data in monthly_theme_data.values():
            all_themes.update(monthly_data.keys())

        for theme in all_themes:
            if not theme or len(theme) <= 2:
                continue

            monthly_counts = []
            periods = []

            for period, themes_dict in monthly_theme_data.items():
                count = themes_dict.get(theme, 0)
                monthly_counts.append(count)
                periods.append(period)

            if (
                len(monthly_counts) >= 3 and sum(monthly_counts) >= 5
            ):  # Minimum threshold
                # Calculate trend using simple linear regression
                x = np.arange(len(monthly_counts))
                if len(monthly_counts) > 1 and np.std(monthly_counts) > 0:
                    # Simple trend calculation
                    trend_slope = np.polyfit(x, monthly_counts, 1)[0]
                    current_avg = (
                        np.mean(monthly_counts[-3:])
                        if len(monthly_counts) >= 3
                        else np.mean(monthly_counts)
                    )

                    # Predict future popularity
                    predicted_growth = trend_slope * forecast_months
                    growth_rate = (
                        (trend_slope / (current_avg + 1)) * 100
                        if current_avg > 0
                        else 0
                    )

                    trend_predictions[theme] = {
                        "current_popularity": int(current_avg),
                        "trend_slope": float(trend_slope),
                        "predicted_growth": float(predicted_growth),
                        "growth_rate_percent": float(growth_rate),
                        "confidence": min(
                            1.0, sum(monthly_counts) / 50
                        ),  # Confidence based on sample size
                    }

        # Sort by growth rate and filter
        emerging_trends = {
            theme: data
            for theme, data in trend_predictions.items()
            if data["growth_rate_percent"] > 10 and data["confidence"] > 0.3
        }

        # Sort by growth rate
        emerging_trends = dict(
            sorted(
                emerging_trends.items(),
                key=lambda x: x[1]["growth_rate_percent"],
                reverse=True,
            )[:15]
        )

        logger.info(f"✅ Identified {len(emerging_trends)} emerging trends")
        return {
            "emerging_trends": emerging_trends,
            "forecast_period": forecast_months,
            "analysis_date": datetime.now().isoformat(),
        }

    def analyze_viral_patterns(self, df: pd.DataFrame) -> dict:
        """Analyze patterns in viral/popular tracks"""
        logger.info("🚀 Analyzing viral patterns...")

        # Define viral tracks based on quality score (top 20%)
        quality_threshold = df["quality_score"].quantile(0.8)
        viral_tracks = df[df["quality_score"] >= quality_threshold]

        viral_analysis = {
            "total_viral_tracks": len(viral_tracks),
            "viral_percentage": float(len(viral_tracks) / len(df) * 100),
            # Viral track characteristics
            "avg_characteristics": {
                "word_count": float(viral_tracks["word_count"].mean()),
                "lyrics_length": float(viral_tracks["lyrics_length"].mean()),
                "lines_count": float(viral_tracks["lines_count"].mean()),
                "complexity": float(viral_tracks["qwen_complexity"].mean()),
                "confidence": float(viral_tracks["qwen_confidence"].mean()),
            },
            # Popular themes in viral tracks
            "viral_themes": self._extract_theme_patterns(viral_tracks["qwen_themes"]),
            # Sentiment distribution
            "sentiment_distribution": viral_tracks["qwen_sentiment"]
            .value_counts()
            .to_dict(),
            # Top viral artists
            "top_viral_artists": viral_tracks["artist"]
            .value_counts()
            .head(10)
            .to_dict(),
            # Theme categories
            "theme_category_distribution": viral_tracks["theme_category"]
            .value_counts()
            .to_dict(),
        }

        # Compare with non-viral tracks
        non_viral_tracks = df[df["quality_score"] < quality_threshold]

        viral_analysis["comparison"] = {
            "viral_vs_normal": {
                "word_count_ratio": float(
                    viral_tracks["word_count"].mean()
                    / non_viral_tracks["word_count"].mean()
                ),
                "complexity_ratio": float(
                    viral_tracks["qwen_complexity"].mean()
                    / non_viral_tracks["qwen_complexity"].mean()
                ),
                "confidence_ratio": float(
                    viral_tracks["qwen_confidence"].mean()
                    / non_viral_tracks["qwen_confidence"].mean()
                ),
            }
        }

        self.viral_patterns = viral_analysis
        logger.info(f"✅ Viral patterns analyzed for {len(viral_tracks)} viral tracks")

        return viral_analysis

    def _extract_theme_patterns(self, themes_series: pd.Series) -> dict:
        """Extract common theme patterns"""
        all_themes = []
        for themes_str in themes_series.dropna():
            if pd.notna(themes_str) and themes_str.strip():
                themes = [t.strip().lower() for t in str(themes_str).split(",")]
                themes = [t for t in themes if len(t) > 2]  # Filter short themes
                all_themes.extend(themes)

        if all_themes:
            theme_counts = Counter(all_themes)
            return dict(theme_counts.most_common(15))
        return {}

    def generate_trend_report(
        self, dataset_path: str = "data/ml/quick_dataset.pkl"
    ) -> dict:
        """Generate comprehensive trend analysis report"""
        logger.info("📊 Generating comprehensive trend report...")

        try:
            # Load data
            df = self.load_data(dataset_path)

            # Run all analyses
            temporal_trends = self.analyze_temporal_trends(df)
            style_clusters = self.cluster_musical_styles(df)
            emerging_trends = self.predict_emerging_trends(df)
            viral_patterns = self.analyze_viral_patterns(df)

            # Compile report
            report = {
                "metadata": {
                    "generation_date": datetime.now().isoformat(),
                    "dataset_size": len(df),
                    "analysis_period": f"{df['analysis_date'].min()} to {df['analysis_date'].max()}",
                    "unique_artists": df["artist"].nunique(),
                    "unique_themes": len(
                        self._extract_theme_patterns(df["qwen_themes"])
                    ),
                },
                "temporal_trends": temporal_trends,
                "style_clusters": style_clusters,
                "emerging_trends": emerging_trends,
                "viral_patterns": viral_patterns,
                "key_insights": self._generate_key_insights(
                    temporal_trends, emerging_trends, viral_patterns
                ),
            }

            # Save report
            output_path = "./models/trend_analysis_report.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"✅ Trend report saved to {output_path}")
            return report

        except Exception as e:
            logger.error(f"❌ Trend report generation failed: {e}")
            raise

    def _generate_key_insights(
        self, temporal_trends: dict, emerging_trends: dict, viral_patterns: dict
    ) -> list[str]:
        """Generate key insights from trend analysis"""
        insights = []

        # Emerging trend insight
        if emerging_trends.get("emerging_trends"):
            top_trend = list(emerging_trends["emerging_trends"].keys())[0]
            growth_rate = emerging_trends["emerging_trends"][top_trend][
                "growth_rate_percent"
            ]
            insights.append(
                f"🔥 Fastest growing theme: '{top_trend}' (+{growth_rate:.1f}% growth rate)"
            )

        # Viral pattern insight
        if viral_patterns.get("avg_characteristics"):
            viral_word_count = viral_patterns["avg_characteristics"]["word_count"]
            insights.append(f"🚀 Viral tracks average {viral_word_count:.0f} words")

        # Sentiment trend
        if temporal_trends.get("sentiment_over_time"):
            # Get latest period data
            latest_sentiments = temporal_trends["sentiment_over_time"]["data"]
            if latest_sentiments:
                periods = list(latest_sentiments.keys())
                if periods:
                    latest_period = periods[-1]
                    if latest_period in latest_sentiments:
                        dominant_sentiment = max(
                            latest_sentiments[latest_period],
                            key=latest_sentiments[latest_period].get,
                        )
                        insights.append(
                            f"😊 Current dominant sentiment: {dominant_sentiment}"
                        )

        # Quality insight
        insights.append(
            f"💎 {viral_patterns.get('total_viral_tracks', 0)} tracks identified as viral-potential"
        )

        return insights

    def create_visualization_dashboard(self, report: dict) -> str:
        """Create interactive visualization dashboard"""
        logger.info("📊 Creating trend visualization dashboard...")

        if not PLOTLY_AVAILABLE:
            logger.warning("⚠️ Plotly not available, creating simple matplotlib plots")
            return self._create_matplotlib_dashboard(report)

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Emerging Trends Growth Rate",
                "Style Clusters (PCA)",
                "Viral Track Characteristics",
                "Theme Evolution",
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}],
            ],
        )

        # Plot 1: Emerging trends
        emerging = report.get("emerging_trends", {}).get("emerging_trends", {})
        if emerging:
            themes = list(emerging.keys())[:10]
            growth_rates = [emerging[theme]["growth_rate_percent"] for theme in themes]

            fig.add_trace(
                go.Bar(
                    x=growth_rates,
                    y=themes,
                    orientation="h",
                    name="Growth Rate %",
                    marker=dict(color=growth_rates, colorscale="Viridis"),
                ),
                row=1,
                col=1,
            )

        # Plot 2: Style clusters PCA
        clusters = report.get("style_clusters", {})
        if "pca_visualization" in clusters:
            pca_data = clusters["pca_visualization"]
            unique_clusters = list(set(pca_data["clusters"]))

            for cluster_id in unique_clusters:
                cluster_indices = [
                    i for i, c in enumerate(pca_data["clusters"]) if c == cluster_id
                ]
                x_vals = [pca_data["x"][i] for i in cluster_indices]
                y_vals = [pca_data["y"][i] for i in cluster_indices]

                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="markers",
                        name=f"Cluster {cluster_id}",
                        marker=dict(size=5),
                    ),
                    row=1,
                    col=2,
                )

        # Plot 3: Viral characteristics
        viral = report.get("viral_patterns", {}).get("avg_characteristics", {})
        if viral:
            characteristics = list(viral.keys())
            values = list(viral.values())

            fig.add_trace(
                go.Bar(
                    x=characteristics,
                    y=values,
                    name="Viral Features",
                    marker=dict(color="red", opacity=0.7),
                ),
                row=2,
                col=1,
            )

        # Plot 4: Theme evolution (latest period)
        temporal = report.get("temporal_trends", {}).get("theme_evolution", {})
        if temporal:
            latest_period = list(temporal.keys())[-1] if temporal else None
            if latest_period and latest_period in temporal:
                themes = list(temporal[latest_period].keys())[:10]
                counts = [temporal[latest_period][theme] for theme in themes]

                fig.add_trace(
                    go.Bar(
                        x=themes,
                        y=counts,
                        name="Theme Frequency",
                        marker=dict(color="blue", opacity=0.7),
                    ),
                    row=2,
                    col=2,
                )

        # Update layout
        fig.update_layout(
            height=800,
            title_text="Rap Music Trend Analysis Dashboard",
            showlegend=False,
        )

        # Save dashboard
        dashboard_path = "./models/trend_dashboard.html"
        fig.write_html(dashboard_path)

        logger.info(f"✅ Interactive dashboard saved to {dashboard_path}")
        return dashboard_path

    def _create_matplotlib_dashboard(self, report: dict) -> str:
        """Create simple matplotlib dashboard as fallback"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Rap Music Trend Analysis Dashboard", fontsize=16)

        # Plot emerging trends
        emerging = report.get("emerging_trends", {}).get("emerging_trends", {})
        if emerging:
            themes = list(emerging.keys())[:8]
            growth_rates = [emerging[theme]["growth_rate_percent"] for theme in themes]
            axes[0, 0].barh(themes, growth_rates)
            axes[0, 0].set_title("Emerging Trends Growth Rate")
            axes[0, 0].set_xlabel("Growth Rate %")

        # Plot viral characteristics
        viral = report.get("viral_patterns", {}).get("avg_characteristics", {})
        if viral:
            characteristics = list(viral.keys())
            values = list(viral.values())
            axes[0, 1].bar(characteristics, values)
            axes[0, 1].set_title("Viral Track Characteristics")
            axes[0, 1].tick_params(axis="x", rotation=45)

        # Plot style clusters sizes
        clusters = report.get("style_clusters", {})
        if clusters:
            cluster_ids = []
            cluster_sizes = []
            for cluster_id, data in clusters.items():
                if isinstance(cluster_id, int) and "size" in data:
                    cluster_ids.append(f"Cluster {cluster_id}")
                    cluster_sizes.append(data["size"])

            if cluster_ids:
                axes[1, 0].bar(cluster_ids, cluster_sizes)
                axes[1, 0].set_title("Style Cluster Sizes")
                axes[1, 0].tick_params(axis="x", rotation=45)

        # Plot theme evolution
        temporal = report.get("temporal_trends", {}).get("theme_evolution", {})
        if temporal:
            latest_period = list(temporal.keys())[-1] if temporal else None
            if latest_period and latest_period in temporal:
                themes = list(temporal[latest_period].keys())[:8]
                counts = [temporal[latest_period][theme] for theme in themes]
                axes[1, 1].bar(themes, counts)
                axes[1, 1].set_title(f"Theme Frequency ({latest_period})")
                axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()

        # Save dashboard
        dashboard_path = "./models/trend_dashboard.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"✅ Matplotlib dashboard saved to {dashboard_path}")
        return dashboard_path


def run_trend_analysis():
    """Complete trend analysis pipeline"""
    logger.info("🚀 RAP TREND ANALYSIS - FULL PIPELINE")
    logger.info("=" * 60)

    try:
        # Initialize analyzer
        analyzer = RapTrendAnalyzer()

        # Generate comprehensive report
        report = analyzer.generate_trend_report("data/ml/quick_dataset.pkl")

        # Create visualization dashboard
        dashboard_path = analyzer.create_visualization_dashboard(report)

        # Print key insights
        print("\n🔍 KEY INSIGHTS:")
        print("=" * 50)
        for insight in report["key_insights"]:
            print(f"  {insight}")

        # Print emerging trends
        emerging = report.get("emerging_trends", {}).get("emerging_trends", {})
        if emerging:
            print("\n📈 TOP 10 EMERGING TRENDS:")
            print("-" * 50)
            for i, (theme, data) in enumerate(list(emerging.items())[:10], 1):
                growth = data["growth_rate_percent"]
                confidence = data["confidence"]
                print(
                    f"  {i:2d}. {theme:20} | Growth: +{growth:5.1f}% | Confidence: {confidence:.2f}"
                )

        # Print viral patterns
        viral = report.get("viral_patterns", {})
        if viral:
            print("\n🚀 VIRAL PATTERNS:")
            print("-" * 50)
            print(f"  Viral tracks identified: {viral.get('total_viral_tracks', 0)}")
            print(
                f"  Average word count: {viral.get('avg_characteristics', {}).get('word_count', 0):.0f}"
            )
            print(
                f"  Top viral theme: {list(viral.get('viral_themes', {}).keys())[0] if viral.get('viral_themes') else 'N/A'}"
            )

        print("\n📊 ANALYSIS COMPLETE!")
        print("Report saved to: ./models/trend_analysis_report.json")
        print(f"Dashboard saved to: {dashboard_path}")

        return True

    except Exception as e:
        logger.error(f"❌ Trend analysis failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rap Trend Analysis")
    parser.add_argument(
        "--mode",
        choices=["analyze"],
        default="analyze",
        help="Mode: run trend analysis",
    )

    args = parser.parse_args()

    if args.mode == "analyze":
        success = run_trend_analysis()
        sys.exit(0 if success else 1)
