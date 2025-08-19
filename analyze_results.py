"""
Анализ и визуализация результатов AI обогащения данных
"""
import sqlite3
import json
import pandas as pd
from pathlib import Path
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultsAnalyzer:
    """Анализатор результатов AI обогащения"""
    
    def __init__(self, db_path="rap_lyrics.db"):
        self.conn = sqlite3.connect(db_path)
        
    def get_comprehensive_stats(self):
        """Получаем полную статистику обогащенных данных"""
        
        # Основная статистика
        query = """
        SELECT 
            COUNT(DISTINCT s.id) as total_songs,
            COUNT(DISTINCT a.id) as analyzed_songs,
            AVG(a.authenticity_score) as avg_authenticity,
            AVG(a.lyrical_creativity) as avg_creativity,
            AVG(a.commercial_appeal) as avg_commercial,
            AVG(a.uniqueness) as avg_uniqueness,
            AVG(a.ai_likelihood) as avg_ai_likelihood,
            COUNT(DISTINCT a.genre) as unique_genres,
            COUNT(DISTINCT s.artist) as unique_artists
        FROM songs s
        LEFT JOIN ai_analysis a ON s.id = a.song_id
        """
        
        df = pd.read_sql_query(query, self.conn)
        return df.iloc[0].to_dict()
    
    def get_genre_distribution(self):
        """Распределение по жанрам"""
        query = """
        SELECT 
            genre, 
            COUNT(*) as count,
            AVG(authenticity_score) as avg_authenticity,
            AVG(commercial_appeal) as avg_commercial,
            AVG(lyrical_creativity) as avg_creativity
        FROM ai_analysis
        WHERE genre IS NOT NULL
        GROUP BY genre
        ORDER BY count DESC
        """
        return pd.read_sql_query(query, self.conn)
    
    def get_quality_metrics(self):
        """Анализ качественных метрик"""
        query = """
        SELECT 
            overall_quality,
            COUNT(*) as count,
            AVG(authenticity_score) as avg_authenticity,
            AVG(commercial_appeal) as avg_commercial,
            AVG(ai_likelihood) as avg_ai_likelihood
        FROM ai_analysis
        WHERE overall_quality IS NOT NULL
        GROUP BY overall_quality
        ORDER BY 
            CASE overall_quality 
                WHEN 'excellent' THEN 4
                WHEN 'good' THEN 3  
                WHEN 'average' THEN 2
                WHEN 'poor' THEN 1
                ELSE 0
            END DESC
        """
        return pd.read_sql_query(query, self.conn)
    
    def get_mood_analysis(self):
        """Анализ настроений"""
        query = """
        SELECT 
            mood,
            COUNT(*) as count,
            AVG(energy_level = 'high') as high_energy_ratio,
            AVG(explicit_content) as explicit_ratio
        FROM ai_analysis
        WHERE mood IS NOT NULL
        GROUP BY mood
        ORDER BY count DESC
        """
        return pd.read_sql_query(query, self.conn)
    
    def get_top_artists_by_quality(self, limit=10):
        """Топ артисты по качеству"""
        query = f"""
        SELECT 
            s.artist,
            COUNT(*) as song_count,
            AVG(a.authenticity_score) as avg_authenticity,
            AVG(a.lyrical_creativity) as avg_creativity,
            AVG(a.commercial_appeal) as avg_commercial,
            AVG(a.overall_quality = 'excellent') as excellent_ratio
        FROM songs s
        JOIN ai_analysis a ON s.id = a.song_id
        GROUP BY s.artist
        HAVING song_count >= 3
        ORDER BY avg_authenticity DESC, avg_creativity DESC
        LIMIT {limit}
        """
        return pd.read_sql_query(query, self.conn)
    
    def get_complexity_analysis(self):
        """Анализ сложности текстов"""
        query = """
        SELECT 
            complexity_level,
            COUNT(*) as count,
            AVG(authenticity_score) as avg_authenticity,
            AVG(lyrical_creativity) as avg_creativity,
            AVG(s.word_count) as avg_word_count
        FROM ai_analysis a
        JOIN songs s ON a.song_id = s.id
        WHERE complexity_level IS NOT NULL
        GROUP BY complexity_level
        ORDER BY 
            CASE complexity_level
                WHEN 'complex' THEN 3
                WHEN 'medium' THEN 2  
                WHEN 'simple' THEN 1
                ELSE 0
            END DESC
        """
        return pd.read_sql_query(query, self.conn)
    
    def generate_report(self):
        """Генерируем полный отчет"""
        
        print("\\n" + "="*80)
        print("🎵 AI-ENHANCED RAP LYRICS ANALYSIS REPORT 🎵")
        print("="*80)
        
        # Основная статистика
        stats = self.get_comprehensive_stats()
        print(f"\\n📊 GENERAL STATISTICS:")
        print(f"   Total songs in database: {stats['total_songs']:,}")
        print(f"   Songs with AI analysis: {stats['analyzed_songs']:,}")
        print(f"   Analysis coverage: {(stats['analyzed_songs']/stats['total_songs']*100):.1f}%")
        print(f"   Unique artists: {stats['unique_artists']:,}")
        print(f"   Unique genres discovered: {stats['unique_genres']}")
        
        print(f"\\n⭐ QUALITY METRICS (Average scores 0-1):")
        print(f"   Authenticity: {stats['avg_authenticity']:.3f}")
        print(f"   Lyrical Creativity: {stats['avg_creativity']:.3f}")
        print(f"   Commercial Appeal: {stats['avg_commercial']:.3f}")
        print(f"   Uniqueness: {stats['avg_uniqueness']:.3f}")
        print(f"   AI Likelihood: {stats['avg_ai_likelihood']:.3f}")
        
        # Жанры
        print(f"\\n🎼 TOP GENRES:")
        genres = self.get_genre_distribution()
        for _, row in genres.head(10).iterrows():
            print(f"   {row['genre']}: {row['count']} songs (Auth: {row['avg_authenticity']:.2f})")
        
        # Качество
        print(f"\\n🏆 QUALITY DISTRIBUTION:")
        quality = self.get_quality_metrics()
        for _, row in quality.iterrows():
            print(f"   {row['overall_quality'].title()}: {row['count']} songs ({row['count']/stats['analyzed_songs']*100:.1f}%)")
        
        # Настроения
        print(f"\\n😊 MOOD ANALYSIS:")
        moods = self.get_mood_analysis()
        for _, row in moods.head(8).iterrows():
            explicit_pct = row['explicit_ratio'] * 100 if row['explicit_ratio'] else 0
            print(f"   {row['mood']}: {row['count']} songs (Explicit: {explicit_pct:.0f}%)")
        
        # Топ артисты
        print(f"\\n🌟 TOP ARTISTS BY AUTHENTICITY (min 3 songs):")
        artists = self.get_top_artists_by_quality()
        for _, row in artists.head(8).iterrows():
            print(f"   {row['artist']}: {row['song_count']} songs (Auth: {row['avg_authenticity']:.3f})")
        
        # Сложность
        print(f"\\n📚 LYRICAL COMPLEXITY:")
        complexity = self.get_complexity_analysis()
        for _, row in complexity.iterrows():
            print(f"   {row['complexity_level'].title()}: {row['count']} songs (Creativity: {row['avg_creativity']:.3f})")
        
        print("\\n" + "="*80)
        
        # Сохраняем детальные данные
        self.save_detailed_data()
        
    def save_detailed_data(self):
        """Сохраняем детальные данные для дальнейшего анализа"""
        
        results_dir = Path("analysis_results")
        results_dir.mkdir(exist_ok=True)
        
        # Сохраняем все таблицы
        tables = {
            'genre_distribution': self.get_genre_distribution(),
            'quality_metrics': self.get_quality_metrics(),
            'mood_analysis': self.get_mood_analysis(),
            'top_artists': self.get_top_artists_by_quality(20),
            'complexity_analysis': self.get_complexity_analysis()
        }
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        for name, df in tables.items():
            filename = results_dir / f"{name}_{timestamp}.csv"
            df.to_csv(filename, index=False, encoding='utf-8')
            logger.info(f"Saved {name} to {filename}")
        
        print(f"\\n💾 Detailed analysis saved to: {results_dir}/")
        
    def close(self):
        self.conn.close()

def main():
    """Основная функция анализа"""
    analyzer = ResultsAnalyzer()
    
    try:
        analyzer.generate_report()
    finally:
        analyzer.close()

if __name__ == "__main__":
    main()
