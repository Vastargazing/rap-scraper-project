#!/usr/bin/env python3
"""
🎯 Comprehensive AI Analysis Statistics Generator
Генератор подробной статистики по результатам AI-анализов

НАЗНАЧЕНИЕ:
- Сбор и агрегация статистики по AI-анализу корпуса треков
- Жанровый, эмоциональный и качественный анализ
- Экспорт JSON-отчетов для дальнейшей визуализации

ИСПОЛЬЗОВАНИЕ:
python scripts/tools/comprehensive_ai_stats.py
python scripts/tools/comprehensive_ai_stats.py --output results.json

ЗАВИСИМОСТИ:
- Python 3.8+
- SQLite database (data/rap_lyrics.db)

РЕЗУЛЬТАТ:
- JSON-отчеты в results/ и/или вывод в консоль
- Сводные CSV/JSON для дашбордов

АВТОР: Vastargazing | ДАТА: Сентябрь 2025
"""
import sqlite3
import json
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Any

class AIAnalysisStatsGenerator:
    """Генератор статистики AI-анализов"""
    
    def __init__(self, db_path: str = "data/rap_lyrics.db"):
        self.db_path = db_path
        
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Получение всеобъемлющей статистики"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        stats = {}
        
        # 1. ОСНОВНАЯ СТАТИСТИКА
        stats['overview'] = self._get_overview_stats(conn)
        
        # 2. ЖАНРОВЫЙ АНАЛИЗ  
        stats['genre_analysis'] = self._get_genre_stats(conn)
        
        # 3. АНАЛИЗ НАСТРОЕНИЙ
        stats['mood_analysis'] = self._get_mood_stats(conn)
        
        # 4. КАЧЕСТВЕННЫЕ МЕТРИКИ
        stats['quality_metrics'] = self._get_quality_stats(conn)
        
        # 5. АНАЛИЗ ПО АРТИСТАМ
        stats['artist_insights'] = self._get_artist_stats(conn)
        
        # 6. ВРЕМЕННЫЕ ТРЕНДЫ
        stats['temporal_trends'] = self._get_temporal_stats(conn)
        
        # 7. ТЕКСТОВАЯ СЛОЖНОСТЬ
        stats['complexity_analysis'] = self._get_complexity_stats(conn)
        
        # 8. КОММЕРЧЕСКИЙ ПОТЕНЦИАЛ
        stats['commercial_insights'] = self._get_commercial_stats(conn)
        
        conn.close()
        return stats
    
    def _get_overview_stats(self, conn) -> Dict[str, Any]:
        """Общая статистика"""
        cursor = conn.cursor()
        
        # Общие цифры
        cursor.execute("SELECT COUNT(*) FROM songs")
        total_songs = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM ai_analysis")
        analyzed_songs = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT song_id) FROM ai_analysis")
        unique_analyzed = cursor.fetchone()[0]
        
        coverage = (analyzed_songs / total_songs * 100) if total_songs > 0 else 0
        
        return {
            'total_songs': total_songs,
            'analyzed_songs': analyzed_songs,
            'unique_analyzed': unique_analyzed,
            'coverage_percent': round(coverage, 2),
            'remaining_to_analyze': total_songs - unique_analyzed
        }
    
    def _get_genre_stats(self, conn) -> Dict[str, Any]:
        """Статистика по жанрам"""
        cursor = conn.cursor()
        
        # Распределение жанров
        cursor.execute("""
            SELECT genre, COUNT(*) as count
            FROM ai_analysis
            WHERE genre IS NOT NULL AND genre != ''
            GROUP BY genre
            ORDER BY count DESC
        """)
        genre_distribution = dict(cursor.fetchall())
        
        # Топ поджанры
        cursor.execute("""
            SELECT subgenre, COUNT(*) as count
            FROM ai_analysis
            WHERE subgenre IS NOT NULL AND subgenre != ''
            GROUP BY subgenre
            ORDER BY count DESC
            LIMIT 20
        """)
        subgenre_distribution = dict(cursor.fetchall())
        
        return {
            'genre_distribution': genre_distribution,
            'subgenre_distribution': subgenre_distribution,
            'total_unique_genres': len(genre_distribution),
            'total_unique_subgenres': len(subgenre_distribution)
        }
    
    def _get_mood_stats(self, conn) -> Dict[str, Any]:
        """Анализ настроений"""
        cursor = conn.cursor()
        
        # Распределение настроений
        cursor.execute("""
            SELECT mood, COUNT(*) as count
            FROM ai_analysis
            WHERE mood IS NOT NULL AND mood != ''
            GROUP BY mood
            ORDER BY count DESC
        """)
        mood_distribution = dict(cursor.fetchall())
        
        # Эмоциональный тон
        cursor.execute("""
            SELECT emotional_tone, COUNT(*) as count
            FROM ai_analysis
            WHERE emotional_tone IS NOT NULL AND emotional_tone != ''
            GROUP BY emotional_tone
            ORDER BY count DESC
        """)
        emotional_tone_distribution = dict(cursor.fetchall())
        
        # Энергетические уровни
        cursor.execute("""
            SELECT energy_level, COUNT(*) as count
            FROM ai_analysis
            WHERE energy_level IS NOT NULL AND energy_level != ''
            GROUP BY energy_level
            ORDER BY count DESC
        """)
        energy_distribution = dict(cursor.fetchall())
        
        return {
            'mood_distribution': mood_distribution,
            'emotional_tone_distribution': emotional_tone_distribution,
            'energy_distribution': energy_distribution
        }
    
    def _get_quality_stats(self, conn) -> Dict[str, Any]:
        """Метрики качества"""
        cursor = conn.cursor()
        
        # Средние показатели качества
        cursor.execute("""
            SELECT 
                AVG(authenticity_score) as avg_authenticity,
                AVG(lyrical_creativity) as avg_creativity,
                AVG(commercial_appeal) as avg_commercial,
                AVG(uniqueness) as avg_uniqueness,
                AVG(ai_likelihood) as avg_ai_likelihood
            FROM ai_analysis
            WHERE authenticity_score IS NOT NULL
        """)
        quality_averages = cursor.fetchone()
        
        # Распределение общего качества
        cursor.execute("""
            SELECT overall_quality, COUNT(*) as count
            FROM ai_analysis
            WHERE overall_quality IS NOT NULL AND overall_quality != ''
            GROUP BY overall_quality
            ORDER BY count DESC
        """)
        quality_distribution = dict(cursor.fetchall())
        
        # Треки с высоким риском ИИ
        cursor.execute("""
            SELECT COUNT(*) 
            FROM ai_analysis
            WHERE ai_likelihood > 0.7
        """)
        high_ai_risk = cursor.fetchone()[0]
        
        return {
            'quality_averages': {
                'authenticity': round(quality_averages[0], 3) if quality_averages[0] else None,
                'creativity': round(quality_averages[1], 3) if quality_averages[1] else None,
                'commercial_appeal': round(quality_averages[2], 3) if quality_averages[2] else None,
                'uniqueness': round(quality_averages[3], 3) if quality_averages[3] else None,
                'ai_likelihood': round(quality_averages[4], 3) if quality_averages[4] else None
            },
            'quality_distribution': quality_distribution,
            'high_ai_risk_tracks': high_ai_risk
        }
    
    def _get_artist_stats(self, conn) -> Dict[str, Any]:
        """Статистика по артистам"""
        cursor = conn.cursor()
        
        # Топ артисты по количеству анализов
        cursor.execute("""
            SELECT s.artist, COUNT(*) as analyzed_count
            FROM songs s
            INNER JOIN ai_analysis a ON s.id = a.song_id
            GROUP BY s.artist
            ORDER BY analyzed_count DESC
            LIMIT 20
        """)
        top_analyzed_artists = dict(cursor.fetchall())
        
        # Средние показатели качества по топ-артистам
        cursor.execute("""
            SELECT s.artist, 
                   AVG(a.authenticity_score) as avg_authenticity,
                   AVG(a.lyrical_creativity) as avg_creativity,
                   COUNT(*) as track_count
            FROM songs s
            INNER JOIN ai_analysis a ON s.id = a.song_id
            WHERE a.authenticity_score IS NOT NULL
            GROUP BY s.artist
            HAVING track_count >= 10
            ORDER BY avg_authenticity DESC
            LIMIT 15
        """)
        top_quality_artists = cursor.fetchall()
        
        return {
            'top_analyzed_artists': top_analyzed_artists,
            'top_quality_artists': [(artist, round(auth, 3), round(creat, 3), count) 
                                   for artist, auth, creat, count in top_quality_artists]
        }
    
    def _get_complexity_stats(self, conn) -> Dict[str, Any]:
        """Анализ сложности текстов"""
        cursor = conn.cursor()
        
        # Распределение уровней сложности
        cursor.execute("""
            SELECT complexity_level, COUNT(*) as count
            FROM ai_analysis
            WHERE complexity_level IS NOT NULL AND complexity_level != ''
            GROUP BY complexity_level
            ORDER BY count DESC
        """)
        complexity_distribution = dict(cursor.fetchall())
        
        # Качество игры слов
        cursor.execute("""
            SELECT wordplay_quality, COUNT(*) as count
            FROM ai_analysis
            WHERE wordplay_quality IS NOT NULL AND wordplay_quality != ''
            GROUP BY wordplay_quality
            ORDER BY count DESC
        """)
        wordplay_distribution = dict(cursor.fetchall())
        
        # Схемы рифмовки
        cursor.execute("""
            SELECT rhyme_scheme, COUNT(*) as count
            FROM ai_analysis
            WHERE rhyme_scheme IS NOT NULL AND rhyme_scheme != ''
            GROUP BY rhyme_scheme
            ORDER BY count DESC
            LIMIT 15
        """)
        rhyme_distribution = dict(cursor.fetchall())
        
        return {
            'complexity_distribution': complexity_distribution,
            'wordplay_distribution': wordplay_distribution,
            'rhyme_distribution': rhyme_distribution
        }
    
    def _get_commercial_stats(self, conn) -> Dict[str, Any]:
        """Коммерческий анализ"""
        cursor = conn.cursor()
        
        # Треки с высоким коммерческим потенциалом
        cursor.execute("""
            SELECT s.artist, s.title, a.commercial_appeal
            FROM songs s
            INNER JOIN ai_analysis a ON s.id = a.song_id
            WHERE a.commercial_appeal > 0.8
            ORDER BY a.commercial_appeal DESC
            LIMIT 20
        """)
        high_commercial_tracks = [(artist, title, round(appeal, 3)) for artist, title, appeal in cursor.fetchall()]
        
        # Средний коммерческий потенциал по жанрам
        cursor.execute("""
            SELECT genre, AVG(commercial_appeal) as avg_commercial
            FROM ai_analysis
            WHERE commercial_appeal IS NOT NULL AND genre IS NOT NULL
            GROUP BY genre
            ORDER BY avg_commercial DESC
        """)
        commercial_by_genre = [(genre, round(avg, 3)) for genre, avg in cursor.fetchall()]
        
        return {
            'high_commercial_tracks': high_commercial_tracks,
            'commercial_by_genre': commercial_by_genre
        }
    
    def _get_temporal_stats(self, conn) -> Dict[str, Any]:
        """Временные тренды"""
        cursor = conn.cursor()
        
        # Анализы по годам (по year_estimate)
        cursor.execute("""
            SELECT year_estimate, COUNT(*) as count
            FROM ai_analysis
            WHERE year_estimate IS NOT NULL AND year_estimate > 1990 AND year_estimate < 2030
            GROUP BY year_estimate
            ORDER BY year_estimate
        """)
        year_distribution = dict(cursor.fetchall())
        
        # Анализы по дням (по analysis_date)
        cursor.execute("""
            SELECT DATE(analysis_date) as date, COUNT(*) as count
            FROM ai_analysis
            WHERE analysis_date IS NOT NULL
            GROUP BY DATE(analysis_date)
            ORDER BY date DESC
            LIMIT 30
        """)
        daily_analysis = dict(cursor.fetchall())
        
        return {
            'year_distribution': year_distribution,
            'daily_analysis': daily_analysis
        }
    
    def generate_report(self, output_file: str = None) -> str:
        """Генерация полного отчета"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"results/ai_analysis_comprehensive_report_{timestamp}.json"
        elif not output_file.startswith('results/'):
            output_file = f"results/{output_file}"
        
        print("📊 Генерация всеобъемлющей статистики AI-анализов...")
        
        stats = self.get_comprehensive_stats()
        
        # Добавляем метаданные
        stats['metadata'] = {
            'generated_at': datetime.now().isoformat(),
            'database_path': self.db_path,
            'generator_version': '1.0'
        }
        
        # Сохраняем в JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # Печатаем краткий summary
        self._print_summary(stats)
        
        print(f"\n📄 Полный отчет сохранен: {output_file}")
        return output_file
    
    def _print_summary(self, stats: Dict[str, Any]):
        """Печать краткого резюме"""
        overview = stats['overview']
        genre_stats = stats['genre_analysis']
        mood_stats = stats['mood_analysis']
        quality_stats = stats['quality_metrics']
        
        print("\n🎯 КРАТКОЕ РЕЗЮМЕ АНАЛИЗА")
        print("=" * 50)
        
        print(f"📊 Покрытие: {overview['analyzed_songs']:,} анализов из {overview['total_songs']:,} треков ({overview['coverage_percent']}%)")
        print(f"🎵 Осталось проанализировать: {overview['remaining_to_analyze']:,} треков")
        
        print(f"\n🎼 Жанровое разнообразие:")
        top_genres = list(genre_stats['genre_distribution'].items())[:5]
        for genre, count in top_genres:
            print(f"   • {genre}: {count:,} треков")
        
        print(f"\n😊 Топ настроения:")
        top_moods = list(mood_stats['mood_distribution'].items())[:5]
        for mood, count in top_moods:
            print(f"   • {mood}: {count:,} треков")
        
        if quality_stats['quality_averages']['authenticity']:
            print(f"\n⭐ Средние показатели качества:")
            qa = quality_stats['quality_averages']
            print(f"   • Аутентичность: {qa['authenticity']}")
            print(f"   • Креативность: {qa['creativity']}")
            print(f"   • Коммерческий потенциал: {qa['commercial_appeal']}")
            print(f"   • Уникальность: {qa['uniqueness']}")
            print(f"   • Риск ИИ-генерации: {qa['ai_likelihood']}")


def main():
    """CLI для генерации статистики"""
    generator = AIAnalysisStatsGenerator()
    report_file = generator.generate_report()
    
    print(f"\n✅ Анализ завершен! Отчет: {report_file}")
    
    # Предложение дополнительных действий
    print("\n💡 Дополнительные возможности:")
    print("   1. Создать визуализации (требует matplotlib)")
    print("   2. Экспорт в CSV для дальнейшего анализа")
    print("   3. Генерация markdown отчета")


if __name__ == "__main__":
    main()
