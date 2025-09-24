#!/usr/bin/env python3
"""
🔎 Утилиты для анализа и мониторинга процесса обогащения Spotify

НАЗНАЧЕНИЕ:
- Получение статистики, мониторинг прогресса Spotify enhancement
- Анализ необработанных артистов и треков

ИСПОЛЬЗОВАНИЕ:
from src.enhancers.spotify_analysis_utils import SpotifyEnhancementAnalyzer

ЗАВИСИМОСТИ:
- Python 3.8+
- asyncpg (PostgreSQL)
- Используется в скриптах мониторинга и анализа

РЕЗУЛЬТАТ:
- Детальная статистика по Spotify enhancement
- Быстрая диагностика состояния базы

АВТОР: AI Assistant
ДАТА: Сентябрь 2025
"""
import asyncio
import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from database.postgres_adapter import PostgreSQLAdapter

class SpotifyEnhancementAnalyzer:
    """Анализатор процесса обогащения Spotify для PostgreSQL"""
    
    def __init__(self):
        self.db = PostgreSQLAdapter()
    
    async def get_detailed_stats(self) -> Dict[str, Any]:
        """Получение детальной статистики"""
        await self.db.connect()
        
        stats = {}
        
        try:
            # Основная статистика
            result = await self.db.execute_query("SELECT COUNT(*) as total FROM tracks")
            stats['total_songs'] = result[0]['total']
            
            result = await self.db.execute_query("SELECT COUNT(*) as enhanced FROM tracks WHERE spotify_data IS NOT NULL")
            stats['spotify_tracks'] = result[0]['enhanced']
            
            result = await self.db.execute_query("SELECT COUNT(DISTINCT artist) as artists FROM tracks WHERE spotify_data IS NOT NULL")
            stats['spotify_artists'] = result[0]['artists']
            
            # Процент обогащения
            if stats['total_songs'] > 0:
                stats['enhancement_percentage'] = round((stats['spotify_tracks'] / stats['total_songs']) * 100, 2)
            else:
                stats['enhancement_percentage'] = 0
            
            # Топ необработанных артистов
            unprocessed_query = """
                SELECT artist, COUNT(*) as tracks_count
                FROM tracks
                WHERE spotify_data IS NULL
                GROUP BY artist
                ORDER BY tracks_count DESC
                LIMIT 10
            """
            result = await self.db.execute_query(unprocessed_query)
            stats['top_unprocessed_artists'] = [(row['artist'], row['tracks_count']) for row in result]
            
            # Топ обработанных артистов
            processed_query = """
                SELECT artist, COUNT(*) as processed_tracks
                FROM tracks
                WHERE spotify_data IS NOT NULL
                GROUP BY artist
                ORDER BY processed_tracks DESC
                LIMIT 10
            """
            result = await self.db.execute_query(processed_query)
            stats['top_processed_artists'] = [(row['artist'], row['processed_tracks']) for row in result]
            
            # Статистика популярности
            popularity_query = """
                SELECT 
                    AVG((spotify_data->>'popularity')::int) as avg_popularity,
                    MIN((spotify_data->>'popularity')::int) as min_popularity,
                    MAX((spotify_data->>'popularity')::int) as max_popularity
                FROM tracks
                WHERE spotify_data IS NOT NULL 
                AND spotify_data->>'popularity' IS NOT NULL
                AND spotify_data->>'popularity' != '0'
            """
            result = await self.db.execute_query(popularity_query)
            if result and result[0]['avg_popularity']:
                stats['popularity'] = {
                    'avg': round(float(result[0]['avg_popularity']), 2),
                    'min': result[0]['min_popularity'],
                    'max': result[0]['max_popularity']
                }
            
            # Треки с самой высокой популярностью
            popular_query = """
                SELECT title, artist, (spotify_data->>'popularity')::int as popularity
                FROM tracks
                WHERE spotify_data IS NOT NULL 
                AND spotify_data->>'popularity' IS NOT NULL
                ORDER BY (spotify_data->>'popularity')::int DESC
                LIMIT 10
            """
            result = await self.db.execute_query(popular_query)
            stats['most_popular_tracks'] = [(row['title'], row['artist'], row['popularity']) for row in result]
            
        finally:
            await self.db.close()
        
        return stats
    
    async def find_problematic_tracks(self) -> Dict[str, List]:
        """Поиск проблематичных треков для обработки"""
        await self.db.connect()
        
        try:
            # Треки с длинными названиями
            long_query = """
                SELECT id, title, artist, LENGTH(title) as title_length
                FROM tracks
                WHERE spotify_data IS NULL AND LENGTH(title) > 50
                ORDER BY LENGTH(title) DESC
                LIMIT 20
            """
            long_result = await self.db.execute_query(long_query)
            long_titles = [(row['id'], row['title'], row['artist'], row['title_length']) for row in long_result]
            
            # Треки с специальными символами
            special_query = """
                SELECT id, title, artist
                FROM tracks
                WHERE spotify_data IS NULL 
                AND (title LIKE '%[%' OR title LIKE '%(%' 
                     OR title LIKE '%*%' OR title LIKE '%feat%'
                     OR title LIKE '%ft.%' OR title LIKE '%demo%'
                     OR title LIKE '%remix%' OR title LIKE '%version%')
                LIMIT 20
            """
            special_result = await self.db.execute_query(special_query)
            special_chars = [(row['id'], row['title'], row['artist']) for row in special_result]
            
        finally:
            await self.db.close()
        
        return {
            'long_titles': long_titles,
            'special_characters': special_chars
        }
    
    def get_artist_coverage_report(self) -> Dict[str, Any]:
        """Отчет по покрытию артистов"""
        conn = sqlite3.connect(self.db_path)
        
        # Артисты с процентом обработанных треков
        cursor = conn.execute("""
            SELECT 
                s.artist,
                COUNT(*) as total_tracks,
                COUNT(st.song_id) as processed_tracks,
                ROUND(COUNT(st.song_id) * 100.0 / COUNT(*), 2) as coverage_percent
            FROM tracks s
            LEFT JOIN spotify_tracks st ON s.id = st.song_id
            GROUP BY s.artist
            HAVING total_tracks >= 5
            ORDER BY coverage_percent ASC, total_tracks DESC
        """)
        
        coverage_data = cursor.fetchall()
        
        # Группируем по уровню покрытия
        coverage_groups = {
            'no_coverage': [],      # 0%
            'low_coverage': [],     # 1-25%
            'medium_coverage': [],  # 26-75%
            'high_coverage': [],    # 76-99%
            'full_coverage': []     # 100%
        }
        
        for artist, total, processed, percent in coverage_data:
            if percent == 0:
                coverage_groups['no_coverage'].append((artist, total, processed, percent))
            elif percent <= 25:
                coverage_groups['low_coverage'].append((artist, total, processed, percent))
            elif percent <= 75:
                coverage_groups['medium_coverage'].append((artist, total, processed, percent))
            elif percent < 100:
                coverage_groups['high_coverage'].append((artist, total, processed, percent))
            else:
                coverage_groups['full_coverage'].append((artist, total, processed, percent))
        
        conn.close()
        return coverage_groups
    
    def suggest_optimization_targets(self) -> Dict[str, List]:
        """Предложения по оптимизации процесса"""
        conn = sqlite3.connect(self.db_path)
        
        suggestions = {}
        
        # Артисты с большим количеством необработанных треков
        cursor = conn.execute("""
            SELECT s.artist, COUNT(*) as unprocessed_count
            FROM tracks s
            LEFT JOIN spotify_tracks st ON s.id = st.song_id
            WHERE st.song_id IS NULL
            GROUP BY s.artist
            HAVING unprocessed_count >= 10
            ORDER BY unprocessed_count DESC
            LIMIT 15
        """)
        suggestions['high_volume_artists'] = cursor.fetchall()
        
        # Треки с простыми названиями (легче найти)
        cursor = conn.execute("""
            SELECT s.id, s.title, s.artist
            FROM tracks s
            LEFT JOIN spotify_tracks st ON s.id = st.song_id
            WHERE st.song_id IS NULL 
            AND LENGTH(s.title) < 30
            AND s.title NOT LIKE '%[%'
            AND s.title NOT LIKE '%(%'
            AND s.title NOT LIKE '%*%'
            AND s.title NOT LIKE '%feat%'
            AND s.title NOT LIKE '%ft.%'
            AND s.title NOT LIKE '%demo%'
            AND s.title NOT LIKE '%remix%'
            ORDER BY RANDOM()
            LIMIT 100
        """)
        suggestions['easy_targets'] = cursor.fetchall()
        
        conn.close()
        return suggestions
    
    def export_analysis_report(self, filename: str = None):
        """Экспорт полного отчета анализа"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"spotify_analysis_report_{timestamp}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'detailed_stats': self.get_detailed_stats(),
            'problematic_tracks': self.find_problematic_tracks(),
            'artist_coverage': self.get_artist_coverage_report(),
            'optimization_targets': self.suggest_optimization_targets()
        }
        
        # Ensure report goes to results/ directory
        if not filename.startswith('results/'):
            filename = f"results/{filename}"
            
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"📊 Отчет сохранен: {filename}")
        return filename


def main():
    """CLI для анализа"""
    analyzer = SpotifyEnhancementAnalyzer()
    
    print("🔍 АНАЛИЗАТОР SPOTIFY ENHANCEMENT")
    print("=" * 50)
    
    print("\n1. 📊 Детальная статистика")
    stats = analyzer.get_detailed_stats()
    
    print(f"📈 Общая статистика:")
    print(f"   • Всего треков: {stats['total_songs']:,}")
    print(f"   • Обработано треков: {stats['spotify_tracks']:,}")
    print(f"   • Обработано артистов: {stats['spotify_artists']:,}")
    
    if stats['spotify_tracks'] > 0:
        coverage = stats['spotify_tracks'] / stats['total_songs'] * 100
        print(f"   • Покрытие треков: {coverage:.1f}%")
    
    if 'popularity' in stats:
        print(f"📊 Популярность треков:")
        print(f"   • Средняя: {stats['popularity']['avg']}")
        print(f"   • Мин-Макс: {stats['popularity']['min']}-{stats['popularity']['max']}")
    
    print(f"\n🎤 Топ необработанных артистов:")
    for artist, count in stats['top_unprocessed_artists'][:5]:
        print(f"   • {artist}: {count} треков")
    
    print(f"\n✅ Топ обработанных артистов:")
    for artist, count in stats['top_processed_artists'][:5]:
        print(f"   • {artist}: {count} треков")
    
    print("\n2. 🎯 Рекомендации по оптимизации")
    suggestions = analyzer.suggest_optimization_targets()
    
    print(f"🔥 Приоритетные артисты (много необработанных треков):")
    for artist, count in suggestions['high_volume_artists'][:5]:
        print(f"   • {artist}: {count} треков")
    
    print(f"\n🎯 Легкие цели (простые названия): {len(suggestions['easy_targets'])} треков")
    
    print("\n3. 🔍 Анализ покрытия")
    coverage = analyzer.get_artist_coverage_report()
    
    print(f"📊 Покрытие по артистам:")
    print(f"   • Без покрытия (0%): {len(coverage['no_coverage'])} артистов")
    print(f"   • Низкое покрытие (1-25%): {len(coverage['low_coverage'])} артистов")  
    print(f"   • Среднее покрытие (26-75%): {len(coverage['medium_coverage'])} артистов")
    print(f"   • Высокое покрытие (76-99%): {len(coverage['high_coverage'])} артистов")
    print(f"   • Полное покрытие (100%): {len(coverage['full_coverage'])} артистов")
    
    # Экспорт полного отчета
    choice = input("\n📄 Экспортировать полный отчет? (y/N): ").strip().lower()
    if choice == 'y':
        filename = analyzer.export_analysis_report()
        print(f"✅ Отчет экспортирован: {filename}")


if __name__ == "__main__":
    main()
