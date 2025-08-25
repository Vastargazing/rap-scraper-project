#!/usr/bin/env python3
"""
Мониторинг прогресса анализа Gemma 3 27B в реальном времени
"""

import sqlite3
import time
import os
from datetime import datetime, timedelta

def monitor_analysis_progress(db_path="rap_lyrics.db", refresh_interval=60):
    """Мониторинг прогресса анализа в реальном времени"""
    
    print("📊 МОНИТОРИНГ АНАЛИЗА GEMMA 3 27B")
    print("=" * 50)
    print("💡 Обновление каждые {} секунд. Нажмите Ctrl+C для выхода\n".format(refresh_interval))
    
    last_count = 0
    start_time = datetime.now()
    
    try:
        while True:
            # Получаем статистику
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Общая статистика
            cursor.execute("SELECT COUNT(*) FROM songs WHERE lyrics IS NOT NULL AND LENGTH(lyrics) > 100")
            total_songs = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM ai_analysis WHERE model_version = 'gemma-3-27b-it'")
            gemma_analyzed = cursor.fetchone()[0]
            
            # Прогресс за последний период
            progress_delta = gemma_analyzed - last_count
            last_count = gemma_analyzed
            
            # Скорость анализа
            elapsed = datetime.now() - start_time
            if elapsed.total_seconds() > 0:
                overall_rate = gemma_analyzed / elapsed.total_seconds() * 3600  # песен в час
            else:
                overall_rate = 0
            
            # Оставшееся время
            remaining = total_songs - gemma_analyzed
            if overall_rate > 0:
                eta_hours = remaining / overall_rate
                eta = timedelta(hours=eta_hours)
            else:
                eta = "Неизвестно"
            
            # Последние анализы
            cursor.execute("""
                SELECT a.analysis_date, s.artist, s.title
                FROM ai_analysis a
                JOIN songs s ON a.song_id = s.id
                WHERE a.model_version = 'gemma-3-27b-it'
                ORDER BY a.id DESC
                LIMIT 3
            """)
            
            recent = cursor.fetchall()
            conn.close()
            
            # Очищаем экран (работает в большинстве терминалов)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Выводим статистику
            progress_percent = (gemma_analyzed / total_songs * 100) if total_songs > 0 else 0
            
            print("📊 ПРОГРЕСС АНАЛИЗА GEMMA 3 27B")
            print("=" * 50)
            print(f"🎵 Общий прогресс: {gemma_analyzed:,} / {total_songs:,} ({progress_percent:.2f}%)")
            print(f"📈 За последние {refresh_interval}с: +{progress_delta} песен")
            print(f"⚡ Скорость: {overall_rate:.1f} песен/час")
            print(f"⏰ ETA: {eta}")
            print(f"🕐 Время работы: {elapsed}")
            print(f"🎯 Осталось: {remaining:,} песен")
            
            # Прогресс-бар
            bar_length = 40
            filled_length = int(bar_length * progress_percent / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"📊 [{bar}] {progress_percent:.1f}%")
            
            print(f"\n🔄 Последние анализы:")
            for i, (date, artist, title) in enumerate(recent, 1):
                date_str = date[:16] if date else "Unknown"
                print(f"   {i}. {date_str} | {artist} - {title[:30]}...")
            
            print(f"\n🔍 Следующее обновление через {refresh_interval} секунд...")
            print("❌ Нажмите Ctrl+C для выхода")
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n👋 Мониторинг завершен!")
    except Exception as e:
        print(f"\n❌ Ошибка мониторинга: {e}")

def show_quick_stats(db_path="rap_lyrics.db"):
    """Быстрая статистика без мониторинга"""
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM songs WHERE lyrics IS NOT NULL AND LENGTH(lyrics) > 100")
        total_songs = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM ai_analysis WHERE model_version = 'gemma-3-27b-it'")
        gemma_analyzed = cursor.fetchone()[0]
        
        # Анализы за последние 24 часа
        cursor.execute("""
            SELECT COUNT(*) FROM ai_analysis 
            WHERE model_version = 'gemma-3-27b-it' 
            AND analysis_date > datetime('now', '-1 day')
        """)
        recent_24h = cursor.fetchone()[0]
        
        conn.close()
        
        progress_percent = (gemma_analyzed / total_songs * 100) if total_songs > 0 else 0
        remaining = total_songs - gemma_analyzed
        
        print("📊 БЫСТРАЯ СТАТИСТИКА GEMMA 3 27B")
        print("=" * 40)
        print(f"✅ Проанализировано: {gemma_analyzed:,} / {total_songs:,}")
        print(f"📈 Прогресс: {progress_percent:.2f}%")
        print(f"🎯 Осталось: {remaining:,} песен")
        print(f"🔥 За 24 часа: {recent_24h} анализов")
        
        if recent_24h > 0:
            eta_days = remaining / recent_24h
            print(f"⏰ ETA при текущей скорости: {eta_days:.1f} дней")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        show_quick_stats()
    else:
        monitor_analysis_progress()
