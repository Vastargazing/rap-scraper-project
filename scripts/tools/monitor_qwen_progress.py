#!/usr/bin/env python3
"""
📊 Мониторинг прогресса массового анализа Qwen
"""

import sys
import os
import sqlite3
from datetime import datetime, timedelta
import time

# Добавляем корневую папку в path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

def monitor_qwen_progress():
    """Мониторинг прогресса анализа Qwen"""
    
    db_path = os.path.join(project_root, 'data', 'rap_lyrics.db')
    
    if not os.path.exists(db_path):
        print("❌ База данных не найдена!")
        return
    
    print("📊 Qwen Analysis Progress Monitor")
    print("=" * 50)
    
    conn = sqlite3.connect(db_path)
    
    try:
        # Общее количество записей
        total_query = "SELECT COUNT(*) FROM songs WHERE lyrics IS NOT NULL AND lyrics != ''"
        total_records = conn.execute(total_query).fetchone()[0]
        
        # Количество уже проанализированных записей (любой моделью)
        all_analyzed_query = "SELECT COUNT(*) FROM ai_analysis"
        all_analyzed = conn.execute(all_analyzed_query).fetchone()[0]
        
        # Количество проанализированных записей Qwen
        qwen_analyzed_query = "SELECT COUNT(*) FROM ai_analysis WHERE model_version LIKE '%qwen%'"
        qwen_analyzed = conn.execute(qwen_analyzed_query).fetchone()[0]
        
        # Статистика по моделям
        models_query = """
        SELECT model_version, COUNT(*) as count
        FROM ai_analysis 
        GROUP BY model_version 
        ORDER BY count DESC
        """
        models_stats = conn.execute(models_query).fetchall()
        
        # Последние анализы Qwen
        recent_query = """
        SELECT analysis_date, COUNT(*) as count
        FROM ai_analysis 
        WHERE model_version LIKE '%qwen%' 
        AND analysis_date >= datetime('now', '-1 hour')
        GROUP BY datetime(analysis_date, 'localtime')
        ORDER BY analysis_date DESC
        LIMIT 10
        """
        recent_analyses = conn.execute(recent_query).fetchall()
        
        # Статистика по времени для Qwen
        time_stats_query = """
        SELECT 
            MIN(analysis_date) as first_analysis,
            MAX(analysis_date) as last_analysis,
            COUNT(*) as total_analyzed
        FROM ai_analysis 
        WHERE model_version LIKE '%qwen%'
        """
        time_stats = conn.execute(time_stats_query).fetchone()
        
        # Вывод статистики
        qwen_progress_percent = (qwen_analyzed / total_records) * 100 if total_records > 0 else 0
        all_progress_percent = (all_analyzed / total_records) * 100 if total_records > 0 else 0
        remaining_total = total_records - all_analyzed
        remaining_qwen = total_records - qwen_analyzed
        
        print(f"📈 Общий прогресс анализа:")
        print(f"  📊 Всего записей в базе: {total_records:,}")
        print(f"  ✅ Проанализировано всеми моделями: {all_analyzed:,} ({all_progress_percent:.1f}%)")
        print(f"  🤖 Проанализировано Qwen: {qwen_analyzed:,} ({qwen_progress_percent:.2f}%)")
        print(f"  ⏳ Осталось для любого анализа: {remaining_total:,}")
        print(f"  🎯 Осталось для Qwen анализа: {remaining_qwen:,}")
        
        # Статистика по моделям
        if models_stats:
            print(f"\n🤖 Статистика по моделям:")
            for model, count in models_stats:
                percent = (count / all_analyzed) * 100 if all_analyzed > 0 else 0
                print(f"  {model}: {count:,} ({percent:.1f}%)")
        
        if time_stats[0] and qwen_analyzed > 0:  # Если есть Qwen анализы
            first_analysis = datetime.fromisoformat(time_stats[0])
            last_analysis = datetime.fromisoformat(time_stats[1])
            duration = last_analysis - first_analysis
            
            if duration.total_seconds() > 0:
                rate = qwen_analyzed / (duration.total_seconds() / 3600)  # записей в час
                estimated_remaining_hours = remaining_qwen / rate if rate > 0 else float('inf')
                
                print(f"\n⏱️  Qwen временная статистика:")
                print(f"  🚀 Первый анализ: {first_analysis.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  🏁 Последний анализ: {last_analysis.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  ⚡ Скорость: {rate:.1f} записей/час")
                
                if estimated_remaining_hours < float('inf'):
                    if estimated_remaining_hours < 24:
                        print(f"  ⏰ Оставшееся время: {estimated_remaining_hours:.1f} часов")
                    else:
                        print(f"  ⏰ Оставшееся время: {estimated_remaining_hours/24:.1f} дней")
        
        # Недавняя активность
        if recent_analyses:
            print(f"\n🕐 Недавняя активность (последний час):")
            for timestamp, count in recent_analyses:
                dt = datetime.fromisoformat(timestamp)
                print(f"  📅 {dt.strftime('%H:%M:%S')}: {count} записей")
        
        # Статистика по настроениям
        sentiment_query = """
        SELECT mood, COUNT(*) as count
        FROM ai_analysis 
        WHERE model_version LIKE '%qwen%'
        GROUP BY mood
        ORDER BY count DESC
        """
        sentiments = conn.execute(sentiment_query).fetchall()
        
        if sentiments:
            print(f"\n😊 Qwen распределение по настроениям:")
            for sentiment, count in sentiments:
                percent = (count / qwen_analyzed) * 100 if qwen_analyzed > 0 else 0
                print(f"  {sentiment}: {count} ({percent:.1f}%)")
        
        print(f"\n🔄 Обновлено: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    finally:
        conn.close()

def monitor_live(interval=60):
    """Живой мониторинг с обновлением каждые N секунд"""
    print("🔄 Запуск живого мониторинга (Ctrl+C для остановки)")
    print(f"⏱️  Интервал обновления: {interval} секунд")
    print("-" * 50)
    
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')  # Очистка экрана
            monitor_qwen_progress()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n👋 Мониторинг остановлен")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Мониторинг прогресса анализа Qwen")
    parser.add_argument("--live", action="store_true", help="Живой мониторинг")
    parser.add_argument("--interval", type=int, default=60, help="Интервал обновления в секундах")
    
    args = parser.parse_args()
    
    if args.live:
        monitor_live(args.interval)
    else:
        monitor_qwen_progress()
