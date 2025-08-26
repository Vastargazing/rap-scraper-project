#!/usr/bin/env python3
import sqlite3
import os
from .config import DB_PATH

def check_database():
    db_file = str(DB_PATH)
    
    if not os.path.exists(db_file):
        print(f"❌ Файл {db_file} не найден!")
        return
    
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Проверяем таблицы
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"📋 Таблицы в базе: {tables}")
        
        if 'songs' in tables:
            # Общая статистика
            cursor.execute("SELECT COUNT(*) FROM songs")
            total_songs = cursor.fetchone()[0]
            print(f"🎵 Всего песен: {total_songs}")
            
            # Количество артистов
            cursor.execute("SELECT COUNT(DISTINCT artist) FROM songs")
            unique_artists = cursor.fetchone()[0]
            print(f"👤 Уникальных артистов: {unique_artists}")
            
            # Размер базы
            file_size = os.path.getsize(db_file) / (1024 * 1024)  # В МБ
            print(f"💾 Размер файла: {file_size:.2f} МБ")
            
            # Топ 5 артистов по количеству песен
            cursor.execute("""
                SELECT artist, COUNT(*) as song_count 
                FROM songs 
                GROUP BY artist 
                ORDER BY song_count DESC 
                LIMIT 5
            """)
            top_artists = cursor.fetchall()
            print(f"\n🏆 Топ 5 артистов:")
            for artist, count in top_artists:
                print(f"  • {artist}: {count} песен")
            
            # Последние добавленные
            cursor.execute("""
                SELECT artist, title, scraped_date 
                FROM songs 
                ORDER BY id DESC 
                LIMIT 3
            """)
            recent = cursor.fetchall()
            print(f"\n🕒 Последние добавленные:")
            for artist, title, date in recent:
                print(f"  • {artist} - {title} ({date})")
        
        # Проверяем AI анализ если таблица существует
        if 'ai_analysis' in tables:
            cursor.execute("SELECT COUNT(*) FROM ai_analysis")
            analyzed_count = cursor.fetchone()[0]
            print(f"\n🤖 AI анализов: {analyzed_count}")
            
            if analyzed_count > 0:
                # Последние анализы
                cursor.execute("""
                    SELECT s.artist, s.title, a.genre, a.overall_quality, a.analysis_date
                    FROM ai_analysis a
                    JOIN songs s ON a.song_id = s.id
                    ORDER BY a.id DESC
                    LIMIT 3
                """)
                analyses = cursor.fetchall()
                print(f"🔍 Последние анализы:")
                for artist, title, genre, quality, date in analyses:
                    print(f"  • {artist} - {title} | {genre} | {quality} ({date[:10]})")
        else:
            print(f"\n⚠️ Таблица ai_analysis не найдена")
        
        conn.close()
        print(f"\n✅ База данных проверена успешно!")
        
    except Exception as e:
        print(f"❌ Ошибка при проверке базы: {e}")

def main():
    """Entry point function."""
    check_database()

if __name__ == "__main__":
    main()
