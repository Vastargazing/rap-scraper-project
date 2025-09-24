#!/usr/bin/env python3
"""
🔍 Проверка SQLite базы данных для миграции Spotify данных
"""
import sqlite3
import json

def check_sqlite_spotify_data():
    """Проверяем что есть в старой SQLite базе"""
    db_path = 'data/rap_lyrics.db'
    
    try:
        conn = sqlite3.connect(db_path)
        print("🔍 CHECKING SQLITE DATABASE")
        print("=" * 40)
        
        # Проверяем какие таблицы есть
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        print('📋 TABLES:')
        for table in tables:
            print(f'   - {table[0]}')
        
        # Проверяем структуру spotify_tracks
        print('\n🎵 SPOTIFY_TRACKS STRUCTURE:')
        try:
            schema = conn.execute('PRAGMA table_info(spotify_tracks)').fetchall()
            for col in schema:
                print(f'   - {col[1]} ({col[2]})')
        except:
            print('   ❌ No spotify_tracks table')
            return
        
        # Проверяем сколько данных
        count = conn.execute('SELECT COUNT(*) FROM spotify_tracks').fetchone()[0]
        print(f'\n📊 SPOTIFY DATA COUNT:')
        print(f'   Total spotify tracks: {count:,}')
        
        if count == 0:
            print('   ❌ No data to migrate')
            return
        
        # Примеры данных
        print('\n📋 SAMPLE DATA:')
        # Сначала получаем все колонки
        columns_info = conn.execute('PRAGMA table_info(spotify_tracks)').fetchall()
        column_names = [col[1] for col in columns_info]
        print(f'   Available columns: {", ".join(column_names)}')
        
        # Берем первые несколько колонок для примера
        sample_columns = column_names[:6] if len(column_names) >= 6 else column_names
        query = f"SELECT {', '.join(sample_columns)} FROM spotify_tracks LIMIT 5"
        samples = conn.execute(query).fetchall()
        
        for i, sample in enumerate(samples, 1):
            print(f'   {i}. {" | ".join([f"{col}: {val}" for col, val in zip(sample_columns, sample)])}')
        
        # Проверяем связь с songs
        print('\n🔗 LINKING CHECK:')
        linked_query = '''
            SELECT COUNT(*) 
            FROM spotify_tracks st 
            JOIN songs s ON st.song_id = s.id
        '''
        linked_count = conn.execute(linked_query).fetchone()[0]
        print(f'   Linked with songs table: {linked_count:,}')
        
        # Топ артисты в SQLite
        print('\n🏆 TOP ARTISTS IN SQLITE:')
        top_artists_query = '''
            SELECT s.artist, COUNT(*) as count
            FROM spotify_tracks st
            JOIN songs s ON st.song_id = s.id
            GROUP BY s.artist
            ORDER BY count DESC
            LIMIT 5
        '''
        top_artists = conn.execute(top_artists_query).fetchall()
        for artist, count in top_artists:
            print(f'   - {artist}: {count} tracks')
        
        print(f'\n✅ READY FOR MIGRATION: {count:,} tracks!')
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
    
    finally:
        conn.close()

if __name__ == "__main__":
    check_sqlite_spotify_data()