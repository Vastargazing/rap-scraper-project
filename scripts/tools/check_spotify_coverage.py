#!/usr/bin/env python3
"""
🎯 Quick Spotify Coverage Report
Быстрый отчет по покрытию данных Spotify в базе

НАЗНАЧЕНИЕ:
- Подсчет записей и покрытие полей в таблицах Spotify
- Быстрая проверка целостности и полноты обогащения
- Вывод отчета в консоль

ИСПОЛЬЗОВАНИЕ:
python scripts/tools/check_spotify_coverage.py
python scripts/tools/check_spotify_coverage.py --help

ЗАВИСИМОСТИ:
- Python 3.8+
- SQLite database (data/rap_lyrics.db)

РЕЗУЛЬТАТ:
- Консольный отчет с метриками покрытия
- Код возврата 0/1 в зависимости от состояния

АВТОР: Vastargazing | ДАТА: Сентябрь 2025
"""
import sqlite3
from pathlib import Path

DB = Path('data/rap_lyrics.db')
if not DB.exists():
    print(f"Database not found at {DB.resolve()}")
    raise SystemExit(1)

conn = sqlite3.connect(str(DB))
conn.row_factory = None
cursor = conn.cursor()

def table_exists(name):
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cursor.fetchone() is not None

def get_count(table):
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    return cursor.fetchone()[0]

def get_columns(table):
    cursor.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cursor.fetchall()]

def non_null_count(table, column):
    cursor.execute(f"SELECT COUNT({column}) FROM {table}")
    return cursor.fetchone()[0]

tables = [
    'songs',
    'spotify_tracks',
    'spotify_artists',
    'spotify_audio_features'
]

report = {}
for table in tables:
    if not table_exists(table):
        print(f"Table '{table}' not found in DB.")
        report[table] = None
        continue
    total = get_count(table)
    cols = get_columns(table)
    col_stats = []
    for c in cols:
        try:
            nn = non_null_count(table, c)
        except Exception:
            nn = None
        col_stats.append((c, nn))
    report[table] = {
        'total_rows': total,
        'columns': col_stats
    }

# Extra: how many songs have linked spotify_tracks
linked = None
if table_exists('songs') and table_exists('spotify_tracks'):
    try:
        cursor.execute('SELECT COUNT(DISTINCT s.id) FROM songs s JOIN spotify_tracks st ON s.id = st.song_id')
        linked = cursor.fetchone()[0]
    except Exception:
        linked = None

conn.close()

# Print summary
print('\n=== Spotify coverage report ===\n')
for table, info in report.items():
    if info is None:
        continue
    print(f"Table: {table} — rows: {info['total_rows']}")
    for col, nn in info['columns']:
        if nn is None:
            pct = 'N/A'
        else:
            pct = f"{nn}/{info['total_rows']} ({(nn/info['total_rows']*100) if info['total_rows'] else 0:.1f}%)"
        print(f"  - {col}: {pct}")
    print('')

if linked is not None:
    print(f"Songs linked to spotify_tracks: {linked} / {report['songs']['total_rows']} ({linked/report['songs']['total_rows']*100:.1f}%)")

print('\nNotes:')
print('- spotify_tracks.audio_features are stored in separate table `spotify_audio_features` as `track_spotify_id`. Check join coverage if needed.')
print('- To extract additional fields (ISRC, album_id, album_label, track popularity over time, external_ids), update `src/enhancers/spotify_enhancer.py` -> `search_track()` and `create_spotify_tables()` and re-run enhancement scripts.')
