#!/usr/bin/env python3
"""
Скрипт для объединения двух баз данных rap_lyrics.db
Использует новую схему с метаданными
"""
import sqlite3
import os
import shutil
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseMerger:
    def __init__(self, old_db_path: str, new_db_path: str, output_db_path: str = None):
        self.old_db_path = old_db_path
        self.new_db_path = new_db_path
        self.output_db_path = output_db_path or f"merged_rap_lyrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        
    def backup_databases(self):
        """Создание резервных копий"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if os.path.exists(self.old_db_path):
            backup_old = f"backup_old_{timestamp}.db"
            shutil.copy2(self.old_db_path, backup_old)
            logger.info(f"Создана резервная копия старой БД: {backup_old}")
            
        if os.path.exists(self.new_db_path):
            backup_new = f"backup_new_{timestamp}.db"
            shutil.copy2(self.new_db_path, backup_new)
            logger.info(f"Создана резервная копия новой БД: {backup_new}")
    
    def create_new_schema(self, conn):
        """Создание новой схемы с метаданными"""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS songs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                artist TEXT NOT NULL,
                title TEXT NOT NULL,
                lyrics TEXT NOT NULL,
                url TEXT UNIQUE NOT NULL,
                genius_id INTEGER UNIQUE,
                scraped_date TEXT DEFAULT CURRENT_TIMESTAMP,
                word_count INTEGER,
                -- Новые поля для метаданных
                genre TEXT,
                release_date TEXT,
                album TEXT,
                language TEXT,
                explicit BOOLEAN,
                song_art_url TEXT,
                popularity_score INTEGER,
                lyrics_quality_score REAL,
                UNIQUE(artist, title)
            )
        """)
        
        # Создание индексов
        conn.execute("CREATE INDEX IF NOT EXISTS idx_artist ON songs(artist)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_url ON songs(url)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_genius_id ON songs(genius_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_genre ON songs(genre)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_release_date ON songs(release_date)")
        
        conn.commit()
        logger.info("Создана новая схема БД с метаданными")
    
    def get_table_columns(self, conn, table_name):
        """Получение списка колонок таблицы"""
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        return [row[1] for row in cursor.fetchall()]
    
    def merge_databases(self):
        """Основной метод объединения"""
        logger.info("Начинаем объединение баз данных...")
        
        # Создание резервных копий
        self.backup_databases()
        
        # Проверка существования файлов
        if not os.path.exists(self.old_db_path):
            logger.error(f"Старая БД не найдена: {self.old_db_path}")
            return False
            
        # Создание выходной БД
        output_conn = sqlite3.connect(self.output_db_path)
        output_conn.execute("PRAGMA journal_mode=WAL")
        output_conn.execute("PRAGMA synchronous=NORMAL")
        
        try:
            # Создание новой схемы
            self.create_new_schema(output_conn)
            
            # Подключение к старой БД
            old_conn = sqlite3.connect(self.old_db_path)
            old_conn.row_factory = sqlite3.Row
            
            # Получение колонок старой таблицы
            old_columns = self.get_table_columns(old_conn, 'songs')
            logger.info(f"Колонки старой БД: {old_columns}")
            
            # Копирование данных из старой БД
            cursor = old_conn.execute("SELECT * FROM songs")
            copied = 0
            skipped = 0
            
            logger.info("Копируем данные из старой БД...")
            
            while True:
                rows = cursor.fetchmany(1000)  # Батчами по 1000
                if not rows:
                    break
                    
                for row in rows:
                    try:
                        # Подготовка данных для вставки
                        data = {
                            'artist': row['artist'],
                            'title': row['title'],
                            'lyrics': row['lyrics'],
                            'url': row['url'],
                            'genius_id': row.get('genius_id'),
                            'scraped_date': row.get('scraped_date'),
                            'word_count': row.get('word_count'),
                            # Новые поля пока NULL
                            'genre': None,
                            'release_date': None,
                            'album': None,
                            'language': None,
                            'explicit': None,
                            'song_art_url': None,
                            'popularity_score': None,
                            'lyrics_quality_score': None
                        }
                        
                        # Вставка в новую БД
                        output_conn.execute("""
                            INSERT OR IGNORE INTO songs (
                                artist, title, lyrics, url, genius_id, scraped_date, word_count,
                                genre, release_date, album, language, explicit, song_art_url,
                                popularity_score, lyrics_quality_score
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, tuple(data.values()))
                        
                        copied += 1
                        
                        if copied % 1000 == 0:
                            output_conn.commit()
                            logger.info(f"Скопировано {copied} записей...")
                            
                    except sqlite3.IntegrityError:
                        skipped += 1
                        
            old_conn.close()
            
            # Копирование из новой БД (если существует)
            if os.path.exists(self.new_db_path):
                logger.info("Копируем данные из новой БД...")
                new_conn = sqlite3.connect(self.new_db_path)
                new_conn.row_factory = sqlite3.Row
                
                cursor = new_conn.execute("SELECT * FROM songs")
                new_copied = 0
                
                while True:
                    rows = cursor.fetchmany(1000)
                    if not rows:
                        break
                        
                    for row in rows:
                        try:
                            # Все колонки должны совпадать
                            columns = list(row.keys())[1:]  # Исключаем id
                            placeholders = ', '.join(['?' for _ in columns])
                            column_names = ', '.join(columns)
                            
                            output_conn.execute(f"""
                                INSERT OR IGNORE INTO songs ({column_names})
                                VALUES ({placeholders})
                            """, [row[col] for col in columns])
                            
                            new_copied += 1
                            
                            if new_copied % 1000 == 0:
                                output_conn.commit()
                                logger.info(f"Из новой БД скопировано {new_copied} записей...")
                                
                        except sqlite3.IntegrityError:
                            skipped += 1
                
                new_conn.close()
                logger.info(f"Из новой БД скопировано: {new_copied} записей")
            
            # Финальный коммит и статистика
            output_conn.commit()
            
            # Получение финальной статистики
            cursor = output_conn.execute("SELECT COUNT(*) as total, COUNT(DISTINCT artist) as artists FROM songs")
            stats = cursor.fetchone()
            
            logger.info(f"""
            ✅ ОБЪЕДИНЕНИЕ ЗАВЕРШЕНО!
            📁 Выходной файл: {self.output_db_path}
            📊 Статистика:
               • Всего песен: {stats[0]}
               • Уникальных артистов: {stats[1]}
               • Скопировано из старой БД: {copied}
               • Пропущено дубликатов: {skipped}
            """)
            
            output_conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при объединении: {e}")
            output_conn.close()
            return False

def main():
    print("🔄 Объединение баз данных rap_lyrics")
    print("=" * 50)
    
    # Пути к файлам
    old_db = "rap_lyrics.db"  # Текущая БД
    new_db = "rap_lyrics_new.db"  # Новая БД (если есть)
    
    # Проверяем наличие файлов
    if not os.path.exists(old_db):
        print(f"❌ Файл {old_db} не найден!")
        return
    
    print(f"📁 Старая БД: {old_db}")
    if os.path.exists(new_db):
        print(f"📁 Новая БД: {new_db}")
    else:
        print("📁 Новая БД: не найдена (будет объединена только старая)")
    
    # Подтверждение
    response = input("\n❓ Продолжить объединение? (y/N): ")
    if response.lower() != 'y':
        print("❌ Отменено пользователем")
        return
    
    # Объединение
    merger = DatabaseMerger(old_db, new_db)
    success = merger.merge_databases()
    
    if success:
        print(f"\n✅ Объединение успешно завершено!")
        print(f"📁 Результат: {merger.output_db_path}")
        print("\n💡 Рекомендации:")
        print("1. Переименуйте новый файл в rap_lyrics.db")
        print("2. Резервные копии сохранены автоматически")
    else:
        print("❌ Ошибка при объединении")

if __name__ == "__main__":
    main()
