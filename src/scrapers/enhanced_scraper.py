"""
Интеграция LangChain анализа в основной rap_scraper
"""
import sqlite3
import json
import time
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from langchain_analyzer import GeminiLyricsAnalyzer
from models import EnhancedSongData, AnalysisResult

logger = logging.getLogger(__name__)

class EnhancedLyricsDatabase:
    """Расширенная база данных с поддержкой AI-анализа"""
    
    def __init__(self, db_name="rap_lyrics.db"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.create_enhanced_tables()
        self.analyzer = None
        self.batch_count = 0
        self.batch_size = 20
        
        # Создаем директории для результатов
        self.results_dir = Path("enhanced_data")
        self.results_dir.mkdir(exist_ok=True)
        
    def create_enhanced_tables(self):
        """Создает таблицы для хранения AI-анализа"""
        
        # Основная таблица песен (уже существует)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS songs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                artist TEXT NOT NULL,
                title TEXT NOT NULL,
                lyrics TEXT NOT NULL,
                url TEXT UNIQUE NOT NULL,
                genius_id INTEGER UNIQUE,
                scraped_date TEXT DEFAULT CURRENT_TIMESTAMP,
                word_count INTEGER,
                UNIQUE(artist, title)
            )
        """)
        
        # Таблица для AI-анализа
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                song_id INTEGER REFERENCES songs(id),
                
                -- Метаданные
                genre TEXT,
                subgenre TEXT,
                mood TEXT,
                year_estimate INTEGER,
                energy_level TEXT,
                explicit_content BOOLEAN,
                
                -- Анализ текста
                structure TEXT,
                rhyme_scheme TEXT,
                complexity_level TEXT,
                main_themes TEXT, -- JSON array
                emotional_tone TEXT,
                storytelling_type TEXT,
                wordplay_quality TEXT,
                
                -- Качественные метрики
                authenticity_score REAL,
                lyrical_creativity REAL,
                commercial_appeal REAL,
                uniqueness REAL,
                overall_quality TEXT,
                ai_likelihood REAL,
                
                -- Метаинформация
                analysis_date TEXT DEFAULT CURRENT_TIMESTAMP,
                model_version TEXT,
                
                UNIQUE(song_id)
            )
        """)
        
        # Индексы для производительности
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_genre ON ai_analysis(genre)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_quality ON ai_analysis(overall_quality)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_authenticity ON ai_analysis(authenticity_score)")
        
        self.conn.commit()
        logger.info("Enhanced database schema created/updated")
    
    def init_analyzer(self) -> bool:
        """Инициализируем анализатор при необходимости"""
        if self.analyzer is None:
            try:
                self.analyzer = GeminiLyricsAnalyzer()
                logger.info("✅ Gemini analyzer initialized")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to initialize analyzer: {e}")
                return False
        return True
    
    def analyze_existing_songs(self, limit: Optional[int] = None, skip_analyzed: bool = True):
        """Анализируем существующие песни в базе данных"""
        
        if not self.init_analyzer():
            return
        
        # Получаем песни для анализа
        query = """
            SELECT s.id, s.artist, s.title, s.lyrics, s.url, s.genius_id, 
                   s.scraped_date, s.word_count
            FROM songs s
        """
        
        if skip_analyzed:
            query += """
            LEFT JOIN ai_analysis a ON s.id = a.song_id
            WHERE a.song_id IS NULL
            """
        
        query += " ORDER BY s.word_count DESC"  # Начинаем с более содержательных песен
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor = self.conn.execute(query)
        songs_to_analyze = cursor.fetchall()
        
        logger.info(f"Found {len(songs_to_analyze)} songs for AI analysis")
        
        if not songs_to_analyze:
            logger.info("No songs to analyze")
            return
        
        # Обрабатываем каждую песню
        for i, song_row in enumerate(songs_to_analyze, 1):
            song_data = dict(song_row)
            
            logger.info(f"Analyzing {i}/{len(songs_to_analyze)}: {song_data['artist']} - {song_data['title']}")
            
            try:
                # Анализируем песню
                enhanced_data = self.analyzer.analyze_song_complete(song_data)
                
                # Сохраняем в базу данных
                self.save_ai_analysis(song_data['id'], enhanced_data)
                
                # Сохраняем в JSONL файл
                self.save_to_jsonl(enhanced_data)
                
                logger.info(f"✅ Successfully analyzed and saved")
                
                # Пауза между запросами (rate limiting)
                if i < len(songs_to_analyze):
                    time.sleep(5)  # 5 секунд между песнями
                    
            except Exception as e:
                logger.error(f"❌ Failed to analyze {song_data['artist']} - {song_data['title']}: {e}")
                continue
    
    def save_ai_analysis(self, song_id: int, enhanced_data: EnhancedSongData):
        """Сохраняем AI-анализ в базу данных"""
        
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO ai_analysis (
                    song_id, genre, subgenre, mood, year_estimate, energy_level, explicit_content,
                    structure, rhyme_scheme, complexity_level, main_themes, emotional_tone,
                    storytelling_type, wordplay_quality, authenticity_score, lyrical_creativity,
                    commercial_appeal, uniqueness, overall_quality, ai_likelihood,
                    analysis_date, model_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                song_id,
                enhanced_data.ai_metadata.genre,
                enhanced_data.ai_metadata.subgenre,
                enhanced_data.ai_metadata.mood,
                enhanced_data.ai_metadata.year_estimate,
                enhanced_data.ai_metadata.energy_level,
                enhanced_data.ai_metadata.explicit_content,
                enhanced_data.ai_analysis.structure,
                enhanced_data.ai_analysis.rhyme_scheme,
                enhanced_data.ai_analysis.complexity_level,
                json.dumps(enhanced_data.ai_analysis.main_themes),
                enhanced_data.ai_analysis.emotional_tone,
                enhanced_data.ai_analysis.storytelling_type,
                enhanced_data.ai_analysis.wordplay_quality,
                enhanced_data.quality_metrics.authenticity_score,
                enhanced_data.quality_metrics.lyrical_creativity,
                enhanced_data.quality_metrics.commercial_appeal,
                enhanced_data.quality_metrics.uniqueness,
                enhanced_data.quality_metrics.overall_quality,
                enhanced_data.quality_metrics.ai_likelihood,
                enhanced_data.analysis_date,
                enhanced_data.model_version
            ))
            
            self.batch_count += 1
            if self.batch_count >= self.batch_size:
                self.conn.commit()
                self.batch_count = 0
                logger.debug("Database batch committed")
                
        except Exception as e:
            logger.error(f"Failed to save AI analysis to database: {e}")
            raise
    
    def save_to_jsonl(self, enhanced_data: EnhancedSongData):
        """Сохраняем обогащенные данные в JSONL файл"""
        
        timestamp = time.strftime("%Y%m%d")
        jsonl_file = self.results_dir / f"enhanced_songs_{timestamp}.jsonl"
        
        with open(jsonl_file, "a", encoding="utf-8") as f:
            f.write(enhanced_data.model_dump_json() + "\\n")
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Получаем статистику AI-анализа"""
        
        cursor = self.conn.execute("""
            SELECT 
                COUNT(*) as total_songs,
                COUNT(a.id) as analyzed_songs,
                AVG(a.authenticity_score) as avg_authenticity,
                AVG(a.commercial_appeal) as avg_commercial_appeal,
                COUNT(DISTINCT a.genre) as unique_genres
            FROM songs s
            LEFT JOIN ai_analysis a ON s.id = a.song_id
        """)
        
        stats = dict(cursor.fetchone())
        
        # Топ жанры
        cursor = self.conn.execute("""
            SELECT genre, COUNT(*) as count
            FROM ai_analysis
            WHERE genre IS NOT NULL
            GROUP BY genre
            ORDER BY count DESC
            LIMIT 10
        """)
        
        stats['top_genres'] = [dict(row) for row in cursor.fetchall()]
        
        return stats
    
    def close(self):
        """Закрываем соединения"""
        if self.batch_count > 0:
            self.conn.commit()
        self.conn.close()
        
        if self.analyzer:
            logger.info(f"Final analyzer stats: {self.analyzer.get_stats()}")

# Функция для интеграции в существующий scraper
def enhance_scraped_song(song_data: Dict[str, Any], analyzer: GeminiLyricsAnalyzer) -> Optional[EnhancedSongData]:
    """
    Функция для обогащения новой песни AI-анализом при скрапинге
    Вызывается из основного scraper'а
    """
    try:
        enhanced_data = analyzer.analyze_song_complete(song_data)
        return enhanced_data
    except Exception as e:
        logger.error(f"Failed to enhance song {song_data.get('artist')} - {song_data.get('title')}: {e}")
        return None
