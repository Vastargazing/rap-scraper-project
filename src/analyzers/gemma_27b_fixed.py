#!/usr/bin/env python3
"""
Исправленный анализатор песен через Google Gemma 3 27B
Использует Generative AI API вместо Cloud AI Platform
"""

import json
import time
import logging
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import sqlite3
import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Загружаем переменные из .env файла
load_dotenv()

# Импортируем конфигурацию
from ..utils.config import DB_PATH

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/gemma_27b_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class RateLimitTracker:
    """Трекер для отслеживания лимитов Gemma 3 27B API (БЕСПЛАТНЫХ!)"""
    requests_per_minute: int = 30      # 30 запросов/минуту
    requests_per_day: int = 14400      # 14,400 запросов/день
    tokens_per_minute: int = 15000     # 15,000 токенов/минуту
    
    def __post_init__(self):
        self.requests_today = 0
        self.requests_this_minute = 0
        self.last_request_time = datetime.now()
        self.minute_start = datetime.now()
        self.day_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

@dataclass
class SimpleSongMetadata:
    """Простые метаданные песни"""
    song_id: int
    artist: str
    title: str
    lyrics: str
    album: str = "Unknown"
    year: int = 0

@dataclass
class SimpleLyricsAnalysis:
    """Простой анализ текста песни"""
    authenticity_score: float = 0.5
    ai_likelihood: float = 0.5
    emotional_tone: str = "neutral"
    storytelling_type: str = "abstract"
    wordplay_quality: str = "intermediate"
    flow_rating: int = 5
    lyrical_complexity: int = 5
    technical_skill: int = 5
    creativity_score: int = 5
    commercial_appeal: int = 5
    cultural_impact: int = 5
    overall_quality: int = 5
    genre_classification: str = "hip-hop"
    era_indicator: str = "modern"
    explanation: str = "Автоматический анализ"

class Gemma27BAnalyzer:
    """
    Анализатор через Google Gemma 3 27B API
    Лимиты: 15 requests/minute, 1500 requests/day
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rate_limiter = RateLimitTracker()
        
        # Endpoint для Gemma 3 27B Instruct
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemma-3-27b-it:generateContent?key={api_key}"
        
        logger.info(f"✅ Gemma 3 27B Analyzer initialized")
    
    def _check_rate_limits(self) -> bool:
        """Проверяем, можно ли делать запрос с учетом лимитов"""
        now = datetime.now()
        
        # Сброс счетчика минуты
        if (now - self.rate_limiter.minute_start).total_seconds() >= 60:
            self.rate_limiter.requests_this_minute = 0
            self.rate_limiter.minute_start = now
        
        # Сброс счетчика дня
        if now.date() > self.rate_limiter.day_start.date():
            self.rate_limiter.requests_today = 0
            self.rate_limiter.day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Проверка лимитов
        if self.rate_limiter.requests_this_minute >= self.rate_limiter.requests_per_minute:
            wait_time = 60 - (now - self.rate_limiter.minute_start).total_seconds()
            logger.warning(f"⏳ Достигнут лимит запросов в минуту. Ждем {wait_time:.1f} сек...")
            time.sleep(wait_time + 1)
            return self._check_rate_limits()
        
        if self.rate_limiter.requests_today >= self.rate_limiter.requests_per_day:
            logger.error("❌ Достигнут дневной лимит запросов!")
            return False
        
        return True
    
    def _make_request(self, prompt: str) -> Optional[str]:
        """Отправляем запрос к Gemma 3 27B API"""
        if not self._check_rate_limits():
            return None
        
        headers = {
            'Content-Type': 'application/json',
        }
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "topK": 1,
                "topP": 1,
                "maxOutputTokens": 2048,
            }
        }
        
        try:
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=120  # Больше таймаут для 27B модели
            )
            
            # Обновляем счетчики
            self.rate_limiter.requests_this_minute += 1
            self.rate_limiter.requests_today += 1
            self.rate_limiter.last_request_time = datetime.now()
            
            response.raise_for_status()
            
            result = response.json()
            
            if 'candidates' in result and result['candidates']:
                content = result['candidates'][0]['content']['parts'][0]['text']
                return content.strip()
            else:
                logger.error(f"❌ Неожиданный формат ответа: {result}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Gemma 27B API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in API request: {e}")
            return None
    
    def analyze_song(self, song_metadata: SimpleSongMetadata) -> Optional[SimpleLyricsAnalysis]:
        """Анализируем одну песню"""
        
        prompt = f"""
Ты - эксперт по рэп-музыке с глубокими знаниями культуры хип-хопа. Проанализируй следующий рэп-трек и дай детальную оценку по всем критериям.

Исполнитель: {song_metadata.artist}
Название: {song_metadata.title}
Текст песни:
{song_metadata.lyrics}

Проанализируй трек по следующим критериям и верни анализ в точном JSON формате:

{{
    "authenticity_score": <float от 0.0 до 1.0 - насколько трек звучит аутентично>,
    "ai_likelihood": <float от 0.0 до 1.0 - вероятность что текст создан ИИ>,
    "emotional_tone": "<один из: 'angry', 'melancholic', 'confident', 'introspective', 'aggressive', 'playful', 'serious', 'nostalgic'>",
    "storytelling_type": "<один из: 'narrative', 'abstract', 'conceptual', 'autobiographical', 'fictional', 'stream_of_consciousness'>",
    "wordplay_quality": "<один из: 'basic', 'intermediate', 'advanced', 'masterful'>",
    "flow_rating": <int от 1 до 10 - качество флоу и ритма>,
    "lyrical_complexity": <int от 1 до 10 - сложность и глубина текста>,
    "technical_skill": <int от 1 до 10 - техническое мастерство>,
    "creativity_score": <int от 1 до 10 - креативность и оригинальность>,
    "commercial_appeal": <int от 1 до 10 - коммерческая привлекательность>,
    "cultural_impact": <int от 1 до 10 - культурное влияние>,
    "overall_quality": <int от 1 до 10 - общая оценка качества>,
    "genre_classification": "<основной жанр>",
    "era_indicator": "<временной период: 'old-school', 'golden-age', '90s', '2000s', 'modern', 'contemporary'>",
    "explanation": "<краткое объяснение оценки на русском языке>"
}}

Отвечай ТОЛЬКО в формате JSON, без дополнительного текста до или после JSON.
"""
        
        logger.debug(f"Prompt for {song_metadata.artist} - {song_metadata.title}: {prompt[:200]}...")
        
        response = self._make_request(prompt)
        if not response:
            return None
        
        try:
            # Попытка парсинга JSON
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0].strip()
            elif '```' in response:
                response = response.split('```')[1].strip()
            
            analysis_data = json.loads(response)
            
            # Создаем объект с защитой от отсутствующих полей
            return SimpleLyricsAnalysis(
                authenticity_score=float(analysis_data.get('authenticity_score', 0.5)),
                ai_likelihood=float(analysis_data.get('ai_likelihood', 0.5)),
                emotional_tone=analysis_data.get('emotional_tone', 'neutral'),
                storytelling_type=analysis_data.get('storytelling_type', 'abstract'),
                wordplay_quality=analysis_data.get('wordplay_quality', 'intermediate'),
                flow_rating=int(analysis_data.get('flow_rating', 5)),
                lyrical_complexity=int(analysis_data.get('lyrical_complexity', 5)),
                technical_skill=int(analysis_data.get('technical_skill', 5)),
                creativity_score=int(analysis_data.get('creativity_score', 5)),
                commercial_appeal=int(analysis_data.get('commercial_appeal', 5)),
                cultural_impact=int(analysis_data.get('cultural_impact', 5)),
                overall_quality=int(analysis_data.get('overall_quality', 5)),
                genre_classification=analysis_data.get('genre_classification', 'hip-hop'),
                era_indicator=analysis_data.get('era_indicator', 'modern'),
                explanation=analysis_data.get('explanation', 'Автоматический анализ завершен')
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing failed for {song_metadata.artist} - {song_metadata.title}: {e}")
            logger.debug(f"Raw response: {response}")
            return None
        except Exception as e:
            logger.error(f"❌ Error creating LyricsAnalysis: {e}")
            return None

def get_songs_from_db(db_path: str = "rap_lyrics.db", limit: int = 5, offset: int = 0) -> List[SimpleSongMetadata]:
    """Получаем песни из базы данных"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Получаем песни, которые еще не анализировались через Gemma 27B
        cursor.execute("""
            SELECT s.id, s.artist, s.title, s.lyrics, s.album, s.release_date 
            FROM songs s
            LEFT JOIN ai_analysis a ON s.id = a.song_id AND a.model_version = 'gemma-3-27b-it'
            WHERE s.lyrics IS NOT NULL AND LENGTH(s.lyrics) > 100 AND a.id IS NULL
            ORDER BY s.id
            LIMIT ? OFFSET ?
        """, (limit, offset))
        
        songs = []
        for row in cursor.fetchall():
            # Пробуем извлечь год из release_date
            year = 0
            if row[5]:  # release_date
                try:
                    year = int(row[5].split('-')[0]) if '-' in str(row[5]) else int(row[5])
                except:
                    year = 0
            
            songs.append(SimpleSongMetadata(
                song_id=row[0],
                artist=row[1],
                title=row[2],
                lyrics=row[3],
                album=row[4] or "Unknown",
                year=year
            ))
        
        conn.close()
        return songs
        
    except Exception as e:
        logger.error(f"❌ Database error: {e}")
        return []

def save_analysis_to_db(song_id: int, analysis: SimpleLyricsAnalysis, db_path: str = "rap_lyrics.db"):
    """Сохраняем анализ в базу данных"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Проверяем есть ли уже анализ от gemma-3-27b-it
        cursor.execute("""
            SELECT id FROM ai_analysis 
            WHERE song_id = ? AND model_version = 'gemma-3-27b-it'
        """, (song_id,))
        
        if cursor.fetchone():
            logger.info(f"⚠️ Analysis already exists for song {song_id}")
            conn.close()
            return True
        
        # Если есть анализ от другой модели, обновляем его
        cursor.execute("""
            SELECT id FROM ai_analysis WHERE song_id = ?
        """, (song_id,))
        
        existing = cursor.fetchone()
        
        if existing:
            # Обновляем существующий анализ
            cursor.execute("""
                UPDATE ai_analysis SET
                    model_version = ?, analysis_date = ?,
                    authenticity_score = ?, ai_likelihood = ?, emotional_tone = ?, 
                    storytelling_type = ?, wordplay_quality = ?,
                    genre = ?, mood = ?, energy_level = ?, complexity_level = ?,
                    lyrical_creativity = ?, commercial_appeal = ?, uniqueness = ?, overall_quality = ?
                WHERE song_id = ?
            """, (
                "gemma-3-27b-it", datetime.now().isoformat(),
                analysis.authenticity_score, analysis.ai_likelihood, analysis.emotional_tone,
                analysis.storytelling_type, analysis.wordplay_quality,
                analysis.genre_classification, analysis.emotional_tone, 'medium', 'intermediate',
                analysis.creativity_score / 10.0, analysis.commercial_appeal / 10.0, 
                analysis.cultural_impact / 10.0, str(analysis.overall_quality),
                song_id
            ))
        else:
            # Создаем новый анализ
            cursor.execute("""
                INSERT INTO ai_analysis (
                    song_id, model_version, analysis_date,
                    authenticity_score, ai_likelihood, emotional_tone, storytelling_type, wordplay_quality,
                    genre, mood, energy_level, complexity_level,
                    lyrical_creativity, commercial_appeal, uniqueness, overall_quality
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                song_id, "gemma-3-27b-it", datetime.now().isoformat(),
                analysis.authenticity_score, analysis.ai_likelihood, analysis.emotional_tone,
                analysis.storytelling_type, analysis.wordplay_quality,
                analysis.genre_classification, analysis.emotional_tone, 'medium', 'intermediate',
                analysis.creativity_score / 10.0, analysis.commercial_appeal / 10.0, 
                analysis.cultural_impact / 10.0, str(analysis.overall_quality)
            ))
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error saving analysis: {e}")
        return False

def analyze_songs_from_db(api_key: str, db_path: str = "rap_lyrics.db", 
                         limit: int = None, offset: int = 0, resume: bool = True):
    """
    Основная функция анализа песен из базы данных
    
    Args:
        api_key: Google API ключ
        db_path: Путь к базе данных
        limit: Лимит песен (None = все песни)
        offset: Смещение (автоматически рассчитывается при resume=True)
        resume: Продолжить с места остановки
    """
    
    analyzer = Gemma27BAnalyzer(api_key)
    
    # Подсчитываем общее количество песен
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) FROM songs s
            LEFT JOIN ai_analysis a ON s.id = a.song_id AND a.model_version = 'gemma-3-27b-it'
            WHERE s.lyrics IS NOT NULL AND LENGTH(s.lyrics) > 100 AND a.id IS NULL
        """)
        
        total_remaining = cursor.fetchone()[0]
        
        # Если resume=True, начинаем с правильного offset
        if resume:
            cursor.execute("""
                SELECT COUNT(*) FROM ai_analysis 
                WHERE model_version = 'gemma-3-27b-it'
            """)
            already_analyzed = cursor.fetchone()[0]
            
            logger.info(f"📊 Статус базы данных:")
            logger.info(f"   ✅ Уже проанализировано: {already_analyzed}")
            logger.info(f"   🎵 Осталось проанализировать: {total_remaining}")
            
            if total_remaining == 0:
                logger.info("🎉 Все песни уже проанализированы!")
                conn.close()
                return
        
        conn.close()
        
    except Exception as e:
        logger.error(f"❌ Database error: {e}")
        return
    
    # Устанавливаем лимит для полного анализа
    if limit is None:
        limit = total_remaining
        logger.info(f"🚀 ПОЛНЫЙ АНАЛИЗ: {total_remaining} песен")
    else:
        logger.info(f"🔄 ЧАСТИЧНЫЙ АНАЛИЗ: {min(limit, total_remaining)} песен")
    
    # Получаем песни из базы
    songs = get_songs_from_db(db_path, limit, offset)
    if not songs:
        logger.info("❌ Нет песен для анализа")
        return
    
    logger.info(f"🎵 Начинаем анализ {len(songs)} песен через Gemma 3 27B")
    logger.info(f"📈 Прогресс: {analyzer.rate_limiter.requests_today}/{analyzer.rate_limiter.requests_per_day} запросов сегодня")
    
    success_count = 0
    start_time = time.time()
    
    for i, song in enumerate(songs, 1):
        try:
            # Показываем детальный прогресс
            elapsed = time.time() - start_time
            if i > 1:
                avg_time = elapsed / (i - 1)
                eta_seconds = avg_time * (len(songs) - i + 1)
                eta = timedelta(seconds=int(eta_seconds))
                
                logger.info(f"📈 Analyzing {i}/{len(songs)}: {song.artist} - {song.title}")
                logger.info(f"   ⏱️  Среднее время: {avg_time:.1f}с | ETA: {eta}")
                logger.info(f"   📊 API лимиты: {analyzer.rate_limiter.requests_today}/{analyzer.rate_limiter.requests_per_day} день, {analyzer.rate_limiter.requests_this_minute}/{analyzer.rate_limiter.requests_per_minute} минута")
            else:
                logger.info(f"📈 Analyzing {i}/{len(songs)}: {song.artist} - {song.title}")
            
            analysis = analyzer.analyze_song(song)
            if analysis:
                if save_analysis_to_db(song.song_id, analysis, db_path):
                    success_count += 1
                    logger.info(f"✅ Saved analysis for: {song.artist} - {song.title}")
                else:
                    logger.error(f"❌ Failed to save analysis for: {song.artist} - {song.title}")
            else:
                logger.error(f"❌ Failed to analyze: {song.artist} - {song.title}")
            
            # Динамическая пауза (меньше паузы при больших лимитах)
            if i < len(songs):
                time.sleep(1)  # Уменьшили паузу с 2 до 1 секунды
                
            # Промежуточная статистика каждые 100 песен
            if i % 100 == 0:
                elapsed_total = time.time() - start_time
                rate = success_count / elapsed_total * 3600  # песен в час
                
                logger.info(f"""
                📊 ПРОМЕЖУТОЧНАЯ СТАТИСТИКА (песня {i}):
                   ✅ Успешно: {success_count}/{i} ({success_count/i*100:.1f}%)
                   ⏱️  Скорость: {rate:.1f} песен/час
                   🕐 Время работы: {timedelta(seconds=int(elapsed_total))}
                   📈 API использование: {analyzer.rate_limiter.requests_today}/{analyzer.rate_limiter.requests_per_day}
                """)
                
        except KeyboardInterrupt:
            logger.info(f"\n⏸️  Анализ прерван пользователем на песне {i}")
            logger.info(f"✅ Успешно проанализировано: {success_count}")
            logger.info(f"🔄 Для продолжения запустите скрипт снова (resume=True)")
            break
            
        except Exception as e:
            logger.error(f"❌ Unexpected error analyzing {song.artist} - {song.title}: {e}")
            continue
    
    # Финальная статистика
    total_time = time.time() - start_time
    rate = success_count / total_time * 3600 if total_time > 0 else 0
    
    stats = {
        'successful': success_count,
        'total': len(songs),
        'success_rate': success_count / len(songs) * 100 if len(songs) > 0 else 0,
        'total_time': timedelta(seconds=int(total_time)),
        'rate_per_hour': rate,
        'remaining_songs': total_remaining - success_count,
        'rate_limit_stats': {
            'requests_today': analyzer.rate_limiter.requests_today,
            'requests_this_minute': analyzer.rate_limiter.requests_this_minute,
            'daily_limit': analyzer.rate_limiter.requests_per_day,
            'minute_limit': analyzer.rate_limiter.requests_per_minute
        }
    }
    
    logger.info(f"""
        🎉 АНАЛИЗ ЗАВЕРШЕН!
        ✅ Успешно: {success_count}/{len(songs)} ({stats['success_rate']:.1f}%)
        ⏱️  Общее время: {stats['total_time']}
        📈 Скорость: {rate:.1f} песен/час
        🎵 Осталось песен: {stats['remaining_songs']}
        📊 Статистика API: {stats['rate_limit_stats']}
        
        💡 Совет: При такой скорости оставшиеся {stats['remaining_songs']} песен займут ~{stats['remaining_songs']/rate:.1f} часов
    """)

def main():
    """Главная функция для запуска анализа"""
    # Настройки
    API_KEY = os.getenv("GOOGLE_API_KEY")
    
    if not API_KEY:
        print("❌ Установите переменную окружения GOOGLE_API_KEY")
        exit(1)
    
    # Запуск ПОЛНОГО анализа всех песен
    print("🚀 Запуск полного анализа базы данных через Gemma 3 27B")
    print("💡 Для прерывания нажмите Ctrl+C (прогресс сохранится)")
    print("🔄 При перезапуске анализ продолжится с места остановки")
    
    analyze_songs_from_db(
        api_key=API_KEY,
        db_path=DB_PATH,  # Используем путь из конфигурации
        limit=None,       # None = анализировать ВСЕ песни
        offset=0,         # Автоматически рассчитается при resume=True
        resume=True       # Продолжить с места остановки
    )

if __name__ == "__main__":
    main()
