"""
Улучшенная версия анализатора с оптимизацией для лимитов API
Решения:
1. Batch обработка - несколько песен в одном запросе
2. Кэширование результатов
3. Resumed processing - продолжение с места остановки
"""
import os
import time
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import HumanMessage

from models import SongMetadata, LyricsAnalysis, QualityMetrics, EnhancedSongData

load_dotenv()
logger = logging.getLogger(__name__)

class OptimizedGeminiAnalyzer:
    """Оптимизированный анализатор с учетом лимитов API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not found!")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=self.api_key,
            temperature=0.1,
            convert_system_message_to_human=True
        )
        
        # Статистика использования
        self.requests_today = 0
        self.daily_limit = 50
        self.reset_time = self._get_next_reset_time()
        
        # Кэш для результатов
        self.cache_file = "analysis_cache.json"
        self.cache = self._load_cache()
        
    def _get_next_reset_time(self) -> datetime:
        """Время сброса дневного лимита (полночь UTC)"""
        now = datetime.utcnow()
        tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        return tomorrow
    
    def _load_cache(self) -> Dict[str, Any]:
        """Загружаем кэш результатов"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Сохраняем кэш"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
    
    def _get_cache_key(self, artist: str, title: str) -> str:
        """Генерируем ключ кэша для песни"""
        return f"{artist}|{title}".lower().strip()
    
    def can_make_request(self) -> bool:
        """Проверяем, можем ли делать запрос"""
        now = datetime.utcnow()
        
        # Сбрасываем счетчик если прошел день
        if now >= self.reset_time:
            self.requests_today = 0
            self.reset_time = self._get_next_reset_time()
            logger.info("Daily API limit reset")
        
        return self.requests_today < self.daily_limit
    
    def get_remaining_requests(self) -> int:
        """Количество оставшихся запросов сегодня"""
        return max(0, self.daily_limit - self.requests_today)
    
    def analyze_song_batch(self, songs: List[Dict[str, Any]]) -> List[EnhancedSongData]:
        """
        Анализируем несколько песен в одном запросе для экономии лимитов
        """
        if not self.can_make_request():
            raise Exception(f"Daily API limit reached. Reset at {self.reset_time}")
        
        # Проверяем кэш
        cached_results = []
        songs_to_analyze = []
        
        for song in songs:
            cache_key = self._get_cache_key(song['artist'], song['title'])
            if cache_key in self.cache:
                logger.info(f"Using cached result for {song['artist']} - {song['title']}")
                cached_data = self.cache[cache_key]
                cached_results.append(EnhancedSongData(**cached_data))
            else:
                songs_to_analyze.append(song)
        
        # Если все в кэше
        if not songs_to_analyze:
            return cached_results
        
        # Создаем батч-промпт для нескольких песен
        batch_prompt = self._create_batch_prompt(songs_to_analyze)
        
        try:
            self.requests_today += 1
            response = self.llm.invoke([HumanMessage(content=batch_prompt)])
            
            # Парсим результат и создаем EnhancedSongData
            batch_results = self._parse_batch_response(response.content, songs_to_analyze)
            
            # Сохраняем в кэш
            for result in batch_results:
                cache_key = self._get_cache_key(result.artist, result.title)
                self.cache[cache_key] = result.model_dump()
            
            self._save_cache()
            
            return cached_results + batch_results
            
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            raise
    
    def _create_batch_prompt(self, songs: List[Dict[str, Any]]) -> str:
        """Создаем промпт для анализа нескольких песен одновременно"""
        
        prompt = """Проанализируй следующие песни и для каждой верни JSON с полным анализом.

Для каждой песни нужно определить:
1. Жанр и поджанр
2. Настроение и энергетику  
3. Структуру и схему рифмовки
4. Сложность и основные темы
5. Качественные метрики (аутентичность, креативность, коммерческий потенциал, уникальность, вероятность AI)

Формат ответа - массив JSON объектов:

```json
[
  {
    "artist": "Artist Name",
    "title": "Song Title", 
    "genre": "hip-hop",
    "subgenre": "trap",
    "mood": "aggressive",
    "energy_level": "high",
    "explicit_content": true,
    "structure": "verse-chorus-verse-chorus",
    "rhyme_scheme": "AABB", 
    "complexity_level": "medium",
    "main_themes": ["money", "success"],
    "emotional_tone": "confident",
    "storytelling_type": "boastful",
    "wordplay_quality": "good",
    "authenticity_score": 0.8,
    "lyrical_creativity": 0.7,
    "commercial_appeal": 0.9,
    "uniqueness": 0.6,
    "overall_quality": "good",
    "ai_likelihood": 0.1
  }
]
```

Песни для анализа:

"""
        
        for i, song in enumerate(songs, 1):
            prompt += f"""
Песня {i}:
Исполнитель: {song['artist']}
Название: {song['title']}
Текст: {song['lyrics'][:1500]}...

"""
        
        prompt += "\nВерни только JSON массив без дополнительного текста."
        return prompt
    
    def _parse_batch_response(self, response: str, songs: List[Dict[str, Any]]) -> List[EnhancedSongData]:
        """Парсим ответ для батча песен"""
        
        try:
            # Извлекаем JSON из ответа
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON array found in response")
            
            json_str = response[json_start:json_end]
            batch_data = json.loads(json_str)
            
            results = []
            
            for i, (song_data, song_original) in enumerate(zip(batch_data, songs)):
                try:
                    # Создаем компоненты модели
                    metadata = SongMetadata(
                        genre=song_data.get('genre', 'unknown'),
                        subgenre=song_data.get('subgenre'),
                        mood=song_data.get('mood', 'neutral'),
                        energy_level=song_data.get('energy_level', 'medium'),
                        explicit_content=song_data.get('explicit_content', False)
                    )
                    
                    analysis = LyricsAnalysis(
                        structure=song_data.get('structure', 'unknown'),
                        rhyme_scheme=song_data.get('rhyme_scheme', 'unknown'),
                        complexity_level=song_data.get('complexity_level', 'medium'),
                        main_themes=song_data.get('main_themes', ['unknown']),
                        emotional_tone=song_data.get('emotional_tone', 'neutral'),
                        storytelling_type=song_data.get('storytelling_type', 'unknown'),
                        wordplay_quality=song_data.get('wordplay_quality', 'basic')
                    )
                    
                    quality = QualityMetrics(
                        authenticity_score=song_data.get('authenticity_score', 0.5),
                        lyrical_creativity=song_data.get('lyrical_creativity', 0.5),
                        commercial_appeal=song_data.get('commercial_appeal', 0.5),
                        uniqueness=song_data.get('uniqueness', 0.5),
                        overall_quality=song_data.get('overall_quality', 'average'),
                        ai_likelihood=song_data.get('ai_likelihood', 0.5)
                    )
                    
                    enhanced_data = EnhancedSongData(
                        url=song_original.get('url', ''),
                        title=song_original['title'],
                        artist=song_original['artist'],
                        lyrics=song_original['lyrics'],
                        genius_id=song_original.get('genius_id'),
                        scraped_date=song_original.get('scraped_date', ''),
                        word_count=song_original.get('word_count', 0),
                        ai_metadata=metadata,
                        ai_analysis=analysis,
                        quality_metrics=quality,
                        model_version="gemini-1.5-flash-batch"
                    )
                    
                    results.append(enhanced_data)
                    
                except Exception as e:
                    logger.error(f"Failed to parse song {i}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to parse batch response: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика использования"""
        return {
            "requests_today": self.requests_today,
            "daily_limit": self.daily_limit,
            "remaining_requests": self.get_remaining_requests(),
            "reset_time": self.reset_time.isoformat(),
            "cached_songs": len(self.cache),
            "estimated_cost": "FREE (Gemini)"
        }
