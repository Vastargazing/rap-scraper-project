"""
Локальный анализатор с использованием Ollama (бесплатно, без лимитов)
"""
import requests
import json
import time
import logging
from typing import Dict, Any, List, Optional
from models import SongMetadata, LyricsAnalysis, QualityMetrics, EnhancedSongData

logger = logging.getLogger(__name__)

class OllamaAnalyzer:
    """
    Анализатор с использованием локальной модели Ollama
    Преимущества:
    - Полностью бесплатно
    - Нет лимитов API
    - Работает офлайн
    - Высокая скорость (при хорошем железе)
    
    Недостатки:
    - Требует установку Ollama
    - Нужна мощная видеокарта (рекомендуется 8GB+ VRAM)
    - Может быть менее точным чем Gemini
    """
    
    def __init__(self, model_name: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.session = requests.Session()
        
        # Проверяем доступность Ollama
        if not self._check_ollama_available():
            raise ConnectionError("Ollama не запущен! Установите и запустите Ollama")
        
        logger.info(f"✅ Ollama analyzer initialized with model: {model_name}")
    
    def _check_ollama_available(self) -> bool:
        """Проверяем доступность Ollama сервера"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _call_ollama(self, prompt: str) -> str:
        """Выполняем запрос к Ollama"""
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 1000
            }
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120  # 2 минуты на ответ
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            raise
    
    def analyze_song_batch(self, songs: List[Dict[str, Any]]) -> List[EnhancedSongData]:
        """Анализируем батч песен локально"""
        
        # Создаем промпт для анализа нескольких песен
        prompt = self._create_batch_prompt(songs)
        
        start_time = time.time()
        response = self._call_ollama(prompt)
        processing_time = time.time() - start_time
        
        logger.info(f"Ollama processed {len(songs)} songs in {processing_time:.1f}s")
        
        # Парсим результат
        return self._parse_batch_response(response, songs)
    
    def _create_batch_prompt(self, songs: List[Dict[str, Any]]) -> str:
        """Создаем промпт для локальной модели"""
        
        prompt = """Ты эксперт по анализу рэп-музыки. Проанализируй следующие песни и для каждой верни JSON с анализом.

Для каждой песни определи:
1. Жанр (hip-hop, trap, drill, etc.)
2. Настроение (aggressive, calm, melancholic, energetic, etc.)
3. Уровень энергии (low, medium, high)
4. Сложность текста (simple, medium, complex)
5. Основные темы (массив строк)
6. Аутентичность (0.0-1.0) - насколько "живо" звучит
7. Креативность (0.0-1.0)
8. Коммерческий потенциал (0.0-1.0)
9. Вероятность AI-генерации (0.0-1.0)

Отвечай ТОЛЬКО JSON массивом без дополнительного текста:

[
  {
    "artist": "Artist Name",
    "title": "Song Title",
    "genre": "hip-hop",
    "mood": "aggressive", 
    "energy_level": "high",
    "complexity_level": "medium",
    "main_themes": ["money", "success"],
    "authenticity_score": 0.8,
    "lyrical_creativity": 0.7,
    "commercial_appeal": 0.9,
    "ai_likelihood": 0.1
  }
]

Песни для анализа:

"""
        
        for i, song in enumerate(songs, 1):
            prompt += f"""
Песня {i}:
Исполнитель: {song['artist']}
Название: {song['title']}
Текст: {song['lyrics'][:1200]}...

"""
        
        prompt += "\nВерни только JSON массив:"
        return prompt
    
    def _parse_batch_response(self, response: str, songs: List[Dict[str, Any]]) -> List[EnhancedSongData]:
        """Парсим ответ локальной модели"""
        
        try:
            # Извлекаем JSON из ответа
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                logger.error(f"No JSON found in response: {response[:200]}...")
                return []
            
            json_str = response[json_start:json_end]
            batch_data = json.loads(json_str)
            
            results = []
            
            for i, (analysis_data, original_song) in enumerate(zip(batch_data, songs)):
                try:
                    # Создаем структурированные данные
                    metadata = SongMetadata(
                        genre=analysis_data.get('genre', 'hip-hop'),
                        mood=analysis_data.get('mood', 'neutral'),
                        energy_level=analysis_data.get('energy_level', 'medium'),
                        explicit_content=self._detect_explicit(original_song['lyrics'])
                    )
                    
                    analysis = LyricsAnalysis(
                        structure="unknown",  # Ollama может не всегда это определить
                        rhyme_scheme="unknown",
                        complexity_level=analysis_data.get('complexity_level', 'medium'),
                        main_themes=analysis_data.get('main_themes', ['unknown']),
                        emotional_tone=analysis_data.get('mood', 'neutral'),
                        storytelling_type="unknown",
                        wordplay_quality="basic"
                    )
                    
                    quality = QualityMetrics(
                        authenticity_score=float(analysis_data.get('authenticity_score', 0.5)),
                        lyrical_creativity=float(analysis_data.get('lyrical_creativity', 0.5)),
                        commercial_appeal=float(analysis_data.get('commercial_appeal', 0.5)),
                        uniqueness=0.5,  # Ollama может не определять
                        overall_quality=self._calculate_overall_quality(analysis_data),
                        ai_likelihood=float(analysis_data.get('ai_likelihood', 0.5))
                    )
                    
                    enhanced_data = EnhancedSongData(
                        url=original_song.get('url', ''),
                        title=original_song['title'],
                        artist=original_song['artist'],
                        lyrics=original_song['lyrics'],
                        genius_id=original_song.get('genius_id'),
                        scraped_date=original_song.get('scraped_date', ''),
                        word_count=original_song.get('word_count', 0),
                        ai_metadata=metadata,
                        ai_analysis=analysis,
                        quality_metrics=quality,
                        model_version="llama3.1-8b-local"
                    )
                    
                    results.append(enhanced_data)
                    
                except Exception as e:
                    logger.error(f"Failed to parse song {i}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to parse Ollama response: {e}")
            return []
    
    def _detect_explicit(self, lyrics: str) -> bool:
        """Простая детекция explicit content"""
        explicit_words = ['fuck', 'shit', 'bitch', 'nigga', 'ass', 'damn']
        lyrics_lower = lyrics.lower()
        return any(word in lyrics_lower for word in explicit_words)
    
    def _calculate_overall_quality(self, data: Dict) -> str:
        """Рассчитываем общее качество"""
        auth = float(data.get('authenticity_score', 0.5))
        creativity = float(data.get('lyrical_creativity', 0.5))
        
        avg_score = (auth + creativity) / 2
        
        if avg_score >= 0.8:
            return "excellent"
        elif avg_score >= 0.6:
            return "good"
        elif avg_score >= 0.4:
            return "average"
        else:
            return "poor"
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика использования"""
        return {
            "model": self.model_name,
            "cost": "FREE (Local)",
            "rate_limit": "Unlimited",
            "requires_gpu": True,
            "offline_capable": True
        }


# Инструкции по установке Ollama
OLLAMA_SETUP_INSTRUCTIONS = """
🚀 УСТАНОВКА OLLAMA ДЛЯ ЛОКАЛЬНОГО АНАЛИЗА:

1. Скачайте Ollama:
   https://ollama.ai/download

2. Установите и запустите Ollama

3. Скачайте модель Llama 3.1:
   ollama pull llama3.1:8b

4. Проверьте, что модель работает:
   ollama run llama3.1:8b "Hello"

5. Ollama готов! Теперь можно использовать OllamaAnalyzer

⚡ СИСТЕМНЫЕ ТРЕБОВАНИЯ:
- 8GB+ VRAM (для llama3.1:8b)
- Или 16GB+ RAM (CPU режим, медленнее)
- Windows/Mac/Linux поддерживается

💡 АЛЬТЕРНАТИВНЫЕ МОДЕЛИ:
- llama3.1:7b (меньше VRAM)
- phi3:mini (очень быстрая, 2GB)
- codellama:7b (хорошо понимает структуру)
"""
