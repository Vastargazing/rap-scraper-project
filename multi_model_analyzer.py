#!/usr/bin/env python3
"""
Многоуровневый AI анализатор текстов песен:
1. Приоритет: Ollama (бесплатно, локально)
2. Fallback: DeepSeek API (дешево)
3. Резерв: Google Gemini (ограниченно)
"""

import json
import time
import logging
import requests
import os
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import sqlite3
from datetime import datetime

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Импорт моделей данных
from models import SongMetadata, LyricsAnalysis, QualityMetrics

# Создаем модель для анализа (исправленная версия)
class EnhancedSongData(BaseModel):
    """Результат AI анализа песни"""
    artist: str
    title: str
    metadata: SongMetadata
    lyrics_analysis: LyricsAnalysis
    quality_metrics: QualityMetrics
    model_used: str
    analysis_date: str

class ModelProvider:
    """Базовый класс для AI провайдеров"""
    
    def __init__(self, name: str):
        self.name = name
        self.available = False
        self.cost_per_1k_tokens = 0.0
        
    def check_availability(self) -> bool:
        """Проверка доступности модели"""
        raise NotImplementedError
        
    def analyze_song(self, artist: str, title: str, lyrics: str) -> Optional[EnhancedSongData]:
        """Анализ песни"""
        raise NotImplementedError

class OllamaProvider(ModelProvider):
    """Провайдер для локальных моделей Ollama"""
    
    def __init__(self, model_name: str = "llama3.2:3b", base_url: str = "http://localhost:11434"):
        super().__init__("Ollama")
        self.model_name = model_name
        self.base_url = base_url
        self.cost_per_1k_tokens = 0.0  # Бесплатно!
        self.available = self.check_availability()
        
    def check_availability(self) -> bool:
        """Проверка запущен ли Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5, proxies={"http": "", "https": ""})
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [model['name'] for model in models]
                logger.info(f"🦙 Ollama доступен. Модели: {available_models}")
                
                # Проверяем наличие нужной модели
                if any(self.model_name in model for model in available_models):
                    logger.info(f"✅ Модель {self.model_name} найдена")
                    return True
                else:
                    logger.warning(f"⚠️ Модель {self.model_name} не найдена. Попытка загрузки...")
                    return self._pull_model()
            return False
        except requests.exceptions.RequestException as e:
            logger.warning(f"❌ Ollama недоступен: {e}")
            return False
    
    def _pull_model(self) -> bool:
        """Загрузка модели если её нет"""
        try:
            logger.info(f"📥 Загружаем модель {self.model_name}...")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model_name},
                timeout=300,  # 5 минут на загрузку
                proxies={"http": "", "https": ""}
            )
            if response.status_code == 200:
                logger.info(f"✅ Модель {self.model_name} загружена")
                return True
            else:
                logger.error(f"❌ Не удалось загрузить модель: {response.text}")
                return False
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            return False
    
    def analyze_song(self, artist: str, title: str, lyrics: str) -> Optional[EnhancedSongData]:
        """Анализ песни через Ollama"""
        if not self.available:
            return None
            
        try:
            prompt = self._create_analysis_prompt(artist, title, lyrics)
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Низкая температура для консистентности
                        "top_p": 0.9,
                        "max_tokens": 1500
                    }
                },
                timeout=60,
                proxies={"http": "", "https": ""}
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis_text = result.get('response', '')
                return self._parse_analysis(analysis_text, artist, title)
            else:
                logger.error(f"❌ Ollama ошибка: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Ошибка анализа Ollama: {e}")
            return None
    
    def _create_analysis_prompt(self, artist: str, title: str, lyrics: str) -> str:
        """Создание промпта для анализа"""
        return f"""
Проанализируй рэп-песню и верни результат СТРОГО в JSON формате.

Исполнитель: {artist}
Название: {title}
Текст: {lyrics[:2000]}...

Верни ТОЛЬКО валидный JSON без дополнительного текста:

{{
    "metadata": {{
        "genre": "rap",
        "mood": "aggressive",
        "energy_level": "high",
        "explicit_content": true
    }},
    "lyrics_analysis": {{
        "structure": "verse-chorus-verse",
        "rhyme_scheme": "ABAB",
        "complexity_level": "advanced",
        "main_themes": ["street_life", "success", "relationships"],
        "emotional_tone": "mixed",
        "storytelling_type": "narrative",
        "wordplay_quality": "excellent"
    }},
    "quality_metrics": {{
        "authenticity_score": 0.8,
        "lyrical_creativity": 0.9,
        "commercial_appeal": 0.7,
        "uniqueness": 0.6,
        "overall_quality": "excellent",
        "ai_likelihood": 0.1
    }}
}}

ОБЯЗАТЕЛЬНЫЕ ПОЛЯ:
- emotional_tone: positive/negative/neutral/mixed
- storytelling_type: narrative/abstract/conversational
- wordplay_quality: basic/good/excellent

Верни ТОЛЬКО JSON без комментариев!
"""
    
    def _parse_analysis(self, analysis_text: str, artist: str, title: str) -> Optional[EnhancedSongData]:
        """Парсинг результата анализа"""
        try:
            # Извлекаем JSON из ответа
            json_start = analysis_text.find('{')
            json_end = analysis_text.rfind('}') + 1
            
            if json_start == -1 or json_end <= json_start:
                logger.error("❌ JSON не найден в ответе")
                return None
                
            json_str = analysis_text[json_start:json_end]
            data = json.loads(json_str)
            
            # Проверяем и дополняем отсутствующие поля
            metadata_data = data.get('metadata', {})
            lyrics_data = data.get('lyrics_analysis', {})
            quality_data = data.get('quality_metrics', {})
            
            # Дополняем отсутствующие поля в lyrics_analysis
            if 'emotional_tone' not in lyrics_data:
                lyrics_data['emotional_tone'] = 'neutral'
                logger.warning("⚠️ Добавлено значение по умолчанию для emotional_tone")
            
            if 'storytelling_type' not in lyrics_data:
                lyrics_data['storytelling_type'] = 'conversational'
                logger.warning("⚠️ Добавлено значение по умолчанию для storytelling_type")
            
            if 'wordplay_quality' not in lyrics_data:
                lyrics_data['wordplay_quality'] = 'basic'
                logger.warning("⚠️ Добавлено значение по умолчанию для wordplay_quality")
            
            # Создаем структурированный анализ
            metadata = SongMetadata(**metadata_data)
            lyrics_analysis = LyricsAnalysis(**lyrics_data)
            quality_metrics = QualityMetrics(**quality_data)
            
            return EnhancedSongData(
                artist=artist,
                title=title,
                metadata=metadata,
                lyrics_analysis=lyrics_analysis,
                quality_metrics=quality_metrics,
                model_used=f"ollama:{self.model_name}",
                analysis_date=datetime.now().isoformat()
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Ошибка парсинга JSON: {e}")
            logger.debug(f"Ответ модели: {analysis_text[:500]}")
            return None
        except Exception as e:
            logger.error(f"❌ Ошибка создания анализа: {e}")
            return None

class DeepSeekProvider(ModelProvider):
    """Провайдер для DeepSeek API"""
    
    def __init__(self):
        super().__init__("DeepSeek")
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.model_name = "deepseek-chat"
        self.cost_per_1k_tokens = 0.00056  # $0.56 per 1M input tokens (cache miss)
        self.available = self.check_availability()
        
    def check_availability(self) -> bool:
        """Проверка API ключа DeepSeek"""
        if not self.api_key:
            logger.warning("❌ DEEPSEEK_API_KEY не найден в .env")
            return False
            
        try:
            # Тестовый запрос
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 10
                },
                timeout=3
            )
            
            if response.status_code == 200:
                logger.info("✅ DeepSeek API доступен")
                return True
            else:
                logger.error(f"❌ DeepSeek API ошибка: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка проверки DeepSeek: {e}")
            return False
    
    def analyze_song(self, artist: str, title: str, lyrics: str) -> Optional[EnhancedSongData]:
        """Анализ песни через DeepSeek"""
        if not self.available:
            return None
            
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = self._create_analysis_prompt(artist, title, lyrics)
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json={
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert music analyst. Return only valid JSON format."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1500
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis_text = result['choices'][0]['message']['content']
                
                # Подсчет токенов для статистики
                usage = result.get('usage', {})
                input_tokens = usage.get('prompt_tokens', 0)
                output_tokens = usage.get('completion_tokens', 0)
                cost = (input_tokens * self.cost_per_1k_tokens / 1000) + (output_tokens * 0.00168)
                
                logger.info(f"💰 DeepSeek: {input_tokens}+{output_tokens} токенов, ${cost:.4f}")
                
                return self._parse_analysis(analysis_text, artist, title)
            else:
                logger.error(f"❌ DeepSeek ошибка: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Ошибка анализа DeepSeek: {e}")
            return None
    
    def _create_analysis_prompt(self, artist: str, title: str, lyrics: str) -> str:
        """Создание промпта для DeepSeek"""
        return f"""
Analyze this rap song and return results in STRICT JSON format:

Artist: {artist}
Title: {title}
Lyrics: {lyrics[:2000]}...

Return JSON with ALL these exact fields:
{{
    "metadata": {{
        "genre": "hip-hop/rap/trap/drill/old-school/...",
        "mood": "aggressive/melancholic/energetic/chill/...",
        "energy_level": "low/medium/high",
        "explicit_content": true/false
    }},
    "lyrics_analysis": {{
        "structure": "verse-chorus-verse/freestyle/storytelling/...",
        "rhyme_scheme": "AABA/ABAB/complex/simple/...",
        "complexity_level": "beginner/intermediate/advanced",
        "main_themes": ["money", "relationships", "street_life", "success", "..."],
        "emotional_tone": "positive/negative/neutral/mixed",
        "storytelling_type": "narrative/abstract/conversational/...",
        "wordplay_quality": "basic/good/excellent"
    }},
    "quality_metrics": {{
        "authenticity_score": 0.0-1.0,
        "lyrical_creativity": 0.0-1.0,
        "commercial_appeal": 0.0-1.0,
        "uniqueness": 0.0-1.0,
        "overall_quality": "poor/fair/good/excellent",
        "ai_likelihood": 0.0-1.0
    }}
}}

IMPORTANT: Include ALL fields including emotional_tone, storytelling_type, wordplay_quality!
Return ONLY JSON, no additional text!
"""
    
    def _parse_analysis(self, analysis_text: str, artist: str, title: str) -> Optional[EnhancedSongData]:
        """Парсинг результата анализа"""
        try:
            # Извлекаем JSON из ответа
            json_start = analysis_text.find('{')
            json_end = analysis_text.rfind('}') + 1
            
            if json_start == -1 or json_end <= json_start:
                logger.error("❌ JSON не найден в ответе DeepSeek")
                return None
                
            json_str = analysis_text[json_start:json_end]
            data = json.loads(json_str)
            
            # Проверяем и дополняем отсутствующие поля
            metadata_data = data.get('metadata', {})
            lyrics_data = data.get('lyrics_analysis', {})
            quality_data = data.get('quality_metrics', {})
            
            # Дополняем отсутствующие поля в lyrics_analysis
            if 'emotional_tone' not in lyrics_data:
                lyrics_data['emotional_tone'] = 'neutral'
                logger.warning("⚠️ Добавлено значение по умолчанию для emotional_tone")
            
            if 'storytelling_type' not in lyrics_data:
                lyrics_data['storytelling_type'] = 'conversational'
                logger.warning("⚠️ Добавлено значение по умолчанию для storytelling_type")
            
            if 'wordplay_quality' not in lyrics_data:
                lyrics_data['wordplay_quality'] = 'basic'
                logger.warning("⚠️ Добавлено значение по умолчанию для wordplay_quality")
            
            # Создаем структурированный анализ
            metadata = SongMetadata(**metadata_data)
            lyrics_analysis = LyricsAnalysis(**lyrics_data)
            quality_metrics = QualityMetrics(**quality_data)
            
            return EnhancedSongData(
                artist=artist,
                title=title,
                metadata=metadata,
                lyrics_analysis=lyrics_analysis,
                quality_metrics=quality_metrics,
                model_used=f"deepseek:{self.model_name}",
                analysis_date=datetime.now().isoformat()
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Ошибка парсинга JSON DeepSeek: {e}")
            logger.debug(f"Ответ модели: {analysis_text[:500]}")
            return None
        except Exception as e:
            logger.error(f"❌ Ошибка создания анализа DeepSeek: {e}")
            return None

class MultiModelAnalyzer:
    """Анализатор с поддержкой нескольких провайдеров"""
    
    def __init__(self):
        self.providers = []
        self.current_provider = None
        self.stats = {
            "total_analyzed": 0,
            "ollama_used": 0,
            "deepseek_used": 0,
            "total_cost": 0.0
        }
        
        # Инициализация провайдеров в порядке приоритета
        self._init_providers()
        
    def _init_providers(self):
        """Инициализация провайдеров в порядке приоритета"""
        logger.info("🔍 Инициализация AI провайдеров...")
        
        # 1. Ollama (приоритет - бесплатно)
        ollama = OllamaProvider()
        if ollama.available:
            self.providers.append(ollama)
            logger.info("✅ Ollama готов к использованию")
        
        # 2. DeepSeek (fallback - дешево)
        deepseek = DeepSeekProvider()
        if deepseek.available:
            self.providers.append(deepseek)
            logger.info("✅ DeepSeek готов к использованию")
        
        if not self.providers:
            logger.error("❌ Ни один AI провайдер недоступен!")
            raise Exception("No AI providers available")
        
        self.current_provider = self.providers[0]
        logger.info(f"🎯 Активный провайдер: {self.current_provider.name}")
    
    def analyze_song(self, artist: str, title: str, lyrics: str) -> Optional[EnhancedSongData]:
        """Анализ песни с fallback между провайдерами"""
        
        for provider in self.providers:
            try:
                logger.info(f"🤖 Анализируем через {provider.name}: {artist} - {title}")
                
                result = provider.analyze_song(artist, title, lyrics)
                
                if result:
                    # Обновляем статистику
                    self.stats["total_analyzed"] += 1
                    if provider.name == "Ollama":
                        self.stats["ollama_used"] += 1
                    elif provider.name == "DeepSeek":
                        self.stats["deepseek_used"] += 1
                        # Примерная стоимость (1500 токенов на анализ)
                        estimated_cost = 1.5 * provider.cost_per_1k_tokens
                        self.stats["total_cost"] += estimated_cost
                    
                    logger.info(f"✅ Анализ завершен через {provider.name}")
                    return result
                else:
                    logger.warning(f"⚠️ {provider.name} не смог проанализировать")
                    
            except Exception as e:
                logger.error(f"❌ Ошибка {provider.name}: {e}")
                continue
        
        logger.error(f"❌ Все провайдеры не смогли проанализировать: {artist} - {title}")
        return None
    
    def get_stats(self) -> Dict:
        """Получение статистики использования"""
        return {
            **self.stats,
            "available_providers": [p.name for p in self.providers],
            "current_provider": self.current_provider.name if self.current_provider else None
        }
    
    def batch_analyze_from_db(self, db_path: str = "rap_lyrics.db", limit: int = 100, offset: int = 0):
        """Массовый анализ песен из базы данных"""
        
        logger.info(f"🎵 Начинаем batch анализ: {limit} песен с offset {offset}")
        
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            
            # Получаем песни для анализа
            cursor = conn.execute("""
                SELECT s.id, s.artist, s.title, s.lyrics 
                FROM songs s
                LEFT JOIN ai_analysis a ON s.id = a.song_id
                WHERE a.id IS NULL  -- Только неанализированные
                LIMIT ? OFFSET ?
            """, (limit, offset))
            
            songs = cursor.fetchall()
            logger.info(f"📊 Найдено {len(songs)} песен для анализа")
            
            successful = 0
            failed = 0
            
            for i, song in enumerate(songs, 1):
                try:
                    logger.info(f"📈 Прогресс: {i}/{len(songs)} - {song['artist']} - {song['title']}")
                    
                    analysis = self.analyze_song(song['artist'], song['title'], song['lyrics'])
                    
                    if analysis:
                        # Сохраняем в БД
                        self._save_analysis_to_db(conn, song['id'], analysis)
                        successful += 1
                        logger.info(f"✅ Сохранен анализ #{successful}")
                    else:
                        failed += 1
                        logger.warning(f"❌ Не удалось проанализировать")
                    
                    # Пауза между запросами
                    if i < len(songs):  # Не делаем паузу после последней песни
                        time.sleep(2)  # 2 секунды между анализами
                        
                except Exception as e:
                    failed += 1
                    logger.error(f"❌ Ошибка анализа песни {song['id']}: {e}")
                    continue
            
            conn.close()
            
            logger.info(f"""
            🎉 Batch анализ завершен!
            ✅ Успешно: {successful}
            ❌ Ошибок: {failed}
            📊 Статистика: {self.get_stats()}
            """)
            
        except Exception as e:
            logger.error(f"❌ Ошибка batch анализа: {e}")
    
    def _save_analysis_to_db(self, conn: sqlite3.Connection, song_id: int, analysis: EnhancedSongData):
        """Сохранение анализа в базу данных"""
        try:
            conn.execute("""
                INSERT INTO ai_analysis (
                    song_id, genre, mood, energy_level, explicit_content,
                    structure, rhyme_scheme, complexity_level, main_themes,
                    authenticity_score, lyrical_creativity, commercial_appeal,
                    uniqueness, overall_quality, ai_likelihood,
                    analysis_date, model_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                song_id,
                analysis.metadata.genre,
                analysis.metadata.mood,
                analysis.metadata.energy_level,
                analysis.metadata.explicit_content,
                analysis.lyrics_analysis.structure,
                analysis.lyrics_analysis.rhyme_scheme,
                analysis.lyrics_analysis.complexity_level,
                json.dumps(analysis.lyrics_analysis.main_themes),
                analysis.quality_metrics.authenticity_score,
                analysis.quality_metrics.lyrical_creativity,
                analysis.quality_metrics.commercial_appeal,
                analysis.quality_metrics.uniqueness,
                analysis.quality_metrics.overall_quality,
                analysis.quality_metrics.ai_likelihood,
                analysis.analysis_date,
                analysis.model_used
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения в БД: {e}")
            raise

def main():
    """Тестирование многомодельного анализатора"""
    
    print("🤖 Многомодельный AI анализатор текстов песен")
    print("=" * 60)
    
    try:
        analyzer = MultiModelAnalyzer()
        
        print(f"📊 Доступные провайдеры: {[p.name for p in analyzer.providers]}")
        print(f"🎯 Активный провайдер: {analyzer.current_provider.name}")
        
        # Тест на небольшой выборке
        print("\n🧪 Тестирование на 3 песнях...")
        analyzer.batch_analyze_from_db(limit=3, offset=0)
        
        # Показываем статистику
        stats = analyzer.get_stats()
        print(f"\n📈 Статистика:")
        print(f"  • Всего проанализировано: {stats['total_analyzed']}")
        print(f"  • Ollama использован: {stats['ollama_used']} раз")
        print(f"  • DeepSeek использован: {stats['deepseek_used']} раз")
        print(f"  • Общая стоимость: ${stats['total_cost']:.4f}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    main()
