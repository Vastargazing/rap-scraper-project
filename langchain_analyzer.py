"""
LangChain интеграция с Google Gemini для анализа песен
"""
import os
import time
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import HumanMessage

from models import SongMetadata, LyricsAnalysis, QualityMetrics, EnhancedSongData

# Загружаем переменные окружения
load_dotenv()

logger = logging.getLogger(__name__)

class GeminiLyricsAnalyzer:
    """Анализатор текстов песен на основе Google Gemini"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Инициализация анализатора
        
        Args:
            api_key: Google API ключ (если не указан, берется из .env)
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not found! Add GOOGLE_API_KEY to .env file")
        
        # Инициализация модели Gemini
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",  # Обновленное название модели
            google_api_key=self.api_key,
            temperature=0.1,  # Низкая температура для более консистентных результатов
            convert_system_message_to_human=True
        )
        
        # Создаем парсеры для каждого типа данных
        self.metadata_parser = PydanticOutputParser(pydantic_object=SongMetadata)
        self.analysis_parser = PydanticOutputParser(pydantic_object=LyricsAnalysis)
        self.quality_parser = PydanticOutputParser(pydantic_object=QualityMetrics)
        
        # Счетчик запросов для rate limiting
        self.request_count = 0
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Контроль частоты запросов (15 запросов/минуту для бесплатного тарифа)"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # Если прошло меньше 4 секунд с последнего запроса - ждем
        if time_since_last < 4:
            wait_time = 4 - time_since_last
            logger.info(f"Rate limiting: waiting {wait_time:.1f} seconds")
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def analyze_metadata(self, artist: str, title: str, lyrics: str) -> SongMetadata:
        """Анализ базовых метаданных песни"""
        
        prompt = ChatPromptTemplate.from_template("""
Проанализируй следующую песню и определи её основные характеристики.

Исполнитель: {artist}
Название: {title}
Текст песни: {lyrics}

{format_instructions}

Проанализируй жанр, настроение, энергетику и другие характеристики на основе текста и стиля.
Будь точным и используй стандартные термины для жанров музыки.
""")
        
        formatted_prompt = prompt.format(
            artist=artist,
            title=title,
            lyrics=lyrics[:2000],  # Ограничиваем длину для экономии токенов
            format_instructions=self.metadata_parser.get_format_instructions()
        )
        
        self._rate_limit()
        response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
        
        try:
            return self.metadata_parser.parse(response.content)
        except Exception as e:
            logger.error(f"Failed to parse metadata response: {e}")
            # Возвращаем базовые значения в случае ошибки
            return SongMetadata(
                genre="unknown",
                mood="neutral",
                energy_level="medium"
            )
    
    def analyze_lyrics_structure(self, lyrics: str) -> LyricsAnalysis:
        """Детальный анализ структуры и содержания текста"""
        
        prompt = ChatPromptTemplate.from_template("""
Проведи детальный анализ структуры и содержания следующего текста песни:

{lyrics}

{format_instructions}

Проанализируй:
- Структуру песни (куплеты, припевы, бриджи)
- Схему рифмовки
- Сложность языка и образов
- Основные темы и сообщения
- Тип повествования
- Качество игры слов и метафор
""")
        
        formatted_prompt = prompt.format(
            lyrics=lyrics[:3000],  # Больший лимит для анализа структуры
            format_instructions=self.analysis_parser.get_format_instructions()
        )
        
        self._rate_limit()
        response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
        
        try:
            return self.analysis_parser.parse(response.content)
        except Exception as e:
            logger.error(f"Failed to parse analysis response: {e}")
            return LyricsAnalysis(
                structure="unknown",
                rhyme_scheme="unknown",
                complexity_level="medium",
                main_themes=["unknown"],
                emotional_tone="neutral",
                storytelling_type="unknown",
                wordplay_quality="basic"
            )
    
    def evaluate_quality(self, artist: str, title: str, lyrics: str) -> QualityMetrics:
        """Оценка качества и аутентичности трека"""
        
        prompt = ChatPromptTemplate.from_template("""
Оцени качество и аутентичность следующей песни по шкале от 0 до 1:

Исполнитель: {artist}
Название: {title}
Текст: {lyrics}

{format_instructions}

Критерии оценки:
- Аутентичность: насколько "живо" и естественно звучит текст
- Креативность: оригинальность идей и образов
- Коммерческая привлекательность: потенциал популярности
- Уникальность: отличие от других треков
- Вероятность AI-генерации: признаки искусственного создания

Будь объективным и основывайся на конкретных характеристиках текста.
""")
        
        formatted_prompt = prompt.format(
            artist=artist,
            title=title,
            lyrics=lyrics[:2500],
            format_instructions=self.quality_parser.get_format_instructions()
        )
        
        self._rate_limit()
        response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
        
        try:
            return self.quality_parser.parse(response.content)
        except Exception as e:
            logger.error(f"Failed to parse quality response: {e}")
            return QualityMetrics(
                authenticity_score=0.5,
                lyrical_creativity=0.5,
                commercial_appeal=0.5,
                uniqueness=0.5,
                overall_quality="average",
                ai_likelihood=0.1
            )
    
    def analyze_song_complete(self, song_data: Dict[str, Any]) -> EnhancedSongData:
        """Полный анализ песни с всеми компонентами"""
        
        artist = song_data.get('artist', '')
        title = song_data.get('title', '')
        lyrics = song_data.get('lyrics', '')
        
        logger.info(f"Starting analysis for: {artist} - {title}")
        
        try:
            # Выполняем все три типа анализа
            metadata = self.analyze_metadata(artist, title, lyrics)
            analysis = self.analyze_lyrics_structure(lyrics)
            quality = self.evaluate_quality(artist, title, lyrics)
            
            # Создаем обогащенные данные
            enhanced_data = EnhancedSongData(
                url=song_data.get('url', ''),
                title=title,
                artist=artist,
                lyrics=lyrics,
                genius_id=song_data.get('genius_id'),
                scraped_date=song_data.get('scraped_date', ''),
                word_count=song_data.get('word_count', len(lyrics.split())),
                ai_metadata=metadata,
                ai_analysis=analysis,
                quality_metrics=quality
            )
            
            logger.info(f"Analysis completed for: {artist} - {title}")
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Failed to analyze song {artist} - {title}: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику использования анализатора"""
        return {
            "total_requests": self.request_count,
            "estimated_cost": "FREE (Gemini)",
            "rate_limit": "15 requests/minute"
        }
