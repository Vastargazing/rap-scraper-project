"""
Pydantic модели для структурированного анализа песен через LangChain + Gemini
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class SongMetadata(BaseModel):
    """Базовые метаданные песни"""
    genre: str = Field(description="Основной жанр музыки (hip-hop, rap, r&b, pop, etc.)")
    subgenre: Optional[str] = Field(default=None, description="Поджанр если применимо")
    mood: str = Field(description="Настроение трека (energetic, melancholic, aggressive, chill, etc.)")
    year_estimate: Optional[int] = Field(default=None, description="Примерный год создания по стилю")
    energy_level: str = Field(description="Уровень энергии (low, medium, high)")
    explicit_content: bool = Field(default=False, description="Содержит ли явный контент")

class LyricsAnalysis(BaseModel):
    """Детальный анализ текста песни"""
    structure: str = Field(description="Структура песни (verse-chorus-verse-bridge-outro, etc.)")
    rhyme_scheme: str = Field(description="Схема рифмовки (ABAB, AABB, free, etc.)")
    complexity_level: str = Field(description="Сложность текста (simple, medium, complex)")
    main_themes: List[str] = Field(description="Основные темы песни")
    emotional_tone: str = Field(description="Эмоциональный тон (positive, negative, neutral, mixed)")
    storytelling_type: str = Field(description="Тип повествования (narrative, abstract, conversational, etc.)")
    wordplay_quality: str = Field(description="Качество игры слов (basic, good, excellent)")

class QualityMetrics(BaseModel):
    """Метрики качества и 'живости' трека"""
    authenticity_score: float = Field(ge=0.0, le=1.0, description="Оценка аутентичности (0-1)")
    lyrical_creativity: float = Field(ge=0.0, le=1.0, description="Креативность текста (0-1)")
    commercial_appeal: float = Field(ge=0.0, le=1.0, description="Коммерческая привлекательность (0-1)")
    uniqueness: float = Field(ge=0.0, le=1.0, description="Уникальность относительно других треков (0-1)")
    overall_quality: str = Field(description="Общая оценка качества (poor, average, good, excellent)")
    ai_likelihood: float = Field(ge=0.0, le=1.0, description="Вероятность что текст создан ИИ (0-1)")

class EnhancedSongData(BaseModel):
    """Полная обогащенная информация о песне"""
    # Базовые данные из scraper'а
    url: str
    title: str
    artist: str
    lyrics: str
    genius_id: Optional[int] = None
    scraped_date: str
    word_count: int
    
    # AI-обогащенные данные
    ai_metadata: SongMetadata
    ai_analysis: LyricsAnalysis
    quality_metrics: QualityMetrics
    
    # Метаинформация
    analysis_date: str = Field(default_factory=lambda: datetime.now().isoformat())
    model_version: str = Field(default="gemini-1.5-flash")

class AnalysisResult(BaseModel):
    """Результат анализа одной песни"""
    success: bool
    song_data: Optional[EnhancedSongData] = None
    error_message: Optional[str] = None
    processing_time: float
    
class BatchAnalysisStats(BaseModel):
    """Статистика пакетной обработки"""
    total_processed: int
    successful: int
    failed: int
    average_processing_time: float
    start_time: str
    end_time: str
    errors: List[str] = Field(default_factory=list)
