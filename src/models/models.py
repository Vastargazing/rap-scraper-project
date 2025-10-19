"""
🎼 Pydantic модели для структурированного анализа песен и метаданных

НАЗНАЧЕНИЕ:
- Описание структуры данных для песен, анализа текста, метрик качества
- Используется в анализаторах, скрапере, интеграциях с LangChain, Gemini, Spotify API

ИСПОЛЬЗОВАНИЕ:
from src.models.models import SongMetadata, LyricsAnalysis, QualityMetrics, EnhancedSongData

ЗАВИСИМОСТИ:
- Python 3.8+
- pydantic
- Используется во всех компонентах анализа

РЕЗУЛЬТАТ:
- Единая типизация и валидация данных песен и анализа
- Упрощение интеграции между компонентами

АВТОР: AI Assistant
ДАТА: Сентябрь 2025
"""

from datetime import datetime

from pydantic import BaseModel, Field


class SongMetadata(BaseModel):
    """Базовые метаданные песни"""

    genre: str = Field(
        description="Основной жанр музыки (hip-hop, rap, r&b, pop, etc.)"
    )
    subgenre: str | None = Field(default=None, description="Поджанр если применимо")
    mood: str = Field(
        description="Настроение трека (energetic, melancholic, aggressive, chill, etc.)"
    )
    year_estimate: int | None = Field(
        default=None, description="Примерный год создания по стилю"
    )
    energy_level: str = Field(description="Уровень энергии (low, medium, high)")
    explicit_content: bool = Field(
        default=False, description="Содержит ли явный контент"
    )


class LyricsAnalysis(BaseModel):
    """Детальный анализ текста песни"""

    structure: str = Field(
        description="Структура песни (verse-chorus-verse-bridge-outro, etc.)"
    )
    rhyme_scheme: str = Field(description="Схема рифмовки (ABAB, AABB, free, etc.)")
    complexity_level: str = Field(
        description="Сложность текста (simple, medium, complex)"
    )
    main_themes: list[str] = Field(description="Основные темы песни")
    emotional_tone: str = Field(
        description="Эмоциональный тон (positive, negative, neutral, mixed)"
    )
    storytelling_type: str = Field(
        description="Тип повествования (narrative, abstract, conversational, etc.)"
    )
    wordplay_quality: str = Field(
        description="Качество игры слов (basic, good, excellent)"
    )


class QualityMetrics(BaseModel):
    """Метрики качества и 'живости' трека"""

    authenticity_score: float = Field(
        ge=0.0, le=1.0, description="Оценка аутентичности (0-1)"
    )
    lyrical_creativity: float = Field(
        ge=0.0, le=1.0, description="Креативность текста (0-1)"
    )
    commercial_appeal: float = Field(
        ge=0.0, le=1.0, description="Коммерческая привлекательность (0-1)"
    )
    uniqueness: float = Field(
        ge=0.0, le=1.0, description="Уникальность относительно других треков (0-1)"
    )
    overall_quality: str = Field(
        description="Общая оценка качества (poor, average, good, excellent)"
    )
    ai_likelihood: float = Field(
        ge=0.0, le=1.0, description="Вероятность что текст создан ИИ (0-1)"
    )


class EnhancedSongData(BaseModel):
    """Полная обогащенная информация о песне"""

    # Базовые данные из scraper'а
    url: str
    title: str
    artist: str
    lyrics: str
    genius_id: int | None = None
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
    song_data: EnhancedSongData | None = None
    error_message: str | None = None
    processing_time: float


class BatchAnalysisStats(BaseModel):
    """Статистика пакетной обработки"""

    total_processed: int
    successful: int
    failed: int
    average_processing_time: float
    start_time: str
    end_time: str
    errors: list[str] = Field(default_factory=list)


# ===== SPOTIFY API MODELS =====


class SpotifyAudioFeatures(BaseModel):
    """Аудио-характеристики трека из Spotify"""

    danceability: float = Field(ge=0.0, le=1.0, description="Танцевальность (0.0-1.0)")
    energy: float = Field(ge=0.0, le=1.0, description="Энергичность (0.0-1.0)")
    valence: float = Field(ge=0.0, le=1.0, description="Позитивность (0.0-1.0)")
    tempo: float = Field(ge=0.0, description="Темп в BPM")
    acousticness: float = Field(ge=0.0, le=1.0, description="Акустичность (0.0-1.0)")
    instrumentalness: float = Field(
        ge=0.0, le=1.0, description="Инструментальность (0.0-1.0)"
    )
    speechiness: float = Field(ge=0.0, le=1.0, description="Речевость (0.0-1.0)")
    liveness: float = Field(ge=0.0, le=1.0, description="Живое исполнение (0.0-1.0)")
    loudness: float = Field(description="Громкость в dB")


class SpotifyArtist(BaseModel):
    """Информация об артисте из Spotify"""

    spotify_id: str = Field(description="Уникальный ID артиста в Spotify")
    name: str = Field(description="Имя артиста")
    genres: list[str] = Field(default_factory=list, description="Жанры артиста")
    popularity: int = Field(ge=0, le=100, description="Популярность артиста (0-100)")
    followers: int = Field(ge=0, description="Количество подписчиков")
    image_url: str | None = Field(default=None, description="URL изображения артиста")
    spotify_url: str = Field(description="Ссылка на профиль в Spotify")


class SpotifyTrack(BaseModel):
    """Информация о треке из Spotify"""

    spotify_id: str = Field(description="Уникальный ID трека в Spotify")
    name: str = Field(description="Название трека")
    artist_id: str = Field(description="ID артиста в Spotify")
    album_name: str | None = Field(default=None, description="Название альбома")
    release_date: str | None = Field(default=None, description="Дата релиза")
    duration_ms: int | None = Field(
        default=None, description="Длительность в миллисекундах"
    )
    popularity: int = Field(ge=0, le=100, description="Популярность трека (0-100)")
    explicit: bool = Field(default=False, description="Содержит ли трек явный контент")
    spotify_url: str = Field(description="Ссылка на трек в Spotify")
    preview_url: str | None = Field(default=None, description="URL превью трека")
    audio_features: SpotifyAudioFeatures | None = Field(
        default=None, description="Аудио-характеристики"
    )


class SpotifyEnrichmentResult(BaseModel):
    """Результат обогащения данных из Spotify"""

    success: bool
    artist_data: SpotifyArtist | None = None
    track_data: SpotifyTrack | None = None
    error_message: str | None = None
    processing_time: float
    api_calls_used: int = Field(
        default=0, description="Количество использованных API вызовов"
    )
