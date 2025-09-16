"""
🧠 Унифицированные интерфейсы для всех анализаторов проекта Rap Scraper (PostgreSQL Edition)

НАЗНАЧЕНИЕ:
- Единый API для всех типов анализаторов (Qwen, Emotional, Algorithmic, Ollama, Simplified, Multimodal)
- Стандартизация формата результатов анализа для PostgreSQL
- Фабрика и автоматическая регистрация анализаторов
- Интеграция с PostgreSQL через postgres_adapter.py
- Batch processing и concurrent access support

ИСПОЛЬЗОВАНИЕ:
from src.interfaces.analyzer_interface import BaseAnalyzer, AnalysisResult, AnalyzerFactory

# Создание анализатора
analyzer = AnalyzerFactory.create('qwen')
result = analyzer.analyze_song("Kendrick Lamar", "HUMBLE.", lyrics)

# Анализ всеми анализаторами
all_results = await AnalyzerFactory.analyze_with_all(artist, title, lyrics)

# Массовый анализ неанализированных треков
await AnalyzerFactory.mass_analyze('emotional', batch_size=50)

ЗАВИСИМОСТИ:
- Python 3.8+
- asyncpg, psycopg2-binary (PostgreSQL)
- src.database.postgres_adapter (PostgreSQL manager)
- src.utils.config (конфигурация)

РЕЗУЛЬТАТ:
- Единый API для всех анализаторов
- PostgreSQL интеграция с batch operations
- Автоматическое сохранение результатов
- Concurrent processing support
- Comprehensive error handling и logging

АВТОР: AI Assistant  
ДАТА: Сентябрь 2025
"""

import asyncio
import time
import logging
import json
import os
import sys
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import traceback

# PostgreSQL интеграция
try:
    from src.database.postgres_adapter import PostgreSQLManager
    from src.utils.config import get_db_config
    POSTGRES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ PostgreSQL adapter не найден: {e}")
    POSTGRES_AVAILABLE = False

# Если файл запускается напрямую (python src/interfaces/analyzer_interface.py),
# убедимся, что корень репозитория в sys.path чтобы можно было импортировать пакет `src`.
try:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
except Exception:
    pass

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalyzerType(Enum):
    """Типы анализаторов в системе"""
    EMOTIONAL = "emotional"
    QWEN = "qwen" 
    ALGORITHMIC = "algorithmic"
    OLLAMA = "ollama"
    SIMPLIFIED = "simplified"
    MULTIMODAL = "multimodal"
    GEMMA = "gemma"  # Поддержка Gemma из документации


class AnalysisStatus(Enum):
    """Статусы анализа"""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    PARTIAL = "partial"


@dataclass
class AnalysisResult:
    """
    Стандартизированный результат анализа для PostgreSQL
    
    Совместим с таблицей analysis_results:
    - track_id (FK to tracks)
    - analyzer_type (qwen, emotional, etc.)
    - analysis_data (JSONB)
    - confidence_score
    - created_at
    """
    
    # Основные поля
    artist: str
    title: str
    analyzer_type: str
    analysis_data: Dict[str, Any]
    
    # Метрики качества
    confidence: float = field(default=0.0)
    processing_time: float = field(default=0.0)
    
    # PostgreSQL интеграция
    track_id: Optional[int] = field(default=None)
    status: AnalysisStatus = field(default=AnalysisStatus.SUCCESS)
    
    # Метаданные
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Ошибки и отладка
    error_message: Optional[str] = field(default=None)
    raw_output: Optional[Dict[str, Any]] = field(default=None)
    
    def to_postgres_dict(self) -> Dict[str, Any]:
        """Преобразование в формат для PostgreSQL"""
        return {
            'track_id': self.track_id,
            'analyzer_type': self.analyzer_type,
            'analysis_data': self.analysis_data,
            'confidence_score': self.confidence,
            'metadata': {
                **self.metadata,
                'processing_time': self.processing_time,
                'status': self.status.value,
                'timestamp': self.timestamp,
                'error_message': self.error_message
            }
        }


class BaseAnalyzer(ABC):
    """
    Абстрактный базовый класс для всех анализаторов
    
    Обеспечивает:
    - Единый интерфейс для всех типов анализаторов
    - Интеграция с PostgreSQL
    - Batch processing support
    - Error handling и logging
    - Automatic result saving
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Инициализация анализатора"""
        self.config = config or {}
        self.name = self.__class__.__name__
        self.available = True
        self.model_name = None
        self.api_url = None
        
        # PostgreSQL интеграция
        self.db_manager = None
        if POSTGRES_AVAILABLE:
            try:
                self.db_manager = PostgreSQLManager()
                logger.info(f"✅ {self.name}: PostgreSQL adapter подключен")
            except Exception as e:
                logger.warning(f"⚠️ {self.name}: PostgreSQL недоступен: {e}")
                self.available = False
        
        # Статистика
        self.stats = {
            'total_analyzed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'total_time': 0.0,
            'avg_confidence': 0.0
        }

    @abstractmethod
    async def analyze_song(self, artist: str, title: str, lyrics: str, 
                          track_id: Optional[int] = None) -> AnalysisResult:
        """
        Анализ одной песни
        
        Args:
            artist: Имя артиста
            title: Название песни
            lyrics: Текст песни
            track_id: ID трека в PostgreSQL (если есть)
            
        Returns:
            AnalysisResult с результатами анализа
        """
        pass

    @abstractmethod
    def get_analyzer_info(self) -> Dict[str, Any]:
        """Метаданные об анализаторе"""
        pass

    @property
    @abstractmethod
    def analyzer_type(self) -> str:
        """Тип анализатора (должен совпадать с AnalyzerType)"""
        pass

    @property
    @abstractmethod
    def supported_features(self) -> List[str]:
        """Список поддерживаемых фич анализатора"""
        pass

    # PostgreSQL методы
    
    async def save_to_database(self, result: AnalysisResult) -> bool:
        """Сохранение результата в PostgreSQL"""
        if not self.db_manager:
            logger.warning(f"❌ {self.name}: PostgreSQL недоступен для сохранения")
            return False
        
        try:
            await self.db_manager.initialize()
            
            # Проверяем существование анализа
            existing = await self.db_manager.get_analysis(
                result.track_id, result.analyzer_type
            )
            
            if existing:
                logger.debug(f"📋 {self.name}: Анализ уже существует для track_id={result.track_id}")
                return True
            
            # Сохраняем новый анализ
            postgres_data = result.to_postgres_dict()
            success = await self.db_manager.save_analysis(postgres_data)
            
            if success:
                logger.debug(f"💾 {self.name}: Анализ сохранен для track_id={result.track_id}")
                self.stats['successful'] += 1
            else:
                logger.error(f"❌ {self.name}: Ошибка сохранения для track_id={result.track_id}")
                self.stats['failed'] += 1
            
            return success
            
        except Exception as e:
            logger.error(f"❌ {self.name}: Ошибка сохранения в PostgreSQL: {e}")
            self.stats['failed'] += 1
            return False

    async def get_unanalyzed_tracks(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Получение неанализированных треков для данного анализатора"""
        if not self.db_manager:
            logger.warning(f"❌ {self.name}: PostgreSQL недоступен")
            return []
        
        try:
            await self.db_manager.initialize()
            
            tracks = await self.db_manager.get_unanalyzed_tracks(
                analyzer_type=self.analyzer_type,
                limit=limit
            )
            
            logger.info(f"📊 {self.name}: Найдено {len(tracks)} неанализированных треков")
            return tracks
            
        except Exception as e:
            logger.error(f"❌ {self.name}: Ошибка получения треков: {e}")
            return []

    async def mass_analyze(self, batch_size: int = 50, max_tracks: Optional[int] = None) -> Dict[str, int]:
        """
        Массовый анализ неанализированных треков
        
        Args:
            batch_size: Размер батча для обработки
            max_tracks: Максимальное количество треков (None = все)
            
        Returns:
            Статистика: {'processed': N, 'successful': N, 'failed': N}
        """
        logger.info(f"🚀 {self.name}: Запуск массового анализа (batch_size={batch_size})")
        
        stats = {'processed': 0, 'successful': 0, 'failed': 0, 'skipped': 0}
        
        while True:
            # Получаем батч неанализированных треков
            tracks = await self.get_unanalyzed_tracks(limit=batch_size)
            
            if not tracks:
                logger.info(f"✅ {self.name}: Все треки проанализированы")
                break
            
            if max_tracks and stats['processed'] >= max_tracks:
                logger.info(f"✅ {self.name}: Достигнут лимит треков: {max_tracks}")
                break
            
            # Обрабатываем батч
            batch_start = time.time()
            
            for track in tracks:
                try:
                    # Анализируем трек
                    result = await self.analyze_song(
                        artist=track['artist'],
                        title=track['title'], 
                        lyrics=track['lyrics'],
                        track_id=track['id']
                    )
                    
                    # Сохраняем результат
                    if await self.save_to_database(result):
                        stats['successful'] += 1
                    else:
                        stats['failed'] += 1
                    
                except Exception as e:
                    logger.error(f"❌ {self.name}: Ошибка анализа трека {track['id']}: {e}")
                    stats['failed'] += 1
                
                stats['processed'] += 1
                
                # Проверка лимита
                if max_tracks and stats['processed'] >= max_tracks:
                    break
            
            batch_time = time.time() - batch_start
            logger.info(f"📊 {self.name}: Батч обработан за {batch_time:.1f}с "
                       f"(успешно: {stats['successful']}, ошибок: {stats['failed']})")
        
        logger.info(f"🏁 {self.name}: Массовый анализ завершен. Статистика: {stats}")
        return stats

    # Вспомогательные методы
    
    def validate_input(self, artist: str, title: str, lyrics: str) -> bool:
        """Валидация входных данных"""
        if not all([artist, title, lyrics]):
            return False

        if len(lyrics.strip()) < 10:
            return False

        return True

    def preprocess_lyrics(self, lyrics: str) -> str:
        """Предобработка текста песни"""
        lyrics = lyrics.strip()
        
        # Удаление избыточных пробелов
        import re
        lyrics = re.sub(r'\s+', ' ', lyrics)
        
        return lyrics

    def update_stats(self, result: AnalysisResult):
        """Обновление статистики анализатора"""
        self.stats['total_analyzed'] += 1
        self.stats['total_time'] += result.processing_time
        
        if result.status == AnalysisStatus.SUCCESS:
            self.stats['successful'] += 1
            # Обновляем среднюю уверенность
            total_successful = self.stats['successful']
            current_avg = self.stats['avg_confidence']
            self.stats['avg_confidence'] = ((current_avg * (total_successful - 1)) + result.confidence) / total_successful
        elif result.status == AnalysisStatus.FAILED:
            self.stats['failed'] += 1
        else:
            self.stats['skipped'] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики анализатора"""
        avg_time = self.stats['total_time'] / max(self.stats['total_analyzed'], 1)
        
        return {
            **self.stats,
            'avg_processing_time': avg_time,
            'success_rate': self.stats['successful'] / max(self.stats['total_analyzed'], 1),
            'analyzer_name': self.name,
            'analyzer_type': self.analyzer_type
        }


class AnalyzerFactory:
    """
    Фабрика для создания и управления анализаторами
    
    Автоматическая регистрация, создание экземпляров,
    массовый анализ всеми анализаторами
    """

    _analyzers: Dict[str, type] = {}
    _instances: Dict[str, BaseAnalyzer] = {}

    @classmethod
    def register(cls, name: str, analyzer_class: type) -> None:
        """Регистрация класса анализатора"""
        if not issubclass(analyzer_class, BaseAnalyzer):
            raise ValueError(f"Анализатор должен наследоваться от BaseAnalyzer")

        cls._analyzers[name] = analyzer_class
        logger.info(f"📝 Зарегистрирован анализатор: {name}")

    @classmethod
    def create(cls, name: str, config: Optional[Dict[str, Any]] = None, 
               singleton: bool = True) -> BaseAnalyzer:
        """Создание экземпляра анализатора"""
        if name not in cls._analyzers:
            available = list(cls._analyzers.keys())
            raise ValueError(f"Неизвестный анализатор: {name}. Доступные: {available}")

        # Возвращаем singleton если существует
        if singleton and name in cls._instances:
            return cls._instances[name]

        # Создаем новый экземпляр
        analyzer_class = cls._analyzers[name]
        instance = analyzer_class(config)

        # Сохраняем как singleton
        if singleton:
            cls._instances[name] = instance

        logger.info(f"🏭 Создан анализатор: {name}")
        return instance

    @classmethod
    async def analyze_with_all(cls, artist: str, title: str, lyrics: str, 
                              track_id: Optional[int] = None,
                              save_to_db: bool = True) -> Dict[str, AnalysisResult]:
        """
        Анализ одной песни всеми доступными анализаторами
        
        Args:
            artist: Имя артиста
            title: Название песни
            lyrics: Текст песни
            track_id: ID трека в PostgreSQL
            save_to_db: Сохранять ли результаты в базу
            
        Returns:
            Dict с результатами всех анализаторов
        """
        logger.info(f"🎵 Анализ всеми анализаторами: {artist} - {title}")
        
        results = {}
        start_time = time.time()
        
        for name in cls._analyzers.keys():
            try:
                analyzer = cls.create(name)
                
                if not analyzer.available:
                    logger.warning(f"⚠️ Анализатор {name} недоступен")
                    continue
                
                # Выполняем анализ
                result = await analyzer.analyze_song(artist, title, lyrics, track_id)
                results[name] = result
                
                # Сохраняем в базу
                if save_to_db and track_id:
                    await analyzer.save_to_database(result)
                
                logger.info(f"✅ {name}: confidence={result.confidence:.2f}, "
                           f"time={result.processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ Ошибка анализатора {name}: {e}")
                logger.debug(traceback.format_exc())
        
        total_time = time.time() - start_time
        logger.info(f"🏁 Анализ завершен за {total_time:.2f}с. "
                   f"Успешных: {len(results)}/{len(cls._analyzers)}")
        
        return results

    @classmethod
    async def mass_analyze_all(cls, batch_size: int = 25, max_tracks: Optional[int] = None) -> Dict[str, Dict[str, int]]:
        """
        Массовый анализ всеми анализаторами
        
        Args:
            batch_size: Размер батча для каждого анализатора
            max_tracks: Максимальное количество треков для каждого
            
        Returns:
            Статистика по каждому анализатору
        """
        logger.info(f"🚀 Массовый анализ всеми анализаторами (batch_size={batch_size})")
        
        all_stats = {}
        
        for name in cls._analyzers.keys():
            try:
                analyzer = cls.create(name)
                
                if not analyzer.available:
                    logger.warning(f"⚠️ Пропускаем недоступный анализатор: {name}")
                    continue
                
                logger.info(f"🔄 Запуск массового анализа: {name}")
                stats = await analyzer.mass_analyze(batch_size, max_tracks)
                all_stats[name] = stats
                
            except Exception as e:
                logger.error(f"❌ Ошибка массового анализа {name}: {e}")
                all_stats[name] = {'error': str(e)}
        
        # Общая статистика
        total_processed = sum(s.get('processed', 0) for s in all_stats.values())
        total_successful = sum(s.get('successful', 0) for s in all_stats.values())
        
        logger.info(f"🏁 Массовый анализ завершен. Всего обработано: {total_processed}, "
                   f"успешно: {total_successful}")
        
        return all_stats

    @classmethod
    def list_available(cls) -> List[str]:
        """Список доступных анализаторов"""
        return list(cls._analyzers.keys())

    @classmethod
    def get_analyzer_info(cls, name: str) -> Dict[str, Any]:
        """Информация об анализаторе"""
        if name not in cls._analyzers:
            raise ValueError(f"Неизвестный анализатор: {name}")

        analyzer = cls.create(name)
        return analyzer.get_analyzer_info()

    @classmethod
    def get_all_stats(cls) -> Dict[str, Dict[str, Any]]:
        """Статистика всех созданных анализаторов"""
        stats = {}
        
        for name, instance in cls._instances.items():
            stats[name] = instance.get_stats()
        
        return stats


# Декоратор для автоматической регистрации
def register_analyzer(name: str):
    """
    Декоратор для автоматической регистрации анализатора
    
    Использование:
        @register_analyzer("qwen")
        class QwenAnalyzer(BaseAnalyzer):
            ...
    """
    def decorator(analyzer_class):
        AnalyzerFactory.register(name, analyzer_class)
        return analyzer_class
    return decorator


@register_analyzer("qwen")
class QwenAnalyzerWrapper(BaseAnalyzer):
    """
    Wrapper around the existing legacy QwenAnalyzer implementation.

    This wrapper imports the legacy analyzer at runtime to avoid circular
    imports and adapts its synchronous API to the async `BaseAnalyzer`
    contract. It also ensures the returned `AnalysisResult` matches the
    new schema used by this interface.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._legacy_config = config or {}
        self.name = "QwenAnalyzerWrapper"

    @property
    def analyzer_type(self) -> str:
        return AnalyzerType.QWEN.value

    def get_analyzer_info(self) -> Dict[str, Any]:
        # Try to import legacy analyzer and reuse its info if available
        try:
            from archive.qwen_analyzer import QwenAnalyzer as LegacyQwen
            legacy = LegacyQwen(self._legacy_config)
            info = legacy.get_analyzer_info()
            info['type'] = self.analyzer_type
            return info
        except Exception:
            return {
                'name': 'QwenAnalyzerWrapper',
                'version': 'wrapper-1.0',
                'description': 'Wrapper for legacy Qwen analyzer',
                'type': self.analyzer_type,
                'available': self.available,
                'supported_features': []
            }

    @property
    def supported_features(self) -> List[str]:
        try:
            from archive.qwen_analyzer import QwenAnalyzer as LegacyQwen
            legacy = LegacyQwen(self._legacy_config)
            return legacy.supported_features
        except Exception:
            return []

    async def analyze_song(self, artist: str, title: str, lyrics: str,
                           track_id: Optional[int] = None) -> AnalysisResult:
        # Validate input using base helper
        if not self.validate_input(artist, title, lyrics):
            raise ValueError("Invalid input parameters")

        # Run the legacy, synchronous analysis in a thread to avoid blocking
        def sync_analyze():
            from archive.qwen_analyzer import QwenAnalyzer as LegacyQwen

            legacy = LegacyQwen(self._legacy_config)

            if not legacy.available:
                raise RuntimeError("Legacy Qwen analyzer not available")

            start = time.time()

            # reuse legacy helpers to build prompts and call model
            system_prompt, user_prompt = legacy._create_analysis_prompts(artist, title, legacy.preprocess_lyrics(lyrics))

            response = legacy.client.chat.completions.create(
                model=legacy.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=legacy.max_tokens,
                temperature=legacy.temperature
            )

            raw = legacy._parse_response(response.choices[0].message.content)
            confidence = legacy._calculate_confidence(raw)
            processing_time = time.time() - start

            # Adapt to the new AnalysisResult schema
            result = AnalysisResult(
                artist=artist,
                title=title,
                analyzer_type=self.analyzer_type,
                analysis_data=raw,
                confidence=confidence,
                processing_time=processing_time,
                track_id=track_id,
                status=AnalysisStatus.SUCCESS,
                metadata={
                    'model_name': legacy.model_name,
                    'provider': 'Novita AI',
                    'usage': getattr(response, 'usage', {}),
                },
                raw_output=raw,
                timestamp=datetime.now().isoformat()
            )

            return result

        # Execute synchronous legacy logic in background
        return await asyncio.to_thread(sync_analyze)


@register_analyzer("advanced_algorithmic")
class AdvancedAlgorithmicAnalyzerWrapper(BaseAnalyzer):
    """
    Wrapper for the legacy AdvancedAlgorithmicAnalyzer (sync implementation).
    Runs legacy analysis in a background thread and adapts the result to AnalysisResult.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._legacy_config = config or {}
        self.name = "AdvancedAlgorithmicAnalyzerWrapper"

    @property
    def analyzer_type(self) -> str:
        return AnalyzerType.ALGORITHMIC.value

    def get_analyzer_info(self) -> Dict[str, Any]:
        try:
            from src.analyzers.algorithmic_analyzer import AdvancedAlgorithmicAnalyzer as Legacy
            legacy = Legacy(self._legacy_config)
            info = legacy.get_analyzer_info() if hasattr(legacy, 'get_analyzer_info') else {}
            info['type'] = self.analyzer_type
            return info
        except Exception:
            return {
                'name': 'AdvancedAlgorithmicAnalyzerWrapper',
                'version': 'wrapper-1.0',
                'description': 'Wrapper for legacy algorithmic analyzer',
                'type': self.analyzer_type,
                'available': self.available,
                'supported_features': []
            }

    @property
    def supported_features(self) -> List[str]:
        try:
            from src.analyzers.algorithmic_analyzer import AdvancedAlgorithmicAnalyzer as Legacy
            legacy = Legacy(self._legacy_config)
            return getattr(legacy, 'supported_features', [])
        except Exception:
            return []

    async def analyze_song(self, artist: str, title: str, lyrics: str,
                           track_id: Optional[int] = None) -> AnalysisResult:
        if not self.validate_input(artist, title, lyrics):
            raise ValueError("Invalid input parameters")

        def sync_analyze():
            from src.analyzers.algorithmic_analyzer import AdvancedAlgorithmicAnalyzer as Legacy

            legacy = Legacy(self._legacy_config)

            if not getattr(legacy, 'available', True):
                raise RuntimeError("Legacy algorithmic analyzer not available")

            start = time.time()
            raw = legacy.analyze_song(artist, title, lyrics)
            processing_time = time.time() - start
            # Helper to coerce various legacy result types into a dict
            def to_plain_dict(obj):
                if obj is None:
                    return {}
                if isinstance(obj, dict):
                    # ensure keys are str
                    try:
                        return {str(k): v for k, v in obj.items()}
                    except Exception:
                        return obj
                # pydantic
                if hasattr(obj, 'model_dump'):
                    try:
                        return obj.model_dump()
                    except Exception:
                        pass
                if hasattr(obj, 'dict'):
                    try:
                        return obj.dict()
                    except Exception:
                        pass
                # dataclass or simple object
                if hasattr(obj, '__dict__'):
                    try:
                        return {k: v for k, v in vars(obj).items() if not k.startswith('_')}
                    except Exception:
                        pass
                # last resort: string-serialize then parse JSON if possible
                try:
                    import json as _json
                    return _json.loads(_json.dumps(obj, default=lambda o: getattr(o, '__dict__', str(o))))
                except Exception:
                    return {}

            # Extract fields from legacy result
            analyzer_type = getattr(raw, 'analyzer_type', None) or getattr(raw, 'analysis_type', None) or self.analyzer_type
            confidence = getattr(raw, 'confidence', getattr(raw, 'confidence_score', 0.0))
            # analysis payload could be under several attributes
            analysis_payload = getattr(raw, 'analysis_data', None) or getattr(raw, 'raw_output', None) or getattr(raw, 'raw', None) or raw
            analysis_data = to_plain_dict(analysis_payload)

            # Defensive cleanup: if legacy payload still includes wrapper-level keys like 'analysis_type', remove them
            if isinstance(analysis_data, dict):
                if 'analysis_type' in analysis_data:
                    # remove legacy top-level marker from payload to avoid duplication
                    analysis_data.pop('analysis_type', None)
                if 'analyzer_type' in analysis_data and analysis_data.get('analyzer_type') != analyzer_type:
                    # avoid inconsistent analyzer_type inside analysis_data
                    analysis_data.pop('analyzer_type', None)

            result = AnalysisResult(
                artist=artist,
                title=title,
                analyzer_type=analyzer_type,
                analysis_data=analysis_data,
                confidence=float(confidence or 0.0),
                processing_time=processing_time,
                track_id=track_id,
                status=AnalysisStatus.SUCCESS,
                metadata={
                    'source': 'advanced_algorithmic',
                },
                raw_output=analysis_data,
                timestamp=datetime.now().isoformat()
            )

            return result

        return await asyncio.to_thread(sync_analyze)


@register_analyzer("emotion_analyzer")
class EmotionAnalyzerWrapper(BaseAnalyzer):
    """Thin wrapper for the async EmotionAnalyzer implementation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._legacy_config = config or {}
        self.name = "EmotionAnalyzerWrapper"
        self._instance = None

    @property
    def analyzer_type(self) -> str:
        return AnalyzerType.EMOTIONAL.value

    def get_analyzer_info(self) -> Dict[str, Any]:
        try:
            from src.analyzers.emotion_analyzer import EmotionAnalyzer as Legacy
            legacy = Legacy(self._legacy_config)
            return getattr(legacy, 'get_analyzer_info', lambda: {})()
        except Exception:
            return {
                'name': 'EmotionAnalyzerWrapper',
                'version': 'wrapper-1.0',
                'description': 'Wrapper for async emotion analyzer',
                'type': self.analyzer_type,
                'available': self.available,
                'supported_features': []
            }

    @property
    def supported_features(self) -> List[str]:
        try:
            from src.analyzers.emotion_analyzer import EmotionAnalyzer as Legacy
            legacy = Legacy(self._legacy_config)
            return getattr(legacy, 'supported_features', [])
        except Exception:
            return []

    async def analyze_song(self, artist: str, title: str, lyrics: str,
                           track_id: Optional[int] = None) -> AnalysisResult:
        if not self.validate_input(artist, title, lyrics):
            raise ValueError("Invalid input parameters")

        # Lazy create instance
        if self._instance is None:
            from src.analyzers.emotion_analyzer import EmotionAnalyzer as Legacy
            self._instance = Legacy(self._legacy_config)

        legacy = self._instance

        # delegate to async implementation
        emotion_result = await legacy.analyze_song(artist, title, lyrics)

        # If the legacy analyzer already returned the canonical AnalysisResult, return it
        if isinstance(emotion_result, AnalysisResult):
            return emotion_result

        # Helper to coerce various legacy result types into a plain dict
        def to_plain_dict(obj):
            if obj is None:
                return {}
            if isinstance(obj, dict):
                try:
                    return {str(k): v for k, v in obj.items()}
                except Exception:
                    return obj
            # pydantic
            if hasattr(obj, 'model_dump'):
                try:
                    return obj.model_dump()
                except Exception:
                    pass
            if hasattr(obj, 'dict'):
                try:
                    return obj.dict()
                except Exception:
                    pass
            # dataclass or simple object
            if hasattr(obj, '__dict__'):
                try:
                    return {k: v for k, v in vars(obj).items() if not k.startswith('_')}
                except Exception:
                    pass
            # last resort: JSON serialize/deserialize
            try:
                import json as _json
                return _json.loads(_json.dumps(obj, default=lambda o: getattr(o, '__dict__', str(o))))
            except Exception:
                return {}

        # Normalize legacy return into canonical AnalysisResult
        analyzer_type = getattr(emotion_result, 'analyzer_type', None) or getattr(emotion_result, 'analysis_type', None) or self.analyzer_type
        confidence = getattr(emotion_result, 'confidence', getattr(emotion_result, 'confidence_score', 0.0))
        analysis_payload = getattr(emotion_result, 'analysis_data', None) or getattr(emotion_result, 'raw_output', None) or getattr(emotion_result, 'raw', None) or emotion_result
        analysis_data = to_plain_dict(analysis_payload)

        # Defensive cleanup: remove legacy markers from payload
        if isinstance(analysis_data, dict):
            analysis_data.pop('analysis_type', None)
            if 'analyzer_type' in analysis_data and analysis_data.get('analyzer_type') != analyzer_type:
                analysis_data.pop('analyzer_type', None)

        return AnalysisResult(
            artist=artist,
            title=title,
            analyzer_type=analyzer_type,
            analysis_data=analysis_data or {},
            confidence=float(confidence or 0.0),
            processing_time=float(getattr(emotion_result, 'analysis_time', getattr(emotion_result, 'processing_time', 0.0)) or 0.0),
            track_id=track_id,
            status=AnalysisStatus.SUCCESS,
            metadata=getattr(emotion_result, 'metadata', {}) or {},
            raw_output=analysis_data if isinstance(analysis_data, dict) else {},
            timestamp=datetime.now().isoformat()
        )


@register_analyzer("multimodal")
class MultiModelAnalyzerWrapper(BaseAnalyzer):
    """Wrapper for legacy MultiModelAnalyzer orchestrator."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._legacy_config = config or {}
        self.name = "MultiModelAnalyzerWrapper"

    @property
    def analyzer_type(self) -> str:
        return AnalyzerType.MULTIMODAL.value

    def get_analyzer_info(self) -> Dict[str, Any]:
        try:
            from src.analyzers.multi_model_analyzer import MultiModelAnalyzer as Legacy
            legacy = Legacy()
            return getattr(legacy, 'get_analyzer_info', lambda: {})()
        except Exception:
            return {
                'name': 'MultiModelAnalyzerWrapper',
                'version': 'wrapper-1.0',
                'description': 'Wrapper for multi-model analyzer',
                'type': self.analyzer_type,
                'available': self.available,
                'supported_features': []
            }

    @property
    def supported_features(self) -> List[str]:
        try:
            from src.analyzers.multi_model_analyzer import MultiModelAnalyzer as Legacy
            legacy = Legacy()
            return getattr(legacy, 'supported_features', [])
        except Exception:
            return []

    async def analyze_song(self, artist: str, title: str, lyrics: str,
                           track_id: Optional[int] = None) -> AnalysisResult:
        if not self.validate_input(artist, title, lyrics):
            raise ValueError("Invalid input parameters")

        def sync_analyze():
            from src.analyzers.multi_model_analyzer import MultiModelAnalyzer as Legacy

            legacy = Legacy()
            raw = legacy.analyze_song(artist, title, lyrics)
            processing_time = getattr(raw, 'analysis_time', 0.0) if raw else 0.0

            # Convert EnhancedSongData or dict to AnalysisResult
            analysis_data = raw if isinstance(raw, dict) else getattr(raw, 'model_dump', lambda: {})()
            # normalize quality_metrics (could be dict, pydantic model, or object)
            def norm_metrics(m):
                if m is None:
                    return {}
                if isinstance(m, dict):
                    return m
                if hasattr(m, 'model_dump'):
                    try:
                        return m.model_dump()
                    except Exception:
                        pass
                if hasattr(m, 'dict'):
                    try:
                        return m.dict()
                    except Exception:
                        pass
                if hasattr(m, '__dict__'):
                    try:
                        return {k: v for k, v in vars(m).items() if not k.startswith('_')}
                    except Exception:
                        pass
                return {}

            quality_metrics = norm_metrics(getattr(raw, 'quality_metrics', None)) if raw else {}
            confidence = float(quality_metrics.get('authenticity_score', 0.0)) if isinstance(quality_metrics, dict) else 0.0

            return AnalysisResult(
                artist=artist,
                title=title,
                analyzer_type=self.analyzer_type,
                analysis_data=analysis_data or {},
                confidence=confidence,
                processing_time=processing_time,
                track_id=track_id,
                status=AnalysisStatus.SUCCESS,
                metadata={'source': 'multi_model'},
                raw_output=analysis_data if isinstance(analysis_data, dict) else {},
                timestamp=datetime.now().isoformat()
            )

        return await asyncio.to_thread(sync_analyze)


@register_analyzer("ollama")
class OllamaAnalyzerWrapper(BaseAnalyzer):
    """Wrapper for legacy OllamaAnalyzer (sync)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._legacy_config = config or {}
        self.name = "OllamaAnalyzerWrapper"

    @property
    def analyzer_type(self) -> str:
        return AnalyzerType.OLLAMA.value

    def get_analyzer_info(self) -> Dict[str, Any]:
        try:
            from src.analyzers.ollama_analyzer import OllamaAnalyzer as Legacy
            legacy = Legacy(self._legacy_config)
            return getattr(legacy, 'get_analyzer_info', lambda: {})()
        except Exception:
            return {
                'name': 'OllamaAnalyzerWrapper',
                'version': 'wrapper-1.0',
                'description': 'Wrapper for Ollama analyzer',
                'type': self.analyzer_type,
                'available': self.available,
                'supported_features': []
            }

    @property
    def supported_features(self) -> List[str]:
        try:
            from src.analyzers.ollama_analyzer import OllamaAnalyzer as Legacy
            legacy = Legacy(self._legacy_config)
            return getattr(legacy, 'supported_features', [])
        except Exception:
            return []

    async def analyze_song(self, artist: str, title: str, lyrics: str,
                           track_id: Optional[int] = None) -> AnalysisResult:
        if not self.validate_input(artist, title, lyrics):
            raise ValueError("Invalid input parameters")

        def sync_analyze():
            from src.analyzers.ollama_analyzer import OllamaAnalyzer as Legacy
            legacy = Legacy(self._legacy_config)

            if not getattr(legacy, 'available', True):
                raise RuntimeError("Legacy Ollama analyzer not available")

            start = time.time()
            res = legacy.analyze_song(artist, title, lyrics)
            processing_time = time.time() - start

            if isinstance(res, AnalysisResult):
                return res

            # Fallback conversion
            return AnalysisResult(
                artist=artist,
                title=title,
                analyzer_type=self.analyzer_type,
                analysis_data=getattr(res, 'raw_output', {}) or {},
                confidence=getattr(res, 'confidence', 0.0),
                processing_time=processing_time,
                track_id=track_id,
                status=AnalysisStatus.SUCCESS,
                metadata={'model': getattr(res, 'model_name', legacy.model_name if hasattr(legacy, 'model_name') else None)},
                raw_output=getattr(res, 'raw_output', {}) or {},
                timestamp=datetime.now().isoformat()
            )

        return await asyncio.to_thread(sync_analyze)


@register_analyzer("simplified_features")
class SimplifiedFeatureAnalyzerWrapper(BaseAnalyzer):
    """Wrapper for simplified feature analyzer (sync or async)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._legacy_config = config or {}
        self.name = "SimplifiedFeatureAnalyzerWrapper"

    @property
    def analyzer_type(self) -> str:
        return AnalyzerType.SIMPLIFIED.value

    def get_analyzer_info(self) -> Dict[str, Any]:
        try:
            # The simplified feature analyzer exposes a LyricsAnalyzer class
            from src.analyzers.simplified_feature_analyzer import LyricsAnalyzer as Legacy
            legacy = Legacy()
            # Legacy analyzer does not implement get_analyzer_info in older versions
            info = getattr(legacy, 'get_analyzer_info', lambda: {})()
            if not info:
                info = {
                    'name': 'LyricsAnalyzer',
                    'version': getattr(legacy, 'version', 'unknown'),
                    'description': 'Simplified features lyrics analyzer',
                }
            return info
        except Exception:
            return {
                'name': 'SimplifiedFeatureAnalyzerWrapper',
                'version': 'wrapper-1.0',
                'description': 'Wrapper for simplified feature analyzer',
                'type': self.analyzer_type,
                'available': self.available,
                'supported_features': []
            }

    @property
    def supported_features(self) -> List[str]:
        try:
            from src.analyzers.simplified_feature_analyzer import LyricsAnalyzer as Legacy
            legacy = Legacy()
            return getattr(legacy, 'supported_features', [])
        except Exception:
            return []

    async def analyze_song(self, artist: str, title: str, lyrics: str,
                           track_id: Optional[int] = None) -> AnalysisResult:
        if not self.validate_input(artist, title, lyrics):
            raise ValueError("Invalid input parameters")

        # The simplified analyzer exposes LyricsAnalyzer.analyze(lyrics, track_id)
        try:
            from src.analyzers.simplified_feature_analyzer import LyricsAnalyzer as Legacy
            legacy = Legacy()

            def sync_run():
                # Legacy analyze returns a pydantic LyricsFeatures instance
                return legacy.analyze(lyrics, track_id)

            features = await asyncio.to_thread(sync_run)

            # Convert pydantic model to dict
            if hasattr(features, 'model_dump'):
                features_dict = features.model_dump()
            else:
                features_dict = getattr(features, 'dict', lambda: {})()

            confidence = features_dict.get('confidence_score', features_dict.get('confidence', 0.0))
            processing_ms = features_dict.get('processing_time_ms', features_dict.get('processing_time', 0.0))

            return AnalysisResult(
                artist=artist,
                title=title,
                analyzer_type=self.analyzer_type,
                analysis_data=features_dict or {},
                confidence=confidence,
                processing_time=(processing_ms / 1000.0) if processing_ms else 0.0,
                track_id=track_id,
                status=AnalysisStatus.SUCCESS,
                metadata={'analyzer_version': features_dict.get('analyzer_version')},
                raw_output=features_dict,
                timestamp=datetime.now().isoformat()
            )

        except ImportError:
            raise
        except Exception as e:
            logger.error(f"SimplifiedFeatureAnalyzerWrapper failed: {e}")
            raise


# Вспомогательные функции для CLI

async def test_analyzer(analyzer_name: str, test_lyrics: Optional[str] = None) -> None:
    """Тест конкретного анализатора"""
    if test_lyrics is None:
        test_lyrics = """
        I've been working on my confidence
        Every day I'm getting better at it
        Started from the bottom now I'm here
        Money trees is the perfect place for shade
        """
    
    try:
        analyzer = AnalyzerFactory.create(analyzer_name)
        
        print(f"🧪 Тестирование анализатора: {analyzer_name}")
        print(f"📊 Доступность: {analyzer.available}")
        print(f"🎯 Тип: {analyzer.analyzer_type}")
        print(f"🔧 Поддерживаемые фичи: {analyzer.supported_features}")
        
        if analyzer.available:
            result = await analyzer.analyze_song("Test Artist", "Test Song", test_lyrics)
            
            print(f"\n✅ Результат анализа:")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Processing time: {result.processing_time:.2f}s")
            print(f"  Status: {result.status.value}")
            print(f"  Data keys: {list(result.analysis_data.keys())}")
        
    except Exception as e:
        print(f"❌ Ошибка тестирования {analyzer_name}: {e}")


async def test_all_analyzers() -> None:
    """Тест всех доступных анализаторов"""
    print("🧪 Тестирование всех анализаторов...\n")
    
    available = AnalyzerFactory.list_available()
    print(f"📋 Доступные анализаторы: {available}\n")
    
    for name in available:
        await test_analyzer(name)
        print("-" * 50)


if __name__ == "__main__":
    """
    Standalone запуск для тестирования
    
    Использование:
        python src/interfaces/analyzer_interface.py
        python src/interfaces/analyzer_interface.py test qwen
        python src/interfaces/analyzer_interface.py stats
    """
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "test":
            if len(sys.argv) > 2:
                analyzer_name = sys.argv[2]
                asyncio.run(test_analyzer(analyzer_name))
            else:
                asyncio.run(test_all_analyzers())
        
        elif command == "list":
            available = AnalyzerFactory.list_available()
            print(f"📋 Доступные анализаторы: {available}")
        
        elif command == "stats":
            stats = AnalyzerFactory.get_all_stats()
            print("📊 Статистика анализаторов:")
            for name, stat in stats.items():
                print(f"  {name}: {stat}")
    
    else:
        print("🧠 Analyzer Interface для Rap Scraper проекта")
        print("Команды:")
        print("  test [analyzer_name] - тест анализатора(ов)")
        print("  list                 - список анализаторов") 
        print("  stats                - статистика анализаторов")