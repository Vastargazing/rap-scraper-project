"""
🧬 Гибридный анализатор текстов песен (AI + алгоритмы)

НАЗНАЧЕНИЕ:
- Комбинированный анализ с использованием AI и алгоритмических методов
- Объединение результатов нескольких моделей для максимальной точности
- Кросс-валидация и интеллектуальное объединение метрик

ИСПОЛЬЗОВАНИЕ:
Используется через main.py, batch_processor, analyzer_cli

ЗАВИСИМОСТИ:
- Python 3.8+
- src/interfaces/analyzer_interface.py
- Gemma, Qwen, AlgorithmicAnalyzer

РЕЗУЛЬТАТ:
- Совмещенные метрики качества, настроения, сложности
- Повышенная точность анализа

АВТОР: AI Assistant
ДАТА: Сентябрь 2025
"""
import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from interfaces.analyzer_interface import BaseAnalyzer, AnalysisResult, AnalyzerFactory, register_analyzer

logger = logging.getLogger(__name__)


@register_analyzer("hybrid")
class HybridAnalyzer(BaseAnalyzer):
    """
    Гибридный анализатор комбинирующий AI и алгоритмические методы.
    
    Использует несколько анализаторов для получения более точных результатов:
    - Gemma AI анализатор (основной)
    - Алгоритмический анализатор (дополняющий)
    - Интеллектуальное объединение результатов
    - Кросс-валидация между методами
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Инициализация гибридного анализатора"""
        super().__init__(config)
        
        # Настройки компонентов
        self.primary_analyzer = self.config.get('primary_analyzer', 'gemma')
        self.secondary_analyzer = self.config.get('secondary_analyzer', 'algorithmic_basic')
        self.fallback_analyzer = self.config.get('fallback_analyzer', 'ollama')
        
        # Веса для объединения результатов
        self.ai_weight = self.config.get('ai_weight', 0.7)
        self.algorithmic_weight = self.config.get('algorithmic_weight', 0.3)
        
        # Пороги уверенности
        self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.5)
        self.use_fallback_threshold = self.config.get('use_fallback_threshold', 0.3)
        
        # Инициализация компонентов
        self.analyzers = {}
        self._initialize_analyzers()
    
    def _initialize_analyzers(self) -> None:
        """Инициализация составных анализаторов"""
        analyzer_names = [self.primary_analyzer, self.secondary_analyzer, self.fallback_analyzer]
        
        for analyzer_name in analyzer_names:
            try:
                analyzer = AnalyzerFactory.create(
                    analyzer_name, 
                    config=self.config.get(f'{analyzer_name}_config', {}),
                    singleton=True
                )
                self.analyzers[analyzer_name] = analyzer
                logger.info(f"✅ Компонент {analyzer_name} инициализирован")
                
            except Exception as e:
                logger.warning(f"⚠️ Не удалось инициализировать {analyzer_name}: {e}")
                self.analyzers[analyzer_name] = None
        
        # Проверяем доступность хотя бы одного анализатора
        available_analyzers = [name for name, analyzer in self.analyzers.items() if analyzer is not None]
        
        if not available_analyzers:
            logger.error("❌ Ни один анализатор не доступен для гибридного анализа")
            self.available = False
        else:
            logger.info(f"✅ Гибридный анализатор готов. Доступны: {available_analyzers}")
            self.available = True
    
    def analyze_song(self, artist: str, title: str, lyrics: str) -> AnalysisResult:
        """
        Гибридный анализ песни с использованием нескольких методов.
        
        Args:
            artist: Имя исполнителя
            title: Название песни
            lyrics: Текст песни
            
        Returns:
            AnalysisResult с объединенными результатами
        """
        start_time = time.time()
        
        # Валидация входных данных
        if not self.validate_input(artist, title, lyrics):
            raise ValueError("Invalid input parameters")
        
        if not self.available:
            raise RuntimeError("Hybrid analyzer is not available - no component analyzers initialized")
        
        # Предобработка текста
        processed_lyrics = self.preprocess_lyrics(lyrics)
        
        # Запуск анализов
        analysis_results = {}
        
        # 1. Основной AI анализатор
        ai_result = self._run_analyzer(self.primary_analyzer, artist, title, processed_lyrics)
        if ai_result:
            analysis_results['primary'] = ai_result
        
        # 2. Алгоритмический анализатор (всегда запускаем)
        algo_result = self._run_analyzer(self.secondary_analyzer, artist, title, processed_lyrics)
        if algo_result:
            analysis_results['secondary'] = algo_result
        
        # 3. Fallback анализатор (если основной не сработал или низкая уверенность)
        use_fallback = False
        if not ai_result:
            use_fallback = True
            logger.info("🔄 Основной анализатор недоступен, используем fallback")
        elif ai_result.confidence < self.use_fallback_threshold:
            use_fallback = True
            logger.info(f"🔄 Низкая уверенность основного анализатора ({ai_result.confidence:.3f}), используем fallback")
        
        if use_fallback and self.fallback_analyzer != self.primary_analyzer:
            fallback_result = self._run_analyzer(self.fallback_analyzer, artist, title, processed_lyrics)
            if fallback_result:
                analysis_results['fallback'] = fallback_result
        
        # Объединение результатов
        combined_analysis = self._combine_results(analysis_results)
        
        # Вычисление итоговой уверенности
        final_confidence = self._calculate_combined_confidence(analysis_results)
        
        processing_time = time.time() - start_time
        
        return AnalysisResult(
            artist=artist,
            title=title,
            analysis_type="hybrid",
            confidence=final_confidence,
            metadata={
                "analyzers_used": list(analysis_results.keys()),
                "primary_analyzer": self.primary_analyzer,
                "secondary_analyzer": self.secondary_analyzer,
                "fallback_used": 'fallback' in analysis_results,
                "ai_weight": self.ai_weight,
                "algorithmic_weight": self.algorithmic_weight,
                "processing_date": datetime.now().isoformat(),
                "component_confidences": {
                    name: result.confidence for name, result in analysis_results.items()
                }
            },
            raw_output=combined_analysis,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
    
    def _run_analyzer(self, analyzer_name: str, artist: str, title: str, lyrics: str) -> Optional[AnalysisResult]:
        """Запуск конкретного анализатора с обработкой ошибок"""
        analyzer = self.analyzers.get(analyzer_name)
        
        if not analyzer:
            logger.warning(f"⚠️ Анализатор {analyzer_name} недоступен")
            return None
        
        try:
            result = analyzer.analyze_song(artist, title, lyrics)
            logger.info(f"✅ {analyzer_name} анализ завершен (уверенность: {result.confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка {analyzer_name} анализа: {e}")
            return None
    
    def _combine_results(self, analysis_results: Dict[str, AnalysisResult]) -> Dict[str, Any]:
        """Интеллектуальное объединение результатов анализа"""
        combined = {
            "hybrid_summary": {
                "analysis_methods": list(analysis_results.keys()),
                "consensus_metrics": {},
                "method_comparison": {}
            }
        }
        
        # Получаем результаты от разных анализаторов
        primary_data = analysis_results.get('primary', {}).raw_output if 'primary' in analysis_results else {}
        secondary_data = analysis_results.get('secondary', {}).raw_output if 'secondary' in analysis_results else {}
        fallback_data = analysis_results.get('fallback', {}).raw_output if 'fallback' in analysis_results else {}
        
        # 1. Объединение базовой информации
        combined["basic_info"] = self._merge_basic_info(primary_data, secondary_data, fallback_data)
        
        # 2. Объединение анализа настроения
        combined["sentiment_analysis"] = self._merge_sentiment(primary_data, secondary_data)
        
        # 3. Объединение качественных метрик
        combined["quality_metrics"] = self._merge_quality_metrics(primary_data, secondary_data, fallback_data)
        
        # 4. Кросс-валидация
        combined["cross_validation"] = self._cross_validate_results(analysis_results)
        
        # 5. Объединение технических аспектов
        combined["technical_analysis"] = self._merge_technical_analysis(primary_data, secondary_data, fallback_data)
        
        # 6. Сохранение исходных результатов для отладки
        combined["component_results"] = {
            name: result.raw_output for name, result in analysis_results.items()
        }
        
        return combined
    
    def _merge_basic_info(self, primary: Dict, secondary: Dict, fallback: Dict) -> Dict[str, Any]:
        """Объединение базовой информации"""
        merged = {}
        
        # Жанр (приоритет AI анализаторам)
        if primary.get('genre_analysis', {}).get('primary_genre'):
            merged['genre'] = primary['genre_analysis']['primary_genre']
        elif primary.get('basic_analysis', {}).get('genre'):
            merged['genre'] = primary['basic_analysis']['genre']
        elif fallback.get('basic_analysis', {}).get('genre'):
            merged['genre'] = fallback['basic_analysis']['genre']
        else:
            merged['genre'] = 'rap'  # По умолчанию
        
        # Настроение (консенсус)
        moods = []
        if primary.get('mood_analysis', {}).get('primary_mood'):
            moods.append(primary['mood_analysis']['primary_mood'])
        if secondary.get('sentiment_analysis', {}).get('sentiment_label'):
            moods.append(secondary['sentiment_analysis']['sentiment_label'])
        if fallback.get('basic_analysis', {}).get('mood'):
            moods.append(fallback['basic_analysis']['mood'])
        
        if moods:
            # Простое голосование
            from collections import Counter
            mood_votes = Counter(moods)
            merged['mood'] = mood_votes.most_common(1)[0][0]
            merged['mood_consensus'] = len(set(moods)) == 1  # Все согласны?
        
        # Энергия
        energy_sources = []
        if primary.get('mood_analysis', {}).get('energy_level'):
            energy_sources.append(primary['mood_analysis']['energy_level'])
        if fallback.get('basic_analysis', {}).get('energy'):
            energy_sources.append(fallback['basic_analysis']['energy'])
        
        if energy_sources:
            # Усреднение энергии
            energy_map = {'low': 1, 'medium': 2, 'high': 3}
            reverse_map = {1: 'low', 2: 'medium', 3: 'high'}
            avg_energy = sum(energy_map.get(e, 2) for e in energy_sources) / len(energy_sources)
            merged['energy_level'] = reverse_map[round(avg_energy)]
        
        return merged
    
    def _merge_sentiment(self, primary: Dict, secondary: Dict) -> Dict[str, Any]:
        """Объединение анализа настроения"""
        merged = {}
        
        # AI анализ настроения
        if primary.get('mood_analysis'):
            mood_data = primary['mood_analysis']
            merged['ai_mood'] = {
                'primary_mood': mood_data.get('primary_mood'),
                'emotional_intensity': mood_data.get('emotional_intensity'),
                'valence': mood_data.get('valence')
            }
        
        # Алгоритмический анализ настроения
        if secondary.get('sentiment_analysis'):
            sentiment_data = secondary['sentiment_analysis']
            merged['algorithmic_sentiment'] = {
                'sentiment_score': sentiment_data.get('sentiment_score'),
                'sentiment_label': sentiment_data.get('sentiment_label'),
                'positive_words': sentiment_data.get('positive_words_count', 0),
                'negative_words': sentiment_data.get('negative_words_count', 0)
            }
        
        # Консенсус настроения
        ai_mood = merged.get('ai_mood', {}).get('valence', 'neutral')
        algo_sentiment = merged.get('algorithmic_sentiment', {}).get('sentiment_label', 'neutral')
        
        # Маппинг для сравнения
        mood_mapping = {
            'positive': 'positive', 'negative': 'negative', 'neutral': 'neutral',
            'aggressive': 'negative', 'melancholic': 'negative', 
            'energetic': 'positive', 'confident': 'positive'
        }
        
        ai_normalized = mood_mapping.get(ai_mood, 'neutral')
        algo_normalized = mood_mapping.get(algo_sentiment, 'neutral')
        
        merged['consensus'] = {
            'sentiment_agreement': ai_normalized == algo_normalized,
            'final_sentiment': ai_normalized if ai_normalized == algo_normalized else 'mixed'
        }
        
        return merged
    
    def _merge_quality_metrics(self, primary: Dict, secondary: Dict, fallback: Dict) -> Dict[str, Any]:
        """Объединение качественных метрик"""
        merged = {}
        
        # AI метрики качества
        ai_quality = primary.get('quality_metrics', {})
        algo_quality = secondary.get('complexity_analysis', {})
        fallback_quality = fallback.get('quality_assessment', {})
        
        # Объединение с весами
        combined_metrics = {}
        
        # Креативность
        creativity_sources = []
        if ai_quality.get('lyrical_creativity') is not None:
            creativity_sources.append(('ai', ai_quality['lyrical_creativity'], self.ai_weight))
        if fallback_quality.get('creativity') is not None:
            creativity_sources.append(('fallback', fallback_quality['creativity'], 0.3))
        
        if creativity_sources:
            weighted_creativity = sum(score * weight for _, score, weight in creativity_sources)
            total_weight = sum(weight for _, _, weight in creativity_sources)
            combined_metrics['creativity'] = weighted_creativity / total_weight
        
        # Техническое мастерство
        technical_sources = []
        if ai_quality.get('technical_skill') is not None:
            technical_sources.append(('ai', ai_quality['technical_skill'], self.ai_weight))
        if fallback_quality.get('lyrical_skill') is not None:
            technical_sources.append(('fallback', fallback_quality['lyrical_skill'], 0.3))
        
        if technical_sources:
            weighted_technical = sum(score * weight for _, score, weight in technical_sources)
            total_weight = sum(weight for _, _, weight in technical_sources)
            combined_metrics['technical_skill'] = weighted_technical / total_weight
        
        # Читабельность (только алгоритмический)
        if algo_quality.get('readability_score') is not None:
            # Нормализация читабельности в 0-1
            readability = algo_quality['readability_score']
            combined_metrics['readability'] = min(1.0, max(0.0, readability / 100))
        
        # Уникальность словаря
        if algo_quality.get('vocabulary_richness') is not None:
            combined_metrics['vocabulary_richness'] = algo_quality['vocabulary_richness']
        
        # Общая оценка
        if ai_quality.get('overall_quality') is not None:
            combined_metrics['overall_quality'] = ai_quality['overall_quality']
        elif fallback_quality.get('overall_quality') is not None:
            combined_metrics['overall_quality'] = fallback_quality['overall_quality']
        
        merged['combined_metrics'] = combined_metrics
        merged['source_metrics'] = {
            'ai_metrics': ai_quality,
            'algorithmic_metrics': algo_quality,
            'fallback_metrics': fallback_quality
        }
        
        return merged
    
    def _merge_technical_analysis(self, primary: Dict, secondary: Dict, fallback: Dict) -> Dict[str, Any]:
        """Объединение технического анализа"""
        merged = {}
        
        # AI технический анализ
        if primary.get('technical_analysis'):
            merged['ai_technical'] = primary['technical_analysis']
        
        # Алгоритмический структурный анализ
        if secondary.get('structure_analysis'):
            merged['algorithmic_structure'] = secondary['structure_analysis']
        
        # Анализ рифм
        if secondary.get('structure_analysis', {}).get('rhyme_density') is not None:
            merged['rhyme_analysis'] = {
                'density': secondary['structure_analysis']['rhyme_density'],
                'pairs': secondary['structure_analysis'].get('rhyming_pairs', 0)
            }
        
        # Консенсус по сложности
        complexities = []
        if primary.get('technical_analysis', {}).get('complexity_level'):
            complexities.append(primary['technical_analysis']['complexity_level'])
        if fallback.get('technical_aspects', {}).get('rhyme_complexity'):
            complexities.append(fallback['technical_aspects']['rhyme_complexity'])
        
        if complexities:
            # Маппинг уровней сложности
            complexity_map = {'simple': 1, 'basic': 1, 'moderate': 2, 'intermediate': 2, 'complex': 3, 'advanced': 3, 'expert': 4}
            avg_complexity = sum(complexity_map.get(c, 2) for c in complexities) / len(complexities)
            
            if avg_complexity <= 1.5:
                merged['consensus_complexity'] = 'simple'
            elif avg_complexity <= 2.5:
                merged['consensus_complexity'] = 'moderate'
            elif avg_complexity <= 3.5:
                merged['consensus_complexity'] = 'complex'
            else:
                merged['consensus_complexity'] = 'expert'
        
        return merged
    
    def _cross_validate_results(self, analysis_results: Dict[str, AnalysisResult]) -> Dict[str, Any]:
        """Кросс-валидация результатов между анализаторами"""
        validation = {
            'agreement_scores': {},
            'discrepancies': [],
            'reliability_assessment': {}
        }
        
        if len(analysis_results) < 2:
            validation['note'] = 'Insufficient analyzers for cross-validation'
            return validation
        
        # Сравнение уверенности
        confidences = {name: result.confidence for name, result in analysis_results.items()}
        avg_confidence = sum(confidences.values()) / len(confidences)
        confidence_variance = sum((c - avg_confidence) ** 2 for c in confidences.values()) / len(confidences)
        
        validation['confidence_analysis'] = {
            'average_confidence': avg_confidence,
            'confidence_variance': confidence_variance,
            'consistent_confidence': confidence_variance < 0.1  # Низкая вариация
        }
        
        # Проверка согласованности времени обработки
        processing_times = {name: result.processing_time for name, result in analysis_results.items()}
        validation['performance_analysis'] = {
            'processing_times': processing_times,
            'fastest_analyzer': min(processing_times.items(), key=lambda x: x[1])[0],
            'slowest_analyzer': max(processing_times.items(), key=lambda x: x[1])[0]
        }
        
        return validation
    
    def _calculate_combined_confidence(self, analysis_results: Dict[str, AnalysisResult]) -> float:
        """Вычисление итоговой уверенности гибридного анализа"""
        if not analysis_results:
            return 0.0
        
        # Веса для разных анализаторов
        weights = {
            'primary': self.ai_weight,
            'secondary': self.algorithmic_weight,
            'fallback': 0.5  # Средний вес для fallback
        }
        
        weighted_confidences = []
        
        for name, result in analysis_results.items():
            weight = weights.get(name, 0.3)  # Дефолтный вес
            weighted_confidences.append(result.confidence * weight)
        
        # Базовая уверенность
        base_confidence = sum(weighted_confidences) / len(analysis_results)
        
        # Бонус за консенсус
        if len(analysis_results) >= 2:
            # Если все анализаторы показывают высокую уверенность
            min_confidence = min(result.confidence for result in analysis_results.values())
            if min_confidence > 0.7:
                base_confidence = min(1.0, base_confidence * 1.1)  # 10% бонус
        
        # Штраф за использование fallback
        if 'fallback' in analysis_results and 'primary' not in analysis_results:
            base_confidence *= 0.9  # 10% штраф
        
        return min(1.0, max(0.0, base_confidence))
    
    def get_analyzer_info(self) -> Dict[str, Any]:
        """Получение информации о гибридном анализаторе"""
        component_info = {}
        for name, analyzer in self.analyzers.items():
            if analyzer:
                try:
                    component_info[name] = analyzer.get_analyzer_info()
                except:
                    component_info[name] = {"status": "available", "info": "basic"}
            else:
                component_info[name] = {"status": "unavailable"}
        
        return {
            "name": "HybridAnalyzer",
            "version": "2.0.0",
            "description": "Combines AI and algorithmic analysis methods for comprehensive lyrics analysis",
            "author": "Rap Scraper Project",
            "type": self.analyzer_type,
            "supported_features": self.supported_features,
            "components": component_info,
            "configuration": {
                "primary_analyzer": self.primary_analyzer,
                "secondary_analyzer": self.secondary_analyzer,
                "fallback_analyzer": self.fallback_analyzer,
                "ai_weight": self.ai_weight,
                "algorithmic_weight": self.algorithmic_weight
            },
            "available": self.available,
            "config_options": {
                "primary_analyzer": "Main AI analyzer (default: gemma)",
                "secondary_analyzer": "Algorithmic analyzer (default: algorithmic_basic)",
                "fallback_analyzer": "Backup analyzer (default: ollama)",
                "ai_weight": "Weight for AI results (default: 0.7)",
                "algorithmic_weight": "Weight for algorithmic results (default: 0.3)",
                "min_confidence_threshold": "Minimum confidence threshold (default: 0.5)",
                "use_fallback_threshold": "When to use fallback analyzer (default: 0.3)"
            }
        }
    
    @property
    def analyzer_type(self) -> str:
        """Тип анализатора"""
        return "hybrid"
    
    @property
    def supported_features(self) -> List[str]:
        """Поддерживаемые функции анализа"""
        return [
            "multi_method_analysis",
            "cross_validation",
            "consensus_building",
            "quality_assessment",
            "technical_analysis",
            "sentiment_analysis",
            "fallback_support",
            "confidence_weighting",
            "result_merging",
            "reliability_assessment"
        ]
