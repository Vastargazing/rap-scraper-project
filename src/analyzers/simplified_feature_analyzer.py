#!/usr/bin/env python3
"""
Simplified Advanced Feature Analyzer (без NLTK зависимостей)

Базовая версия расширенного анализатора для демонстрации возможностей
Feature Engineering без внешних NLP библиотек.
"""

import re
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Set
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

# ===== УПРОЩЕННЫЕ МОДЕЛИ ДАННЫХ =====

class SimplifiedRhymeAnalysis(BaseModel):
    """Упрощенный анализ рифм"""
    rhyme_density: float = Field(ge=0.0, le=1.0, description="Плотность рифм")
    rhyme_detection_confidence: float = Field(ge=0.0, le=1.0, description="Уверенность в детекции рифм")
    end_rhyme_scheme: str = Field(description="Схема рифмовки")
    internal_rhymes: int = Field(ge=0, description="Внутренние рифмы")
    perfect_rhymes: int = Field(ge=0, description="Точные рифмы")
    alliteration_score: float = Field(ge=0.0, le=1.0, description="Аллитерация")
    rhyme_scheme_confidence: float = Field(ge=0.0, le=1.0, description="Уверенность в определении схемы рифм")

class SimplifiedVocabularyAnalysis(BaseModel):
    """Упрощенный анализ словаря"""
    ttr_score: float = Field(ge=0.0, le=1.0, description="Type-Token Ratio")
    unique_words: int = Field(ge=0, description="Уникальные слова")
    total_words: int = Field(ge=0, description="Всего слов")
    average_word_length: float = Field(ge=0.0, description="Средняя длина слова")
    complex_words_ratio: float = Field(ge=0.0, le=1.0, description="Доля сложных слов")

class SimplifiedMetaphorAnalysis(BaseModel):
    """Упрощенный анализ метафор"""
    metaphor_count: int = Field(ge=0, description="Количество метафор")
    metaphor_confidence: float = Field(ge=0.0, le=1.0, description="Уверенность в детекции метафор")
    wordplay_instances: int = Field(ge=0, description="Игра слов")
    wordplay_confidence: float = Field(ge=0.0, le=1.0, description="Уверенность в детекции игры слов")
    creativity_score: float = Field(ge=0.0, le=1.0, description="Креативность")
    creativity_confidence: float = Field(ge=0.0, le=1.0, description="Уверенность в оценке креативности")

class SimplifiedFlowAnalysis(BaseModel):
    """Упрощенный анализ ритма"""
    syllable_count: int = Field(ge=0, description="Количество слогов")
    average_syllables_per_line: float = Field(ge=0.0, description="Слогов на строку")
    stress_pattern_consistency: float = Field(ge=0.0, le=1.0, description="Консистентность")
    stress_pattern_confidence: float = Field(ge=0.0, le=1.0, description="Уверенность в анализе ударений")
    flow_breaks: int = Field(ge=0, description="Паузы в потоке")
    flow_analysis_confidence: float = Field(ge=0.0, le=1.0, description="Уверенность в анализе потока")

class SimplifiedAdvancedFeatures(BaseModel):
    """Упрощенный набор расширенных фичей"""
    rhyme_analysis: SimplifiedRhymeAnalysis
    vocabulary_analysis: SimplifiedVocabularyAnalysis
    metaphor_analysis: SimplifiedMetaphorAnalysis
    flow_analysis: SimplifiedFlowAnalysis
    
    # Композитные метрики
    overall_complexity: float = Field(ge=0.0, le=1.0)
    artistic_sophistication: float = Field(ge=0.0, le=1.0)
    technical_skill: float = Field(ge=0.0, le=1.0)
    innovation_score: float = Field(ge=0.0, le=1.0)

# ===== УПРОЩЕННЫЙ АНАЛИЗАТОР =====

class SimplifiedFeatureAnalyzer:
    """Упрощенный анализатор без внешних зависимостей"""
    
    def __init__(self):
        # Простые паттерны для анализа
        self.metaphor_keywords = [
            'как', 'словно', 'будто', 'точно', 'похож на', 'like', 'as'
        ]
        
        self.wordplay_patterns = [
            r'\b(\w+)\s+\w*\1\w*\b',  # Повторения
            r'\b\w*([аеиоуыэя]{2})\w*\s+\w*\1\w*\b',  # Ассонанс
        ]
        
        # Русские гласные для подсчета слогов
        self.vowels = 'аеёиоуыэюя'
        
        # Стоп-слова
        self.stop_words = {
            'и', 'в', 'на', 'с', 'по', 'за', 'из', 'к', 'у', 'о', 'от', 'до', 'для',
            'но', 'а', 'что', 'как', 'не', 'я', 'ты', 'он', 'она', 'мы', 'вы', 'они',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of'
        }

    def analyze_lyrics(self, lyrics: str) -> SimplifiedAdvancedFeatures:
        """Анализ текста с упрощенными методами"""
        
        # Предобработка
        lines = [line.strip() for line in lyrics.split('\n') if line.strip()]
        words = self._tokenize_lyrics(lyrics)
        
        # Анализ компонентов
        rhyme_analysis = self._analyze_rhymes_simple(lines, words)
        vocabulary_analysis = self._analyze_vocabulary_simple(words)
        metaphor_analysis = self._analyze_metaphors_simple(lyrics, words)
        flow_analysis = self._analyze_flow_simple(lines, words)
        
        # Композитные метрики
        overall_complexity = (
            rhyme_analysis.rhyme_density * 0.25 +
            vocabulary_analysis.ttr_score * 0.25 +
            metaphor_analysis.creativity_score * 0.25 +
            flow_analysis.stress_pattern_consistency * 0.25
        )
        
        artistic_sophistication = (
            metaphor_analysis.creativity_score * 0.4 +
            vocabulary_analysis.ttr_score * 0.3 +
            rhyme_analysis.alliteration_score * 0.3
        )
        
        technical_skill = (
            rhyme_analysis.rhyme_density * 0.4 +
            flow_analysis.stress_pattern_consistency * 0.3 +
            min(vocabulary_analysis.ttr_score * 2, 1.0) * 0.3
        )
        
        innovation_score = (
            metaphor_analysis.creativity_score * 0.5 +
            min(rhyme_analysis.internal_rhymes / 10, 1.0) * 0.3 +
            min(vocabulary_analysis.complex_words_ratio * 2, 1.0) * 0.2
        )
        
        return SimplifiedAdvancedFeatures(
            rhyme_analysis=rhyme_analysis,
            vocabulary_analysis=vocabulary_analysis,
            metaphor_analysis=metaphor_analysis,
            flow_analysis=flow_analysis,
            overall_complexity=overall_complexity,
            artistic_sophistication=artistic_sophistication,
            technical_skill=technical_skill,
            innovation_score=innovation_score
        )

    def _tokenize_lyrics(self, lyrics: str) -> List[str]:
        """Простая токенизация"""
        words = re.findall(r'\b\w+\b', lyrics.lower())
        return [word for word in words if len(word) > 2 and word not in self.stop_words]

    def _analyze_rhymes_simple(self, lines: List[str], words: List[str]) -> SimplifiedRhymeAnalysis:
        """Упрощенный анализ рифм"""
        
        # Получаем концовки строк
        line_endings = []
        for line in lines:
            words_in_line = line.strip().split()
            if words_in_line:
                ending = words_in_line[-1].lower().strip('.,!?')
                line_endings.append(ending)
        
        # Схема рифмовки
        rhyme_scheme = self._detect_rhyme_scheme_simple(line_endings)
        
        # Подсчет рифм
        perfect_rhymes = self._count_perfect_rhymes_simple(line_endings)
        internal_rhymes = self._count_internal_rhymes_simple(lines)
        
        # Аллитерация
        alliteration_score = self._calculate_alliteration_simple(lines)
        
        # Плотность рифм
        rhyme_density = perfect_rhymes / max(len(lines), 1)
        
        # Confidence для детекции рифм: базируется на количестве найденных рифм и длине текста
        rhyme_detection_confidence = min(
            (perfect_rhymes / max(len(lines) / 2, 1)) * 0.6 +  # Относительное количество рифм
            (1 if len(line_endings) >= 8 else len(line_endings) / 8) * 0.4,  # Достаточность материала
            0.85  # Максимум 0.85 для простого алгоритма
        )
        
        # Confidence для схемы рифм: зависит от сложности схемы и количества строк
        if rhyme_scheme == "insufficient":
            rhyme_scheme_confidence = 0.1
        elif len(set(rhyme_scheme)) == 1:  # Монорифма
            rhyme_scheme_confidence = 0.9
        elif len(rhyme_scheme) >= 6:  # Достаточно строк для анализа
            rhyme_scheme_confidence = 0.75
        else:
            rhyme_scheme_confidence = 0.5
        
        return SimplifiedRhymeAnalysis(
            rhyme_density=min(rhyme_density, 1.0),
            rhyme_detection_confidence=rhyme_detection_confidence,
            end_rhyme_scheme=rhyme_scheme,
            internal_rhymes=internal_rhymes,
            perfect_rhymes=perfect_rhymes,
            alliteration_score=alliteration_score,
            rhyme_scheme_confidence=rhyme_scheme_confidence
        )

    def _detect_rhyme_scheme_simple(self, endings: List[str]) -> str:
        """Простое определение схемы рифм"""
        if len(endings) < 4:
            return "insufficient"
        
        scheme = []
        rhyme_groups = {}
        current_letter = 'A'
        
        for ending in endings[:8]:
            found_rhyme = False
            
            for grouped_ending, letter in rhyme_groups.items():
                if self._words_rhyme_simple(ending, grouped_ending):
                    scheme.append(letter)
                    found_rhyme = True
                    break
            
            if not found_rhyme:
                rhyme_groups[ending] = current_letter
                scheme.append(current_letter)
                current_letter = chr(ord(current_letter) + 1)
        
        return ''.join(scheme)

    def _words_rhyme_simple(self, word1: str, word2: str) -> bool:
        """Простая проверка рифмы по окончанию"""
        if len(word1) < 3 or len(word2) < 3:
            return False
        
        # Проверяем последние 2-3 символа
        return word1[-2:] == word2[-2:] or word1[-3:] == word2[-3:]

    def _count_perfect_rhymes_simple(self, endings: List[str]) -> int:
        """Подсчет точных рифм"""
        rhyme_count = 0
        for i in range(len(endings)):
            for j in range(i + 1, len(endings)):
                if self._words_rhyme_simple(endings[i], endings[j]):
                    rhyme_count += 1
        return rhyme_count

    def _count_internal_rhymes_simple(self, lines: List[str]) -> int:
        """Подсчет внутренних рифм"""
        internal_count = 0
        
        for line in lines:
            words = [w.lower().strip('.,!?') for w in line.split()]
            if len(words) < 2:
                continue
            
            for i in range(len(words) - 1):
                for j in range(i + 1, len(words)):
                    if self._words_rhyme_simple(words[i], words[j]):
                        internal_count += 1
        
        return internal_count

    def _calculate_alliteration_simple(self, lines: List[str]) -> float:
        """Подсчет аллитерации"""
        alliteration_count = 0
        total_pairs = 0
        
        for line in lines:
            words = [word.lower() for word in line.split() if len(word) > 2]
            if len(words) < 2:
                continue
            
            total_pairs += len(words) - 1
            
            for i in range(len(words) - 1):
                if words[i][0] == words[i + 1][0]:
                    alliteration_count += 1
        
        return alliteration_count / max(total_pairs, 1)

    def _analyze_vocabulary_simple(self, words: List[str]) -> SimplifiedVocabularyAnalysis:
        """Упрощенный анализ словаря"""
        
        unique_words = len(set(words))
        total_words = len(words)
        
        # TTR
        ttr_score = unique_words / max(total_words, 1)
        
        # Средняя длина слова
        average_word_length = sum(len(word) for word in words) / max(len(words), 1)
        
        # Сложные слова (длиннее 6 символов)
        complex_words = sum(1 for word in words if len(word) > 6)
        complex_words_ratio = complex_words / max(total_words, 1)
        
        return SimplifiedVocabularyAnalysis(
            ttr_score=min(ttr_score, 1.0),
            unique_words=unique_words,
            total_words=total_words,
            average_word_length=average_word_length,
            complex_words_ratio=complex_words_ratio
        )

    def _analyze_metaphors_simple(self, lyrics: str, words: List[str]) -> SimplifiedMetaphorAnalysis:
        """Упрощенный анализ метафор с confidence scores"""
        
        metaphor_count = 0
        metaphor_matches = []
        for keyword in self.metaphor_keywords:
            matches = lyrics.lower().count(keyword)
            metaphor_count += matches
            if matches > 0:
                metaphor_matches.append(keyword)
        
        # Confidence для метафор: базируется на количестве найденных ключевых слов и их разнообразии
        metaphor_confidence = min(len(metaphor_matches) / max(len(self.metaphor_keywords), 1) + 
                                metaphor_count / max(len(words), 1) * 5, 1.0)
        
        # Игра слов (простые паттерны)
        wordplay_count = 0
        wordplay_patterns_found = 0
        for pattern in self.wordplay_patterns:
            matches = re.findall(pattern, lyrics, re.IGNORECASE)
            if matches:
                wordplay_count += len(matches)
                wordplay_patterns_found += 1
        
        # Confidence для игры слов: низкая, так как простые паттерны могут давать ложные срабатывания
        wordplay_confidence = min(wordplay_patterns_found / max(len(self.wordplay_patterns), 1) * 0.6 + 
                                wordplay_count / max(len(words), 1) * 3, 0.8)  # Максимум 0.8
        
        # Креативность (на основе разнообразия и художественных приемов)
        creativity_score = min((metaphor_count + wordplay_count) / max(len(words), 1) * 10, 1.0)
        
        # Confidence креативности: средняя между confidence метафор и игры слов
        creativity_confidence = (metaphor_confidence + wordplay_confidence) / 2
        
        return SimplifiedMetaphorAnalysis(
            metaphor_count=metaphor_count,
            metaphor_confidence=metaphor_confidence,
            wordplay_instances=wordplay_count,
            wordplay_confidence=wordplay_confidence,
            creativity_score=creativity_score,
            creativity_confidence=creativity_confidence
        )

    def _analyze_flow_simple(self, lines: List[str], words: List[str]) -> SimplifiedFlowAnalysis:
        """Упрощенный анализ ритма с confidence scores"""
        
        # Подсчет слогов
        syllable_count = self._count_syllables_simple(words)
        average_syllables_per_line = syllable_count / max(len(lines), 1)
        
        # Консистентность (на основе длины строк)
        line_lengths = [len(line.split()) for line in lines]
        if line_lengths:
            length_variance = self._calculate_variance(line_lengths)
            stress_consistency = max(0, 1 - length_variance / 10)
            # Confidence для ударений: высокая при низкой дисперсии длин строк
            stress_pattern_confidence = max(0.3, 1 - length_variance / 8)  # Минимум 0.3
        else:
            stress_consistency = 0
            stress_pattern_confidence = 0.2  # Низкая confidence без данных
        
        # Паузы (знаки препинания)
        flow_breaks = sum(line.count(',') + line.count('.') + line.count('!') + line.count('?') for line in lines)
        
        # Confidence для анализа flow: зависит от количества строк и их заполненности
        lines_with_content = sum(1 for line in lines if len(line.strip()) > 5)
        flow_analysis_confidence = min(lines_with_content / max(len(lines), 1) * 0.8 + 
                                     (1 if len(lines) >= 8 else len(lines) / 8) * 0.2, 0.9)
        
        return SimplifiedFlowAnalysis(
            syllable_count=syllable_count,
            average_syllables_per_line=average_syllables_per_line,
            stress_pattern_consistency=stress_consistency,
            stress_pattern_confidence=stress_pattern_confidence,
            flow_breaks=flow_breaks,
            flow_analysis_confidence=flow_analysis_confidence
        )

    def _count_syllables_simple(self, words: List[str]) -> int:
        """Простой подсчет слогов по гласным"""
        total_syllables = 0
        
        for word in words:
            syllables = 0
            prev_was_vowel = False
            
            for char in word.lower():
                is_vowel = char in self.vowels
                if is_vowel and not prev_was_vowel:
                    syllables += 1
                prev_was_vowel = is_vowel
            
            # Минимум 1 слог на слово
            total_syllables += max(syllables, 1)
        
        return total_syllables

    def _calculate_variance(self, values: List[int]) -> float:
        """Расчет дисперсии"""
        if not values:
            return 0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance


# ===== УТИЛИТЫ =====

def extract_simplified_features(lyrics: str) -> Dict[str, float]:
    """Извлечение упрощенных фичей в плоском формате с confidence scores"""
    
    analyzer = SimplifiedFeatureAnalyzer()
    features = analyzer.analyze_lyrics(lyrics)
    
    return {
        # Rhyme features
        'rhyme_density': features.rhyme_analysis.rhyme_density,
        'rhyme_detection_confidence': features.rhyme_analysis.rhyme_detection_confidence,
        'perfect_rhymes': float(features.rhyme_analysis.perfect_rhymes),
        'internal_rhymes': float(features.rhyme_analysis.internal_rhymes),
        'alliteration_score': features.rhyme_analysis.alliteration_score,
        'rhyme_scheme_confidence': features.rhyme_analysis.rhyme_scheme_confidence,
        
        # Vocabulary features
        'ttr_score': features.vocabulary_analysis.ttr_score,
        'average_word_length': features.vocabulary_analysis.average_word_length,
        'complex_words_ratio': features.vocabulary_analysis.complex_words_ratio,
        
        # Metaphor features
        'metaphor_count': float(features.metaphor_analysis.metaphor_count),
        'metaphor_confidence': features.metaphor_analysis.metaphor_confidence,
        'wordplay_instances': float(features.metaphor_analysis.wordplay_instances),
        'wordplay_confidence': features.metaphor_analysis.wordplay_confidence,
        'creativity_score': features.metaphor_analysis.creativity_score,
        'creativity_confidence': features.metaphor_analysis.creativity_confidence,
        
        # Flow features
        'average_syllables_per_line': features.flow_analysis.average_syllables_per_line,
        'stress_pattern_consistency': features.flow_analysis.stress_pattern_consistency,
        'stress_pattern_confidence': features.flow_analysis.stress_pattern_confidence,
        'flow_breaks': float(features.flow_analysis.flow_breaks),
        'flow_analysis_confidence': features.flow_analysis.flow_analysis_confidence,
        
        # Composite features
        'overall_complexity': features.overall_complexity,
        'artistic_sophistication': features.artistic_sophistication,
        'technical_skill': features.technical_skill,
        'innovation_score': features.innovation_score,
    }


def demo_simplified_analysis():
    """Демонстрация упрощенного анализа"""
    
    sample_lyrics = """
    Я поднимаюсь как солнце над городом серым
    Мои слова как пули, попадают в цель верно
    В лабиринте из строчек я нашёл свою дорогу
    Рифмы льются как река, несут меня к итогу
    
    Время — деньги, деньги — власть, власть — это иллюзия
    В игре теней и отражений я создаю конфузию
    Мой флоу как водопад, стремительный и чистый
    Слова танцуют на битах, движения артистичны
    """
    
    analyzer = SimplifiedFeatureAnalyzer()
    features = analyzer.analyze_lyrics(sample_lyrics)
    
    print("=== УПРОЩЕННЫЙ АНАЛИЗ РЭПА ===")
    print(f"🎵 Rhyme Density: {features.rhyme_analysis.rhyme_density:.3f}")
    print(f"📚 TTR Score: {features.vocabulary_analysis.ttr_score:.3f}")
    print(f"🎨 Creativity Score: {features.metaphor_analysis.creativity_score:.3f}")
    print(f"🎯 Technical Skill: {features.technical_skill:.3f}")
    print(f"💡 Innovation Score: {features.innovation_score:.3f}")
    
    # Плоские фичи
    flat_features = extract_simplified_features(sample_lyrics)
    print(f"\n📊 Всего извлечено фичей: {len(flat_features)}")
    
    return features


if __name__ == "__main__":
    demo_simplified_analysis()
