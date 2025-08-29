"""
Тест новых анализаторов после рефакторинга multi_model_analyzer.py.

Проверяет работу всех созданных анализаторов: Algorithmic, Gemma, Ollama, Hybrid.
"""

import sys
from pathlib import Path

# Добавляем корневую папку проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core import create_app, AppContext
from src.interfaces import AnalyzerFactory
from src.analyzers import AlgorithmicAnalyzer, GemmaAnalyzer, OllamaAnalyzer, HybridAnalyzer


def test_all_analyzers():
    """Тест всех доступных анализаторов"""
    print("🔬 Тестирование всех анализаторов после рефакторинга...\n")
    
    test_lyrics = """
    Started from the bottom now we here
    Working every day to make it clear
    Money on my mind, success in sight
    Grinding through the struggle, day and night
    
    They said I'd never make it to the top
    But I kept pushing, never gonna stop
    Dreams become reality when you believe
    Hard work and dedication help you achieve
    
    Now I'm living life like I imagined
    All the pain and struggle, now it's happened
    From the streets to the studio booth
    Speaking my reality, nothing but truth
    """
    
    with AppContext() as app:
        available_analyzers = app.list_analyzers()
        print(f"📋 Доступные анализаторы: {available_analyzers}\n")
        
        results = {}
        
        for analyzer_name in available_analyzers:
            print(f"🔍 Тестирование {analyzer_name}...")
            
            try:
                analyzer = app.get_analyzer(analyzer_name)
                
                # Получаем информацию об анализаторе
                info = analyzer.get_analyzer_info()
                print(f"   Тип: {info['type']}")
                print(f"   Описание: {info['description']}")
                print(f"   Доступен: {info.get('available', 'unknown')}")
                
                # Если анализатор недоступен, пропускаем
                if not info.get('available', True):
                    print(f"   ⚠️ Анализатор недоступен: {info.get('requirements', 'Unknown reason')}")
                    continue
                
                # Выполняем анализ
                result = analyzer.analyze_song(
                    artist="Test Artist",
                    title="Success Story", 
                    lyrics=test_lyrics
                )
                
                results[analyzer_name] = result
                
                print(f"   ✅ Анализ завершен:")
                print(f"      Уверенность: {result.confidence:.3f}")
                print(f"      Время: {result.processing_time:.3f}s")
                print(f"      Функции: {len(analyzer.supported_features)}")
                
                # Показываем ключевые результаты
                _show_key_results(analyzer_name, result)
                
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
                if "not available" in str(e) or "недоступен" in str(e):
                    print(f"   💡 Совет: проверьте настройки и зависимости для {analyzer_name}")
                
            print()
        
        # Сравнение результатов
        if len(results) > 1:
            print("📊 Сравнение результатов анализаторов:")
            _compare_results(results)
        
        return len(results) > 0


def _show_key_results(analyzer_name: str, result):
    """Показать ключевые результаты анализа"""
    raw_output = result.raw_output
    
    if analyzer_name == "algorithmic_basic":
        sentiment = raw_output.get("sentiment_analysis", {})
        complexity = raw_output.get("complexity_analysis", {})
        
        if sentiment:
            print(f"      Настроение: {sentiment.get('sentiment_label')} (score: {sentiment.get('sentiment_score', 0):.2f})")
        if complexity:
            print(f"      Читабельность: {complexity.get('readability_score', 0):.1f}")
    
    elif analyzer_name == "gemma":
        genre = raw_output.get("genre_analysis", {})
        mood = raw_output.get("mood_analysis", {}) 
        quality = raw_output.get("quality_metrics", {})
        
        if genre:
            print(f"      Жанр: {genre.get('primary_genre', 'unknown')}")
        if mood:
            print(f"      Настроение: {mood.get('primary_mood', 'unknown')}")
        if quality:
            print(f"      Качество: {quality.get('overall_quality', 0):.2f}")
    
    elif analyzer_name == "ollama":
        basic = raw_output.get("basic_analysis", {})
        quality = raw_output.get("quality_assessment", {})
        
        if basic:
            print(f"      Жанр: {basic.get('genre', 'unknown')}")
            print(f"      Настроение: {basic.get('mood', 'unknown')}")
        if quality:
            print(f"      Общее качество: {quality.get('overall_quality', 0):.2f}")
    
    elif analyzer_name == "hybrid":
        basic_info = raw_output.get("basic_info", {})
        quality = raw_output.get("quality_metrics", {})
        meta = result.metadata
        
        if basic_info:
            print(f"      Консенсус жанр: {basic_info.get('genre', 'unknown')}")
            print(f"      Консенсус настроение: {basic_info.get('mood', 'unknown')}")
        
        analyzers_used = meta.get("analyzers_used", [])
        print(f"      Использованы: {', '.join(analyzers_used)}")
        
        if meta.get("fallback_used"):
            print(f"      🔄 Использован fallback")


def _compare_results(results: dict):
    """Сравнение результатов разных анализаторов"""
    print("\n🔍 Анализ консенсуса:")
    
    # Сравнение уверенности
    confidences = {name: result.confidence for name, result in results.items()}
    print(f"   Уверенность анализаторов:")
    for name, conf in sorted(confidences.items(), key=lambda x: x[1], reverse=True):
        print(f"      {name}: {conf:.3f}")
    
    # Сравнение времени обработки
    times = {name: result.processing_time for name, result in results.items()}
    print(f"   Время обработки:")
    for name, time_val in sorted(times.items(), key=lambda x: x[1]):
        print(f"      {name}: {time_val:.3f}s")
    
    # Поиск настроения из разных анализаторов
    moods = {}
    for name, result in results.items():
        raw = result.raw_output
        
        if name == "algorithmic_basic":
            mood = raw.get("sentiment_analysis", {}).get("sentiment_label")
        elif name == "gemma":
            mood = raw.get("mood_analysis", {}).get("primary_mood")
        elif name == "ollama":
            mood = raw.get("basic_analysis", {}).get("mood")
        elif name == "hybrid":
            mood = raw.get("basic_info", {}).get("mood")
        else:
            mood = None
        
        if mood:
            moods[name] = mood
    
    if moods:
        print(f"   Определенное настроение:")
        for name, mood in moods.items():
            print(f"      {name}: {mood}")
        
        # Проверка консенсуса
        unique_moods = set(moods.values())
        if len(unique_moods) == 1:
            print(f"   ✅ Полный консенсус по настроению: {list(unique_moods)[0]}")
        else:
            print(f"   ⚠️ Разногласия в настроении: {unique_moods}")


def test_analyzer_factory():
    """Тест фабрики анализаторов"""
    print("\n🏭 Тестирование фабрики анализаторов:")
    
    # Получение списка анализаторов
    available = AnalyzerFactory.list_available()
    print(f"   Зарегистрированные анализаторы: {available}")
    
    # Тест создания синглтонов
    for analyzer_name in available:
        try:
            analyzer1 = AnalyzerFactory.create(analyzer_name, singleton=True)
            analyzer2 = AnalyzerFactory.create(analyzer_name, singleton=True)
            
            is_singleton = analyzer1 is analyzer2
            print(f"   {analyzer_name}: singleton={'✅' if is_singleton else '❌'}")
            
        except Exception as e:
            print(f"   {analyzer_name}: ❌ ошибка создания - {e}")
    
    return True


def test_configuration():
    """Тест кастомной конфигурации анализаторов"""
    print("\n⚙️ Тестирование кастомной конфигурации:")
    
    try:
        # Создание алгоритмического анализатора с кастомными настройками
        custom_config = {
            'min_word_length': 4,
            'sentiment_threshold': 0.2
        }
        
        analyzer = AnalyzerFactory.create(
            "algorithmic_basic", 
            config=custom_config, 
            singleton=False
        )
        
        print(f"   ✅ Кастомная конфигурация применена")
        print(f"      min_word_length: {analyzer.config.get('min_word_length')}")
        print(f"      sentiment_threshold: {analyzer.config.get('sentiment_threshold')}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Ошибка кастомной конфигурации: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Тестирование новых анализаторов после рефакторинга\n")
    
    # Запуск всех тестов
    tests = [
        ("Анализаторы", test_all_analyzers),
        ("Фабрика", test_analyzer_factory), 
        ("Конфигурация", test_configuration)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Тест: {test_name}")
        print('='*50)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: ПРОЙДЕН")
            else:
                print(f"❌ {test_name}: НЕ ПРОЙДЕН")
        except Exception as e:
            print(f"❌ {test_name}: КРИТИЧЕСКАЯ ОШИБКА - {e}")
    
    print(f"\n📊 Итоговые результаты: {passed}/{len(tests)} тестов пройдено")
    
    if passed == len(tests):
        print("🎉 Все анализаторы работают корректно!")
        print("\n💡 Рекомендации:")
        print("   - Для production используйте 'hybrid' анализатор")
        print("   - Для экспериментов подходит 'ollama' (локально)")
        print("   - Для быстрого анализа - 'algorithmic_basic'")
        print("   - Для качественного AI анализа - 'gemma'")
    else:
        print("⚠️ Некоторые анализаторы требуют настройки")
        print("\n🔧 Возможные проблемы:")
        print("   - Ollama: убедитесь что сервер запущен (ollama serve)")
        print("   - Gemma: проверьте GOOGLE_API_KEY в переменных окружения")
        print("   - Установите зависимости: pip install google-generativeai")
