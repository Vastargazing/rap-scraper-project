"""
Простой пример использования новой архитектуры анализаторов.

Демонстрирует основные возможности рефакторированной системы.
"""

import sys
from pathlib import Path

# Добавляем корневую папку проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core import create_app, AppContext, get_config, load_config
from src.interfaces import AnalyzerFactory

def simple_analysis_example():
    """Простой пример анализа песни"""
    print("🎵 Простой пример анализа песни\n")
    
    # Пример текста песни
    lyrics = """
    I remember being broke, sleeping on the floor
    Now I got money but I still want more
    Success came fast but the pain stayed slow  
    That's the price of fame that they don't show
    
    Started from nothing, built my empire
    Every single bar is pure fire
    They say money changes people but I stayed the same
    Just a kid from the hood playing this game
    """
    
    # Создание приложения с кастомной конфигурацией
    with AppContext(config_file="config.json") as app:
        print(f"📱 Приложение: {app.config.project_name} v{app.config.version}")
        print(f"🗄️ База данных: {app.config.database.path}")
        print(f"📊 Доступные анализаторы: {app.list_analyzers()}\n")
        
        # Тестируем алгоритмический анализатор (всегда доступен)
        print("🔍 Анализ алгоритмическим методом:")
        try:
            algo_analyzer = app.get_analyzer("algorithmic_basic")
            result = algo_analyzer.analyze_song("Example Artist", "Success Story", lyrics)
            
            print(f"   ✅ Уверенность: {result.confidence:.3f}")
            print(f"   ⏱️ Время: {result.processing_time:.3f}s")
            
            # Показать ключевые результаты
            sentiment = result.raw_output.get("sentiment_analysis", {})
            complexity = result.raw_output.get("complexity_analysis", {})
            themes = result.raw_output.get("themes_analysis", {})
            
            if sentiment:
                print(f"   😊 Настроение: {sentiment.get('sentiment_label')} (score: {sentiment.get('sentiment_score', 0):.2f})")
            
            if complexity:
                print(f"   📚 Читабельность: {complexity.get('readability_score', 0):.1f}/100")
                print(f"   🔤 Уникальных слов: {complexity.get('unique_words', 0)}")
            
            if themes:
                dominant_theme = themes.get('dominant_theme', 'none')
                print(f"   🎯 Главная тема: {dominant_theme}")
            
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
        
        print()
        
        # Пробуем гибридный анализатор
        print("🔬 Попытка гибридного анализа:")
        try:
            hybrid_analyzer = app.get_analyzer("hybrid")
            result = hybrid_analyzer.analyze_song("Example Artist", "Success Story", lyrics)
            
            print(f"   ✅ Уверенность: {result.confidence:.3f}")
            print(f"   ⏱️ Время: {result.processing_time:.3f}s")
            
            # Показать какие анализаторы использовались
            analyzers_used = result.metadata.get("analyzers_used", [])
            print(f"   🔧 Использованы: {', '.join(analyzers_used)}")
            
            if result.metadata.get("fallback_used"):
                print(f"   🔄 Использован fallback анализатор")
            
            # Показать консенсус результатов
            basic_info = result.raw_output.get("basic_info", {})
            if basic_info:
                print(f"   🎵 Жанр: {basic_info.get('genre', 'unknown')}")
                print(f"   😊 Настроение: {basic_info.get('mood', 'unknown')}")
                if basic_info.get('mood_consensus'):
                    print(f"   ✅ Полный консенсус по настроению")
            
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")


def analyzer_info_example():
    """Пример получения информации об анализаторах"""
    print("\n📋 Информация о доступных анализаторах:\n")
    
    available_analyzers = AnalyzerFactory.list_available()
    
    for analyzer_name in available_analyzers:
        try:
            info = AnalyzerFactory.get_analyzer_info(analyzer_name)
            
            print(f"🔍 {analyzer_name}:")
            print(f"   Название: {info.get('name', 'Unknown')}")
            print(f"   Версия: {info.get('version', 'Unknown')}")
            print(f"   Тип: {info.get('type', 'Unknown')}")
            print(f"   Описание: {info.get('description', 'No description')}")
            print(f"   Доступен: {'✅' if info.get('available', True) else '❌'}")
            
            features = info.get('supported_features', [])
            if features:
                print(f"   Функции: {', '.join(features[:3])}{'...' if len(features) > 3 else ''}")
            
            requirements = info.get('requirements', [])
            if requirements:
                print(f"   Требования: {', '.join(requirements)}")
            
            print()
            
        except Exception as e:
            print(f"❌ Ошибка получения информации о {analyzer_name}: {e}\n")


def database_integration_example():
    """Пример интеграции с базой данных"""
    print("💾 Пример работы с базой данных:\n")
    
    with AppContext() as app:
        db = app.get_database()
        
        # Получаем статистику
        stats = db.get_analysis_stats()
        print(f"📊 Статистика базы данных:")
        print(f"   Всего песен: {stats['total_songs']:,}")
        print(f"   С текстами: {stats['songs_with_lyrics']:,}")
        print(f"   Проанализировано: {sum(stats['analysis_coverage'].values()) if stats['analysis_coverage'] else 0}")
        
        # Находим песни без анализа алгоритмическим методом
        unanalyzed = db.get_songs_without_analysis("algorithmic_basic", limit=3)
        
        if unanalyzed:
            print(f"\n🔍 Примеры песен без алгоритмического анализа:")
            for song in unanalyzed[:3]:
                print(f"   - {song['artist']} - {song['title']}")
                
                # Можем проанализировать прямо сейчас
                try:
                    analyzer = app.get_analyzer("algorithmic_basic")
                    result = analyzer.analyze_song(
                        song['artist'], 
                        song['title'], 
                        song['lyrics'][:500] + "..."  # Сокращаем для примера
                    )
                    
                    # Сохраняем результат
                    import json
                    success = db.save_analysis_result(
                        song['id'],
                        "algorithmic_basic",
                        json.dumps(result.raw_output),
                        result.confidence,
                        result.processing_time
                    )
                    
                    if success:
                        print(f"     ✅ Анализ сохранен (уверенность: {result.confidence:.3f})")
                    
                except Exception as e:
                    print(f"     ❌ Ошибка анализа: {e}")
        
        # Обновленная статистика
        updated_stats = db.get_analysis_stats()
        analysis_count = sum(updated_stats['analysis_coverage'].values()) if updated_stats['analysis_coverage'] else 0
        print(f"\n📈 Обновленная статистика: {analysis_count} проанализированных песен")


def configuration_example():
    """Пример работы с конфигурацией"""
    print("\n⚙️ Пример работы с конфигурацией:\n")
    
    # Загружаем конфигурацию из файла
    config = load_config(config_file="config.json")
    
    print(f"📱 Проект: {config.project_name} v{config.version}")
    print(f"🌍 Окружение: {config.environment}")
    print(f"🗄️ База данных: {config.database.path}")
    print(f"📊 Размер батча: {config.analysis.batch_size}")
    print(f"👥 Макс. воркеров: {config.analysis.max_workers}")
    
    # Настройки анализаторов
    print(f"\n🔧 Настройки анализаторов:")
    analyzer_configs = getattr(config, 'analyzers', {})
    
    if hasattr(config, 'analyzers'):
        for name, settings in config.analyzers.items():
            print(f"   {name}:")
            for key, value in settings.items():
                print(f"     {key}: {value}")
    else:
        print("   Конфигурация анализаторов не найдена")


if __name__ == "__main__":
    print("🚀 Примеры использования новой архитектуры анализаторов\n")
    print("="*60)
    
    # Запуск примеров
    try:
        simple_analysis_example()
        print("="*60)
        
        analyzer_info_example()
        print("="*60)
        
        database_integration_example()
        print("="*60)
        
        configuration_example()
        print("="*60)
        
        print("\n🎉 Все примеры выполнены успешно!")
        print("\n💡 Следующие шаги:")
        print("   1. Настройте API ключи для AI анализаторов")
        print("   2. Запустите Ollama для локального анализа")
        print("   3. Используйте гибридный анализатор для лучших результатов")
        print("   4. Настройте batch обработку для большого объема данных")
        
    except Exception as e:
        print(f"❌ Ошибка в примерах: {e}")
        import traceback
        traceback.print_exc()
