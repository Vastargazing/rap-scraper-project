"""
Тест новой архитектуры анализаторов.

Демонстрация работы с новыми интерфейсами и алгоритмическим анализатором.
"""

import sys
from pathlib import Path

# Добавляем корневую папку проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core import create_app, AppContext
from src.interfaces import AnalyzerFactory
from src.analyzers import AlgorithmicAnalyzer


def test_basic_architecture():
    """Тест базовой архитектуры"""
    print("🔧 Тестирование новой архитектуры...")
    
    # Тест конфигурации
    with AppContext() as app:
        print(f"✅ Приложение инициализировано: {app.config.project_name} v{app.config.version}")
        
        # Тест базы данных
        db = app.get_database()
        stats = db.get_analysis_stats()
        print(f"✅ База данных подключена: {stats['total_songs']} песен")
        
        # Тест фабрики анализаторов
        available_analyzers = app.list_analyzers()
        print(f"✅ Доступные анализаторы: {available_analyzers}")
        
        # Тест создания анализатора
        if "algorithmic_basic" in available_analyzers:
            analyzer = app.get_analyzer("algorithmic_basic")
            print(f"✅ Анализатор создан: {analyzer.name}")
            
            # Тест анализа
            test_lyrics = """
            I wake up every morning feeling blessed
            Life is good, I'm doing my best
            Money in my pocket, dreams in my head
            Working hard until the day I'm dead
            
            Sometimes the world seems dark and cold
            But I keep fighting, I'm brave and bold
            Love conquers all, that's what I believe
            Never give up, always achieve
            """
            
            try:
                result = analyzer.analyze_song(
                    artist="Test Artist",
                    title="Test Song", 
                    lyrics=test_lyrics
                )
                
                print(f"✅ Анализ выполнен успешно!")
                print(f"   Тип анализа: {result.analysis_type}")
                print(f"   Уверенность: {result.confidence:.3f}")
                print(f"   Время обработки: {result.processing_time:.3f}s")
                
                # Показываем некоторые результаты
                sentiment = result.raw_output.get("sentiment_analysis", {})
                if sentiment:
                    print(f"   Настроение: {sentiment.get('sentiment_label')} (score: {sentiment.get('sentiment_score')})")
                
                complexity = result.raw_output.get("complexity_analysis", {})
                if complexity:
                    print(f"   Читабельность: {complexity.get('readability_score')}")
                    print(f"   Уникальных слов: {complexity.get('unique_words')}")
                
                themes = result.raw_output.get("themes_analysis", {})
                if themes:
                    print(f"   Главная тема: {themes.get('dominant_theme')}")
                
            except Exception as e:
                print(f"❌ Ошибка анализа: {e}")
                return False
        
        else:
            print("⚠️ Алгоритмический анализатор не найден")
            return False
    
    print("🎉 Все тесты прошли успешно!")
    return True


def test_analyzer_info():
    """Тест получения информации об анализаторах"""
    print("\n📋 Информация об анализаторах:")
    
    try:
        available = AnalyzerFactory.list_available()
        
        for analyzer_name in available:
            info = AnalyzerFactory.get_analyzer_info(analyzer_name)
            print(f"\n🔍 {analyzer_name}:")
            print(f"   Название: {info.get('name')}")
            print(f"   Версия: {info.get('version')}")
            print(f"   Тип: {info.get('type')}")
            print(f"   Описание: {info.get('description')}")
            print(f"   Функции: {', '.join(info.get('supported_features', []))}")
    
    except Exception as e:
        print(f"❌ Ошибка получения информации: {e}")
        return False
    
    return True


def test_database_operations():
    """Тест операций с базой данных"""
    print("\n💾 Тестирование операций с базой данных...")
    
    with AppContext() as app:
        db = app.get_database()
        
        try:
            # Тест вставки тестовой песни
            song_id = db.insert_song(
                artist="Test Artist",
                title="Test Song for Architecture", 
                lyrics="This is a test song for testing the new architecture"
            )
            
            if song_id:
                print(f"✅ Тестовая песня добавлена с ID: {song_id}")
                
                # Тест получения песни
                song = db.get_song_by_id(song_id)
                if song:
                    print(f"✅ Песня найдена: {song['artist']} - {song['title']}")
                else:
                    print("❌ Песня не найдена после вставки")
                    return False
            else:
                print("✅ Песня уже существует (это нормально)")
            
            # Тест статистики
            stats = db.get_analysis_stats()
            print(f"✅ Статистика БД:")
            print(f"   Всего песен: {stats['total_songs']}")
            print(f"   С текстами: {stats['songs_with_lyrics']}")
            print(f"   Анализы: {stats.get('analysis_coverage', {})}")
            
        except Exception as e:
            print(f"❌ Ошибка операций с БД: {e}")
            return False
    
    return True


if __name__ == "__main__":
    print("🚀 Запуск тестов новой архитектуры rap-scraper-project\n")
    
    # Запуск всех тестов
    tests = [
        test_basic_architecture,
        test_analyzer_info,
        test_database_operations
    ]
    
    passed = 0
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                break
        except Exception as e:
            print(f"❌ Критическая ошибка в {test_func.__name__}: {e}")
            break
    
    print(f"\n📊 Результаты: {passed}/{len(tests)} тестов пройдено")
    
    if passed == len(tests):
        print("🎉 Новая архитектура работает корректно!")
    else:
        print("⚠️ Требуются исправления в архитектуре")
        sys.exit(1)
