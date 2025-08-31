#!/usr/bin/env python3
"""
🔍 Просмотр доступных анализаторов
"""

import sys
import os
from pathlib import Path

# Добавляем путь к проекту
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

def list_available_analyzers():
    """Список всех доступных анализаторов"""
    
    print("🤖 Доступные анализаторы в системе:")
    print("=" * 50)
    
    try:
        # Импортируем фабрику анализаторов
        from interfaces.analyzer_interface import AnalyzerFactory
        
        # Принудительно импортируем все анализаторы для регистрации
        try:
            import analyzers.qwen_analyzer
            import analyzers.gemma_analyzer
            import analyzers.ollama_analyzer
            import analyzers.algorithmic_analyzer
            import analyzers.hybrid_analyzer
        except ImportError as e:
            print(f"⚠️ Не удалось импортировать некоторые анализаторы: {e}")
        
        # Получаем все зарегистрированные анализаторы
        available_analyzers = AnalyzerFactory.list_available()
        
        if not available_analyzers:
            print("❌ Анализаторы не найдены")
            return
        
        for i, analyzer_name in enumerate(available_analyzers, 1):
            print(f"{i}. 📊 {analyzer_name}")
            
            try:
                # Пытаемся создать анализатор и получить информацию
                analyzer = AnalyzerFactory.create(analyzer_name)
                info = analyzer.get_analyzer_info()
                
                print(f"   🔧 Тип: {info.get('type', 'Unknown')}")
                print(f"   📝 Описание: {info.get('description', 'No description')}")
                print(f"   ✅ Доступен: {'Да' if getattr(analyzer, 'available', True) else 'Нет'}")
                
                if hasattr(analyzer, 'model_name'):
                    print(f"   🧠 Модель: {analyzer.model_name}")
                
            except Exception as e:
                print(f"   ❌ Ошибка инициализации: {e}")
            
            print()
    
    except Exception as e:
        print(f"❌ Ошибка загрузки фабрики: {e}")
        return
    
    print("\n💡 Использование:")
    print("   python scripts/analyze_with_model.py --model <название>")

def test_specific_analyzers():
    """Тестирует конкретные анализаторы"""
    
    print("\n🧪 Тестирование основных анализаторов:")
    print("=" * 50)
    
    test_analyzers = ['qwen', 'gemma', 'ollama', 'algorithmic_basic']
    
    for analyzer_name in test_analyzers:
        print(f"\n🔍 Тестируем {analyzer_name}:")
        
        try:
            from interfaces.analyzer_interface import AnalyzerFactory
            
            if analyzer_name in AnalyzerFactory.list_available():
                analyzer = AnalyzerFactory.create(analyzer_name)
                
                # Проверяем доступность
                if hasattr(analyzer, 'available'):
                    status = "✅ Доступен" if analyzer.available else "❌ Недоступен"
                else:
                    status = "✅ Доступен (предполагается)"
                
                print(f"   Статус: {status}")
                
                # Показываем дополнительную информацию
                if hasattr(analyzer, 'model_name'):
                    print(f"   Модель: {analyzer.model_name}")
                
                if hasattr(analyzer, 'api_url'):
                    print(f"   API: {analyzer.api_url}")
                
            else:
                print(f"   ❌ Не зарегистрирован")
                
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")

if __name__ == "__main__":
    list_available_analyzers()
    test_specific_analyzers()
