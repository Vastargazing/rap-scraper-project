#!/usr/bin/env python3
"""
Финальная демонстрация confidence scores в продакшене

Показывает практическое применение confidence scores для оценки качества ML-анализа
на реальных данных из базы.
"""

import json

def analyze_confidence_results():
    """Анализ результатов с confidence scores"""
    
    # Читаем результаты
    with open('confidence_test_updated.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("🎯 АНАЛИЗ CONFIDENCE SCORES НА РЕАЛЬНЫХ ДАННЫХ")
    print("=" * 60)
    
    for result in data['results']:
        artist = result['artist']
        title = result['title']
        features = result['features']
        
        print(f"\n🎵 {artist} - {title}")
        print("-" * 40)
        
        # Извлекаем confidence scores
        confidences = {
            'rhyme_detection': features.get('rhyme_detection_confidence', 0),
            'rhyme_scheme': features.get('rhyme_scheme_confidence', 0),
            'metaphor': features.get('metaphor_confidence', 0),
            'wordplay': features.get('wordplay_confidence', 0),
            'creativity': features.get('creativity_confidence', 0),
            'stress_pattern': features.get('stress_pattern_confidence', 0),
            'flow_analysis': features.get('flow_analysis_confidence', 0),
        }
        
        # Средний confidence
        avg_confidence = sum(confidences.values()) / len(confidences)
        
        print(f"📊 CONFIDENCE SCORES:")
        for metric, score in confidences.items():
            status = "🟢" if score >= 0.7 else "🟡" if score >= 0.5 else "🔴"
            print(f"   {status} {metric.replace('_', ' ').title()}: {score:.3f}")
        
        print(f"\n🎯 OVERALL CONFIDENCE: {avg_confidence:.3f}")
        
        # Рекомендации на основе confidence
        recommendations = []
        
        if confidences['metaphor'] < 0.5:
            recommendations.append("⚠️  Низкая уверенность в детекции метафор - требуется ручная проверка")
        
        if confidences['wordplay'] < 0.5:
            recommendations.append("⚠️  Игра слов может быть недооценена - сложная детекция")
        
        if confidences['rhyme_detection'] < 0.7:
            recommendations.append("⚠️  Рифмы могут быть пропущены - проверьте сложные схемы")
        
        if avg_confidence >= 0.7:
            print("✅ ВЫСОКАЯ НАДЁЖНОСТЬ - Результаты можно использовать для ML")
        elif avg_confidence >= 0.5:
            print("⚠️  СРЕДНЯЯ НАДЁЖНОСТЬ - Рекомендуется дополнительная валидация")
        else:
            print("❌ НИЗКАЯ НАДЁЖНОСТЬ - Требуется ручная проверка")
        
        if recommendations:
            print("\n💡 РЕКОМЕНДАЦИИ:")
            for rec in recommendations:
                print(f"   {rec}")
        
        # Показываем конкретные значения метрик
        print(f"\n📈 КЛЮЧЕВЫЕ МЕТРИКИ:")
        print(f"   Metaphor Count: {features.get('metaphor_count', 0)} (confidence: {confidences['metaphor']:.3f})")
        print(f"   Wordplay Instances: {features.get('wordplay_instances', 0)} (confidence: {confidences['wordplay']:.3f})")
        print(f"   Rhyme Density: {features.get('rhyme_density', 0):.3f} (confidence: {confidences['rhyme_detection']:.3f})")
        
        print("\n" + "=" * 60)

def show_best_practices():
    """Лучшие практики использования confidence scores"""
    
    print("\n🏆 ЛУЧШИЕ ПРАКТИКИ ИСПОЛЬЗОВАНИЯ CONFIDENCE SCORES:")
    print("-" * 50)
    
    practices = [
        "🎯 Confidence >= 0.8: Автоматическое принятие результатов",
        "🔍 Confidence 0.5-0.8: Селективная ручная проверка",
        "⚠️  Confidence < 0.5: Обязательная валидация или исключение",
        "📊 Для ML-моделей: Используйте confidence как feature weight",
        "🔄 Для active learning: Фокусируйтесь на низких confidence",
        "📈 Для reporting: Всегда показывайте confidence вместе с метриками"
    ]
    
    for practice in practices:
        print(f"   {practice}")
    
    print(f"\n💡 ПРИМЕНЕНИЕ В ML PIPELINE:")
    print(f"   • Фильтрация данных по минимальному confidence")
    print(f"   • Weighted loss functions на основе confidence")
    print(f"   • Uncertainty-aware predictions")
    print(f"   • Автоматическое лабелирование с high confidence")

if __name__ == "__main__":
    try:
        analyze_confidence_results()
        show_best_practices()
        
        print(f"\n🎉 ЗАДАЧА ВЫПОЛНЕНА УСПЕШНО!")
        print(f"✅ Добавлены confidence scores для:")
        print(f"   • Rhyme detection & scheme analysis")
        print(f"   • Metaphor & wordplay detection")
        print(f"   • Flow & stress pattern analysis")
        print(f"   • Creativity assessment")
        print(f"\n📦 Интегрировано в:")
        print(f"   • SimplifiedFeatureAnalyzer")
        print(f"   • CLI экспорт (JSON/CSV)")
        print(f"   • Pydantic модели данных")
        
    except FileNotFoundError:
        print("❌ Файл confidence_test_updated.json не найден")
        print("Запустите сначала: python scripts/rap_scraper_cli.py mlfeatures --batch 2 --export json --output confidence_test_updated.json")
