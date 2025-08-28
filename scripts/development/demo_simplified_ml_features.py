#!/usr/bin/env python3
"""
Упрощенная демонстрация новых ML-фичей (без NLTK)

Демонстрирует базовые возможности Feature Engineering:
- Rhyme density и схемы рифмовки  
- Vocabulary diversity (TTR)
- Metaphor/wordplay detection (упрощенно)
- Flow patterns (базовый анализ)
"""

import sys
import json
import time
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.analyzers.simplified_feature_analyzer import (
    SimplifiedFeatureAnalyzer, 
    extract_simplified_features,
    demo_simplified_analysis
)

def demo_rhyme_analysis():
    """Демонстрация анализа рифм"""
    print("🎵 === ДЕМОНСТРАЦИЯ АНАЛИЗА РИФМ ===\n")
    
    rhyme_samples = {
        "Простые рифмы (AABB)": """
            Я иду по дороге домой
            Со мной моя любовь и покой
            В кармане звенят монеты  
            Как песни весенние ветра
        """,
        
        "Сложные рифмы (ABAB)": """
            Время течёт как река бесконечная
            В ритме сердца стучит мой флоу
            Жизнь пролетает стрелой быстротечною
            Я остаюсь здесь назло всему злу
        """,
        
        "Внутренние рифмы": """
            Бит качает, сердце скачет, значит всё нормально
            Рифма к рифме, строчка к строчке, складывается грамотно
            В каждом слове есть история, в каждой строчке — правда
            Это рэп из глубины души, а не пустая забава
        """
    }
    
    analyzer = SimplifiedFeatureAnalyzer()
    
    for sample_name, lyrics in rhyme_samples.items():
        print(f"📝 {sample_name}:")
        features = extract_simplified_features(lyrics.strip())
        
        print(f"   Плотность рифм: {features.get('rhyme_density', 0):.3f}")
        print(f"   Точные рифмы: {features.get('perfect_rhymes', 0):.0f}")
        print(f"   Внутренние рифмы: {features.get('internal_rhymes', 0):.0f}")
        print(f"   Аллитерация: {features.get('alliteration_score', 0):.3f}")
        print()

def demo_vocabulary_diversity():
    """Демонстрация анализа разнообразия словаря"""
    print("📚 === ДЕМОНСТРАЦИЯ АНАЛИЗА СЛОВАРЯ ===\n")
    
    vocab_samples = {
        "Базовый словарь": """
            Я иду домой быстро
            Дом мой дом хороший
            Хорошо дома быстро
            Быстро иду дом хорошо
        """,
        
        "Богатый словарь": """
            Блуждаю по лабиринтам мегаполиса современного
            Архитектура сознания формирует восприятие реальности
            Интеллектуальные конструкции переплетаются с эмоциями
            Создавая симфонию мыслей в хаосе урбанистической среды
        """,
        
        "Средний уровень": """
            Города меняются, люди остаются прежними
            История повторяется, но в новых декорациях
            Мудрость поколений передается через музыку
            Ритм жизни ускоряется, но смысл остается вечным
        """
    }
    
    for sample_name, lyrics in vocab_samples.items():
        print(f"📖 {sample_name}:")
        features = extract_simplified_features(lyrics.strip())
        
        print(f"   TTR Score: {features.get('ttr_score', 0):.3f}")
        print(f"   Средняя длина слова: {features.get('average_word_length', 0):.1f}")
        print(f"   Доля сложных слов: {features.get('complex_words_ratio', 0):.3f}")
        print()

def demo_metaphor_detection():
    """Демонстрация обнаружения метафор"""
    print("🎨 === ДЕМОНСТРАЦИЯ АНАЛИЗА МЕТАФОР ===\n")
    
    metaphor_samples = {
        "Много метафор": """
            Жизнь как шахматная партия с судьбой
            Мои слова как пули в войне за правду
            Сердце горит как огонь в ночи
            Время течёт как река бесконечная
            Душа парит как птица в небесах
        """,
        
        "Игра слов": """
            Money talks, но я говорю на языке денег
            Flow как река, но я плыву против течения
            Bars как тюрьма, но я свободен в своих рифмах
            Beat как сердце, и я чувствую его пульс
            Words как оружие, и я мастер боя
        """,
        
        "Простой текст": """
            Я иду в магазин купить хлеб
            Встречаю друга на улице
            Мы говорим о погоде
            Потом расходимся по домам
        """
    }
    
    for sample_name, lyrics in metaphor_samples.items():
        print(f"🎭 {sample_name}:")
        features = extract_simplified_features(lyrics.strip())
        
        print(f"   Метафоры: {features.get('metaphor_count', 0):.0f}")
        print(f"   Игра слов: {features.get('wordplay_instances', 0):.0f}")
        print(f"   Креативность: {features.get('creativity_score', 0):.3f}")
        print()

def demo_flow_analysis():
    """Демонстрация анализа флоу"""
    print("🎼 === ДЕМОНСТРАЦИЯ АНАЛИЗА ФЛОУ ===\n")
    
    flow_samples = {
        "Ровный флоу": """
            Каждая строчка звучит одинаково
            Ритм не меняется никогда совсем
            Слоги ложатся ровно и красиво
            Это простой и понятный флоу
        """,
        
        "Сложный флоу": """
            Быстро-медленно, тихо-громко — это игра!
            Синкопы, паузы... (стоп) — начинаем с нуля
            Слоги прыгают: раз-два-три-четыре-пять
            А потом — длинная пауза... чтобы дать подумать
        """,
        
        "Переменный темп": """
            Медленно начинаю свой рассказ о том
            Как быстробыстробыстро может звучать рэп
            Потом... опять... замедляюсь... и снова
            Ускоряюсь-ускоряюсь-до-предела!
        """
    }
    
    for sample_name, lyrics in flow_samples.items():
        print(f"🎵 {sample_name}:")
        features = extract_simplified_features(lyrics.strip())
        
        print(f"   Слогов на строку: {features.get('average_syllables_per_line', 0):.1f}")
        print(f"   Консистентность: {features.get('stress_pattern_consistency', 0):.3f}")
        print(f"   Паузы во флоу: {features.get('flow_breaks', 0):.0f}")
        print()

def demo_composite_metrics():
    """Демонстрация композитных метрик"""
    print("🏆 === КОМПОЗИТНЫЕ МЕТРИКИ МАСТЕРСТВА ===\n")
    
    skill_samples = {
        "Начинающий": """
            Я рэпер новый
            Пишу простые тексты
            Рифмы не очень
            Но стараюсь лучше
        """,
        
        "Опытный": """
            Мастерство оттачивал годами в студии звукозаписи
            Рифмы сложные плету, как паутину искусно  
            Метафоры и аллегории в каждой строчке прописаны
            Флоу меняется плавно, техника отработана чисто
        """,
        
        "Мастер": """
            Архитектор лирических конструкций сложносочинённых
            Синестезия слов рождает симфонии многослойные
            Интеллектуальные каламбуры переплетаются искусно
            С социальной философией современности постмодернистской
        """
    }
    
    for sample_name, lyrics in skill_samples.items():
        print(f"🎯 {sample_name}:")
        features = extract_simplified_features(lyrics.strip())
        
        print(f"   Общая сложность: {features.get('overall_complexity', 0):.3f}")
        print(f"   Техническое мастерство: {features.get('technical_skill', 0):.3f}")
        print(f"   Художественность: {features.get('artistic_sophistication', 0):.3f}")
        print(f"   Инновационность: {features.get('innovation_score', 0):.3f}")
        
        # Простая интерпретация
        technical_level = features.get('technical_skill', 0)
        if technical_level > 0.7:
            level_desc = "Expert"
        elif technical_level > 0.5:
            level_desc = "Skilled"
        else:
            level_desc = "Developing"
        
        print(f"   Уровень мастерства: {level_desc}")
        print()

def demo_performance():
    """Демонстрация производительности"""
    print("⚡ === ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ ===\n")
    
    test_lyrics = """
    Это тестовый текст для проверки скорости работы
    Алгоритма извлечения фичей из рэп-текстов в реальном времени
    Рифмы, метафоры, флоу и словарное разнообразие
    Анализируются быстро и эффективно для ML pipeline
    """
    
    # Тест одиночного извлечения
    start_time = time.time()
    features = extract_simplified_features(test_lyrics)
    single_time = time.time() - start_time
    
    print(f"⏱️  Время извлечения фичей (1 текст): {single_time:.4f}с")
    print(f"📊 Количество извлеченных фичей: {len(features)}")
    
    # Тест пакетного извлечения
    lyrics_list = [test_lyrics] * 100
    start_time = time.time()
    for lyrics in lyrics_list:
        extract_simplified_features(lyrics)
    batch_time = time.time() - start_time
    
    print(f"⏱️  Время пакетного извлечения (100 текстов): {batch_time:.4f}с")
    print(f"📈 Средняя скорость: {batch_time/100:.4f}с на текст")
    print(f"🚀 Потенциальная обработка: ~{int(3600/batch_time*100)} текстов/час")
    print()

def save_sample_features():
    """Сохранение примера фичей"""
    print("💾 === СОХРАНЕНИЕ ПРИМЕРА ФИЧЕЙ ===\n")
    
    sample_lyrics = """
    Я поднимаюсь над городом как солнце рассветное
    Мои рифмы — это пули, летящие в цель метко
    В лабиринте из слов я нашёл дорогу к свету
    Флоу льётся как река, несёт меня к победе
    
    Время — деньги, но мудрость дороже золота
    В игре теней я создаю новую эпоху
    Слова танцуют на битах, рождая магию звука
    Это искусство rap'а — моя вечная наука
    """
    
    features = extract_simplified_features(sample_lyrics)
    
    # Создаем полный набор данных
    sample_data = {
        "lyrics": sample_lyrics.strip(),
        "extracted_features": features,
        "feature_count": len(features),
        "extraction_info": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "analyzer_version": "simplified_v1.0",
            "method": "basic_nlp_without_external_libs"
        }
    }
    
    # Сохраняем в файл
    output_file = "sample_simplified_ml_features.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Упрощенные фичи сохранены в: {output_file}")
    print(f"📊 Всего фичей: {len(features)}")
    print("🔍 Основные метрики:")
    for key, value in features.items():
        if 'score' in key or 'complexity' in key or 'skill' in key:
            print(f"   {key}: {value:.3f}")
    print()

def create_feature_comparison():
    """Сравнение различных типов текстов"""
    print("📊 === СРАВНИТЕЛЬНЫЙ АНАЛИЗ ===\n")
    
    comparison_samples = {
        "Коммерческий поп-рэп": """
            Party party party всю ночь до утра
            Money money money это моя игра
            Танцуй танцуй танцуй пока молодой
            Живи живи живи будто последний бой
        """,
        
        "Социальный рэп": """
            Улицы говорят правду сквозь асфальт и бетон
            Поколение потерянных ищет свой дом
            В системе координат где деньги решают всё
            Мы пишем манифест нового времени своё
        """,
        
        "Экспериментальный рэп": """
            Диссонанс мыслительных парадигм в урбанистической среде
            Фрагментация сознания через призму постмодернистской эстетики
            Интертекстуальность нарративов в контексте социокультурных трансформаций
            Деконструкция лингвистических структур в пространстве художественного дискурса
        """
    }
    
    comparison_results = {}
    
    for style_name, lyrics in comparison_samples.items():
        features = extract_simplified_features(lyrics.strip())
        comparison_results[style_name] = features
        
        print(f"🎨 {style_name}:")
        print(f"   Техническое мастерство: {features.get('technical_skill', 0):.3f}")
        print(f"   Художественность: {features.get('artistic_sophistication', 0):.3f}")
        print(f"   Инновационность: {features.get('innovation_score', 0):.3f}")
        print(f"   TTR (разнообразие): {features.get('ttr_score', 0):.3f}")
        print(f"   Плотность рифм: {features.get('rhyme_density', 0):.3f}")
        print()
    
    # Сохраняем сравнение
    with open("style_comparison.json", 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)
    
    print("💾 Сравнительный анализ сохранен в style_comparison.json")

def main():
    """Основная функция демонстрации"""
    print("🎤 === ДЕМОНСТРАЦИЯ РАСШИРЕННЫХ ML-ФИЧЕЙ (Упрощенная версия) ===\n")
    print("Новые возможности Feature Engineering:")
    print("✅ Rhyme density и схемы рифмовки (базовый анализ)")
    print("✅ Vocabulary diversity (TTR - Type-Token Ratio)")  
    print("✅ Metaphor/wordplay detection (ключевые слова)")
    print("✅ Flow patterns (анализ ритма)")
    print("✅ Композитные метрики мастерства")
    print("🔧 Версия: упрощенная (без NLTK зависимостей)")
    print("\n" + "="*60 + "\n")
    
    try:
        # Запускаем все демонстрации
        demo_rhyme_analysis()
        demo_vocabulary_diversity()
        demo_metaphor_detection()
        demo_flow_analysis()
        demo_composite_metrics()
        demo_performance()
        save_sample_features()
        create_feature_comparison()
        
        print("🎉 === ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА УСПЕШНО ===")
        print("\n📚 Что дальше:")
        print("1. Установите NLTK для полнофункциональной версии:")
        print("   pip install nltk")
        print("2. Используйте extract_simplified_features() для быстрого ML")
        print("3. Интегрируйте в основной pipeline:")
        print("   from src.analyzers.simplified_feature_analyzer import extract_simplified_features")
        print("4. Запустите пакетную обработку на полном датасете")
        
    except Exception as e:
        print(f"❌ Ошибка в демонстрации: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
