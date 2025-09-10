#!/usr/bin/env python3
"""
Демонстрация новых confidence scores в ML feature analysis

Показывает, как новые метрики уверенности помогают оценить надёжность анализа.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.analyzers.simplified_feature_analyzer import SimplifiedFeatureAnalyzer
import json

def test_confidence_scores():
    """Тестирование confidence scores на разных типах текстов"""
    
    analyzer = SimplifiedFeatureAnalyzer()
    
    # Тестовые тексты разного качества
    test_texts = {
        "high_quality": """
        Money on my mind, time to shine and grind
        Every single line designed to blow your mind
        Rhyme after rhyme, climbing every time
        Living life sublime, never falling behind
        """,
        
        "medium_quality": """
        I wake up in the morning feeling good
        Life is crazy but I do what I should
        Working hard in my neighborhood
        Everything is going as it could
        """,
        
        "low_quality": """
        Yo yo yo check it out
        I'm the best without a doubt
        Money cars and girls I shout
        That's what hip hop is about
        """,
        
        "complex_wordplay": """
        Intellectual spitting, my syntax is terrific
        Metaphysical with syllables that's hierarchical
        Lyrical miracle, spiritual, empirical
        Satirical and clerical, numerical and spherical
        """
    }
    
    print("🎤 CONFIDENCE SCORES DEMONSTRATION\n")
    print("=" * 60)
    
    for text_type, lyrics in test_texts.items():
        print(f"\n📝 Testing: {text_type.upper()}")
        print("-" * 40)
        
        # Анализируем
        features = analyzer.analyze_lyrics(lyrics)
        
        # Показываем ключевые метрики с confidence scores
        print(f"🎯 RHYME ANALYSIS:")
        print(f"   Rhyme Density: {features.rhyme_analysis.rhyme_density:.3f}")
        print(f"   ✅ Rhyme Detection Confidence: {features.rhyme_analysis.rhyme_detection_confidence:.3f}")
        print(f"   Rhyme Scheme: {features.rhyme_analysis.end_rhyme_scheme}")
        print(f"   ✅ Rhyme Scheme Confidence: {features.rhyme_analysis.rhyme_scheme_confidence:.3f}")
        
        print(f"\n🎭 METAPHOR ANALYSIS:")
        print(f"   Metaphor Count: {features.metaphor_analysis.metaphor_count}")
        print(f"   ✅ Metaphor Confidence: {features.metaphor_analysis.metaphor_confidence:.3f}")
        print(f"   Wordplay Instances: {features.metaphor_analysis.wordplay_instances}")
        print(f"   ✅ Wordplay Confidence: {features.metaphor_analysis.wordplay_confidence:.3f}")
        print(f"   Creativity Score: {features.metaphor_analysis.creativity_score:.3f}")
        print(f"   ✅ Creativity Confidence: {features.metaphor_analysis.creativity_confidence:.3f}")
        
        print(f"\n🌊 FLOW ANALYSIS:")
        print(f"   Syllable Count: {features.flow_analysis.syllable_count}")
        print(f"   Stress Pattern Consistency: {features.flow_analysis.stress_pattern_consistency:.3f}")
        print(f"   ✅ Stress Pattern Confidence: {features.flow_analysis.stress_pattern_confidence:.3f}")
        print(f"   ✅ Flow Analysis Confidence: {features.flow_analysis.flow_analysis_confidence:.3f}")
        
        # Общая оценка надёжности
        avg_confidence = (
            features.rhyme_analysis.rhyme_detection_confidence +
            features.rhyme_analysis.rhyme_scheme_confidence +
            features.metaphor_analysis.metaphor_confidence +
            features.metaphor_analysis.wordplay_confidence +
            features.flow_analysis.stress_pattern_confidence +
            features.flow_analysis.flow_analysis_confidence
        ) / 6
        
        print(f"\n📊 OVERALL ANALYSIS CONFIDENCE: {avg_confidence:.3f}")
        
        if avg_confidence >= 0.7:
            print("   🟢 HIGH RELIABILITY - Results are trustworthy")
        elif avg_confidence >= 0.5:
            print("   🟡 MEDIUM RELIABILITY - Results may need validation")
        else:
            print("   🔴 LOW RELIABILITY - Results should be used with caution")
        
        print("\n" + "=" * 60)

def demonstrate_json_output():
    """Показать JSON вывод с confidence scores"""
    
    analyzer = SimplifiedFeatureAnalyzer()
    
    sample_lyrics = """
    Metaphorical bars, I'm spitting fire like a dragon
    Internal rhymes align, my flow is never laggin'
    Syllable patterns scatter, but the rhythm stays tight
    Wordplay displays in maze-like lyrical sight
    """
    
    features = analyzer.analyze_lyrics(sample_lyrics)
    
    print("\n🔧 JSON OUTPUT WITH CONFIDENCE SCORES:")
    print("-" * 50)
    
    # Конвертируем в словарь для красивого JSON
    result_dict = features.model_dump()
    
    # Выводим только ключевые метрики с confidence
    key_metrics = {
        'metaphor_analysis': {
            'metaphor_count': result_dict['metaphor_analysis']['metaphor_count'],
            'metaphor_confidence': result_dict['metaphor_analysis']['metaphor_confidence'],
            'wordplay_instances': result_dict['metaphor_analysis']['wordplay_instances'],
            'wordplay_confidence': result_dict['metaphor_analysis']['wordplay_confidence'],
        },
        'rhyme_analysis': {
            'rhyme_density': result_dict['rhyme_analysis']['rhyme_density'],
            'rhyme_detection_confidence': result_dict['rhyme_analysis']['rhyme_detection_confidence'],
            'rhyme_scheme_confidence': result_dict['rhyme_analysis']['rhyme_scheme_confidence'],
        },
        'flow_analysis': {
            'stress_pattern_confidence': result_dict['flow_analysis']['stress_pattern_confidence'],
            'flow_analysis_confidence': result_dict['flow_analysis']['flow_analysis_confidence'],
        }
    }
    
    print(json.dumps(key_metrics, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    test_confidence_scores()
    demonstrate_json_output()
    
    print("\n💡 EXPLANATION:")
    print("Confidence scores help you understand:")
    print("• How reliable each metric is")
    print("• Which results to trust vs. validate manually")
    print("• Quality of the source material for analysis")
    print("• Limitations of automated detection algorithms")
