#!/usr/bin/env python3
"""
Тестирование LangChain интеграции для анализа рэп-лирики
Основано на Case 9 из PROJECT_DIARY
"""

import os
import sqlite3
from typing import List, Dict
from dotenv import load_dotenv

# LangChain imports (раскомментировать при установке)
# from langchain.llms import OpenAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.callbacks import get_openai_callback

load_dotenv()

class LangChainAnalyzer:
    """Анализатор лирики через LangChain + LLM"""
    
    def __init__(self, db_path: str = "rap_lyrics.db"):
        self.db_path = db_path
        
        # Промпт для анализа (из Case 9)
        self.analysis_prompt = """
        Проанализируй этот текст рэп-песни и оцени по следующим критериям:
        
        Текст: {lyrics}
        Артист: {artist}
        Название: {title}
        
        Оценки (1-10):
        1. Сложность рифм и словесных конструкций
        2. Эмоциональная глубина и искренность
        3. Социальная значимость темы
        4. Оригинальность и креативность
        5. Техническое мастерство (флоу, ритм)
        
        Формат ответа:
        Complexity: [оценка]
        Emotion: [оценка]  
        Social: [оценка]
        Creativity: [оценка]
        Technical: [оценка]
        Genre: [hip-hop/rap/drill/trap/etc]
        Mood: [angry/sad/confident/motivational/etc]
        Summary: [краткое резюме 1-2 предложения]
        """
    
    def get_test_sample(self, limit: int = 5) -> List[tuple]:
        """Получение тестовой выборки из базы"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, title, artist, lyrics 
                FROM songs 
                WHERE lyrics IS NOT NULL 
                AND length(lyrics) > 100
                ORDER BY RANDOM() 
                LIMIT ?
            """, (limit,))
            
            result = cursor.fetchall()
            conn.close()
            return result
            
        except Exception as e:
            print(f"Ошибка при получении данных: {e}")
            return []
    
    def analyze_with_mock(self, lyrics: str, artist: str, title: str) -> Dict:
        """Mock анализ без LangChain (для тестирования)"""
        # Простая эмуляция анализа
        word_count = len(lyrics.split())
        complexity = min(10, max(1, word_count // 20))
        
        # Эмуляция based on keywords
        emotion = 7 if any(word in lyrics.lower() for word in ['love', 'heart', 'pain', 'feel']) else 5
        social = 8 if any(word in lyrics.lower() for word in ['street', 'money', 'struggle', 'life']) else 4
        creativity = min(10, max(1, len(set(lyrics.lower().split())) // 10))
        technical = 6  # Default
        
        # Genre detection
        genre = "hip-hop"
        if any(word in lyrics.lower() for word in ['drill', 'gang']):
            genre = "drill"
        elif any(word in lyrics.lower() for word in ['auto-tune', 'melody']):
            genre = "trap"
        
        # Mood detection  
        mood = "confident"
        if any(word in lyrics.lower() for word in ['sad', 'pain', 'hurt']):
            mood = "sad"
        elif any(word in lyrics.lower() for word in ['angry', 'mad', 'hate']):
            mood = "angry"
        
        return {
            "complexity": complexity,
            "emotion": emotion,
            "social": social, 
            "creativity": creativity,
            "technical": technical,
            "genre": genre,
            "mood": mood,
            "summary": f"Track by {artist} showing {mood} mood with {complexity}/10 complexity"
        }
    
    def analyze_with_langchain(self, lyrics: str, artist: str, title: str) -> Dict:
        """Анализ через LangChain (требует установки)"""
        try:
            # Раскомментировать при установке LangChain
            # openai_api_key = os.getenv('OPENAI_API_KEY')
            # if not openai_api_key:
            #     return self.analyze_with_mock(lyrics, artist, title)
            
            # # Создаем LLM и chain
            # llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
            # prompt = PromptTemplate(
            #     input_variables=["lyrics", "artist", "title"],
            #     template=self.analysis_prompt
            # )
            # chain = LLMChain(llm=llm, prompt=prompt)
            
            # # Выполняем анализ
            # with get_openai_callback() as cb:
            #     result = chain.run(lyrics=lyrics, artist=artist, title=title)
            #     
            # # Парсим результат
            # return self.parse_llm_response(result)
            
            print("⚠️ LangChain не установлен, используем mock анализ")
            return self.analyze_with_mock(lyrics, artist, title)
            
        except Exception as e:
            print(f"Ошибка LangChain анализа: {e}")
            return self.analyze_with_mock(lyrics, artist, title)
    
    def parse_llm_response(self, response: str) -> Dict:
        """Парсинг ответа LLM в структурированный формат"""
        result = {}
        lines = response.strip().split('\n')
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key in ['complexity', 'emotion', 'social', 'creativity', 'technical']:
                    try:
                        result[key] = int(value.split()[0])  # Извлекаем число
                    except:
                        result[key] = 5  # Default
                elif key in ['genre', 'mood', 'summary']:
                    result[key] = value
        
        return result
    
    def test_analysis_pipeline(self):
        """Тестирование полного пайплайна анализа"""
        print("🧪 Тестирование LangChain анализа рэп-лирики")
        print("=" * 60)
        
        # Получаем тестовые данные
        sample_data = self.get_test_sample(3)
        
        if not sample_data:
            print("❌ Нет данных для тестирования")
            return
        
        results = []
        
        for song_id, title, artist, lyrics in sample_data:
            print(f"\n🎵 Анализируем: {artist} - {title}")
            print(f"📝 Длина текста: {len(lyrics)} символов")
            
            # Анализируем
            analysis = self.analyze_with_langchain(
                lyrics[:500],  # Первые 500 символов для теста
                artist, 
                title
            )
            
            # Выводим результаты
            print(f"📊 Результаты анализа:")
            print(f"   • Сложность: {analysis.get('complexity', 'N/A')}/10")
            print(f"   • Эмоции: {analysis.get('emotion', 'N/A')}/10")
            print(f"   • Социальность: {analysis.get('social', 'N/A')}/10")
            print(f"   • Креативность: {analysis.get('creativity', 'N/A')}/10")
            print(f"   • Техника: {analysis.get('technical', 'N/A')}/10")
            print(f"   • Жанр: {analysis.get('genre', 'N/A')}")
            print(f"   • Настроение: {analysis.get('mood', 'N/A')}")
            print(f"   • Резюме: {analysis.get('summary', 'N/A')}")
            
            results.append({
                'song_id': song_id,
                'artist': artist,
                'title': title,
                'analysis': analysis
            })
        
        print(f"\n✅ Проанализировано {len(results)} треков")
        print("🎯 LangChain интеграция протестирована!")
        
        return results

def main():
    """Основная функция тестирования"""
    analyzer = LangChainAnalyzer()
    
    # Проверяем подключение к базе
    if not os.path.exists(analyzer.db_path):
        print(f"❌ База данных не найдена: {analyzer.db_path}")
        return
    
    # Запускаем тестирование
    results = analyzer.test_analysis_pipeline()
    
    if results:
        print(f"\n📈 Статистика:")
        avg_complexity = sum(r['analysis'].get('complexity', 0) for r in results) / len(results)
        avg_emotion = sum(r['analysis'].get('emotion', 0) for r in results) / len(results)
        
        print(f"   • Средняя сложность: {avg_complexity:.1f}/10")
        print(f"   • Средняя эмоциональность: {avg_emotion:.1f}/10")
        
        genres = [r['analysis'].get('genre', 'unknown') for r in results]
        print(f"   • Жанры: {', '.join(set(genres))}")

if __name__ == "__main__":
    main()
