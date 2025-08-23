#!/usr/bin/env python3
"""
Быстрый тест Ollama с llama3.2:3b
"""

import requests
import json
import time

def test_ollama_connection():
    """Проверка подключения к Ollama"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print("✅ Ollama запущен!")
            print(f"📋 Доступные модели: {[m['name'] for m in models]}")
            return True
        else:
            print("❌ Ollama не отвечает")
            return False
    except Exception as e:
        print(f"❌ Ошибка подключения к Ollama: {e}")
        return False

def test_analysis(model_name="llama3.2:3b"):
    """Тест анализа текста песни"""
    
    # Простой текст для теста
    test_lyrics = """
    Started from the bottom now we're here
    Started from the bottom now my whole team here
    I don't know what you've been told
    But time never stops and the story goes on
    Money, power, respect, that's what we chase
    But I keep my family close, that's my base
    """
    
    prompt = f"""
    Analyze this rap song lyrics and return result in JSON format:
    
    Lyrics: {test_lyrics}
    
    Return JSON with:
    {{
        "genre": "trap/hip-hop/old-school/drill",
        "mood": "energetic/aggressive/chill/melancholic", 
        "main_themes": ["success", "money", "family"],
        "energy_level": "low/medium/high",
        "authenticity_score": 0.0-1.0,
        "quality": "poor/fair/good/excellent"
    }}
    
    Return ONLY JSON, no other text!
    """
    
    print(f"🤖 Тестируем анализ с моделью {model_name}...")
    start_time = time.time()
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": 1000
                }
            },
            timeout=60
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            analysis = result.get('response', '')
            
            print(f"✅ Анализ завершен за {duration:.1f} секунд")
            print(f"📄 Ответ модели:")
            print(analysis)
            print("\n" + "="*50)
            
            # Попытка парсинга JSON
            try:
                json_start = analysis.find('{')
                json_end = analysis.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = analysis[json_start:json_end]
                    parsed = json.loads(json_str)
                    print("✅ JSON успешно распарсен:")
                    for key, value in parsed.items():
                        print(f"  • {key}: {value}")
                else:
                    print("⚠️ JSON не найден в ответе")
            except json.JSONDecodeError as e:
                print(f"❌ Ошибка парсинга JSON: {e}")
                
        else:
            print(f"❌ Ошибка API: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Ошибка анализа: {e}")

def main():
    print("🦙 Тест Ollama с llama3.2:3b")
    print("=" * 40)
    
    # Проверяем подключение
    if not test_ollama_connection():
        print("\n💡 Запустите Ollama командой: ollama serve")
        return
    
    print("\n🧪 Запускаем тест анализа...")
    test_analysis()
    
    print("\n🎉 Тест завершен!")
    print("💡 Если все работает, можно запускать полный анализ с multi_model_analyzer.py")

if __name__ == "__main__":
    main()
