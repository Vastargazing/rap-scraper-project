#!/usr/bin/env python3
"""
Простой тест подключения к Ollama
"""
import requests
import json

def test_ollama_connection():
    """Тест подключения к Ollama API"""
    try:
        # Проверяем доступность сервера
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        print(f"✅ Ollama сервер доступен: {response.status_code}")
        print(f"Доступные модели: {response.json()}")
        
        # Тест простого запроса
        payload = {
            "model": "llama3.2:3b",
            "prompt": "Hello! Please respond briefly.",
            "stream": False
        }
        
        response = requests.post("http://localhost:11434/api/generate", 
                               json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Тест запроса успешен:")
            print(f"Ответ: {result.get('response', 'Нет ответа')}")
        else:
            print(f"❌ Ошибка запроса: {response.status_code}")
            print(f"Ответ: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка подключения: {e}")
    except Exception as e:
        print(f"❌ Общая ошибка: {e}")

if __name__ == "__main__":
    print("🔍 Тест подключения к Ollama...")
    test_ollama_connection()
