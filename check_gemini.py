"""
Проверяем доступные модели Gemini API
"""
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ GOOGLE_API_KEY not found in .env file")
    exit(1)

# Настраиваем API
genai.configure(api_key=api_key)

print("🔍 Checking available Gemini models...")
print("="*50)

try:
    # Получаем список доступных моделей
    models = genai.list_models()
    
    print("📋 Available models:")
    for model in models:
        if 'generateContent' in model.supported_generation_methods:
            print(f"✅ {model.name}")
            print(f"   Display name: {model.display_name}")
            print(f"   Description: {model.description}")
            print(f"   Input token limit: {model.input_token_limit}")
            print(f"   Output token limit: {model.output_token_limit}")
            print()
    
    # Тестируем простой запрос
    print("🧪 Testing simple request with gemini-1.5-flash...")
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Say hello in 3 words")
    print(f"✅ Response: {response.text}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Please check your API key and internet connection")
