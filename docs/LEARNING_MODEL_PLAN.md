
## 📋 О документе LEARNING_MODEL_PLAN.md

**Это детальный план создания собственной ML-модели для генерации рэп-текстов** на базе твоих 57K треков из PostgreSQL. 

### 🎯 **Главная цель документа:**
Создать **custom rap generator** (генератор рэп-текстов), который будет обучен на твоих данных и сможет создавать аутентичные рэп-тексты в стиле различных исполнителей.

### 📖 **Структура плана:**

**Phase 1: Data Preparation (Week 1)**
- Экспорт 57K треков из PostgreSQL в формат для обучения
- Подготовка training data в формате JSONL
- Фильтрация качественных треков (quality_score > 0.7)

**Phase 2: Model Selection & Training (Week 2)**
- Выбор между разными подходами (GPT fine-tuning, local models)
- Подготовка данных в формате "system/user/assistant"
- Обучение модели на твоих данных

**Phase 3: Production Integration (Week 3)**
- Интеграция обученной модели в твой проект
- CLI и Web интерфейсы
- Docker deployment

### 🔥 **Ключевые особенности:**

1. **Использует твои реальные данные** - 57K треков как training dataset
2. **Production-ready подход** - не просто эксперимент, а готовая система
3. **Качественная фильтрация** - только треки с хорошим quality_score
4. **Тематическая классификация** - автоматическое определение тем (love, money, struggle, etc.)
5. **Интеграция с существующей архитектурой** - PostgreSQL, Docker, FastAPI

### 🎵 **Что на выходе:**
- **Custom rap generator**, который генерирует тексты в стиле разных исполнителей
- **API endpoints** для интеграции
- **Web интерфейс** для удобного использования
- **Production deployment** с Docker


--

## 🎯 Phase 1: Data Preparation (Week 1)

### **Step 1.1: Export из PostgreSQL**
```bash
# Создай рабочую папку
mkdir rap_ml_project
cd rap_ml_project
mkdir data scripts logs models
```

```python
# scripts/data_exporter.py
import asyncpg
import json
import asyncio
from pathlib import Path

class RapDataExporter:
    def __init__(self):
        self.db_url = "postgresql://user:pass@localhost/rap_db"  # Твоя БД
        self.output_dir = Path("./data")
        
    async def export_training_data(self):
        """Экспорт всех 57K треков в training format"""
        
        conn = await asyncpg.connect(self.db_url)
        
        # Выбираем только качественные треки
        query = """
        SELECT 
            track_id, title, artist, lyrics, genre, mood, 
            popularity_score, quality_score, release_year
        FROM tracks 
        WHERE 
            lyrics IS NOT NULL 
            AND length(lyrics) > 100 
            AND quality_score > 0.7
        ORDER BY popularity_score DESC
        """
        
        tracks = await conn.fetch(query)
        await conn.close()
        
        print(f"🎵 Found {len(tracks)} quality tracks")
        
        # Конвертируем в training format
        training_data = []
        
        for track in tracks:
            # Определяем тему из lyrics (упрощённо)
            theme = self.extract_theme(track['lyrics'], track['genre'])
            
            training_example = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a skilled rapper creating authentic hip-hop lyrics with complex rhymes and natural flow."
                    },
                    {
                        "role": "user",
                        "content": f"Create rap about: {theme}. Style: {track['genre']}. Mood: {track['mood']}"
                    },
                    {
                        "role": "assistant", 
                        "content": track['lyrics']
                    }
                ]
            }
            
            training_data.append(training_example)
        
        # Сохраняем в JSONL
        output_file = self.output_dir / "rap_training_full.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in training_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        print(f"✅ Saved {len(training_data)} examples to {output_file}")
        print(f"📊 File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
        
        return output_file, len(training_data)
    
    def extract_theme(self, lyrics: str, genre: str) -> str:
        """Простое извлечение темы из текста"""
        
        # Ключевые слова для тем
        themes_keywords = {
            "love": ["love", "baby", "girl", "heart", "kiss"],
            "money": ["money", "cash", "rich", "bank", "dollars"],
            "struggle": ["struggle", "pain", "hard", "fight", "survive"],
            "success": ["success", "win", "top", "boss", "champion"],
            "party": ["party", "club", "dance", "night", "drinks"],
            "street": ["street", "block", "hood", "real", "gangsta"]
        }
        
        lyrics_lower = lyrics.lower()
        
        for theme, keywords in themes_keywords.items():
            if any(keyword in lyrics_lower for keyword in keywords):
                return theme
        
        return genre  # Fallback to genre

# Запуск экспорта
async def main():
    exporter = RapDataExporter()
    file_path, count = await exporter.export_training_data()
    print(f"🚀 Ready for training: {count} examples in {file_path}")

if __name__ == "__main__":
    asyncio.run(main())
```

### **Step 1.2: Validation данных**
```python
# scripts/data_validator.py
import json
from pathlib import Path

def validate_training_data(file_path: str):
    """Проверяем качество training data"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"🔍 Validating {len(lines)} training examples...")
    
    valid_count = 0
    issues = []
    
    for i, line in enumerate(lines):
        try:
            example = json.loads(line)
            
            # Проверяем структуру
            if not example.get('messages'):
                issues.append(f"Line {i}: No messages")
                continue
                
            if len(example['messages']) != 3:
                issues.append(f"Line {i}: Wrong message count")
                continue
            
            lyrics = example['messages'][2]['content']
            
            # Проверяем качество lyrics
            if len(lyrics) < 50:
                issues.append(f"Line {i}: Lyrics too short")
                continue
                
            if len(lyrics) > 2000:
                issues.append(f"Line {i}: Lyrics too long") 
                continue
            
            valid_count += 1
            
        except json.JSONDecodeError:
            issues.append(f"Line {i}: Invalid JSON")
    
    print(f"✅ Valid examples: {valid_count}")
    print(f"❌ Issues found: {len(issues)}")
    
    if issues:
        print("\nFirst 5 issues:")
        for issue in issues[:5]:
            print(f"  - {issue}")
    
    # Показываем примеры
    print("\n📝 Sample training examples:")
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:  # Показываем первые 3
                break
            example = json.loads(line)
            print(f"\nExample {i+1}:")
            print(f"User: {example['messages'][1]['content']}")
            print(f"Assistant: {example['messages'][2]['content'][:100]}...")

# Запуск
validate_training_data("./data/rap_training_full.jsonl")
```

### **Step 1.3: Создание test set**
```python
# scripts/split_data.py
import json
import random
from pathlib import Path

def split_training_data(input_file: str, train_ratio: float = 0.9):
    """Разделяем на train/test sets"""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Перемешиваем данные
    random.shuffle(lines)
    
    # Делим на train/test
    split_idx = int(len(lines) * train_ratio)
    train_lines = lines[:split_idx]
    test_lines = lines[split_idx:]
    
    # Сохраняем train set
    train_file = Path(input_file).parent / "rap_training_train.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    
    # Сохраняем test set
    test_file = Path(input_file).parent / "rap_training_test.jsonl"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.writelines(test_lines)
    
    print(f"📊 Split complete:")
    print(f"  Train: {len(train_lines)} examples → {train_file}")
    print(f"  Test: {len(test_lines)} examples → {test_file}")
    
    return train_file, test_file

# Запуск
train_file, test_file = split_training_data("./data/rap_training_full.jsonl")
```

---

## 🧠 Phase 2: Fine-tuning через Navita (Week 2)

### **Step 2.1: Настройка Navita API**
```python
# scripts/navita_trainer.py
import httpx
import asyncio
import json
from pathlib import Path

class NavitaFineTuner:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.navita.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def upload_training_file(self, file_path: str):
        """Загружаем training file"""
        
        async with httpx.AsyncClient() as client:
            with open(file_path, 'rb') as f:
                files = {"file": f}
                data = {"purpose": "fine-tune"}
                
                response = await client.post(
                    f"{self.base_url}/files",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files=files,
                    data=data
                )
        
        if response.status_code == 200:
            file_info = response.json()
            print(f"✅ File uploaded: {file_info['id']}")
            return file_info['id']
        else:
            print(f"❌ Upload failed: {response.text}")
            return None
    
    async def start_fine_tuning(self, training_file_id: str, model: str = "qwen2.5-32b-instruct"):
        """Запускаем fine-tuning job"""
        
        payload = {
            "training_file": training_file_id,
            "model": model,
            "hyperparameters": {
                "n_epochs": 3,
                "batch_size": 8,
                "learning_rate_multiplier": 0.1
            }
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/fine_tuning/jobs",
                headers=self.headers,
                json=payload
            )
        
        if response.status_code == 200:
            job_info = response.json()
            print(f"🚀 Fine-tuning started: {job_info['id']}")
            print(f"📊 Model: {model}")
            print(f"⏱️ Estimated time: 2-6 hours")
            return job_info['id']
        else:
            print(f"❌ Fine-tuning failed: {response.text}")
            return None
    
    async def check_job_status(self, job_id: str):
        """Проверяем статус обучения"""
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/fine_tuning/jobs/{job_id}",
                headers=self.headers
            )
        
        if response.status_code == 200:
            job_info = response.json()
            status = job_info['status']
            
            print(f"📊 Job {job_id}: {status}")
            
            if status == "succeeded":
                model_id = job_info['fine_tuned_model']
                print(f"🎉 Fine-tuning complete! Model: {model_id}")
                return model_id
            elif status == "failed":
                print(f"❌ Fine-tuning failed: {job_info.get('error', 'Unknown error')}")
                return None
            else:
                print(f"⏳ Still running... Check again in 10 minutes")
                return "running"
        
        return None

# Запуск fine-tuning
async def main():
    API_KEY = "your_navita_api_key"  # Получи на navita.ai
    
    trainer = NavitaFineTuner(API_KEY)
    
    # 1. Загружаем training file
    print("📤 Uploading training data...")
    file_id = await trainer.upload_training_file("./data/rap_training_train.jsonl")
    
    if not file_id:
        return
    
    # 2. Запускаем обучение
    print("🧠 Starting fine-tuning...")
    job_id = await trainer.start_fine_tuning(file_id)
    
    if not job_id:
        return
    
    # 3. Мониторинг (проверяй каждые 10 минут)
    print("⏳ Monitoring training progress...")
    while True:
        await asyncio.sleep(600)  # 10 минут
        result = await trainer.check_job_status(job_id)
        
        if result and result != "running":
            break
    
    print("🎯 Fine-tuning complete!")

if __name__ == "__main__":
    asyncio.run(main())
```

### **Step 2.2: Test fine-tuned model**
```python
# scripts/test_finetuned.py
import httpx
import asyncio

class FineTunedTester:
    def __init__(self, api_key: str, model_id: str):
        self.api_key = api_key
        self.model_id = model_id  # ft:qwen2.5-32b:your-model
        self.base_url = "https://api.navita.ai/v1"
    
    async def test_generation(self, theme: str, style: str):
        """Тестируем fine-tuned модель"""
        
        prompt = f"Create rap about: {theme}. Style: {style}. Mood: energetic"
        
        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a skilled rapper creating authentic hip-hop lyrics."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.8,
            "max_tokens": 500
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
        
        if response.status_code == 200:
            result = response.json()
            lyrics = result['choices'][0]['message']['content']
            
            print(f"🎤 Generated rap about {theme} in {style} style:")
            print("=" * 50)
            print(lyrics)
            print("=" * 50)
            
            return lyrics
        else:
            print(f"❌ Generation failed: {response.text}")
            return None

# Тестирование
async def test_model():
    API_KEY = "your_navita_api_key"
    MODEL_ID = "ft:qwen2.5-32b:your-rap-model"  # Из результата fine-tuning
    
    tester = FineTunedTester(API_KEY, MODEL_ID)
    
    test_cases = [
        ("love", "R&B"),
        ("success", "trap"),
        ("struggle", "conscious rap"),
        ("party", "club rap")
    ]
    
    for theme, style in test_cases:
        print(f"\n🎯 Testing: {theme} + {style}")
        await tester.test_generation(theme, style)
        await asyncio.sleep(2)  # Rate limiting

if __name__ == "__main__":
    asyncio.run(test_model())
```

---

## 🔍 Phase 3: RAG System Setup (Week 3)

### **Step 3.1: Подготовка vector embeddings**
```python
# scripts/rag_setup.py
import asyncpg
import asyncio
from sentence_transformers import SentenceTransformer
import numpy as np

class RapRAGSetup:
    def __init__(self):
        self.db_url = "postgresql://user:pass@localhost/rap_db"
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    async def create_embeddings_table(self):
        """Создаём таблицу для embeddings если её нет"""
        
        conn = await asyncpg.connect(self.db_url)
        
        await conn.execute("""
            CREATE EXTENSION IF NOT EXISTS vector;
            
            CREATE TABLE IF NOT EXISTS track_embeddings (
                track_id UUID PRIMARY KEY,
                title TEXT,
                artist TEXT,
                genre TEXT,
                theme TEXT,
                lyrics_snippet TEXT,
                embedding vector(384)  -- MiniLM dimension
            );
            
            CREATE INDEX IF NOT EXISTS track_embeddings_idx 
            ON track_embeddings USING ivfflat (embedding vector_cosine_ops);
        """)
        
        await conn.close()
        print("✅ Embeddings table ready")
    
    async def generate_embeddings(self, batch_size: int = 100):
        """Генерируем embeddings для всех треков"""
        
        conn = await asyncpg.connect(self.db_url)
        
        # Получаем все треки
        tracks = await conn.fetch("""
            SELECT track_id, title, artist, genre, lyrics
            FROM tracks 
            WHERE lyrics IS NOT NULL
        """)
        
        print(f"🔄 Generating embeddings for {len(tracks)} tracks...")
        
        for i in range(0, len(tracks), batch_size):
            batch = tracks[i:i + batch_size]
            
            # Подготавливаем тексты для embedding
            texts = []
            track_data = []
            
            for track in batch:
                # Создаём текст для embedding (title + snippet)
                lyrics_snippet = track['lyrics'][:500]  # Первые 500 символов
                text = f"{track['title']} {track['artist']} {lyrics_snippet}"
                
                texts.append(text)
                
                # Определяем тему (упрощённо)
                theme = self.extract_theme(track['lyrics'])
                
                track_data.append({
                    'track_id': track['track_id'],
                    'title': track['title'],
                    'artist': track['artist'], 
                    'genre': track['genre'],
                    'theme': theme,
                    'lyrics_snippet': lyrics_snippet
                })
            
            # Генерируем embeddings
            embeddings = self.model.encode(texts)
            
            # Сохраняем в БД
            for track_info, embedding in zip(track_data, embeddings):
                await conn.execute("""
                    INSERT INTO track_embeddings 
                    (track_id, title, artist, genre, theme, lyrics_snippet, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (track_id) DO UPDATE SET
                    embedding = $7
                """, 
                    track_info['track_id'],
                    track_info['title'],
                    track_info['artist'],
                    track_info['genre'], 
                    track_info['theme'],
                    track_info['lyrics_snippet'],
                    embedding.tolist()
                )
            
            print(f"✅ Processed batch {i//batch_size + 1}/{(len(tracks)-1)//batch_size + 1}")
        
        await conn.close()
        print("🎉 All embeddings generated!")
    
    def extract_theme(self, lyrics: str) -> str:
        """Извлекаем тему из lyrics"""
        # (Same as before)
        themes_keywords = {
            "love": ["love", "baby", "girl", "heart"],
            "money": ["money", "cash", "rich", "bank"],
            "struggle": ["struggle", "pain", "hard", "fight"],
            "success": ["success", "win", "top", "boss"],
            "party": ["party", "club", "dance", "night"]
        }
        
        lyrics_lower = lyrics.lower()
        for theme, keywords in themes_keywords.items():
            if any(keyword in lyrics_lower for keyword in keywords):
                return theme
        return "general"

# Запуск
async def setup_rag():
    rag_setup = RapRAGSetup()
    
    print("🏗️ Setting up RAG system...")
    await rag_setup.create_embeddings_table()
    await rag_setup.generate_embeddings()
    print("🚀 RAG system ready!")

if __name__ == "__main__":
    asyncio.run(setup_rag())
```

### **Step 3.2: RAG Search Engine**
```python
# scripts/rag_engine.py
import asyncpg
from sentence_transformers import SentenceTransformer
import numpy as np

class RapRAGEngine:
    def __init__(self):
        self.db_url = "postgresql://user:pass@localhost/rap_db"
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    async def search_similar_tracks(self, query: str, limit: int = 5):
        """Поиск похожих треков для RAG"""
        
        # Генерируем embedding для запроса
        query_embedding = self.model.encode([query])[0]
        
        conn = await asyncpg.connect(self.db_url)
        
        # Поиск похожих треков
        results = await conn.fetch("""
            SELECT 
                track_id, title, artist, genre, theme, lyrics_snippet,
                embedding <=> $1 as distance
            FROM track_embeddings
            ORDER BY embedding <=> $1
            LIMIT $2
        """, query_embedding.tolist(), limit)
        
        await conn.close()
        
        # Форматируем результаты
        similar_tracks = []
        for row in results:
            similar_tracks.append({
                'track_id': row['track_id'],
                'title': row['title'],
                'artist': row['artist'],
                'genre': row['genre'],
                'theme': row['theme'],
                'lyrics_snippet': row['lyrics_snippet'],
                'similarity': 1 - row['distance']  # Конвертируем distance в similarity
            })
        
        return similar_tracks
    
    def extract_patterns(self, similar_tracks: list) -> dict:
        """Извлекаем паттерны из найденных треков"""
        
        patterns = {
            'themes': [],
            'genres': [],
            'common_words': [],
            'rhyme_patterns': [],
            'emotional_tones': []
        }
        
        for track in similar_tracks:
            patterns['themes'].append(track['theme'])
            patterns['genres'].append(track['genre'])
            
            # Простой анализ слов
            words = track['lyrics_snippet'].lower().split()
            patterns['common_words'].extend(words[:10])  # Первые 10 слов
        
        # Убираем дубликаты и сортируем по частоте
        patterns['themes'] = list(set(patterns['themes']))
        patterns['genres'] = list(set(patterns['genres']))
        patterns['common_words'] = list(set(patterns['common_words']))
        
        return patterns

# Тест RAG
async def test_rag():
    rag = RapRAGEngine()
    
    test_queries = [
        "sad love song about heartbreak",
        "energetic party rap about success",
        "deep conscious rap about struggle"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Searching for: {query}")
        
        similar_tracks = await rag.search_similar_tracks(query)
        patterns = rag.extract_patterns(similar_tracks)
        
        print(f"Found {len(similar_tracks)} similar tracks:")
        for track in similar_tracks:
            print(f"  - {track['title']} by {track['artist']} ({track['similarity']:.2f})")
        
        print(f"Patterns: themes={patterns['themes']}, genres={patterns['genres']}")

if __name__ == "__main__":
    asyncio.run(test_rag())
```

---

## 🔥 Phase 4: Hybrid System (Week 4)

### **Step 4.1: Полный RAG + Fine-tuned Generator**
```python
# scripts/hybrid_generator.py
import asyncio
import httpx
from rag_engine import RapRAGEngine

class HybridRapGenerator:
    def __init__(self, api_key: str, fine_tuned_model: str):
        self.api_key = api_key
        self.fine_tuned_model = fine_tuned_model
        self.rag_engine = RapRAGEngine()
        self.base_url = "https://api.navita.ai/v1"
    
    async def generate_rap_with_context(self, user_request: str):
        """Главная функция: RAG + Fine-tuned generation"""
        
        print(f"🎯 User request: {user_request}")
        
        # 1. RAG: Поиск похожих треков
        print("🔍 Searching for similar tracks...")
        similar_tracks = await self.rag_engine.search_similar_tracks(user_request, limit=5)
        
        # 2. Извлечение паттернов
        patterns = self.rag_engine.extract_patterns(similar_tracks)
        
        # 3. Создание контекстного промпта
        context_prompt = self.create_context_prompt(user_request, similar_tracks, patterns)
        
        # 4. Генерация с fine-tuned моделью
        print("🧠 Generating with fine-tuned model...")
        generated_rap = await self.generate_with_finetuned(context_prompt)
        
        return {
            'user_request': user_request,
            'similar_tracks': similar_tracks,
            'patterns': patterns,
            'generated_rap': generated_rap,
            'inspiration_sources': [f"{t['title']} - {t['artist']}" for t in similar_tracks]
        }
    
    def create_context_prompt(self, request: str, similar_tracks: list, patterns: dict) -> str:
        """Создаём умный промпт с контекстом"""
        
        # Форматируем информацию о похожих треках
        context_info = "Based on these successful tracks from your database:\n\n"
        
        for i, track in enumerate(similar_tracks[:3], 1):
            context_info += f"{i}. \"{track['title']}\" by {track['artist']}\n"
            context_info += f"   Theme: {track['theme']}, Genre: {track['genre']}\n"
            context_info += f"   Sample: {track['lyrics_snippet'][:100]}...\n\n"
        
        # Добавляем паттерны
        patterns_info = f"""
Successful patterns found:
- Common themes: {', '.join(patterns['themes'])}
- Popular genres: {', '.join(patterns['genres'])}
- Effective approaches: Use similar emotional tone and structure

"""
        
        # Финальный промпт
        prompt = f"""
{context_info}
{patterns_info}

User request: {request}

Create an original rap inspired by these successful patterns. Make it:
- Completely original (no copying)
- Authentic rap style with complex rhymes
- 16-32 bars
- Natural flow and rhythm
- Incorporate the successful patterns above

Generated rap:
"""
        
        return prompt
    
    async def generate_with_finetuned(self, prompt: str):
        """Генерация с fine-tuned моделью"""
        
        payload = {
            "model": self.fine_tuned_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a skilled rapper creating authentic hip-hop lyrics with complex rhymes and natural flow."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.8,
            "max_tokens": 800
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            print(f"❌ Generation failed: {response.text}")
            return None

# Главный интерфейс
async def main():
    API_KEY = "your_navita_api_key"
    FINE_TUNED_MODEL = "ft:qwen2.5-32b:your-rap-model"
    
    generator = HybridRapGenerator(API_KEY, FINE_TUNED_MODEL)
    
    # Тестовые запросы
    test_requests = [
        "Create energetic party rap about celebrating success",
        "Make sad love rap about missing someone",
        "Write conscious rap about overcoming struggles in the hood",
        "Create trap rap about making money and staying loyal"
    ]
    
    for request in test_requests:
        print("="*60)
        result = await generator.generate_rap_with_context(request)
        
        print(f"🎤 GENERATED RAP:")
        print(result['generated_rap'])
        print(f"\n💡 Inspired by: {', '.join(result['inspiration_sources'])}")
        print("="*60)
        
        await asyncio.sleep(2)  # Rate limiting

if __name__ == "__main__":
    asyncio.run(main())
```

### **Step 4.2: CLI Interface**
```python
# scripts/rap_cli.py
import asyncio
import click
from hybrid_generator import HybridRapGenerator

@click.command()
@click.option('--request', prompt='Describe the rap you want', help='Rap description')
@click.option('--api-key', prompt='Navita API key', help='Your Navita API key')
@click.option('--model', prompt='Fine-tuned model ID', help='Your fine-tuned model')
def create_rap(request: str, api_key: str, model: str):
    """Create rap using your fine-tuned model + RAG system"""
    
    async def generate():
        generator = HybridRapGenerator(api_key, model)
        
        print("🎵 Creating your rap...")
        print("🔍 Analyzing your 57K tracks database...")
        
        result = await generator.generate_rap_with_context(request)
        
        print("\n" + "="*60)
        print("🎤 YOUR GENERATED RAP:")
        print("="*60)
        print(result['generated_rap'])
        print("="*60)
        print(f"💡 Inspired by tracks: {', '.join(result['inspiration_sources'][:3])}")
        print(f"🎯 Patterns used: {', '.join(result['patterns']['themes'])}")
        print("="*60)
        
        # Сохраняем результат
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"./logs/generated_rap_{timestamp}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Request: {request}\n\n")
            f.write(f"Generated Rap:\n{result['generated_rap']}\n\n")
            f.write(f"Inspiration: {', '.join(result['inspiration_sources'])}\n")
        
        print(f"💾 Saved to: {output_file}")
    
    asyncio.run(generate())

if __name__ == "__main__":
    create_rap()

# Usage examples:
# python rap_cli.py --request "energetic party rap about success" --api-key "your_key" --model "ft:qwen2.5-32b:your-model"
```

### **Step 4.3: Web Interface (Bonus)**
```python
# scripts/web_app.py
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from hybrid_generator import HybridRapGenerator
import os

app = FastAPI(title="RAP ML Generator")
templates = Jinja2Templates(directory="templates")

# Configuration
API_KEY = os.getenv("NAVITA_API_KEY")
FINE_TUNED_MODEL = os.getenv("FINE_TUNED_MODEL")

generator = HybridRapGenerator(API_KEY, FINE_TUNED_MODEL)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate", response_class=HTMLResponse)
async def generate_rap(request: Request, rap_request: str = Form(...)):
    
    try:
        # Генерируем рэп
        result = await generator.generate_rap_with_context(rap_request)
        
        return templates.TemplateResponse("result.html", {
            "request": request,
            "user_request": rap_request,
            "generated_rap": result['generated_rap'],
            "inspiration_sources": result['inspiration_sources'],
            "patterns": result['patterns']
        })
        
    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request, 
            "error": str(e)
        })

# Запуск: uvicorn web_app:app --reload
```

```html
<!-- templates/index.html -->
<!DOCTYPE html>
<html>
<head>
    <title>🎤 RAP ML Generator</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .form-group { margin-bottom: 20px; }
        textarea { width: 100%; height: 100px; padding: 10px; }
        button { background: #ff6b6b; color: white; padding: 15px 30px; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #ff5252; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎤 RAP ML Generator</h1>
        <p>Powered by your 57K tracks + Fine-tuned QWEN + RAG</p>
    </div>
    
    <form action="/generate" method="post">
        <div class="form-group">
            <label for="rap_request">Describe the rap you want:</label>
            <textarea name="rap_request" placeholder="e.g., energetic party rap about success and money" required></textarea>
        </div>
        <button type="submit">🎵 Generate Rap</button>
    </form>
</body>
</html>
```

```html
<!-- templates/result.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Generated Rap</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .rap-output { background: #f5f5f5; padding: 20px; border-radius: 10px; white-space: pre-line; }
        .inspiration { background: #e3f2fd; padding: 15px; border-radius: 5px; margin-top: 20px; }
        .back-btn { background: #2196f3; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>🎤 Your Generated Rap</h1>
    
    <h3>Request: "{{ user_request }}"</h3>
    
    <div class="rap-output">{{ generated_rap }}</div>
    
    <div class="inspiration">
        <h4>💡 Inspired by these tracks from your database:</h4>
        <ul>
            {% for source in inspiration_sources %}
            <li>{{ source }}</li>
            {% endfor %}
        </ul>
        
        <p><strong>Patterns used:</strong> {{ ', '.join(patterns.themes) }}</p>
    </div>
    
    <br>
    <a href="/" class="back-btn">🔄 Generate Another</a>
</body>
</html>
```

---

## 📊 Phase 5: Testing & Optimization (Week 5)

### **Step 5.1: Quality Testing**
```python
# scripts/quality_tester.py
import asyncio
from hybrid_generator import HybridRapGenerator

class QualityTester:
    def __init__(self, api_key: str, model: str):
        self.generator = HybridRapGenerator(api_key, model)
        
    async def run_quality_tests(self):
        """Тестируем качество генерации"""
        
        test_cases = [
            {
                "request": "sad love rap about heartbreak",
                "expected_themes": ["love", "sadness", "relationships"],
                "expected_style": "emotional, slow"
            },
            {
                "request": "energetic party rap about success", 
                "expected_themes": ["success", "party", "celebration"],
                "expected_style": "upbeat, confident"
            },
            {
                "request": "conscious rap about social issues",
                "expected_themes": ["society", "struggle", "awareness"], 
                "expected_style": "thoughtful, serious"
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 Test {i}: {test_case['request']}")
            
            result = await self.generator.generate_rap_with_context(test_case['request'])
            
            # Анализируем качество
            quality_score = self.analyze_quality(result, test_case)
            
            results.append({
                'test_case': test_case,
                'result': result,
                'quality_score': quality_score
            })
            
            print(f"📊 Quality Score: {quality_score}/10")
            print(f"🎤 Generated: {result['generated_rap'][:100]}...")
        
        # Общий отчёт
        avg_score = sum(r['quality_score'] for r in results) / len(results)
        print(f"\n📈 Average Quality Score: {avg_score:.1f}/10")
        
        return results
    
    def analyze_quality(self, result: dict, test_case: dict) -> float:
        """Простой анализ качества (0-10)"""
        
        score = 0
        lyrics = result['generated_rap'].lower()
        
        # Проверяем длину (должно быть достаточно текста)
        if 100 < len(lyrics) < 2000:
            score += 2
        
        # Проверяем наличие рифм (упрощённо)
        lines = lyrics.split('\n')
        if len(lines) >= 8:  # Минимум 8 строк
            score += 2
            
        # Проверяем соответствие теме
        expected_themes = test_case['expected_themes']
        theme_matches = sum(1 for theme in expected_themes if theme in lyrics)
        score += min(theme_matches * 2, 4)  # Максимум 4 балла за темы
        
        # Проверяем использование RAG (есть ли inspiration sources)
        if result.get('inspiration_sources'):
            score += 2
        
        return min(score, 10)

# Запуск тестов
async def run_tests():
    API_KEY = "your_navita_api_key"
    MODEL = "ft:qwen2.5-32b:your-model"
    
    tester = QualityTester(API_KEY, MODEL)
    results = await tester.run_quality_tests()
    
    print("\n🎯 Testing complete!")

if __name__ == "__main__":
    asyncio.run(run_tests())
```

### **Step 5.2: Performance Monitoring**
```python
# scripts/performance_monitor.py
import time
import asyncio
import psutil
from hybrid_generator import HybridRapGenerator

class PerformanceMonitor:
    def __init__(self):
        self.metrics = []
    
    async def benchmark_generation(self, api_key: str, model: str):
        """Бенчмарк производительности"""
        
        generator = HybridRapGenerator(api_key, model)
        
        test_requests = [
            "Create love rap",
            "Make party anthem", 
            "Write conscious rap about struggle",
            "Generate trap song about money"
        ]
        
        for request in test_requests:
            # Засекаем время и ресурсы
            start_time = time.time()
            start_memory = psutil.virtual_memory().used
            
            print(f"⏱️ Benchmarking: {request}")
            
            # Генерируем
            result = await generator.generate_rap_with_context(request)
            
            # Замеряем результаты
            end_time = time.time()
            end_memory = psutil.virtual_memory().used
            
            metrics = {
                'request': request,
                'generation_time': end_time - start_time,
                'memory_used': end_memory - start_memory,
                'output_length': len(result['generated_rap']),
                'rag_sources_found': len(result['similar_tracks'])
            }
            
            self.metrics.append(metrics)
            
            print(f"  ⏰ Time: {metrics['generation_time']:.2f}s")
            print(f"  🧠 Memory: {metrics['memory_used']/1024/1024:.1f}MB")
            print(f"  📝 Output: {metrics['output_length']} chars")
            print(f"  🔍 RAG sources: {metrics['rag_sources_found']}")
        
        self.print_summary()
    
    def print_summary(self):
        """Печатаем сводку производительности"""
        
        if not self.metrics:
            return
            
        avg_time = sum(m['generation_time'] for m in self.metrics) / len(self.metrics)
        avg_memory = sum(m['memory_used'] for m in self.metrics) / len(self.metrics)
        avg_length = sum(m['output_length'] for m in self.metrics) / len(self.metrics)
        
        print("\n📊 PERFORMANCE SUMMARY")
        print("="*40)
        print(f"Average generation time: {avg_time:.2f}s")
        print(f"Average memory usage: {avg_memory/1024/1024:.1f}MB") 
        print(f"Average output length: {avg_length:.0f} chars")
        print(f"Total tests: {len(self.metrics)}")
        print("="*40)

# Запуск бенчмарка
async def run_benchmark():
    API_KEY = "your_navita_api_key"
    MODEL = "ft:qwen2.5-32b:your-model"
    
    monitor = PerformanceMonitor()
    await monitor.benchmark_generation(API_KEY, MODEL)

if __name__ == "__main__":
    asyncio.run(run_benchmark())
```

---

## 🚀 Phase 6: Production Deployment

### **Step 6.1: Docker Setup**
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY scripts/ ./scripts/
COPY templates/ ./templates/
COPY data/ ./data/

# Install PostgreSQL client
RUN apt-get update && apt-get install -y postgresql-client

EXPOSE 8000

CMD ["uvicorn", "scripts.web_app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  rap-generator:
    build: .
    ports:
      - "8000:8000"
    environment:
      - NAVITA_API_KEY=${NAVITA_API_KEY}
      - FINE_TUNED_MODEL=${FINE_TUNED_MODEL}
      - DATABASE_URL=${DATABASE_URL}
    depends_on:
      - postgres
    volumes:
      - ./logs:/app/logs

  postgres:
    image: pgvector/pgvector:pg15
    environment:
      - POSTGRES_DB=rap_db
      - POSTGRES_USER=rap_user
      - POSTGRES_PASSWORD=rap_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./data/backup.sql:/docker-entrypoint-initdb.d/backup.sql
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

### **Step 6.2: Environment Setup**
```bash
# requirements.txt
fastapi==0.104.1
uvicorn==0.24.0
httpx==0.25.2
asyncpg==0.29.0
sentence-transformers==2.2.2
psutil==5.9.6
click==8.1.7
jinja2==3.1.2
python-multipart==0.0.6
numpy==1.24.3
scikit-learn==1.3.2
```

```bash
# .env file (НЕ коммить в git!)
NAVITA_API_KEY=your_navita_api_key_here
FINE_TUNED_MODEL=ft:qwen2.5-32b:your-rap-model
DATABASE_URL=postgresql://rap_user:rap_password@localhost:5432/rap_db
```

```bash
# setup.sh - Автоматическая установка
#!/bin/bash

echo "🎤 Setting up RAP ML Generator..."

# Create virtual environment
python -m venv rap_ml_env
source rap_ml_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data logs models templates

# Setup database
echo "🗄️ Setting up PostgreSQL..."
docker-compose up -d postgres

# Wait for DB to be ready
sleep 10

# Run data export
echo "📊 Exporting training data..."
python scripts/data_exporter.py

# Setup RAG
echo "🔍 Setting up RAG system..."
python scripts/rag_setup.py

echo "✅ Setup complete! Ready to fine-tune."
echo "Next steps:"
echo "1. Get Navita API key: https://navita.ai"
echo "2. Run: python scripts/navita_trainer.py"
echo "3. Test: python scripts/test_finetuned.py"
```

---

## 📋 Complete Project Structure

```
rap_ml_project/
├── data/
│   ├── rap_training_full.jsonl      # Все 57K треков
│   ├── rap_training_train.jsonl     # Training set  
│   └── rap_training_test.jsonl      # Test set
├── scripts/
│   ├── data_exporter.py             # Экспорт из PostgreSQL
│   ├── data_validator.py            # Валидация данных
│   ├── navita_trainer.py            # Fine-tuning через Navita
│   ├── test_finetuned.py           # Тест fine-tuned модели
│   ├── rag_setup.py                # Настройка RAG системы
│   ├── rag_engine.py               # RAG поисковик
│   ├── hybrid_generator.py         # Основной генератор
│   ├── rap_cli.py                  # CLI интерфейс
│   ├── web_app.py                  # Web интерфейс
│   ├── quality_tester.py           # Тестирование качества
│   └── performance_monitor.py      # Мониторинг производительности
├── templates/
│   ├── index.html                  # Главная страница
│   ├── result.html                 # Результаты генерации
│   └── error.html                  # Страница ошибок
├── logs/                           # Логи и сохранённые рэпы
├── models/                         # Локальные модели (если есть)
├── requirements.txt                # Python зависимости
├── docker-compose.yml             # Docker конфигурация
├── Dockerfile                     # Docker образ
├── .env                           # Environment variables
├── setup.sh                       # Автоматическая установка
└── README.md                      # Документация проекта
```

---

## 🎯 Execution Timeline

### **Week 1: Data Preparation**
- ✅ День 1-2: Экспорт из PostgreSQL (`data_exporter.py`)
- ✅ День 3-4: Валидация и очистка (`data_validator.py`)  
- ✅ День 5-7: Разделение на train/test (`split_data.py`)

### **Week 2: Fine-tuning**
- ✅ День 1-2: Настройка Navita API (`navita_trainer.py`)
- ✅ День 3-5: Обучение модели (автоматически)
- ✅ День 6-7: Тестирование fine-tuned модели

### **Week 3: RAG System**
- ✅ День 1-3: Генерация embeddings (`rag_setup.py`)
- ✅ День 4-5: Поисковая система (`rag_engine.py`)
- ✅ День 6-7: Тестирование RAG

### **Week 4: Integration**
- ✅ День 1-3: Hybrid система (`hybrid_generator.py`)
- ✅ День 4-5: CLI интерфейс (`rap_cli.py`)
- ✅ День 6-7: Web интерфейс (`web_app.py`)

### **Week 5: Testing & Polish**
- ✅ День 1-3: Quality testing (`quality_tester.py`)
- ✅ День 4-5: Performance optimization
- ✅ День 6-7: Docker deployment

---

## 💰 Total Cost Estimate

```python
cost_breakdown = {
    "Navita Fine-tuning": "$15-25 (57K треков)",
    "API calls (testing)": "$5-10", 
    "Server hosting": "$20-50/month",
    "PostgreSQL hosting": "$20/month (или бесплатно локально)",
    
    "Total initial": "$40-85",
    "Monthly running": "$40-70"
}
```

---

## 🎉 Expected Results

После выполнения плана у тебя будет:

1. **Fine-tuned QWEN модель** обученная на твоих 57K треков
2. **RAG система** с semantic search по твоей базе
3. **Hybrid generator** комбинирующий оба подхода  
4. **CLI + Web интерфейсы** для удобного использования
5. **Production-ready система** с Docker deployment

**Качество:** Генерирует authentic rap lyrics с natural flow, основанные на patterns из твоих successful треков

**Производительность:** ~3-10 секунд на генерацию, зависит от complexity

**Масштабируемость:** Готово к добавлению новых треков и обновлению модели

---

## 🚀 Next Level Features (После MVP)

1. **Multi-language support** (русский рэп)
2. **Voice synthesis** интеграция  
3. **Beat matching** под музыку
4. **A/B testing** разных моделей
5. **User feedback** система для improvement
6. **API endpoints** для интеграции в другие приложения

**Братан, это полный battle plan от нуля до production rap generator! Готов начинать?** 🔥💪
