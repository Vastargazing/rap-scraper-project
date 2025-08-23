# AI Engineer Guide: От Новичка до Junior (2025) (Вконце идёт краткий вариант, можно
использовать в проекте)

> **Цель**: За 30 дней превратиться из новичка в Junior AI Engineer
> 
> **Требования**: Python 3.9+, базовые знания программирования
> 
> **Время изучения**: 2-3 часа в день

## 📋 Содержание
1. [Неделя 1: Основы Промт-Инжиниринга](#неделя-1-основы-промт-инжиниринга)
2. [Неделя 2: LangChain и Цепочки](#неделя-2-langchain-и-цепочки)  
3. [Неделя 3: RAG и Векторные Базы](#неделя-3-rag-и-векторные-базы)
4. [Неделя 4: Агенты и Production](#неделя-4-агенты-и-production)

---

# Неделя 1: Основы Промт-Инжиниринга

## День 1-2: Настройка Окружения и Первые Промпты

### 🔧 Настройка
```bash
# Создаем виртуальное окружение
python -m venv ai_env
source ai_env/bin/activate  # Linux/Mac
# ai_env\Scripts\activate     # Windows

# Устанавливаем базовые библиотеки
pip install openai python-dotenv jupyter
```

### 📝 Создаем .env файл
```bash
OPENAI_API_KEY=your_api_key_here
```

### 💻 Первый код - проверяем подключение
```python
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def ask_gpt(prompt, model="gpt-3.5-turbo"):
    """Простая функция для общения с GPT"""
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

# Тест
result = ask_gpt("Привет! Напиши короткое стихотворение про Python")
print(result)
```

**💡 Что происходит?**
- Мы создаем функцию-обертку для удобства
- `temperature=0.7` означает средний уровень креативности (0 = предсказуемо, 1 = очень креативно)
- Модель получает наш промпт и возвращает ответ

---

## День 3-4: Пять Золотых Принципов Промтинга

### 🎯 Принцип 1: Задайте Четкое Направление

**❌ Плохо:**
```python
prompt = "Напиши про собак"
```

**✅ Хорошо:**
```python
prompt = """
Роль: Ты - ветеринар с 10-летним опытом
Задача: Напиши статью для владельцев собак о правильном питании
Аудитория: Новички-собаководы
Тон: Дружелюбный и понятный
Длина: 300-400 слов
"""
```

### 📊 Принцип 2: Укажите Формат Вывода

```python
def get_structured_response(topic):
    prompt = f"""
    Создай план изучения темы: {topic}
    
    Верни ответ строго в формате JSON:
    {{
        "topic": "название темы",
        "duration_days": число дней,
        "sections": [
            {{"day": 1, "title": "название", "tasks": ["задача1", "задача2"]}},
            {{"day": 2, "title": "название", "tasks": ["задача1", "задача2"]}}
        ]
    }}
    """
    return ask_gpt(prompt)

# Тест
plan = get_structured_response("Машинное обучение")
print(plan)
```

### 🎓 Принцип 3: Покажите Примеры (Few-Shot Learning)

```python
def classify_text_sentiment():
    prompt = """
    Определи тональность текста: позитивная, негативная или нейтральная.
    
    Примеры:
    Текст: "Обожаю этот фильм! Невероятные спецэффекты!"
    Тональность: позитивная
    
    Текст: "Ужасный сервис, никому не рекомендую"
    Тональность: негативная
    
    Текст: "Вчера был дождь, сегодня солнечно"
    Тональность: нейтральная
    
    Теперь классифицируй:
    Текст: "Этот продукт превзошел все мои ожидания!"
    Тональность:
    """
    return ask_gpt(prompt, temperature=0.1)  # Низкая температура для точности

result = classify_text_sentiment()
print(result)
```

### 🔍 Принцип 4: Просите Самооценку (Метапромтинг)

```python
def get_response_with_evaluation(question):
    prompt = f"""
    Ответь на вопрос: {question}
    
    После ответа оцени свой ответ по критериям:
    - Точность (1-10)
    - Полнота (1-10)  
    - Понятность (1-10)
    
    Формат оценки:
    Оценка: Точность: X/10, Полнота: Y/10, Понятность: Z/10
    Комментарий: что можно улучшить
    """
    return ask_gpt(prompt)

result = get_response_with_evaluation("Как работает нейронная сеть?")
print(result)
```

### ⛓️ Принцип 5: Разделите Сложную Задачу (Chain of Thought)

```python
def solve_complex_task(task):
    # Шаг 1: Планирование
    planning_prompt = f"""
    Задача: {task}
    
    Разбей эту задачу на логические шаги.
    Верни только список шагов, пронумерованных 1, 2, 3...
    """
    
    plan = ask_gpt(planning_prompt, temperature=0.3)
    print("План выполнения:")
    print(plan)
    
    # Шаг 2: Выполнение
    execution_prompt = f"""
    План: {plan}
    
    Теперь выполни каждый шаг детально для задачи: {task}
    Рассуждай пошагово, показывай свои мысли.
    """
    
    result = ask_gpt(execution_prompt, temperature=0.5)
    return result

# Пример
task = "Создать план запуска интернет-магазина"
solution = solve_complex_task(task)
print("\nРешение:")
print(solution)
```

---

## День 5-7: Практические Техники и Избежание Ошибок

### 🚫 Борьба с Галлюцинациями

```python
def get_factual_answer(question):
    prompt = f"""
    Вопрос: {question}
    
    Инструкции:
    1. Если ты знаешь точный ответ - дай его
    2. Если не уверен - скажи "Не уверен, рекомендую проверить"
    3. Не выдумывай факты и цифры
    4. Укажи источники, если знаешь
    
    Ответ:
    """
    return ask_gpt(prompt, temperature=0.1)

# Тест
answer = get_factual_answer("Сколько планет в солнечной системе?")
print(answer)
```

### 🔢 Подсчет Токенов

```python
import tiktoken

def count_tokens(text, model="gpt-3.5-turbo"):
    """Подсчитывает количество токенов в тексте"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def split_text_by_tokens(text, max_tokens=1000, overlap=100):
    """Разбивает текст на части по токенам"""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(text)
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        if end == len(tokens):
            break
        start = end - overlap
    
    return chunks

# Тест
long_text = "Очень длинный текст..." * 100
print(f"Токенов в тексте: {count_tokens(long_text)}")
chunks = split_text_by_tokens(long_text, max_tokens=500)
print(f"Разбито на {len(chunks)} частей")
```

---

## 📚 Вопросы для Закрепления - Неделя 1

### Теория (ответьте письменно)
1. Почему важно указывать роль в промпте? Приведите пример.
2. В чем разница между temperature=0.1 и temperature=0.9?
3. Когда использовать Few-Shot Learning вместо Zero-Shot?
4. Как Chain of Thought улучшает качество ответов?

### Практика (напишите код)
1. Создайте функцию, которая переводит текст на любой язык, используя все 5 принципов
2. Реализуйте классификатор эмоций в тексте (радость, грусть, злость, страх, удивление)
3. Напишите промпт для создания резюме по тексту с указанием ключевых фактов в JSON
4. Создайте систему оценки качества статей (критерии: читабельность, полезность, структура)

### Проект недели
Создайте "Умного помощника для изучения" - программу, которая:
- Принимает тему для изучения
- Создает план изучения на неделю
- Генерирует вопросы для самопроверки
- Оценивает сложность темы (1-10)

```python
def create_learning_assistant(topic):
    # Ваш код здесь
    pass
```

---

# Неделя 2: LangChain и Цепочки

## День 8-10: Основы LangChain

### 📦 Установка и Настройка

```bash
pip install langchain langchain-openai langchain-community
```

### 🔗 Ваш Первый Chain

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Настройка модели
llm = ChatOpenAI(
    temperature=0.7,
    model="gpt-3.5-turbo"
)

# Создание шаблона промпта
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Ты - эксперт по {domain}. Отвечай кратко и по делу."),
    ("human", "{question}")
])

# Создание цепочки
chain = prompt_template | llm | StrOutputParser()

# Использование
result = chain.invoke({
    "domain": "машинному обучению", 
    "question": "Что такое градиентный спуск?"
})
print(result)
```

**💡 Что такое LCEL?**
`|` - это LangChain Expression Language (LCEL). Читается как: "промпт передается в модель, результат парсится"

### 🔄 Последовательные Цепочки

```python
from langchain.chains import SequentialChain
from langchain.chains import LLMChain

# Цепочка 1: Генерация идеи
idea_prompt = ChatPromptTemplate.from_template(
    "Придумай идею для {topic}. Верни только одну лучшую идею."
)
idea_chain = LLMChain(llm=llm, prompt=idea_prompt, output_key="idea")

# Цепочка 2: Развитие идеи
develop_prompt = ChatPromptTemplate.from_template(
    "Развей эту идею детально: {idea}. Добавь план реализации."
)
develop_chain = LLMChain(llm=llm, prompt=develop_prompt, output_key="developed_idea")

# Объединение цепочек
overall_chain = SequentialChain(
    chains=[idea_chain, develop_chain],
    input_variables=["topic"],
    output_variables=["idea", "developed_idea"],
    verbose=True
)

# Тест
result = overall_chain.invoke({"topic": "мобильное приложение для изучения языков"})
print("Идея:", result["idea"])
print("\nРазвитие:", result["developed_idea"])
```

---

## День 11-12: Память и Контекст

### 🧠 Типы Памяти в LangChain

```python
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import ConversationChain

# 1. Буферная память (хранит все сообщения)
buffer_memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=buffer_memory,
    verbose=True
)

# Диалог
print(conversation.predict(input="Привет! Меня зовут Алексей"))
print(conversation.predict(input="Какое у меня имя?"))
print(conversation.predict(input="Что ты знаешь обо мне?"))

# Просмотр памяти
print("\nПамять:", buffer_memory.buffer)
```

```python
# 2. Память с суммаризацией (экономит токены)
summary_memory = ConversationSummaryMemory(llm=llm)

conversation_with_summary = ConversationChain(
    llm=llm,
    memory=summary_memory,
    verbose=True
)

# Длинный диалог
topics = [
    "Расскажи о машинном обучении",
    "Какие есть алгоритмы классификации?", 
    "Объясни переобучение",
    "Как выбрать метрики для оценки модели?"
]

for topic in topics:
    response = conversation_with_summary.predict(input=topic)
    print(f"Q: {topic}")
    print(f"A: {response[:100]}...\n")

# Смотрим суммарную память
print("Суммарная память:", summary_memory.buffer)
```

### 💾 Кастомная Память

```python
class SimpleMemory:
    def __init__(self):
        self.facts = {}  # Факты о пользователе
        self.preferences = {}  # Предпочтения
        self.history = []  # История разговоров
    
    def add_fact(self, key, value):
        self.facts[key] = value
    
    def add_preference(self, key, value):  
        self.preferences[key] = value
        
    def add_to_history(self, human_msg, ai_msg):
        self.history.append({"human": human_msg, "ai": ai_msg})
        if len(self.history) > 5:  # Храним только последние 5
            self.history.pop(0)
    
    def get_context(self):
        context = ""
        if self.facts:
            context += f"Факты о пользователе: {self.facts}\n"
        if self.preferences:
            context += f"Предпочтения: {self.preferences}\n"
        if self.history:
            context += "Недавняя история:\n"
            for h in self.history[-3:]:  # Последние 3 сообщения
                context += f"Человек: {h['human']}\nИИ: {h['ai']}\n"
        return context

# Использование
memory = SimpleMemory()

def chat_with_memory(user_input):
    # Добавляем контекст в промпт
    context = memory.get_context()
    
    full_prompt = f"""
    {context}
    
    Пользователь: {user_input}
    
    Ответь с учетом контекста и запомни новую информацию о пользователе.
    """
    
    response = ask_gpt(full_prompt)
    
    # Сохраняем в историю
    memory.add_to_history(user_input, response)
    
    return response

# Тест
print(chat_with_memory("Привет! Я изучаю Python уже 2 месяца"))
memory.add_fact("experience", "Python 2 месяца")

print(chat_with_memory("Мне нравится работать с данными"))
memory.add_preference("interests", "работа с данными")

print(chat_with_memory("Посоветуй книгу для изучения"))
```

---

## День 13-14: Загрузка и Обработка Документов

### 📄 Загрузчики Документов

```python
from langchain.document_loaders import TextLoader, WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Загрузка текстового файла
def load_text_file(file_path):
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    return documents

# 2. Загрузка веб-страницы
def load_webpage(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    return documents

# 3. Загрузка PDF (требует: pip install pypdf)
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

# Тест загрузки веб-страницы
docs = load_webpage("https://ru.wikipedia.org/wiki/Машинное_обучение")
print(f"Загружено {len(docs)} документов")
print(f"Длина первого документа: {len(docs[0].page_content)} символов")
print(f"Первые 200 символов: {docs[0].page_content[:200]}")
```

### ✂️ Разбиение Документов

```python
def smart_text_split(documents, chunk_size=1000, chunk_overlap=200):
    """Умное разбиение текста с сохранением контекста"""
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Приоритет разделителей
    )
    
    splits = text_splitter.split_documents(documents)
    
    # Добавляем метаданные к каждому фрагменту
    for i, split in enumerate(splits):
        split.metadata["chunk_id"] = i
        split.metadata["chunk_size"] = len(split.page_content)
    
    return splits

# Тест
docs = load_webpage("https://ru.wikipedia.org/wiki/Искусственный_интеллект")
chunks = smart_text_split(docs, chunk_size=500, chunk_overlap=50)

print(f"Разбито на {len(chunks)} частей")
for i, chunk in enumerate(chunks[:3]):  # Показываем первые 3
    print(f"\nЧасть {i+1}:")
    print(f"Размер: {len(chunk.page_content)} символов")
    print(f"Содержание: {chunk.page_content[:150]}...")
```

### 📊 Анализ и Суммаризация Документов

```python
from langchain.chains.summarize import load_summarize_chain

def analyze_document(file_path_or_url, analysis_type="summary"):
    """Анализирует документ и возвращает результат"""
    
    # Загрузка
    if file_path_or_url.startswith("http"):
        docs = load_webpage(file_path_or_url)
    else:
        docs = load_text_file(file_path_or_url)
    
    # Разбиение
    chunks = smart_text_split(docs, chunk_size=2000)
    
    if analysis_type == "summary":
        # Суммаризация
        chain = load_summarize_chain(
            llm=llm, 
            chain_type="map_reduce",  # Стратегия для длинных текстов
            verbose=True
        )
        summary = chain.run(chunks)
        return summary
    
    elif analysis_type == "key_points":
        # Ключевые моменты
        prompt = ChatPromptTemplate.from_template(
            "Выдели 5 самых важных моментов из этого текста:\n{text}"
        )
        chain = prompt | llm | StrOutputParser()
        
        # Обрабатываем каждый кусок
        all_points = []
        for chunk in chunks[:5]:  # Берем первые 5 кусков
            points = chain.invoke({"text": chunk.page_content})
            all_points.append(points)
        
        return "\n\n".join(all_points)

# Тест
summary = analyze_document("https://ru.wikipedia.org/wiki/Нейронная_сеть", "summary")
print("Краткое содержание:")
print(summary)
```

---

## 📚 Вопросы для Закрепления - Неделя 2

### Теория
1. В чем преимущество LCEL перед обычными функциями?
2. Когда использовать ConversationBufferMemory, а когда ConversationSummaryMemory?
3. Почему важно правильно разбивать документы на чанки?
4. Какие стратегии суммаризации есть в LangChain и когда какую использовать?

### Практика
1. Создайте цепочку для анализа тональности отзывов с сохранением результатов
2. Реализуйте чат-бота с памятью, который запоминает интересы пользователя
3. Создайте систему обработки PDF-документов с извлечением ключевых фактов
4. Напишите цепочку для перевода и улучшения текстов

### Проект недели
Создайте "Личного Аналитика Документов":
- Загружает документы разных форматов
- Создает краткое содержание
- Отвечает на вопросы по содержанию
- Сравнивает несколько документов
- Сохраняет историю анализа

---

# Неделя 3: RAG и Векторные Базы

## День 15-17: Основы RAG (Retrieval-Augmented Generation)

### 🎯 Что такое RAG и зачем он нужен?

**Проблема**: LLM могут "галлюцинировать" - придумывать факты
**Решение**: RAG = поиск реальной информации + генерация ответа

```python
# Простой пример без RAG vs с RAG

def ask_without_rag(question):
    """Обычный запрос к LLM"""
    return ask_gpt(f"Ответь на вопрос: {question}")

def ask_with_rag(question, knowledge_base):
    """Запрос с добавлением контекста"""
    # Ищем релевантную информацию
    relevant_info = search_in_knowledge(question, knowledge_base)
    
    prompt = f"""
    Контекст: {relevant_info}
    
    Вопрос: {question}
    
    Ответь на вопрос, используя только информацию из контекста.
    Если информации недостаточно, скажи об этом.
    """
    return ask_gpt(prompt)

# Тест
question = "Когда была основана компания OpenAI?"

print("Без RAG:", ask_without_rag(question))
# Может дать неточный ответ

knowledge = ["OpenAI была основана в декабре 2015 года Сэмом Альтманом и Илоном Маском"]
print("С RAG:", ask_with_rag(question, knowledge))
# Точный ответ на основе фактов
```

### 🔢 Эмбеддинги - превращаем текст в числа

```python
from langchain.embeddings import OpenAIEmbeddings
import numpy as np

# Инициализация
embeddings_model = OpenAIEmbeddings()

def get_embeddings_example():
    """Показывает, как работают эмбеддинги"""
    
    texts = [
        "Собака - домашнее животное",
        "Кот любит рыбу", 
        "Python - язык программирования",
        "Machine Learning - область ИИ",
        "Щенок играет во дворе"
    ]
    
    # Получаем векторы
    vectors = embeddings_model.embed_documents(texts)
    
    print(f"Количество текстов: {len(texts)}")
    print(f"Размер вектора: {len(vectors[0])}")
    
    # Считаем схожесть
    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    print("\nСхожесть между текстами:")
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            similarity = cosine_similarity(vectors[i], vectors[j])
            print(f"'{texts[i]}' <-> '{texts[j]}': {similarity:.3f}")

# Запуск
get_embeddings_example()
```

**💡 Что мы увидим?**
- Тексты про животных будут похожи друг на друга
- Тексты про программирование - тоже
- Схожесть измеряется числами от -1 до 1

---

## День 18-19: Векторные Базы Данных

### 📚 FAISS - локальная векторная БД

```bash
pip install faiss-cpu langchain-community
```

```python
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

def create_local_knowledge_base(documents_folder):
    """Создает локальную базу знаний"""
    
    # Загружаем документы
    all_docs = []
    import os
    for filename in os.listdir(documents_folder):
        if filename.endswith('.txt'):
            loader = TextLoader(os.path.join(documents_folder, filename))
            docs = loader.load()
            all_docs.extend(docs)
    
    # Разбиваем на чанки
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(all_docs)
    
    # Создаем векторную БД
    vectorstore = FAISS.from_documents(chunks, embeddings_model)
    
    # Сохраняем локально
    vectorstore.save_local("local_knowledge_base")
    
    return vectorstore

def load_knowledge_base():
    """Загружает сохраненную базу знаний"""
    return FAISS.load_local("local_knowledge_base", embeddings_model)

def search_knowledge(query, vectorstore, k=3):
    """Поиск в базе знаний"""
    results = vectorstore.similarity_search(query, k=k)
    
    print(f"Найдено {len(results)} релевантных документов для запроса: '{query}'\n")
    
    for i, doc in enumerate(results, 1):
        print(f"Документ {i}:")
        print(f"Содержание: {doc.page_content[:200]}...")
        print(f"Метаданные: {doc.metadata}")
        print("-" * 50)
    
    return results

# Пример использования
# vectorstore = create_local_knowledge_base("my_documents/")
# results = search_knowledge("машинное обучение", vectorstore)
```

### ☁️ Pinecone - облачная векторная БД

```bash
pip install pinecone-client
```

```python
import pinecone
from langchain.vectorstores import Pinecone

def setup_pinecone_db(api_key, environment, index_name):
    """Настройка Pinecone"""
    
    # Инициализация
    pinecone.init(api_key=api_key, environment=environment)
    
    # Создание индекса (если не существует)
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            dimension=1536,  # Размер векторов OpenAI
            metric="cosine"
        )
    
    return pinecone.Index(index_name)

def create_pinecone_vectorstore(texts, index_name="ai-knowledge"):
    """Создает векторное хранилище в Pinecone"""
    
    # Настройка (замените на ваши данные)
    index = setup_pinecone_db(
        api_key="your-pinecone-api-key",
        environment="your-environment", 
        index_name=index_name
    )
    
    # Создание векторного хранилища
    vectorstore = Pinecone.from_texts(
        texts=texts,
        embedding=embeddings_model,
        index_name=index_name
    )
    
    return vectorstore

# Пример использования
texts = [
    "Python - высокоуровневый язык программирования",
    "Машинное обучение - подраздел искусственного интеллекта",
    "Нейронные сети вдохновлены работой человеческого мозга"
]

# vectorstore = create_pinecone_vectorstore(texts)
```

---

## День 20-21: Продвинутый RAG

### 🔍 Умный RAG с улучшенным поиском

```python
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

class SmartRAG:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(temperature=0)
        
        # Создаем компрессор для улучшения релевантности
        compressor = LLMChainExtractor.from_llm(self.llm)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
        )
        
        # Создаем QA цепочку
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            verbose=True
        )
    
    def ask(self, question):
        """Задает вопрос и возвращает ответ с источниками"""
        result = self.qa_chain.invoke({"query": question})
        
        response = {
            "answer": result["result"],
            "sources": []
        }
        
        # Добавляем информацию об источниках
        for doc in result["source_documents"]:
            response["sources"].append({
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata
            })
        
        return response
    
    def multi_query_search(self, question):
        """Поиск по нескольким переформулированным запросам"""
        
        # Генерируем альтернативные формулировки вопроса
        reformulate_prompt = f"""
        Исходный вопрос: {question}
        
        Создай 3 альтернативные формулировки этого вопроса для поиска:
        1.
        2. 
        3.
        """
        
        reformulated = ask_gpt(reformulate_prompt, temperature=0.3)
        queries = [question] + reformulated.split('\n')[1:4]
        
        # Поиск по каждому запросу
        all_results = []
        for query in queries:
            if query.strip():
                results = self.vectorstore.similarity_search(query.strip(), k=2)
                all_results.extend(results)
        
        # Удаляем дубликаты
        unique_results = []
        seen_content = set()
        for doc in all_results:
            if doc.page_content not in seen_content:
                unique_results.append(doc)
                seen_content.add(doc.page_content)
        
        return unique_results[:5]  # Топ-5 результатов

# Пример использования
def demo_smart_rag():
    # Создаем тестовые документы
    documents = [
        "Машинное обучение - это метод анализа данных, который автоматизирует построение аналитических моделей.",
        "Глубокое обучение - подмножество машинного обучения, использующее нейронные сети с множеством слоев.",
        "Supervised learning требует размеченных данных для обучения модели.",
        "Unsupervised learning работает с неразмеченными данными, ищет скрытые паттерны.",
        "Reinforcement learning - обучение через взаимодействие с средой и получение наград."
    ]
    
    # Создаем векторное хранилище
    vectorstore = FAISS.from_texts(documents, embeddings_model)
    
    # Инициализируем SmartRAG
    smart_rag = SmartRAG(vectorstore)
    
    # Тестируем
    questions = [
        "Что такое machine learning?",
        "Какие виды обучения существуют?",
        "Нужны ли размеченные данные для supervised learning?"
    ]
    
    for question in questions:
        print(f"Вопрос: {question}")
        result = smart_rag.ask(question)
        print(f"Ответ: {result['answer']}")
        print(f"Источников: {len(result['sources'])}")
        print("-" * 50)

# demo_smart_rag()
```

### 🔄 RAG с переранжированием

```python
from sentence_transformers import CrossEncoder

class RerankingRAG:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(temperature=0)
        
        # Модель для переранжирования (требует: pip install sentence-transformers)
        # self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def search_and_rerank(self, query, k=10, rerank_top=3):
        """Поиск с переранжированием результатов"""
        
        # Первичный поиск - берем больше результатов
        initial_results = self.vectorstore.similarity_search(query, k=k)
        
        # Простое переранжирование по длине совпадающих слов
        # (в реальности используйте CrossEncoder выше)
        query_words = set(query.lower().split())
        
        scored_results = []
        for doc in initial_results:
            doc_words = set(doc.page_content.lower().split())
            score = len(query_words.intersection(doc_words))
            scored_results.append((score, doc))
        
        # Сортируем по релевантности
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        # Возвращаем топ результаты
        return [doc for score, doc in scored_results[:rerank_top]]
    
    def answer_with_reranking(self, question):
        """Ответ с переранжированием"""
        
        # Получаем лучшие документы
        relevant_docs = self.search_and_rerank(question, k=8, rerank_top=3)
        
        # Формируем контекст
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        prompt = f"""
        Контекст:
        {context}
        
        Вопрос: {question}
        
        Ответь на вопрос, используя информацию из контекста.
        Если информации недостаточно, скажи об этом честно.
        """
        
        answer = ask_gpt(prompt, temperature=0.1)
        return {
            "answer": answer,
            "context_docs": len(relevant_docs),
            "context_length": len(context)
        }
```

---

## 📚 Вопросы для Закрепления - Неделя 3

### Теория
1. Объясните схему работы RAG простыми словами
2. В чем разница между FAISS и Pinecone? Когда что использовать?
3. Зачем нужно переранжирование результатов поиска?
4. Как эмбеддинги помогают находить семантически похожие тексты?

### Практика
1. Создайте RAG систему для анализа научных статей
2. Реализуйте поиск по базе знаний с фильтрацией по метаданным
3. Создайте систему сравнения документов через векторный поиск
4. Напишите функцию для автоматического улучшения поисковых запросов

### Проект недели
Создайте "Персональную Базу Знаний":
- Загружает документы разных форматов (PDF, TXT, веб-страницы)
- Создает семантический индекс
- Отвечает на вопросы по загруженным документам
- Показывает источники ответов
- Позволяет добавлять/удалять документы

```python
class PersonalKnowledgeBase:
    def __init__(self):
        # Ваша реализация
        pass
    
    def add_document(self, path_or_url):
        # Загрузка и индексация
        pass
    
    def ask(self, question):
        # Поиск и ответ
        pass
    
    def list_sources(self):
        # Список всех источников
        pass
```

---

# Неделя 4: Агенты и Production

## День 22-24: Создание AI Агентов

### 🤖 Что такое AI Агент?

AI Агент = LLM + Инструменты + Способность к рассуждению и действию

```python
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
import requests
import json

# Создаем инструменты для агента
def search_wikipedia(query):
    """Поиск в Википедии"""
    try:
        url = f"https://ru.wikipedia.org/api/rest_v1/page/summary/{query}"
        response = requests.get(url)
        data = response.json()
        return data.get('extract', 'Информация не найдена')
    except:
        return "Ошибка при поиске в Википедии"

def calculate(expression):
    """Простой калькулятор"""
    try:
        # ВНИМАНИЕ: в продакшене используйте безопасные методы!
        result = eval(expression)
        return f"Результат: {result}"
    except:
        return "Ошибка в вычислении"

def get_weather(city):
    """Получение погоды (заглушка)"""
    # В реальности подключите API погоды
    return f"В городе {city} сегодня солнечно, +20°C"

# Создаем список инструментов
tools = [
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="Поиск информации в Википедии. Используй когда нужны факты или определения."
    ),
    Tool(
        name="Calculator", 
        func=calculate,
        description="Вычисления. Используй для математических задач. Входной параметр - математическое выражение."
    ),
    Tool(
        name="Weather",
        func=get_weather,
        description="Получение информации о погоде. Входной параметр - название города."
    )
]

# Создаем агента
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate"
)

# Тестируем агента
def test_agent():
    test_queries = [
        "Что такое машинное обучение и посчитай 25 * 4?",
        "Какая погода в Москве и найди информацию про Python?", 
        "Посчитай площадь круга с радиусом 5 и расскажи про Альберта Эйнштейна"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"ЗАПРОС: {query}")
        print(f"{'='*50}")
        
        try:
            result = agent.run(query)
            print(f"ОТВЕТ: {result}")
        except Exception as e:
            print(f"ОШИБКА: {e}")

# test_agent()
```

### 🔧 Продвинутый Агент с Памятью

```python
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory

class SmartAgent:
    def __init__(self, tools, memory_window=10):
        self.llm = ChatOpenAI(temperature=0.1, model="gpt-4")
        self.tools = tools
        
        # Память для контекста разговора
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=memory_window,
            return_messages=True
        )
        
        # Создаем агента
        self.agent = ConversationalChatAgent.from_llm_and_tools(
            llm=self.llm,
            tools=self.tools,
            verbose=True
        )
        
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=5
        )
    
    def chat(self, message):
        """Диалог с агентом"""
        try:
            response = self.agent_executor.run(input=message)
            return response
        except Exception as e:
            return f"Произошла ошибка: {e}"
    
    def get_chat_history(self):
        """Получить историю разговора"""
        return self.memory.chat_memory.messages

# Создаем расширенные инструменты
def web_search(query):
    """Поиск в интернете (заглушка)"""
    return f"Результаты поиска для '{query}': [здесь были бы результаты поиска]"

def save_note(note):
    """Сохранение заметки"""
    with open("agent_notes.txt", "a", encoding="utf-8") as f:
        f.write(f"{note}\n")
    return "Заметка сохранена"

def read_notes():
    """Чтение всех заметок"""
    try:
        with open("agent_notes.txt", "r", encoding="utf-8") as f:
            notes = f.read()
        return notes if notes else "Заметок пока нет"
    except FileNotFoundError:
        return "Файл с заметками не найден"

# Расширенный набор инструментов
advanced_tools = [
    Tool(name="WebSearch", func=web_search, description="Поиск информации в интернете"),
    Tool(name="Calculator", func=calculate, description="Математические вычисления"),
    Tool(name="SaveNote", func=save_note, description="Сохранение важной информации"),
    Tool(name="ReadNotes", func=read_notes, description="Чтение сохраненных заметок"),
    Tool(name="Weather", func=get_weather, description="Информация о погоде")
]

# Создаем умного агента
smart_agent = SmartAgent(advanced_tools)

# Тест
def test_smart_agent():
    conversations = [
        "Привет! Меня зовут Алексей. Сохрани это в заметках.",
        "Какая погода в Санкт-Петербурге?",
        "Посчитай 15% от 1000",
        "Как меня зовут? Проверь в заметках.",
        "Сохрани результат предыдущего вычисления"
    ]
    
    for msg in conversations:
        print(f"\nПользователь: {msg}")
        response = smart_agent.chat(msg)
        print(f"Агент: {response}")

# test_smart_agent()
```

---

## День 25-26: Развертывание в Production

### 🚀 FastAPI для AI сервисов

```bash
pip install fastapi uvicorn python-multipart
```

```python
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import uvicorn
import os
from typing import List, Optional

app = FastAPI(title="AI Engineer API", version="1.0.0")

# Модели данных
class ChatRequest(BaseModel):
    message: str
    temperature: Optional[float] = 0.7
    use_rag: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    tokens_used: Optional[int] = None
    sources: Optional[List[str]] = None

class DocumentRequest(BaseModel):
    text: str
    task: str  # "summarize", "analyze", "translate"

# Глобальные переменные для сервисов
llm_service = None
rag_service = None
agent_service = None

@app.on_event("startup")
async def startup_event():
    """Инициализация сервисов при запуске"""
    global llm_service, rag_service, agent_service
    
    print("Загрузка AI сервисов...")
    
    # Инициализация LLM
    llm_service = ChatOpenAI(temperature=0.7)
    
    # Инициализация RAG (если есть индекс)
    if os.path.exists("local_knowledge_base"):
        rag_service = FAISS.load_local("local_knowledge_base", OpenAIEmbeddings())
    
    print("AI сервисы загружены!")

@app.get("/")
async def root():
    return {"message": "AI Engineer API готов к работе!"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Основной чат эндпоинт"""
    try:
        if request.use_rag and rag_service:
            # Используем RAG
            relevant_docs = rag_service.similarity_search(request.message, k=3)
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            prompt = f"""Контекст: {context}
            
            Вопрос: {request.message}
            
            Ответь на основе контекста."""
            
            response = ask_gpt(prompt, temperature=request.temperature)
            sources = [doc.metadata.get("source", "unknown") for doc in relevant_docs]
            
            return ChatResponse(
                response=response,
                sources=sources
            )
        else:
            # Обычный чат
            response = ask_gpt(request.message, temperature=request.temperature)
            return ChatResponse(response=response)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/document/process")
async def process_document(request: DocumentRequest):
    """Обработка документа"""
    try:
        if request.task == "summarize":
            prompt = f"Кратко перескажи основные моменты:\n{request.text}"
        elif request.task == "analyze":
            prompt = f"Проанализируй тональность и ключевые темы:\n{request.text}"
        elif request.task == "translate":
            prompt = f"Переведи на английский:\n{request.text}"
        else:
            raise HTTPException(status_code=400, detail="Неподдерживаемая задача")
        
        result = ask_gpt(prompt, temperature=0.3)
        return {"result": result, "task": request.task}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Загрузка файла для добавления в базу знаний"""
    try:
        # Сохраняем файл
        file_path = f"uploads/{file.filename}"
        os.makedirs("uploads", exist_ok=True)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Обрабатываем файл (добавляем в RAG)
        # Здесь был бы код для добавления в векторную БД
        
        return {"message": f"Файл {file.filename} загружен и обработан"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Проверка состояния сервиса"""
    return {
        "status": "healthy",
        "llm_available": llm_service is not None,
        "rag_available": rag_service is not None,
        "timestamp": "2025-01-01T00:00:00Z"
    }

# Запуск сервера
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 📊 Мониторинг и Логирование

```python
import logging
import time
from functools import wraps
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AIMetrics:
    """Класс для сбора метрик"""
    def __init__(self):
        self.requests_count = 0
        self.total_tokens = 0
        self.error_count = 0
        self.response_times = []
    
    def log_request(self, tokens_used, response_time, success=True):
        self.requests_count += 1
        if tokens_used:
            self.total_tokens += tokens_used
        self.response_times.append(response_time)
        if not success:
            self.error_count += 1
    
    def get_stats(self):
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        return {
            "total_requests": self.requests_count,
            "total_tokens": self.total_tokens,
            "error_rate": self.error_count / max(self.requests_count, 1),
            "avg_response_time": avg_response_time
        }

metrics = AIMetrics()

def monitor_performance(func):
    """Декоратор для мониторинга производительности"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        success = True
        tokens_used = None
        
        try:
            result = func(*args, **kwargs)
            # Попытка извлечь информацию о токенах
            if hasattr(result, 'usage'):
                tokens_used = result.usage.total_tokens
            return result
        except Exception as e:
            success = False
            logger.error(f"Ошибка в {func.__name__}: {str(e)}")
            raise
        finally:
            response_time = time.time() - start_time
            metrics.log_request(tokens_used, response_time, success)
            logger.info(f"{func.__name__} выполнен за {response_time:.2f}с")
    
    return wrapper

# Добавляем эндпоинт для метрик
@app.get("/metrics")
async def get_metrics():
    """Получение метрик сервиса"""
    return metrics.get_stats()

# Пример использования декоратора
@monitor_performance
def monitored_gpt_call(prompt, temperature=0.7):
    return ask_gpt(prompt, temperature)
```

---

## День 27-28: Тестирование и Оптимизация

### 🧪 Тестирование AI Систем

```python
import pytest
import json
from unittest.mock import patch, MagicMock

class TestAISystem:
    """Тесты для AI системы"""
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.test_prompts = [
            "Что такое машинное обучение?",
            "Объясни разницу между AI и ML",
            "Как работают нейронные сети?"
        ]
    
    def test_prompt_generation(self):
        """Тест генерации промптов"""
        from your_ai_module import create_prompt
        
        prompt = create_prompt("Объясни квантовую физику", role="учитель")
        
        assert "учитель" in prompt.lower()
        assert "квантовую физику" in prompt
        assert len(prompt) > 50  # Промпт должен быть достаточно детальным
    
    @patch('openai.ChatCompletion.create')
    def test_llm_response(self, mock_openai):
        """Тест ответа LLM с мокированием"""
        # Мокируем ответ OpenAI
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Тестовый ответ"
        mock_openai.return_value = mock_response
        
        from your_ai_module import ask_gpt
        result = ask_gpt("Тестовый вопрос")
        
        assert result == "Тестовый ответ"
        mock_openai.assert_called_once()
    
    def test_rag_retrieval(self):
        """Тест системы RAG"""
        # Создаем тестовую базу знаний
        test_docs = [
            "Python - язык программирования",
            "Machine learning - область AI",
            "Deep learning использует нейронные сети"
        ]
        
        # Тестируем поиск
        from your_ai_module import create_test_vectorstore, search_relevant
        vectorstore = create_test_vectorstore(test_docs)
        results = search_relevant("программирование", vectorstore, k=1)
        
        assert len(results) == 1
        assert "Python" in results[0].page_content
    
    def test_agent_tools(self):
        """Тест инструментов агента"""
        from your_ai_module import calculate, search_wikipedia
        
        # Тест калькулятора
        result = calculate("2 + 2")
        assert "4" in result
        
        # Тест поиска (с мокированием)
        with patch('requests.get') as mock_get:
            mock_get.return_value.json.return_value = {
                'extract': 'Тестовая статья'
            }
            result = search_wikipedia("Python")
            assert "Тестовая статья" in result

def run_quality_tests():
    """Тесты качества ответов"""
    
    test_cases = [
        {
            "input": "Объясни машинное обучение простыми словами",
            "expected_keywords": ["данные", "алгоритм", "обучение", "модель"],
            "max_length": 500
        },
        {
            "input": "Переведи на английский: 'Привет, как дела?'",
            "expected_keywords": ["Hello", "how", "are", "you"],
            "max_length": 100
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"Тест {i+1}: {test_case['input']}")
        
        response = ask_gpt(test_case['input'])
        
        # Проверка длины
        assert len(response) <= test_case['max_length'], f"Ответ слишком длинный: {len(response)}"
        
        # Проверка ключевых слов
        found_keywords = sum(1 for keyword in test_case['expected_keywords'] 
                           if keyword.lower() in response.lower())
        
        coverage = found_keywords / len(test_case['expected_keywords'])
        assert coverage >= 0.5, f"Недостаточно релевантных слов: {coverage:.2%}"
        
        print(f"✅ Пройден (покрытие ключевых слов: {coverage:.2%})")

# Запуск тестов
if __name__ == "__main__":
    run_quality_tests()
```

### ⚡ Оптимизация Производительности

```python
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import time

class OptimizedAIService:
    """Оптимизированный AI сервис с кэшированием и асинхронностью"""
    
    def __init__(self):
        self.cache = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.llm = ChatOpenAI(temperature=0.7)
    
    def _cache_key(self, prompt, temperature):
        """Создание ключа для кэша"""
        return f"{hash(prompt)}_{temperature}"
    
    async def cached_gpt_call(self, prompt, temperature=0.7):
        """LLM вызов с кэшированием"""
        cache_key = self._cache_key(prompt, temperature)
        
        # Проверяем кэш
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Асинхронный вызов LLM
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            lambda: ask_gpt(prompt, temperature)
        )
        
        # Сохраняем в кэш
        self.cache[cache_key] = result
        return result
    
    async def batch_process(self, prompts, temperature=0.7):
        """Пакетная обработка промптов"""
        tasks = [
            self.cached_gpt_call(prompt, temperature) 
            for prompt in prompts
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    def optimize_prompt_length(self, prompt, max_tokens=3000):
        """Оптимизация длины промпта"""
        if count_tokens(prompt) <= max_tokens:
            return prompt
        
        # Сокращаем промпт сохраняя важные части
        lines = prompt.split('\n')
        important_lines = []
        current_tokens = 0
        
        # Приоритет: начало и конец промпта
        for line in lines[:5] + lines[-5:]:  # Первые и последние 5 строк
            line_tokens = count_tokens(line)
            if current_tokens + line_tokens <= max_tokens:
                important_lines.append(line)
                current_tokens += line_tokens
            else:
                break
        
        return '\n'.join(important_lines)

# Пример использования оптимизированного сервиса
async def demo_optimized_service():
    service = OptimizedAIService()
    
    # Тест пакетной обработки
    test_prompts = [
        "Что такое Python?",
        "Объясни машинное обучение",
        "Как работает Git?",
        "Что такое Docker?"
    ]
    
    start_time = time.time()
    results = await service.batch_process(test_prompts)
    end_time = time.time()
    
    print(f"Обработано {len(test_prompts)} промптов за {end_time - start_time:.2f}с")
    
    # Тест кэширования
    start_time = time.time()
    cached_result = await service.cached_gpt_call("Что такое Python?")  # Должен быть из кэша
    end_time = time.time()
    
    print(f"Кэшированный запрос выполнен за {end_time - start_time:.4f}с")

# asyncio.run(demo_optimized_service())
```

### 💰 Управление Затратами

```python
class CostTracker:
    """Отслеживание затрат на API вызовы"""
    
    def __init__(self):
        # Цены за 1K токенов (примерные, проверьте актуальные)
        self.pricing = {
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "text-embedding-ada-002": {"input": 0.0001, "output": 0}
        }
        self.usage_log = []
    
    def log_usage(self, model, input_tokens, output_tokens):
        """Логирование использования токенов"""
        if model not in self.pricing:
            model = "gpt-3.5-turbo"  # По умолчанию
        
        input_cost = (input_tokens / 1000) * self.pricing[model]["input"]
        output_cost = (output_tokens / 1000) * self.pricing[model]["output"]
        total_cost = input_cost + output_cost
        
        usage_record = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": total_cost
        }
        
        self.usage_log.append(usage_record)
        return total_cost
    
    def get_daily_cost(self, date=None):
        """Получение затрат за день"""
        if not date:
            date = datetime.now().date()
        
        daily_cost = sum(
            record["cost"] for record in self.usage_log
            if datetime.fromisoformat(record["timestamp"]).date() == date
        )
        return daily_cost
    
    def estimate_monthly_cost(self):
        """Оценка месячных затрат"""
        if not self.usage_log:
            return 0
        
        # Берем затраты за последние 7 дней и экстраполируем на месяц
        week_ago = datetime.now() - timedelta(days=7)
        recent_cost = sum(
            record["cost"] for record in self.usage_log
            if datetime.fromisoformat(record["timestamp"]) >= week_ago
        )
        
        return (recent_cost / 7) * 30  # Оценка на месяц

cost_tracker = CostTracker()

def cost_aware_gpt_call(prompt, temperature=0.7, model="gpt-3.5-turbo"):
    """LLM вызов с отслеживанием затрат"""
    
    # Подсчитываем входные токены
    input_tokens = count_tokens(prompt, model)
    
    # Проверяем бюджет
    daily_cost = cost_tracker.get_daily_cost()
    if daily_cost > 10.0:  # Лимит $10 в день
        raise Exception(f"Превышен дневной лимит затрат: ${daily_cost:.2f}")
    
    # Делаем запрос
    response = ask_gpt(prompt, temperature)
    
    # Подсчитываем выходные токены
    output_tokens = count_tokens(response, model)
    
    # Логируем затраты
    cost = cost_tracker.log_usage(model, input_tokens, output_tokens)
    
    print(f"Затрачено: ${cost:.4f} (всего за день: ${daily_cost + cost:.2f})")
    
    return response
```

---

## 📚 Финальные Вопросы для Закрепления - Неделя 4

### Теория
1. Объясните разницу между простым чат-ботом и AI агентом
2. Какие инструменты может использовать AI агент и как их создавать?
3. Почему важно мониторить производительность AI сервисов?
4. Как оптимизировать затраты на API вызовы?

### Практика
1. Создайте агента для автоматизации рабочих задач (email, календарь, заметки)
2. Реализуйте FastAPI сервис с аутентификацией и ограничением запросов
3. Создайте систему A/B тестирования разных промптов
4. Напишите скрипт для мониторинга качества ответов AI

### Итоговый Проект
Создайте полноценного "AI Помощника Разработчика":

```python
class DeveloperAIAssistant:
    """AI помощник для разработчиков"""
    
    def __init__(self):
        self.tools = self._setup_tools()
        self.agent = self._create_agent()
        self.knowledge_base = self._load_knowledge_base()
    
    def _setup_tools(self):
        """Настройка инструментов"""
        return [
            # Поиск в документации
            # Анализ кода
            # Генерация тестов
            # Поиск багов
            # Объяснение ошибок
        ]
    
    def analyze_code(self, code):
        """Анализ качества кода"""
        pass
    
    def explain_error(self, error_message):
        """Объяснение ошибки и предложение решения"""
        pass
    
    def generate_tests(self, function_code):
        """Генерация unit тестов"""
        pass
    
    def suggest_improvements(self, code):
        """Предложения по улучшению кода"""
        pass
    
    def search_documentation(self, query, library):
        """Поиск в документации библиотек"""
        pass
```

**Требования к итоговому проекту:**
- Использует все изученные техники (промптинг, RAG, агенты)
- Имеет веб-интерфейс (FastAPI + простой HTML)
- Логирует все действия
- Оптимизирован для production
- Покрыт тестами
- Документирован

---

## 🎓 Поздравляем! Что дальше?

### Вы изучили:
✅ Основы промт-инжиниринга и работы с LLM  
✅ LangChain для создания сложных цепочек  
✅ RAG для работы с внешними знаниями  
✅ AI агентов с инструментами  
✅ Развертывание в production  

### Следующие шаги для Junior AI Engineer:

1. **Углубленное изучение:**
   - Fine-tuning моделей
   - Векторные базы данных (Weaviate, Qdrant)
   - Мультимодальные модели (текст + изображения)

2. **Специализация:**
   - Computer Vision AI
   - NLP специалист  
   - MLOps Engineer
   - AI Product Manager

3. **Практика:**
   - Участие в Open Source проектах
   - Kaggle соревнования
   - Создание собственных AI продуктов

4. **Сообщество:**
   - AI/ML митапы
   - Конференции (NeurIPS, ICML)
   - Telegram/Discord каналы

### Полезные ресурсы:
- **Документация**: OpenAI, Anthropic, LangChain
- **Курсы**: Andrew Ng (Coursera), Fast.ai
- **Книги**: "Hands-on Machine Learning", "Deep Learning" (Ian Goodfellow)
- **YouTube**: Two Minute Papers, 3Blue1Brown
- **Подкасты**: Lex Fridman, The AI Podcast

**Помните**: AI развивается очень быстро. Главное - постоянно учиться и практиковаться!

---

## 📖 Дополнительные Материалы

### Чек-лист Junior AI Engineer:

**Технические навыки:**
- [ ] Python (pandas, numpy, requests)
- [ ] Работа с API (OpenAI, Anthropic)
- [ ] LangChain/LlamaIndex
- [ ] Векторные БД (FAISS, Pinecone)
- [ ] FastAPI/Flask
- [ ] Docker основы
- [ ] Git/GitHub

**Понимание концепций:**
- [ ] Transformer архитектура
- [ ] Эмбеддинги и семантический поиск
- [ ] Prompt engineering лучшие практики
- [ ] RAG паттерны
- [ ] AI агенты и инструменты
- [ ] Оценка качества LLM
- [ ] AI этика и безопасность

**Практические навыки:**
- [ ] Создание чат-ботов
- [ ] Системы вопрос-ответ
- [ ] Обработка документов
- [ ] Интеграция с внешними API
- [ ] Мониторинг и логирование
- [ ] A/B тестирование промптов

Удачи в изучении AI! 🚀

# Актуальные Темы для AI Engineer: Промт-Инжиниринг для GenAI (2025)

Этот Markdown содержит ключевые извлечения и summaries из книги "Промт-инжиниринг для GenAI" (Джеймс Феникс, Майк Тейлор). Я исключил ненужные детали (введение, отзывы, предисловия), фокусируясь на практических концепциях, примерах и коде. Материал структурирован для обучения: теория + примеры + советы. Начну с одной темы за раз, как предложил. Если нужно углубить/расширить, скажи.

Для обучения:
- Читайте последовательно.
- Практикуйте код в Jupyter/Colab (требования: Python 3.9+, OpenAI API, LangChain и др.).
- Ключевые модели: GPT-4, Claude, Gemini, Llama (актуальны в 2025).
- Установка: `pip install -r requirements.txt` из репозитория GitHub (https://oreil.ly/BrightPool).

## Тема 1: Основы Промтинга и ChatGPT Техники (Главы 1-3)

### Пять Принципов Промтинга (Глава 1)
Фундаментальные правила для надежных запросов в LLM (работают в GPT-4, Claude, Gemini, Llama).

1. **Задайте Направление**: Укажите цель, роль, контекст. Избегайте неоднозначности.
   - Пример: "Ты - SEO-эксперт. Опиши стратегию для сайта e-commerce."
   - Совет: Добавляйте "шаг за шагом" для CoT (Chain of Thought) - улучшает reasoning.

2. **Укажите Формат**: Определите вывод (JSON, список, таблица).
   - Пример: "Верни в формате JSON: {'title': str, 'summary': str}."
   - Совет: Используйте для парсинга вывода, избегайте галлюцинаций.

3. **Приведите Примеры** (Few-Shot Learning): 1-3 примера для демонстрации.
   - Пример: "Пример 1: Вход: 'apple'. Выход: 'fruit'. Вход: 'car'. Выход: ?"
   - Совет: Few-shot работает во всех LLM; для CoT: "Рассуждай шаг за шагом."

4. **Оцените Качество**: Попросите LLM самооценить ответ (метапромтинг).
   - Пример: "Оцени свой ответ по шкале 1-10 по точности и полноте."
   - Совет: Используйте для итераций; добавляйте критерии (точность, оригинальность).

5. **Разделите Задачу**: Разбейте на подзадачи (chains).
   - Пример: Сначала сгенерируй план, потом текст.
   - Совет: Снижает ошибки; актуально для агентов.

Практика: Тестируйте в ChatGPT. Для кода - используйте OpenAI API:
```python
import openai
openai.api_key = "your_key"
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "system", "content": "Ты - эксперт."}, {"role": "user", "content": "Запрос с принципами."}]
)
print(response.choices[0].message.content)
```

### Введение в LLM (Глава 2)
- **Архитектура**: Трансформеры (attention для контекста). Вероятностная генерация (next-token prediction).
- **Модели 2025**:
  - GPT-4: Мультимодальный (текст+видео), сильный в reasoning.
  - Gemini (Google): Интегрирует поиск, мультимодален.
  - Llama (Meta): Open-source, квантизация/LoRA для efficiency.
  - Claude (Anthropic): Фокус на safety, длинный контекст.
- Совет: Сравнивайте по метрикам (точность, скорость). Используйте LoRA для fine-tuning на локальных GPU.

### Стандартные Техники ChatGPT (Глава 3)
- **Few-Shot/CoT**: Для задач вроде классификации/анализа.
  - CoT Пример: "Решай задачу шаг за шагом: 2+2=?"
- **Ролевые Запросы**: "Ты - учитель. Объясни квантовая механика как 5-летнему."
- **Анализ Настроений/Классификация**: "Классифицируй текст: позитив/негатив."
- **Разбиение Текста**: Используйте tiktoken для чанков (chunk_size=1500, overlap=400).
  - Код:
    ```python
    import tiktoken
    encoding = tiktoken.encoding_for_model("gpt-4")
    tokens = encoding.encode("Текст")
    print(len(tokens))  # Оцени число токенов
    ```
- **Избегание Галлюцинаций**: Добавляйте референсы, "Думай шаг за шагом".
- Практика: Генерируйте JSON/YAML, анализируйте sentiment. Тестируйте на 10+ примерах.

Обучение: Практикуйте 5 принципов на 5 задачах (генерация текста, классификация). Перейди к следующей теме после.

## Тема 2: LangChain (Глава 4)

### Введение и Настройка
LangChain - фреймворк для chains/agents с LLM. Установка: `pip install langchain openai`.
- **Модели Чатов**: 
  ```python
  from langchain_openai import ChatOpenAI
  llm = ChatOpenAI(temperature=0.7)  # Temperature для креативности
  ```
- **Шаблоны Запросов** (PromptTemplate):
  ```python
  from langchain.prompts import PromptTemplate
  template = "Переведи {text} на {lang}."
  prompt = PromptTemplate.from_template(template)
  chain = prompt | llm
  result = chain.invoke({"text": "Hello", "lang": "Russian"})
  ```

### Ключевые Концепции
- **Цепочки (Chains)**: Последовательные (SequentialChain), LCEL (prompt | llm | parser).
  - Пример: Разбиение + обобщение.
- **Парсеры Выводов**: Pydantic для структурированного вывода.
  ```python
  from langchain.output_parsers import PydanticOutputParser
  from pydantic import BaseModel
  class Response(BaseModel):
      answer: str
  parser = PydanticOutputParser(pydantic_object=Response)
  ```
- **Вызов Функций**: Для tools (e.g., API calls).
  - Пример: Параллельный вызов.
- **Память**: ConversationBufferMemory для контекста.
  ```python
  from langchain.memory import ConversationBufferMemory
  memory = ConversationBufferMemory()
  memory.save_context({"input": "Hi"}, {"output": "Hello"})
  ```
- **Загрузчики Документов**: Для PDF/web.
  ```python
  from langchain.document_loaders import WebBaseLoader
  loader = WebBaseLoader("url")
  docs = loader.load()
  ```

### Продвинутые: Декомпозиция Задач
- Разбиение: RecursiveCharacterTextSplitter.
- Цепочки Документов: Stuff/Refine/MapReduce.
  - MapReduce: Разбить, обобщить части, свести.

Практика: Создайте chain для суммаризации текста. Тестируйте на 3 документах.

## Тема 3: RAG (Retrieval-Augmented Generation) (Глава 5)

### Основы RAG
RAG: Дополняет LLM внешними данными для точности (избегание галлюцинаций).
- Шаги: Embed текст → Храни в векторной БД → Query → Добавь в prompt.

### Эмбеддинги и Векторные БД
- **Эмбеддинги**: Числовые векторы (OpenAI embeddings).
  ```python
  from langchain.embeddings import OpenAIEmbeddings
  embeddings = OpenAIEmbeddings()
  vector = embeddings.embed_query("Текст")
  ```
- **FAISS (локальная)**: 
  ```python
  from langchain.vectorstores import FAISS
  db = FAISS.from_documents(docs, embeddings)
  results = db.similarity_search("Query", k=3)
  ```
- **Pinecone (hosted)**: Установка: `pip install pinecone-client`.
  ```python
  import pinecone
  pinecone.init(api_key="key", environment="env")
  index = pinecone.Index("index_name")
  # Upsert vectors
  ```

### Реализация RAG в LangChain
- Цепочка: 
  ```python
  from langchain.chains import RetrievalQA
  qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
  result = qa.run("Вопрос")
  ```
- Альтернативы: Self-query, multi-query.

Практика: Загрузите PDF, создайте RAG для Q&A. Тестируйте на 5 вопросах (сравните с/без RAG).

## Тема 4: Агенты и Инструменты (Глава 6)

### Агенты
Автономные системы: Reasoning + Action (ReAct).
- **ReAct**: "Think → Act → Observe".
  ```python
  from langchain.agents import initialize_agent, Tool
  tools = [Tool(name="Search", func=google_search)]
  agent = initialize_agent(tools, llm, agent="zero-shot-react-description")
  agent.run("Задача")
  ```

### Инструменты и Память
- **Tools**: Search, calculator, API.
- **Память**: Краткосрочная (buffer), долгосрочная (vector store).
  - Примеры: ConversationBufferWindowMemory (окно), ConversationSummaryMemory (суммаризация).
  ```python
  from langchain.memory import ConversationSummaryBufferMemory
  memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=400)
  ```
- **Callbacks**: Для логирования (verbose=True).

### Продвинутые
- Агенты с функциями OpenAI: Параллельные calls.
- Фреймворки: Planning agents, Tree of Thoughts.

Практика: Создайте агента с 2 tools (search + math). Решите задачу вроде "Найди人口 Китая и умножь на 2".