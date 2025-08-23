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