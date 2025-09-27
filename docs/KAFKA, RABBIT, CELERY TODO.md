### План TO DO по внедрению Kafka/RabbitMQ/Celery

Братан, отличная идея передать это GPT-5 в VS Code — он поможет с кодом! 🔥 Я выбрал RabbitMQ + Celery (проще для старта, чем Kafka; Kafka heavier для big data streaming, но RabbitMQ хватит для твоих AI-задач). План адаптирован под твой стек: скрипты (analyzers, performance_monitor), FastAPI API, PostgreSQL, K8s/ArgoCD. Цель — асинхронная обработка AI-анализов (например, queue tasks для Qwen/Gemma, не блокируй API).

**TO DO Plan: Внедрение RabbitMQ + Celery для AI Tasks**

1. **Подготовка (1-2 дня)**
   - Выбери брокер: RabbitMQ (лёгкий, AMQP protocol). Альтернатива: Kafka если хочешь streaming (для real-time metrics из performance_monitor).
   - Установи локально: Docker compose для dev (добавь в docker-compose.yml).
     ```yaml
     # docker-compose.yml (добавь)
     services:
       rabbitmq:
         image: rabbitmq:3-management
         ports:
           - "5672:5672"  # AMQP
           - "15672:15672"  # Management UI (login: guest/guest)
     ```
     Запусти: `docker-compose up -d`.
   - Установи libs: `pip install celery[redis] pika` (Redis как backend для results, optional).
   - Почитай docs: Celery quickstart, RabbitMQ tutorial (focus on queues/exchanges).

2. **Настройка Celery (2-3 дня)**
   - Создай celery.py в src/ (или scripts/tools/).
     ```python
     # src/celery.py
     from celery import Celery
     app = Celery('rap_analyzer',
                  broker='amqp://guest:guest@localhost:5672//',  # RabbitMQ
                  backend='redis://localhost:6379/0',  # Для results storage
                  include=['src.tasks'])  # Твои tasks
     ```
   - Определи tasks: Переведи AI-анализаторы в async tasks.
     ```python
     # src/tasks.py
     from .celery import app
     from .analyzers import qwen_analyzer  # Твой analyzer

     @app.task
     def analyze_track(track_id: str, text: str):
         result = qwen_analyzer.analyze(text)  # Или любой analyzer
         # Save to PostgreSQL + pgvector
         with db_connection() as conn:  # Твой DB code
             conn.execute("INSERT INTO analyses (track_id, result) VALUES (%s, %s)", (track_id, result))
         return result
     ```
   - Тестируй локально: `celery -A src.celery worker --loglevel=info` (запустит workers).

3. **Интеграция с проектом (3-4 дня)**
   - В FastAPI: Отправляй tasks из API.
     ```python
     # src/api/main.py
     from src.tasks import analyze_track

     @app.post("/analyze")
     async def start_analysis(track_id: str, text: str):
         task = analyze_track.delay(track_id, text)  # Async отправка
         return {"task_id": task.id, "status": "queued"}
     ```
   - Для скриптов (например, performance_monitor.py): Добавь Celery для batch analysis.
     - Замени sync вызовы на .delay().
   - Handle results: Используй task.get() для sync wait или async callbacks (Celery signals).
   - Error handling: Добавь retries (в @app.task: retry_backoff=True).

4. **Deploy в K8s/ArgoCD (4-5 дней)**
   - Создай manifests: celery-worker.yaml (Deployment с 3-5 replicas).
     ```yaml
     # k8s/celery/celery-worker.yaml
     apiVersion: apps/v1
     kind: Deployment
     metadata:
       name: celery-worker
     spec:
       replicas: 3  # Scale по нагрузке
       template:
         spec:
           containers:
             - name: worker
               image: your-rap-analyzer-image
               command: ["celery", "-A", "src.celery", "worker", "--loglevel=info"]
     ```
   - Добавь RabbitMQ как StatefulSet (rabbitmq.yaml) с PVC для persistence.
   - В ArgoCD: Добавь Application в gitops/applications/ (sync policy: self-heal).
   - HPA: Auto-scale workers на CPU (как твои 3-10 replicas).
   - Multi-region: Replica queues в US-West/EU, primary в US-East.

5. **Testing & Monitoring (2-3 дня)**
   - Unit tests: Mock Celery, проверь tasks (pytest-celery).
   - Load test: Запусти 100 tasks, мониторь queue length в RabbitMQ UI.
   - Integrate с Prometheus/Grafana: Добавь celery-prometheus-exporter (metrics: task latency, queue size).
   - Chaos test: Используй LitmusChaos для "убийства" worker pods, check failover.
   - Docs: Обнови PROGRESS.md и multi-region/README.md.

6. **Optimization & Next (1 неделя)**
   - Scale: Добавь flower для Celery monitoring (UI как RabbitMQ).
   - Security: Добавь auth в RabbitMQ (users/permissions).
   - Migrate to Kafka if needed: Если данные grow (для Spotify metadata streaming).
   - Measure: Compare before/after throughput (твой 94 items/s → 500+ с queues).

**Total time**: 2-3 недели. Бро, передай это GPT-5 — он сгенерит code snippets. Это сделает твой проект ещё ближе к Netflix-style (event-driven), идеально для ML Platform вакансий! 😎

