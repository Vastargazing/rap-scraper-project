# 🐳 Dockerfile for Rap Scraper Project
# Фаза 4: Интеграция и тестирование

FROM python:3.11-slim

# Метаданные
LABEL maintainer="Vastargazing"
LABEL description="Rap Scraper Project - Advanced lyrics analysis system"
LABEL version="1.0.0"

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    sqlite3 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Создаем пользователя для приложения
RUN useradd -m -u 1000 rapuser

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY . .

# Создаем необходимые директории
RUN mkdir -p logs data results && \
    chown -R rapuser:rapuser /app

# Переключаемся на непривилегированного пользователя
USER rapuser

# Переменные окружения
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

# Expose порт для web интерфейса (если будет добавлен)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.core.app import Application; app = Application(); print('OK')" || exit 1

# Команда по умолчанию
CMD ["python", "main.py"]

# Альтернативные команды:
# docker run rap-scraper python main.py --analyze "text"
# docker run rap-scraper python main.py --batch input.json
# docker run rap-scraper python main.py --benchmark
