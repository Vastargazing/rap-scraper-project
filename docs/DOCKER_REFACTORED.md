# Docker Compose - Новая Структура

## ✅ Рефакторинг завершен!

Структура Docker Compose файлов приведена к best practices согласно плану в `docs/dockerprod.md`.

---

## 📁 Финальная структура файлов:

```
docker-compose.yml          # Production (минимальный: API + PostgreSQL + Redis)
docker-compose.dev.yml      # Development extensions (+ pgAdmin + Grafana + Prometheus)
docker-compose.pgvector.yml # Database only (только PostgreSQL + Redis для локалки)
```

---

## 🚀 Команды запуска:

### Production
```bash
make docker-up
# или
docker-compose up -d
```
**Запускает:** API + PostgreSQL + Redis

### Development (полный стек)
```bash
make docker-dev
# или
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```
**Запускает:** API + PostgreSQL + Redis + pgAdmin + Grafana + Prometheus

### Только база данных (локальная разработка)
```bash
make docker-db
# или
docker-compose -f docker-compose.pgvector.yml up -d
```
**Запускает:** PostgreSQL + Redis

### Остановить все
```bash
make docker-down
```

---

## 🔧 Доступные сервисы:

| Сервис | Production | Development | Database Only | Порт |
|--------|------------|-------------|---------------|------|
| **API** | ✅ | ✅ | ❌ | 8000 |
| **PostgreSQL** | ✅ | ✅ | ✅ | 5432 |
| **Redis** | ✅ | ✅ | ✅ | 6379 |
| **pgAdmin** | ❌ | ✅ | ❌ | 5050 |
| **Grafana** | ❌ | ✅ | ❌ | 3000 |
| **Prometheus** | ❌ | ✅ | ❌ | 9090 |

---

## 🎯 Use Cases:

### 1. **Локальная разработка** (только API локально)
```bash
make docker-db      # Запустить только базы
make run-fastapi    # Запустить API локально через Poetry
```

### 2. **Full Development** (все в контейнерах)
```bash
make docker-dev
```
- Доступен hot reload
- Полный monitoring stack
- pgAdmin для управления БД

### 3. **Production Deployment**
```bash
make docker-up
```
- Минимальный footprint
- Только необходимые сервисы
- Production-ready конфигурация

---

## 📊 Что было исправлено:

### ❌ ДО рефакторинга:
- 3 разных compose файла с 80% дублирования
- `docker-compose.yml` с SQLite + Ollama (не для production)
- `docker-compose.postgres.yml` с множеством ненужных сервисов
- Сложные команды запуска

### ✅ ПОСЛЕ рефакторинга:
- Четкое разделение prod/dev/db-only
- Использование `extends` для переиспользования
- Environment variables для конфигурации
- Простые команды через Makefile
- Минимализм в production
- Удален Ollama (используется Novita API)

---

## 🔑 Environment Variables:

Создайте `.env` файл:
```env
# Database
POSTGRES_PASSWORD=your-secure-password
POSTGRES_DB=rap_lyrics
POSTGRES_USER=rap_user

# API Keys
NOVITA_API_KEY=your-novita-key
SPOTIFY_CLIENT_ID=your-spotify-id
SPOTIFY_CLIENT_SECRET=your-spotify-secret

# Optional overrides
API_PORT=8000
POSTGRES_PORT=5432
REDIS_PORT=6379
VERSION=latest
```

---

## 📝 Быстрый старт:

```bash
# 1. Установить зависимости
make quick-start

# 2. Создать .env файл (см. выше)

# 3. Запустить нужный стек:
make docker-db    # Только база
make docker-dev   # Полная разработка
make docker-up    # Production
```

**Результат:** Чистая, maintainable структура Docker Compose без дублирования! 🎉