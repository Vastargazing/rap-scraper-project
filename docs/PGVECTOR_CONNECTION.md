# 🔐 pgvector Connection Information

## Актуальные параметры подключения к базе данных

### Для psql, pgAdmin, DBeaver и других клиентов:
```
Host: localhost
Port: 5433
Database: rap_lyrics
Username: rap_user
Password: secure_password_2024
```

### Для Python приложений (config.yaml):
```yaml
database:
  type: "postgresql"
  host: "localhost"
  port: 5433
  name: "rap_lyrics"
  username: "rap_user"
  password: "secure_password_2024"
```

### Connection String (для библиотек типа SQLAlchemy):
```
postgresql://rap_user:secure_password_2024@localhost:5433/rap_lyrics
```

### Через Docker exec (прямое подключение без пароля):
```powershell
docker exec -it rap-analyzer-postgres-vector psql -U rap_user -d rap_lyrics
```

## ✅ Проверка работы pgvector:
```sql
-- Проверить установку расширения
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';

-- Тест векторных операций
SELECT vector('[1,2,3]') AS test_vector;

-- Посмотреть доступные функции
SELECT proname FROM pg_proc WHERE proname LIKE '%vector%' ORDER BY proname;
```

## 🐳 Docker команды:
```powershell
# Запустить контейнер
docker-compose -f docker-compose.pgvector.yml up -d

# Остановить и удалить данные (для пересоздания)
docker-compose -f docker-compose.pgvector.yml down -v

# Проверить статус
docker ps

# Посмотреть логи
docker logs rap-analyzer-postgres-vector
```

## ⚠️ Важные заметки:
- Порт **5433** (не стандартный 5432), чтобы не конфликтовать с локальной установкой PostgreSQL
- Пароль: `secure_password_2024` (указан в docker-compose.pgvector.yml)
- База автоматически создается при первом запуске контейнера
- pgvector версии 0.5.1 установлен и протестирован
