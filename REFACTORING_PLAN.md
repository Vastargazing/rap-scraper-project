# 🔧 Refactoring Plan - Legacy Cleanup

**Date:** 2025-10-27
**Branch:** `refactor/cleanup-legacy-and-duplicates`
**Goal:** Удалить legacy код, дубликаты логики, обновить архитектуру

---

## 📋 Анализ проекта

### 🗑️ Файлы для удаления

#### 1. **Archive директория (полностью legacy)**
```
archive/
├── ai_context_manager_legacy.py        # Старая версия context manager
├── config.json                          # Legacy config
├── gemma_27b_fixed.py                  # Дубликат src/analyzers/
├── performance_monitor_legacy.py       # Старый monitoring
├── performance_monitor.py              # Старый monitoring
├── qwen_analyzer.py                    # Дубликат, используется новый в src/
├── spotify_enhancer_sqlite_backup.py   # SQLite backup (не нужен)
└── __pycache__/                        # Скомпилированные файлы
```

**Действие:** Удалить весь `archive/` (уже есть git history)

#### 2. **SQLite Imports (нужно убрать)**

**Файлы с `import sqlite3` (нужно обновить или удалить):**

1. ❌ `tests/test_spotify_enhancer.py` - **УДАЛИТЬ** (тесты для SQLite)
2. ❌ `tests/conftest.py` - **ОБНОВИТЬ** (убрать SQLite fixture)
3. ❌ `src/enhancers/bulk_spotify_enhancement.py` - **ПРОВЕРИТЬ** (возможно legacy)
4. ❌ `src/enhancers/spotify_analysis_utils.py` - **ОБНОВИТЬ** (есть SQLite методы)
5. ❌ `src/analyzers/multi_model_analyzer_backup.py` - **УДАЛИТЬ** (backup файл)
6. ❌ `src/analyzers/create_visual_analysis.py` - **ПРОВЕРИТЬ** (SQLite методы)
7. ❌ `scripts/tools/comprehensive_ai_stats.py` - **ОБНОВИТЬ** (PostgreSQL версия)
8. ❌ `scripts/tools/monitor_qwen_progress.py` - **ОБНОВИТЬ** (PostgreSQL)
9. ❌ `scripts/tools/create_cli_showcase.py` - **ОБНОВИТЬ** (PostgreSQL)
10. ❌ `scripts/tools/batch_ai_analysis.py` - **ОБНОВИТЬ** (PostgreSQL)

#### 3. **Legacy/Backup файлы**

```
src/analyzers/multi_model_analyzer_backup.py   # Backup - удалить
src/utils/config.py (line 39)                   # DB_PATH = "rap_lyrics.db" - обновить
```

#### 4. **Data файлы**

```
data/rap_lyrics.db                              # SQLite база - удалить (есть PostgreSQL)
```

---

## ✅ Что НЕ трогаем

### ✨ Актуальные компоненты

1. **Type-safe Config System** ✅
   ```
   src/config/
   ├── __init__.py
   ├── config_loader.py
   ├── test_loader.py
   └── README.md
   ```

2. **ML Models** ✅
   ```
   models/
   ├── test_qwen.py              # PRIMARY ML MODEL
   ├── quality_prediction.py
   ├── style_transfer.py
   └── trend_analysis.py
   ```

3. **PostgreSQL Infrastructure** ✅
   ```
   src/database/
   ├── postgres_adapter.py
   └── connection.py
   ```

4. **Production Scripts** ✅
   ```
   scripts/mass_qwen_analysis.py
   scripts/db_browser.py
   scripts/tools/database_diagnostics.py
   ```

---

## 🎯 Действия (по порядку)

### Шаг 1: Создать резервную копию (на всякий случай)
```bash
# Уже в Git, но можно создать tag
git tag -a before-refactoring-2025-10-27 -m "Before legacy cleanup"
```

### Шаг 2: Удалить archive/ полностью
```bash
rm -rf archive/
```

### Шаг 3: Удалить SQLite database
```bash
rm -f data/rap_lyrics.db
```

### Шаг 4: Удалить backup файлы
```bash
rm -f src/analyzers/multi_model_analyzer_backup.py
rm -f tests/test_spotify_enhancer.py  # SQLite tests
```

### Шаг 5: Обновить файлы с SQLite импортами

#### 5.1 `src/utils/config.py`
```python
# БЫЛО:
DB_PATH = DATA_DIR / "rap_lyrics.db"

# СТАЛО:
# Removed - using PostgreSQL only (see src/config/config_loader.py)
```

#### 5.2 `tests/conftest.py`
Убрать SQLite fixture, оставить только PostgreSQL тесты

#### 5.3 Scripts в `scripts/tools/`
Заменить `sqlite3` на `PostgreSQLManager` (если еще не заменено)

### Шаг 6: Проверить дубликаты логики

**Потенциальные дубликаты:**
- `src/enhancers/spotify_analysis_utils.py` - есть методы и для SQLite и для PostgreSQL
- `scripts/tools/` - несколько скриптов делают похожие вещи

**Решение:**
- Оставить только PostgreSQL версии
- Удалить SQLite fallback методы

---

## 📊 Статистика

**Файлов для удаления:** ~15-20
**Строк кода для удаления:** ~3000-5000
**Директорий для удаления:** 1 (archive/)
**SQLite импортов для замены:** ~10

---

## 🚀 После рефакторинга

### Проверка
```bash
# 1. Проверить что нет SQLite импортов
grep -r "import sqlite3" src/ scripts/ --exclude-dir=archive

# 2. Проверить что нет .db файлов
find . -name "*.db" -type f

# 3. Запустить тесты
python -m pytest tests/ -v

# 4. Проверить main.py работает
python main.py --info
```

### Обновить документацию
- ✅ `docs/ARCHITECTURE.md` - убрать упоминания SQLite
- ✅ `docs/PROGRESS.md` - добавить запись о рефакторинге
- ✅ `docs/claude.md` - обновить структуру файлов

---

## ⚠️ Риски

**Минимальные риски:**
- ✅ Все в Git - можем откатиться
- ✅ PostgreSQL migration уже завершена (100% data integrity)
- ✅ Production скрипты не трогаем
- ✅ Тесты проверят что ничего не сломалось

**Если что-то пошло не так:**
```bash
git checkout master
git branch -D refactor/cleanup-legacy-and-duplicates
```

---

**Автор:** RapAnalyst 🎤🤖
**Status:** Ready to execute ✅
