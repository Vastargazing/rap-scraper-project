# Git Best Practices для ML Platform Engineers

## 1. Commit Messages - Правило 50/72

```
[type]: [description]

[detailed explanation if needed]
```

### Types:
- `feat:` - новая фишка
- `fix:` - исправление баги
- `refactor:` - улучшение кода
- `docs:` - документация
- `test:` - тесты
- `chore:` - техническое обслуживание

### Примеры ХОРОШИЕ ✅:
```
feat: Add sentiment analysis to rap tracks

This feature analyzes emotional tone using transformers
and returns sentiment scores for each verse.
```

### Примеры ПЛОХИЕ ❌:
```
update stuff
s
fixed bug
```

---

## 2. Feature Branch Workflow

### Правило: НИКОГДА не коммитишь в master!

```bash
# 1. Создаёшь новую ветку
git checkout -b feature/sentiment-analyzer

# 2. Работаешь на этой ветке
git add .
git commit -m "feat: Add sentiment analyzer"

# 3. Пушишь ветку
git push origin feature/sentiment-analyzer

# 4. Создаёшь PR на GitHub
# (запрашиваешь код ревью)

# 5. После одобрения мержишь
git checkout master
git merge feature/sentiment-analyzer
```

---

## 3. Branching Strategy

### Naming Convention:
- `feature/...` - новые фишки
- `fix/...` - баги
- `refactor/...` - рефакторинг
- `docs/...` - документация

### Пример:
```
feature/add-spotify-integration
fix/handle-null-values
refactor/extract-cache-logic
docs/api-documentation
```

---

## 4. Pull Request Best Practices

### Перед созданием PR:
```bash
# 1. Обновляешь master
git checkout master
git pull origin master

# 2. Переключаешься на свою ветку
git checkout feature/my-feature

# 3. Переbase на latest master (чтобы не было конфликтов)
git rebase master

# 4. Пушишь
git push -f origin feature/my-feature
```

### Описание PR:
```
## What
- Добавил sentiment analyzer

## Why
- Нужно анализировать эмоции в текстах

## How
- Использовал transformers library
- Добавил unit tests

## Testing
- Все тесты проходят ✅
```

---

## 5. Undo Commands (БЕЗОПАСНО!)

### Если неправильно закоммитил (но не пушил):
```bash
git reset --soft HEAD~1
# Коммит отмен, но изменения остаются в staging
```

### Если уже пушил:
```bash
git revert abc123
# Создаёт новый коммит который отменяет старый
```

---

## 6. Merge vs Rebase

### Merge (безопаснее для PR):
```bash
git merge feature/my-feature
# Создаёт merge commit
# История показывает оба пути
```

### Rebase (чище история):
```bash
git rebase master
# Переносит твои коммиты на top of master
# История линейная
```

**ПРАВИЛО:** В team projects используй merge для PR!

---

## 7. Conflict Resolution

Когда git говорит CONFLICT:
```python
<<<<<<< HEAD
    # твоя версия
    return sentiment_analysis(track)
=======
    # их версия
    return emotion_analysis(track)
>>>>>>> feature/new-feature
```

**Что делать:**
1. Открыть файл
2. Выбрать правильную версию (или объединить)
3. Удалить конфликт маркеры
4. `git add файл.py`
5. `git commit -m "Resolve merge conflict"`

---

## 8. Git Log - Как читать историю

```bash
git log --oneline --graph --all
```

Output:
```
* 5adc5a1 (HEAD -> feature/add-best-practices) docs: Add Git best practices
* e5d1efc (master) chore: Remove legacy code
* 6a82d5d fix: Handle null values
```

**Что видишь:**
- `5adc5a1` = commit hash (первые 7 символов)
- `(HEAD -> feature/...)` = где ты сейчас
- `docs: Add Git...` = commit message
- Звёздочки и линии = визуализация веток

---

## 9. Atomic Commits - Маленькие шаги!

### ❌ ПЛОХО - один большой коммит:
```bash
git commit -m "Add sentiment analysis, refactor API, update docs"
```

### ✅ ХОРОШО - несколько маленьких:
```bash
git commit -m "feat: Add sentiment analyzer class"
git commit -m "test: Add tests for sentiment analyzer"
git commit -m "docs: Update README with sentiment feature"
```

**Почему:**
- Легче найти bug (git bisect)
- Понятная история
- Легче отменить одно изменение

---

## 10. Golden Rules 🌟

1. ✅ **Коммитишь рано и часто** - больше save points!
2. ✅ **Хорошие commit messages** - будущий ты скажет спасибо
3. ✅ **Feature branches ВСЕГДА** - master остаётся чистым
4. ✅ **PR перед merge** - code review спасает баги
5. ✅ **Pull перед push** - избегаешь конфликтов
6. ❌ **НИКОГДА force push в shared branch** - кикнешься из проекта
7. ❌ **НИКОГДА не коммитишь secrets** - используй .gitignore

---

Сделано! 🚀 Это твой первый commit будет!
