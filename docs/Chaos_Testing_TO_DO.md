## 🎯 TO DO: Chaos Testing для Rap Analyzer

### **Phase 1: Базовая Настройка**
```markdown
□ Установить Toxiproxy локально
□ Создать папку `/tests/chaos/` в проекте
□ Настроить pytest для chaos тестов
□ Добавить chaos dependencies в requirements-test.txt
□ Создать базовый chaos test runner
```

### **Phase 2: Network Chaos Tests**
```markdown
□ Тест таймаутов API (Spotify/Genius/LastFM)
□ Тест медленных ответов API (latency injection)
□ Тест полного отказа внешних API
□ Тест rate limiting (429 errors)
□ Тест прерванных соединений
```

### **Phase 3: Database Chaos Tests**
```markdown
□ Тест отвала PostgreSQL connection
□ Тест медленных SQL запросов
□ Тест полной БД (disk space)
□ Тест pgvector индексов под нагрузкой
□ Тест concurrent connections limit
```

### **Phase 4: Resource Chaos Tests**
```markdown
□ Memory pressure тесты (для 57K треков)
□ CPU throttling тесты
□ Disk I/O saturation
□ Container restart simulation
□ OOM killer simulation
```

### **Phase 5: AI Pipeline Chaos**
```markdown
□ Тест падения одного из 5 AI analyzers
□ Тест медленной обработки embeddings
□ Тест vector search при высокой нагрузке
□ Тест batch processing failures
□ Тест ML model timeout scenarios
```

### **Phase 6: Integration & Monitoring**
```markdown
□ Chaos dashboard (Grafana metrics)
□ Автоматический chaos в CI/CD
□ Recovery time измерения
□ Alerting на chaos events
□ Chaos experiment docs
```

---

