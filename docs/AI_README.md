# 🤖 AI Assistant Quick Context

> **CRITICAL:** This project uses PostgreSQL, not SQLite. All legacy SQLite code is archived.

## ⚡ 30-Second Context
- **Architecture:** Microservices with main.py orchestration
- **Database:** PostgreSQL 15 with connection pooling  
- **AI Models:** 5 analyzers (algorithmic_basic, qwen, ollama, emotion, hybrid)
- **Entry Point:** `python main.py` (unified interface)
- **Data:** 57,718 tracks, 54,170+ analyzed, concurrent processing capable

## 🎯 AI Assistant Commands

### Quick Status
```bash
python main.py --info          # Complete system status
python main.py --test          # Validate all components
python check_stats.py          # PostgreSQL database health
```

### Analysis Tasks  
```bash
python main.py --analyze "text" --analyzer qwen     # Single analysis
python main.py --benchmark                          # Performance comparison
python scripts/mass_qwen_analysis.py --test         # Batch analysis test
```

### Development
```bash
python main.py --help          # All available options
pytest tests/ -v               # Run test suite
docker-compose up -d            # Deploy full stack
```

## 🚨 Critical Reminders for AI

1. **PostgreSQL ONLY** - No sqlite3 imports, use PostgreSQLManager
2. **Unified Interface** - Use main.py, not direct component calls
3. **Microservices** - Respect src/{analyzers,cli,models}/ boundaries
4. **Testing** - Always validate with python main.py --test
5. **Concurrent Processing** - Multiple scripts can run simultaneously

## 📁 Priority Files for AI Analysis

| File | Priority | Purpose |
|------|----------|---------|
| `docs/claude.md` | 🔥🔥🔥🔥🔥 | Complete AI context |
| `main.py` | 🔥🔥🔥🔥🔥 | Central orchestration |  
| `src/database/postgres_adapter.py` | 🔥🔥🔥🔥🔥 | Database layer |
| `config.yaml` | 🔥🔥🔥🔥 | System configuration |
| `scripts/mass_qwen_analysis.py` | 🔥🔥🔥🔥 | Main analysis script |

## ⚠️ Deprecated/Legacy (Reference Only)
- `scripts/archive/` - SQLite legacy code
- `data/data_backup_*.db` - SQLite backups  
- Any file with `_sqlite.py` suffix

---
*Auto-generated for AI assistants. Human-readable docs in README.md*
