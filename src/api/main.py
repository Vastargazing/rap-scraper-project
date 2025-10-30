"""
🚀 ОСНОВНОЕ ПРИЛОЖЕНИЕ FASTAPI - RAP ML API (v3.0.0)

📋 ЧТО ЭТО ФАЙЛ:
    Главный файл приложения FastAPI для анализа рэп-текстов с помощью AI.
    Здесь:
    - Инициализируется весь API
    - Загружаются конфигурации и маршруты
    - Управляется жизненный цикл приложения (startup/shutdown)
    - Инициализируются ML модели (QWEN analyzer)
    - Настраивается CORS и middleware

🔧 АРХИТЕКТУРА:
    ✅ Config-based (конфиги из .env через Pydantic)
    ✅ Graceful imports (импорты с fallback если что-то сломается)
    ✅ Modular routes (6 отдельных маршрутов в папке routes/)
    ✅ Lifespan management (инициализация ресурсов при старте)
    ✅ Exception handling (обработка ошибок при инициализации)

📊 ОСНОВНЫЕ КОМПОНЕНТЫ:
    1. Config management - загрузка конфигурации с fallback
    2. Route imports - импорт всех маршрутов (health, analyze, ml_models, batch, web, models_info)
    3. Lifespan context manager - управление startup/shutdown
    4. ML model initialization - инициализация QWEN analyzer при старте
    5. CORS middleware - разрешение кросс-доменных запросов
    6. FastAPI app creation - создание и конфигурирование приложения

🎯 ПОТОК ВЫПОЛНЕНИЯ:
    1️⃣ Python загружает main.py (defines)
    2️⃣ Импортируются конфиги и маршруты (graceful fallback)
    3️⃣ Uvicorn запускает FastAPI app
    4️⃣ lifespan.__aenter__() вызывается (startup):
       - Логирование конфигурации
       - Инициализация QWEN analyzer (3-5 секунд!)
       - Exception handling если что-то не инициализировалось
    5️⃣ API готова! Обслуживает запросы
    6️⃣ При Ctrl+C: lifespan.__aexit__() (shutdown):
       - Логирование завершения
       - Cleanup ресурсов (если будет добавлен)

⚙️ ПЕРЕМЕННЫЕ И ФУНКЦИИ:
    - config: Конфигурация приложения (с fallback значениями)
    - ROUTES_AVAILABLE: Словарь импортированных маршрутов
    - INIT_FUNCTIONS: Словарь функций инициализации (initialize_analyzer)
    - lifespan(): Контекст-менеджер для управления жизненным циклом
    - app: FastAPI instance (главное приложение)

🔑 КЛЮЧЕВЫЕ ТЕХНОЛОГИИ:
    📦 FastAPI - веб-фреймворк для API
    🐍 asynccontextmanager - управление ресурсами (startup/shutdown)
    ⚡ async/await - асинхронное выполнение
    🤖 QWEN Analyzer - AI модель для анализа текстов
    🔐 CORS - кросс-доменные запросы
    📝 Pydantic - валидация конфигурации

🚀 КАК ЗАПУСТИТЬ:
    python -m uvicorn src.api.main:app --reload
    (для разработки с автоперезагрузкой)

    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
    (для production с 4 рабочими процессами)

📍 АДРЕСА:
    API Swagger docs: http://localhost:8000/docs
    API ReDoc docs: http://localhost:8000/redoc
    API Health check: http://localhost:8000/health

⚠️ ВАЖНО:
    - QWEN инициализируется при КАЖДОМ запуске (3-5 секунд)
    - Если инициализация failse - API все равно запускается но без анализа
    - Graceful imports значит что если маршрут не загружается - API не упадет
    - Uvicorn с --workers запускает несколько процессов (каждый инициализирует QWEN!)

👤 Автор: ML Platform Team
📅 Дата: October 2025
🔗 Версия: v3.0.0
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure logging FIRST
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Graceful imports with fallbacks
try:
    from src.config import get_config

    config = get_config()
    CONFIG_AVAILABLE = True
except Exception as e:
    logger.warning(f"⚠️ Config not available: {e}")

    # Fallback config
    class FallbackConfig:
        class application:
            environment = "development"
            name = "Rap ML API"
            version = "3.0.0"

        class api:
            host = "127.0.0.1"
            port = 8000
            workers = 1
            reload = True
            log_level = "info"

            class cors:
                enabled = True
                origins = ["*"]
                allow_credentials = True
                allow_methods = ["*"]
                allow_headers = ["*"]

            class docs:
                enabled = True
                title = "Rap ML API"
                version = "3.0.0"
                swagger_url = "/docs"
                redoc_url = "/redoc"

        class database:
            type = "postgresql"
            pool_size = 20

        class redis:
            enabled = False

    config = FallbackConfig()
    CONFIG_AVAILABLE = False

# Import routes with graceful fallback
ROUTES_AVAILABLE = {}
try:
    from .routes import health

    ROUTES_AVAILABLE["health"] = health
except Exception as e:
    logger.warning(f"⚠️ Health routes not available: {e}")
    ROUTES_AVAILABLE["health"] = None

try:
    from .routes import analyze

    ROUTES_AVAILABLE["analyze"] = analyze
except Exception as e:
    logger.warning(f"⚠️ Analyze routes not available: {e}")
    ROUTES_AVAILABLE["analyze"] = None

try:
    from .routes import ml_models

    ROUTES_AVAILABLE["ml_models"] = ml_models
except Exception as e:
    logger.warning(f"⚠️ ML models routes not available: {e}")
    ROUTES_AVAILABLE["ml_models"] = None

try:
    from .routes import batch

    ROUTES_AVAILABLE["batch"] = batch
except Exception as e:
    logger.warning(f"⚠️ Batch routes not available: {e}")
    ROUTES_AVAILABLE["batch"] = None

try:
    from .routes import web

    ROUTES_AVAILABLE["web"] = web
except Exception as e:
    logger.warning(f"⚠️ Web routes not available: {e}")
    ROUTES_AVAILABLE["web"] = None

try:
    from .routes import models_info

    ROUTES_AVAILABLE["models_info"] = models_info
except Exception as e:
    logger.warning(f"⚠️ Models info routes not available: {e}")
    ROUTES_AVAILABLE["models_info"] = None

# 🆕 Import initialization functions for models
INIT_FUNCTIONS = {}
try:
    from .routes.analyze import initialize_analyzer

    INIT_FUNCTIONS["initialize_analyzer"] = initialize_analyzer
    logger.debug("✅ initialize_analyzer imported successfully")
except Exception as e:
    logger.warning(f"⚠️ initialize_analyzer not available: {e}")
    INIT_FUNCTIONS["initialize_analyzer"] = None

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load configuration
config = get_config()


# ============================================================================
# STARTUP / SHUTDOWN EVENTS
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # STARTUP
    logger.info("=" * 80)
    logger.info("🚀 STARTING UNIFIED RAP ML API (v3.0.0)")
    logger.info("=" * 80)
    logger.info(f"   Environment: {config.application.environment}")
    logger.info(f"   API Host: {config.api.host}:{config.api.port}")
    logger.info(f"   Workers: {config.api.workers}")
    logger.info(
        f"   Database: {config.database.type} (pool: {config.database.pool_size})"
    )
    logger.info(f"   Redis: {'enabled' if config.redis.enabled else 'disabled'}")
    logger.info("=" * 80)

    # 🆕 INITIALIZE ML MODELS
    logger.info("\n📊 INITIALIZING ML MODELS...")
    if INIT_FUNCTIONS.get("initialize_analyzer"):
        try:
            logger.info("   🤖 Initializing QWEN analyzer...")
            success = await INIT_FUNCTIONS["initialize_analyzer"]()
            if success:
                logger.info("   ✅ QWEN analyzer initialized successfully!")
            else:
                logger.warning(
                    "   ⚠️ Failed to initialize QWEN analyzer (check logs above)"
                )
        except Exception as e:
            logger.error(f"   ❌ Error initializing QWEN analyzer: {e}", exc_info=True)
    else:
        logger.warning("   ⚠️ Initialize function not available (check route imports)")

    logger.info("=" * 80 + "\n")

    yield

    # SHUTDOWN
    logger.info("=" * 80)
    logger.info("🛑 SHUTTING DOWN RAP ML API")
    logger.info("=" * 80)


# ============================================================================
# CREATE FASTAPI APP
# ============================================================================

app = FastAPI(
    title=config.api.docs.title,
    description="Unified ML API for rap lyrics analysis with RAG systems",
    version=config.api.docs.version,
    docs_url=config.api.docs.swagger_url if config.api.docs.enabled else None,
    redoc_url=config.api.docs.redoc_url if config.api.docs.enabled else None,
    lifespan=lifespan,
)

# ============================================================================
# MIDDLEWARE SETUP
# ============================================================================

# CORS middleware
if config.api.cors.enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors.origins,
        allow_credentials=config.api.cors.allow_credentials,
        allow_methods=config.api.cors.allow_methods,
        allow_headers=config.api.cors.allow_headers,
    )
    logger.debug(f"✅ CORS enabled for origins: {config.api.cors.origins}")

# ============================================================================
# INCLUDE ROUTES (All route modules)
# ============================================================================

# Register all available route modules
routes_registered = []

if ROUTES_AVAILABLE.get("health"):
    app.include_router(ROUTES_AVAILABLE["health"].router, tags=["health"])
    routes_registered.append("health")
if ROUTES_AVAILABLE.get("analyze"):
    app.include_router(ROUTES_AVAILABLE["analyze"].router, tags=["analysis"])
    routes_registered.append("analyze")
if ROUTES_AVAILABLE.get("ml_models"):
    app.include_router(ROUTES_AVAILABLE["ml_models"].router, tags=["ml-models"])
    routes_registered.append("ml_models")
if ROUTES_AVAILABLE.get("batch"):
    app.include_router(ROUTES_AVAILABLE["batch"].router, tags=["batch"])
    routes_registered.append("batch")
if ROUTES_AVAILABLE.get("web"):
    app.include_router(ROUTES_AVAILABLE["web"].router, tags=["web"])
    routes_registered.append("web")
if ROUTES_AVAILABLE.get("models_info"):
    app.include_router(ROUTES_AVAILABLE["models_info"].router, tags=["models"])
    routes_registered.append("models_info")

logger.info("✅ Unified FastAPI application initialized (v3.0.0)")
logger.info(
    f"   Routes registered: {', '.join(routes_registered) if routes_registered else 'NONE'}"
)

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting uvicorn server...")
    uvicorn.run(
        "src.api.main:app",
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers if not config.api.reload else 1,
        reload=config.api.reload,
        log_level=config.api.log_level,
    )
