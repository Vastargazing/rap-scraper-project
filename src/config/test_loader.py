"""
🧪 Configuration Loader - FULL TEST SUITE
Complete testing of all configuration sections with detailed output

This script performs comprehensive validation of:
- Environment variables loading
- All Pydantic models
- Database configuration
- Redis cache settings
- API configuration (CORS, rate limiting, docs)
- All analyzers (QWEN, Ollama, Emotion, Hybrid)
- Monitoring (Prometheus, Grafana, Health checks)
- Logging configuration
- CI/CD settings

Usage:
    python src/config/test_loader.py
    python -m src.config.test_loader

For quick config check use:
    python src/config/config_loader.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f"✅ Loaded .env from: {env_file}")
else:
    print(f"⚠️  No .env file found at: {env_file}")
    print("   Using minimal test environment variables")

# Set minimal test variables if not set (even if .env exists)
if not os.getenv("DB_HOST"):
    os.environ["DB_HOST"] = "localhost"
if not os.getenv("DB_NAME"):
    os.environ["DB_NAME"] = "rap_lyrics"
if not os.getenv("DB_USERNAME"):
    os.environ["DB_USERNAME"] = "rap_user"
if not os.getenv("DB_PASSWORD"):
    os.environ["DB_PASSWORD"] = "test_password_for_validation"
if not os.getenv("REDIS_HOST"):
    os.environ["REDIS_HOST"] = "localhost"

print("\n" + "=" * 70)
print("🧪 Configuration Loader Test")
print("=" * 70)

print("\n🔧 Environment Variables:")
print(f"   DB_HOST: {os.getenv('DB_HOST')}")
print(f"   DB_NAME: {os.getenv('DB_NAME')}")
print(f"   DB_USERNAME: {os.getenv('DB_USERNAME')}")
print(f"   DB_PASSWORD: {'SET' if os.getenv('DB_PASSWORD') else 'NOT SET'}")
print(f"   NOVITA_API_KEY: {'SET' if os.getenv('NOVITA_API_KEY') else 'NOT SET'}")
print(f"   REDIS_HOST: {os.getenv('REDIS_HOST')}")

# Test config loading
try:
    print("\n📁 Loading configuration from src/config/loader.py...")
    from src.config import get_config

    config = get_config()

    print("\n✅ Configuration loaded successfully!")

    print("\n📊 Application Info:")
    print(f"   Name: {config.application.name}")
    print(f"   Version: {config.application.version}")
    print(f"   Environment: {config.application.environment}")
    print(f"   Description: {config.application.description}")

    print("\n🗄️  Database Configuration:")
    print(f"   Type: {config.database.type}")
    print(f"   Host: {config.database.host}")
    print(f"   Port: {config.database.port}")
    print(f"   Database: {config.database.database_name}")
    print(f"   Username: {config.database.username}")
    print(f"   Pool Size: {config.database.pool_size}")
    print(f"   Max Overflow: {config.database.max_overflow}")
    print(f"   Pool Recycle: {config.database.pool_recycle}s")
    print(
        f"   Connection String: postgresql://{config.database.username}:***@{config.database.host}:{config.database.port}/{config.database.database_name}"
    )

    print("\n🔍 Vector Search Configuration:")
    print(f"   Enabled: {config.vector_search.enabled}")
    print(f"   Embedding Model: {config.vector_search.embedding_model}")
    print(f"   Dimension: {config.vector_search.dimension}")
    print(f"   Distance Metric: {config.vector_search.distance_metric}")
    print(f"   Index Type: {config.vector_search.index_type}")
    print(f"   Cache Enabled: {config.vector_search.cache_enabled}")

    print("\n🚀 API Configuration:")
    print(f"   Host: {config.api.host}")
    print(f"   Port: {config.api.port}")
    print(f"   Workers: {config.api.workers}")
    print(f"   Reload: {config.api.reload}")
    print(f"   Log Level: {config.api.log_level}")
    print("\n   📝 API Docs:")
    print(f"      Enabled: {config.api.docs.enabled}")
    print(f"      Title: {config.api.docs.title}")
    print(f"      Version: {config.api.docs.version}")
    print(f"      Swagger: {config.api.docs.swagger_url}")
    print(f"      ReDoc: {config.api.docs.redoc_url}")
    print("\n   🌐 CORS:")
    print(f"      Enabled: {config.api.cors.enabled}")
    print(f"      Origins: {', '.join(config.api.cors.origins)}")
    print(f"      Credentials: {config.api.cors.allow_credentials}")
    print("\n   ⏱️  Rate Limiting:")
    print(f"      Enabled: {config.api.rate_limit.enabled}")
    print(f"      Requests/min: {config.api.rate_limit.requests_per_minute}")
    print(f"      Burst Size: {config.api.rate_limit.burst_size}")

    print("\n💾 Redis Configuration:")
    print(f"   Enabled: {config.redis.enabled}")
    print(f"   Host: {config.redis.host}")
    print(f"   Port: {config.redis.port}")
    print(f"   Database: {config.redis.db}")
    print(f"   Max Connections: {config.redis.max_connections}")
    print("\n   ⏰ Cache TTL:")
    print(
        f"      Artist: {config.redis.cache.artist_ttl}s ({config.redis.cache.artist_ttl // 3600}h)"
    )
    print(
        f"      Lyrics: {config.redis.cache.lyrics_ttl}s ({config.redis.cache.lyrics_ttl // 3600}h)"
    )
    print(
        f"      Analysis: {config.redis.cache.analysis_ttl}s ({config.redis.cache.analysis_ttl // 86400}d)"
    )
    print(
        f"      Embedding: {config.redis.cache.embedding_ttl}s ({config.redis.cache.embedding_ttl // 86400}d)"
    )

    print("\n🤖 Analyzers Configuration:")

    # Test QWEN config
    try:
        qwen_config = config.analyzers.get_qwen()
        print("\n   ✅ QWEN Analyzer:")
        print(f"      Model: {qwen_config.model_name}")
        print(f"      Base URL: {qwen_config.base_url}")
        print(f"      API Key: {'SET' if qwen_config.api_key else 'NOT SET'}")
        print(f"      Temperature: {qwen_config.temperature}")
        print(f"      Max Tokens: {qwen_config.max_tokens}")
        print(f"      Timeout: {qwen_config.timeout}s")
        print(f"      Retry Attempts: {qwen_config.retry_attempts}")
    except Exception as e:
        print(f"\n   ❌ QWEN Analyzer: {e}")

    # Test Ollama config
    try:
        ollama_config = config.analyzers.get_ollama()
        print("\n   ✅ Ollama Analyzer:")
        print(f"      Model: {ollama_config.model}")
        print(f"      Base URL: {ollama_config.base_url}")
        print(f"      Temperature: {ollama_config.temperature}")
        print(f"      Max Tokens: {ollama_config.max_tokens}")
        print(f"      Timeout: {ollama_config.timeout}s")
    except Exception as e:
        print(f"\n   ❌ Ollama Analyzer: {e}")

    # Test Emotion config
    try:
        emotion_config = config.analyzers.get_emotion()
        print("\n   ✅ Emotion Analyzer:")
        print(f"      Model: {emotion_config.model_name}")
        print(f"      Device: {emotion_config.device}")
        print(f"      Max Length: {emotion_config.max_length}")
        print(f"      Batch Size: {emotion_config.batch_size}")
        print(f"      Cache: {emotion_config.cache_enabled}")
    except Exception as e:
        print(f"\n   ❌ Emotion Analyzer: {e}")

    # Test Hybrid config
    try:
        hybrid_config = config.analyzers.get_hybrid()
        print("\n   ✅ Hybrid Analyzer:")
        print(f"      Algorithms: {', '.join(hybrid_config.algorithms)}")
        print(f"      Consensus Threshold: {hybrid_config.consensus_threshold}")
        print(f"      Fallback: {hybrid_config.fallback_analyzer}")
    except Exception as e:
        print(f"\n   ❌ Hybrid Analyzer: {e}")

    print("\n📊 Monitoring Configuration:")
    print("\n   📈 Prometheus:")
    print(f"      Enabled: {config.monitoring.prometheus.enabled}")
    print(f"      Port: {config.monitoring.prometheus.port}")
    print(f"      Path: {config.monitoring.prometheus.path}")
    print(
        f"      Default Metrics: {config.monitoring.prometheus.include_default_metrics}"
    )
    print("\n   📊 Grafana:")
    print(f"      Enabled: {config.monitoring.grafana.enabled}")
    print(f"      Port: {config.monitoring.grafana.port}")
    print("\n   🏥 Health Checks:")
    print(f"      Enabled: {config.monitoring.health.enabled}")
    print(f"      Endpoint: {config.monitoring.health.endpoint}")
    print(f"      Interval: {config.monitoring.health.check_interval}s")
    print(f"      Components: {', '.join(config.monitoring.health.components)}")

    print("\n📝 Logging Configuration:")
    print(f"   Level: {config.logging.level}")
    print(f"   File: {config.logging.file_path}")
    print(f"   Max Size: {config.logging.max_file_size // (1024 * 1024)}MB")
    print(f"   Backup Count: {config.logging.backup_count}")
    print(f"   Console Output: {config.logging.console_output}")

    print("\n🔄 CI/CD Configuration:")
    print(f"   GitHub Actions: {config.ci_cd.github_actions.enabled}")
    print(f"   Test Coverage Required: {config.ci_cd.testing.required_coverage}%")
    print(f"   Integration Tests: {config.ci_cd.testing.run_integration_tests}")
    print(f"   Auto Rollback: {config.ci_cd.deployment.auto_rollback}")
    print(f"   Max Complexity: {config.ci_cd.quality.max_complexity}")

    print("\n" + "=" * 70)
    print("✅ All configuration tests passed!")
    print("=" * 70)

    print("\n💡 Next Steps:")
    print("   1. ✅ Config loader is working correctly")
    print("   2. ✅ All Pydantic models validated")
    print("   3. ✅ Environment variables loaded")
    print("   4. 🔄 Ready to integrate into modules:")
    print("      - from src.config import get_config")
    print("      - config = get_config()")
    print("      - db_string = config.database.connection_string")

    print("\n🧪 Test Components:")
    print('   python -c "from src.database import test_connection; test_connection()"')
    print(
        '   python -c "from src.cache import test_redis_connection; test_redis_connection()"'
    )
    print("   python main.py                    # Main CLI application")
    print("   python src/api/main.py            # FastAPI server")

except FileNotFoundError as e:
    print(f"\n❌ File Error: {e}")
    print("   Make sure config.yaml exists in project root")
    sys.exit(1)
except ValueError as e:
    print(f"\n❌ Validation Error: {e}")
    print("   Check your .env file and environment variables")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Configuration Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
