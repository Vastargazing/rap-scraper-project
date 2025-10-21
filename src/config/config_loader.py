"""
🎯 Configuration Loader with Pydantic Validation
Production-ready config management for ML Platform

Features:
- Type-safe configuration with Pydantic
- Environment variable substitution
- API key validation
- Multi-environment support (dev/staging/prod)
- Singleton pattern for global config access
- Comprehensive validation

Author: Vastargazing
Version: 2.0.0
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator

# Load .env file from project root (search up to 3 levels)
project_root = Path(__file__).parent.parent.parent  # src/config/config_loader.py -> project root
dotenv_path = project_root / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path, override=True)
else:
    # Fallback: try to find .env in current directory
    load_dotenv(override=True)

# ============================================================================
# Application Settings
# ============================================================================


class ApplicationConfig(BaseModel):
    """Application metadata and environment settings"""

    name: str
    version: str
    description: str
    author: str
    environment: Literal["development", "staging", "production"] = "production"


# ============================================================================
# Database Configuration
# ============================================================================


class SQLiteConfig(BaseModel):
    """SQLite configuration (legacy support)"""

    path: str = "data/rap_lyrics.db"
    enabled: bool = False


class DatabaseConfig(BaseModel):
    """PostgreSQL database configuration with connection pooling"""

    type: str = "postgresql"
    host_env: str = "DB_HOST"
    port: int = 5432
    name_env: str = "DB_NAME"
    username_env: str = "DB_USERNAME"
    password_env: str = "DB_PASSWORD"
    pool_size: int = 20
    min_pool_size: int = 5
    max_overflow: int = 10
    timeout: int = 30
    backup_enabled: bool = True
    backup_interval: int = 3600
    pool_recycle: int = 3600
    pool_pre_ping: bool = True
    echo: bool = False
    sqlite: SQLiteConfig | None = None

    @property
    def host(self) -> str:
        """Get database host from environment"""
        return os.getenv(self.host_env, "localhost")

    @property
    def database_name(self) -> str:
        """Get database name from environment"""
        return os.getenv(self.name_env, "rap_lyrics")

    @property
    def username(self) -> str:
        """Get database username from environment"""
        return os.getenv(self.username_env, "rap_user")

    @property
    def password(self) -> str:
        """Get database password from environment"""
        password = os.getenv(self.password_env)
        if not password:
            raise ValueError(f"Environment variable {self.password_env} not set!")
        return password

    @property
    def connection_string(self) -> str:
        """Generate SQLAlchemy connection string"""
        return (
            f"postgresql://{self.username}:{self.password}"
            f"@{self.host}:{self.port}/{self.database_name}"
        )


# ============================================================================
# Vector Search Configuration
# ============================================================================


class VectorSearchConfig(BaseModel):
    """pgvector configuration for embeddings"""

    enabled: bool = True
    embedding_model: str = "text-embedding-3-small"
    dimension: int = 1536
    distance_metric: Literal["cosine", "l2", "inner_product"] = "cosine"
    index_type: Literal["ivfflat", "hnsw"] = "ivfflat"
    lists: int = 100
    probes: int = 10
    cache_enabled: bool = True
    cache_ttl: int = 86400
    batch_size: int = 100

    @field_validator("dimension")
    @classmethod
    def validate_dimension(cls, v: int) -> int:
        """Validate embedding dimension"""
        valid_dimensions = [384, 512, 768, 1536, 3072]
        if v not in valid_dimensions:
            raise ValueError(f"Dimension must be one of {valid_dimensions}")
        return v


# ============================================================================
# Logging Configuration
# ============================================================================


class LoggingConfig(BaseModel):
    """Logging configuration"""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "logs/app.log"
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5
    encoding: str = "utf-8"
    console_output: bool = True

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate log level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()


# ============================================================================
# Analyzer Configurations
# ============================================================================


class AlgorithmicBasicConfig(BaseModel):
    """Algorithmic basic analyzer configuration"""

    sentiment_threshold: float = 0.5
    complexity_weights: dict[str, float] = {
        "vocabulary": 0.3,
        "structure": 0.3,
        "themes": 0.4,
    }


class QwenAnalyzerConfig(BaseModel):
    """Qwen LLM analyzer configuration"""

    model_name: str = "qwen/qwen3-4b-fp8"
    base_url: str = "https://api.novita.ai/openai/v1"
    api_key_env: str = "NOVITA_API_KEY"
    temperature: float = 0.1
    max_tokens: int = 1500
    timeout: int = 30
    retry_attempts: int = 3
    validate_api_key: bool = True

    @property
    def api_key(self) -> str:
        """Get API key from environment"""
        api_key = os.getenv(self.api_key_env)
        if self.validate_api_key and not api_key:
            raise ValueError(f"Environment variable {self.api_key_env} not set!")
        return api_key or ""


class OllamaAnalyzerConfig(BaseModel):
    """Ollama local LLM analyzer configuration"""

    model: str = "llama3.1:8b"
    base_url_env: str = "OLLAMA_BASE_URL"
    timeout: int = 60
    temperature: float = 0.3
    max_tokens: int = 1000

    @property
    def base_url(self) -> str:
        """Get Ollama base URL from environment"""
        return os.getenv(self.base_url_env, "http://localhost:11434")


class HybridAnalyzerConfig(BaseModel):
    """Hybrid analyzer configuration"""

    algorithms: list[str] = ["algorithmic_basic", "qwen"]
    consensus_threshold: float = 0.7
    fallback_analyzer: str = "algorithmic_basic"


class EmotionAnalyzerConfig(BaseModel):
    """Emotion analyzer configuration"""

    model_name: str = "j-hartmann/emotion-english-distilroberta-base"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    max_length: int = 512
    batch_size: int = 16
    cache_enabled: bool = True
    fallback_enabled: bool = True
    rap_analysis_enabled: bool = True
    postgres_enabled: bool = True


class AnalyzersConfig(BaseModel):
    """All analyzers configuration"""

    algorithmic_basic: dict[str, Any]
    qwen: dict[str, Any]
    ollama: dict[str, Any]
    hybrid: dict[str, Any]
    emotion_analyzer: dict[str, Any]

    def get_algorithmic_basic(self) -> AlgorithmicBasicConfig:
        return AlgorithmicBasicConfig(**self.algorithmic_basic.get("config", {}))

    def get_qwen(self) -> QwenAnalyzerConfig:
        return QwenAnalyzerConfig(**self.qwen.get("config", {}))

    def get_ollama(self) -> OllamaAnalyzerConfig:
        return OllamaAnalyzerConfig(**self.ollama.get("config", {}))

    def get_hybrid(self) -> HybridAnalyzerConfig:
        return HybridAnalyzerConfig(**self.hybrid.get("config", {}))

    def get_emotion(self) -> EmotionAnalyzerConfig:
        return EmotionAnalyzerConfig(**self.emotion_analyzer.get("config", {}))


# ============================================================================
# API Configuration
# ============================================================================


class CORSConfig(BaseModel):
    """CORS configuration"""

    enabled: bool = True
    origins: list[str] = ["http://localhost:3000"]
    allow_credentials: bool = True
    allow_methods: list[str] = ["*"]
    allow_headers: list[str] = ["*"]


class RateLimitConfig(BaseModel):
    """Rate limiting configuration"""

    enabled: bool = True
    requests_per_minute: int = 100
    burst_size: int = 20


class APIDocsConfig(BaseModel):
    """API documentation configuration"""

    enabled: bool = True
    swagger_url: str = "/docs"
    redoc_url: str = "/redoc"
    title: str = "Rap Analyzer API"
    version: str = "2.0.0"


class APIConfig(BaseModel):
    """FastAPI configuration"""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    log_level: str = "info"
    cors: CORSConfig
    rate_limit: RateLimitConfig
    docs: APIDocsConfig


# ============================================================================
# Redis Configuration
# ============================================================================


class RedisCacheConfig(BaseModel):
    """Redis cache TTL settings"""

    artist_ttl: int = 3600
    lyrics_ttl: int = 86400
    analysis_ttl: int = 604800
    embedding_ttl: int = 2592000


class RedisConfig(BaseModel):
    """Redis configuration"""

    enabled: bool = True
    host_env: str = "REDIS_HOST"
    port: int = 6379
    password_env: str = "REDIS_PASSWORD"
    db: int = 0
    max_connections: int = 50
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    cache: RedisCacheConfig

    @property
    def host(self) -> str:
        """Get Redis host from environment"""
        return os.getenv(self.host_env, "localhost")

    @property
    def password(self) -> str | None:
        """Get Redis password from environment"""
        return os.getenv(self.password_env)


# ============================================================================
# Monitoring Configuration
# ============================================================================


class PrometheusConfig(BaseModel):
    """Prometheus metrics configuration"""

    enabled: bool = True
    port: int = 9090
    path: str = "/metrics"
    include_default_metrics: bool = True


class GrafanaConfig(BaseModel):
    """Grafana dashboard configuration"""

    enabled: bool = True
    port: int = 3000
    admin_password_env: str = "GRAFANA_ADMIN_PASSWORD"
    datasource_url_env: str = "PROMETHEUS_URL"

    @property
    def admin_password(self) -> str:
        """Get Grafana admin password from environment"""
        password = os.getenv(self.admin_password_env, "admin")
        return password


class MetricsConfig(BaseModel):
    """Metrics collection configuration"""

    collect_request_duration: bool = True
    collect_request_count: bool = True
    collect_error_rate: bool = True
    collect_db_pool_metrics: bool = True
    collect_cache_metrics: bool = True


class HealthCheckConfig(BaseModel):
    """Health check configuration"""

    enabled: bool = True
    endpoint: str = "/health"
    check_interval: int = 60
    components: list[str] = ["database", "redis", "ollama", "qwen_api"]


class MonitoringConfig(BaseModel):
    """Monitoring configuration"""

    prometheus: PrometheusConfig
    grafana: GrafanaConfig
    metrics: MetricsConfig
    health: HealthCheckConfig


# ============================================================================
# CI/CD Configuration
# ============================================================================


class GitHubActionsConfig(BaseModel):
    """GitHub Actions CI/CD configuration"""

    enabled: bool = True
    test_on_push: bool = True
    test_on_pr: bool = True
    auto_deploy: bool = False


class TestingConfig(BaseModel):
    """Testing configuration"""

    required_coverage: int = 80
    run_integration_tests: bool = True
    run_performance_tests: bool = False
    parallel_execution: bool = True


class DeploymentConfig(BaseModel):
    """Deployment configuration"""

    environments: list[str] = ["dev", "staging", "prod"]
    auto_rollback: bool = True
    health_check_timeout: int = 300


class QualityConfig(BaseModel):
    """Code quality configuration"""

    max_complexity: int = 10
    max_function_length: int = 50
    max_file_length: int = 500
    enforce_type_hints: bool = True


class CICDConfig(BaseModel):
    """CI/CD configuration"""

    github_actions: GitHubActionsConfig
    testing: TestingConfig
    deployment: DeploymentConfig
    quality: QualityConfig


# ============================================================================
# Main Configuration
# ============================================================================


class Config(BaseModel):
    """Main configuration class with all settings"""

    application: ApplicationConfig
    database: DatabaseConfig
    vector_search: VectorSearchConfig
    logging: LoggingConfig
    analyzers: AnalyzersConfig
    api: APIConfig
    redis: RedisConfig
    monitoring: MonitoringConfig
    ci_cd: CICDConfig

    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> "Config":
        """Load configuration from YAML file"""
        # Try multiple locations for config file
        possible_paths = [
            Path(config_path),  # Direct path (if absolute or relative to CWD)
            Path(__file__).parent.parent.parent / config_path,  # Project root
            Path.cwd() / config_path,  # Current working directory
        ]
        
        config_file = None
        for path in possible_paths:
            if path.exists():
                config_file = path
                break
        
        if config_file is None:
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Searched in: {[str(p) for p in possible_paths]}"
            )

        with open(config_file, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        # Apply environment-specific overrides
        environment = os.getenv("APP_ENVIRONMENT", "production")
        if "environments" in config_data and environment in config_data["environments"]:
            env_overrides = config_data["environments"][environment]
            cls._deep_merge(config_data, env_overrides)

        return cls(**config_data)

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> None:
        """Deep merge override dict into base dict"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                Config._deep_merge(base[key], value)
            else:
                base[key] = value

    def validate_all(self) -> None:
        """Validate all configuration settings"""
        # Validate database connection
        try:
            _ = self.database.connection_string
        except ValueError as e:
            raise ValueError(f"Database configuration error: {e}")

        # Validate API keys if needed
        if self.analyzers.qwen.get("enabled", True):
            qwen_config = self.analyzers.get_qwen()
            if qwen_config.validate_api_key:
                _ = qwen_config.api_key

        print("✅ Configuration validation passed!")


# ============================================================================
# Global Config Singleton
# ============================================================================


@lru_cache
def get_config(config_path: str = "config.yaml") -> Config:
    """
    Get global configuration instance (cached)

    Usage:
        from config_loader import get_config

        config = get_config()
        db_string = config.database.connection_string
        api_port = config.api.port
    """
    config = Config.from_yaml(config_path)
    config.validate_all()
    return config


# ============================================================================
# Helper Functions
# ============================================================================


def load_config(config_path: str = "config.yaml", validate: bool = True) -> Config:
    """
    Load and optionally validate configuration

    Args:
        config_path: Path to YAML config file
        validate: Whether to validate configuration

    Returns:
        Config: Validated configuration object
    """
    config = Config.from_yaml(config_path)
    if validate:
        config.validate_all()
    return config


def get_environment() -> str:
    """Get current environment from ENV variable"""
    return os.getenv("APP_ENVIRONMENT", "production")


# ============================================================================
# CLI for Quick Configuration Check
# ============================================================================

if __name__ == "__main__":
    import sys

    print("⚡ Quick Config Check")
    print("=" * 50)

    try:
        # Load configuration
        config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
        print(f"📁 Loading: {config_file}")

        config = load_config(config_file)

        # Quick validation summary
        print("\n✅ Config Valid!")
        print(f"   • App: {config.application.name} v{config.application.version}")
        print(f"   • Env: {config.application.environment}")
        print(
            f"   • DB: {config.database.type}://{config.database.host}:{config.database.port}/{config.database.database_name}"
        )
        print(f"   • API: http://{config.api.host}:{config.api.port}")
        print(
            f"   • Redis: {config.redis.host}:{config.redis.port} (enabled: {config.redis.enabled})"
        )

        # Component status
        components = []
        if config.vector_search.enabled:
            components.append("Vector Search")
        if config.monitoring.prometheus.enabled:
            components.append("Prometheus")
        if config.monitoring.grafana.enabled:
            components.append("Grafana")

        if components:
            print(f"   • Components: {', '.join(components)}")

        print("\n💡 For detailed testing run:")
        print("   python src/config/test_loader.py")

    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("   Make sure config.yaml exists")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ Validation Error: {e}")
        print("   Check your .env file and config.yaml")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
