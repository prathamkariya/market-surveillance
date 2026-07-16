from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/market_surveillance"

    # JWT — Access tokens
    SECRET_KEY: str = "change-this-in-production-use-openssl-rand-hex-32"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # JWT — Refresh tokens (Phase 2)
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # App
    APP_ENV: str = "development"
    DEBUG: bool = True

    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 60

    # ML model artifacts (Phase 7) — directory produced by
    # mkt_surveillance_ml's scripts/train.py (both multi_pattern_detector
    # and isolation_forest_scratch artifacts, if trained, live here together)
    MODEL_DIR: str = "trained_models"

    # Phase 8: Redis URL
    REDIS_URL: str = "redis://localhost:6379/0"

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings loader.
    lru_cache means this is only instantiated once per process.
    Tests can clear the cache with get_settings.cache_clear().
    """
    return Settings()


settings = get_settings()
