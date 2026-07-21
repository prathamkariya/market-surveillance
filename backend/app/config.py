from pydantic_settings import BaseSettings
from pydantic import model_validator
from functools import lru_cache


class Settings(BaseSettings):
    # Database components
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: str = "5432"
    POSTGRES_DB: str = "market_surveillance"
    DATABASE_URL: str | None = None

    # JWT — Access tokens
    SECRET_KEY: str
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
    # ml's scripts/train.py (both multi_pattern_detector
    # and isolation_forest_scratch artifacts, if trained, live here together)
    MODEL_DIR: str = "trained_models"

    # Phase 8: Redis URL
    REDIS_URL: str = "redis://localhost:6379/0"

    # CORS — comma-separated list of allowed origins.
    # In production, set this to your actual frontend domain(s).
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

    _WEAK_SECRETS = {
        "change-this-in-production-use-openssl-rand-hex-32",
        "dev_secret_key_change_me",
        "super_secret_production_key_change_me",
    }

    @model_validator(mode="after")
    def _check_secrets(self) -> "Settings":
        import warnings
        if self.SECRET_KEY in self._WEAK_SECRETS:
            if self.APP_ENV != "development":
                raise ValueError(
                    "SECRET_KEY is still set to a default placeholder value. "
                    "Generate a secure key with: openssl rand -hex 32 "
                    "and set it in your environment before running in production."
                )
            warnings.warn(
                "SECRET_KEY is using a known insecure placeholder. "
                "This is only acceptable in APP_ENV=development.",
                stacklevel=2,
            )
            
        if self.POSTGRES_PASSWORD == "password":
            if self.APP_ENV != "development":
                raise ValueError(
                    "POSTGRES_PASSWORD is still set to the default 'password'. "
                    "Please use a secure password in production."
                )
            warnings.warn(
                "POSTGRES_PASSWORD is using a known insecure placeholder. "
                "This is only acceptable in APP_ENV=development.",
                stacklevel=2,
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def _apply_dev_defaults(cls, data: dict) -> dict:
        import os
        # APP_ENV may not be in data if relying on the class default or env file,
        # but pydantic hasn't merged env vars fully into `data` in mode="before"
        # wait, actually for BaseSettings, `data` contains the env vars + init kwargs.
        # But just in case, we also check os.getenv.
        env = data.get("APP_ENV", os.getenv("APP_ENV", "development"))
        if env == "development":
            data.setdefault("SECRET_KEY", "change-this-in-production-use-openssl-rand-hex-32")
            data.setdefault("POSTGRES_PASSWORD", "password")
        return data

    @model_validator(mode="after")
    def _build_db_url(self) -> "Settings":
        if not self.DATABASE_URL:
            from sqlalchemy.engine import URL
            self.DATABASE_URL = URL.create(
                drivername="postgresql",
                username=self.POSTGRES_USER,
                password=self.POSTGRES_PASSWORD,
                host=self.POSTGRES_HOST,
                port=int(self.POSTGRES_PORT),
                database=self.POSTGRES_DB
            ).render_as_string(hide_password=False)
            
        if self.APP_ENV != "development" and self.DATABASE_URL and "REPLACE_ME_WITH_SECURE_PASSWORD" in self.DATABASE_URL:
            raise ValueError(
                "DATABASE_URL contains the default placeholder REPLACE_ME_WITH_SECURE_PASSWORD. "
                "You must replace it with a real secure password before running in production."
            )
        return self


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings loader.
    lru_cache means this is only instantiated once per process.
    Tests can clear the cache with get_settings.cache_clear().
    """
    return Settings()


settings = get_settings()
