from pydantic_settings import BaseSettings
from pydantic import model_validator
import os

class Settings(BaseSettings):
    APP_ENV: str = "development"
    SECRET_KEY: str
    POSTGRES_PASSWORD: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _apply_dev_defaults(cls, data: dict) -> dict:
        # data might not contain APP_ENV if it defaults, but pydantic hasn't applied defaults yet
        env = data.get("APP_ENV", os.getenv("APP_ENV", "development"))
        if env == "development":
            data.setdefault("SECRET_KEY", "dev-secret")
            data.setdefault("POSTGRES_PASSWORD", "dev-pass")
        return data

s = Settings()
print("Dev:", s.SECRET_KEY, s.POSTGRES_PASSWORD)

try:
    Settings(APP_ENV="production")
    print("Prod should have failed!")
except Exception as e:
    print("Prod failed as expected:", e)
