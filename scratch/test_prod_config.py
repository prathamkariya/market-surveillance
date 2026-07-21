import os
try:
    os.environ["APP_ENV"] = "production"
    # Ensure it's not picking up from `.env` in the current dir if it has dummy values
    # Actually pydantic-settings reads `.env` by default if env_file=".env"
    from app.config import get_settings
    get_settings.cache_clear()
    s = get_settings()
    print("Should not happen! Built:", s.dict())
except Exception as e:
    print("Error:", type(e).__name__, e)
