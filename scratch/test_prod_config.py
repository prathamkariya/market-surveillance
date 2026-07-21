import os
try:
    os.environ["APP_ENV"] = "production"
    os.environ["SECRET_KEY"] = "change-this-in-production-use-openssl-rand-hex-32"
    os.environ["POSTGRES_PASSWORD"] = "password"
    from app.config import get_settings
    get_settings.cache_clear()
    s = get_settings()
    print("Should not happen! Built:", s.dict())
except Exception as e:
    print("Error:", type(e).__name__, e)
