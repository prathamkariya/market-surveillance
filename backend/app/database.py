from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.pool import NullPool

from app.config import settings


# ──────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────
engine = create_engine(
    settings.DATABASE_URL,
    # Pool settings for production
    pool_size=5,           # Connections kept alive
    max_overflow=10,       # Extra connections when pool exhausted
    pool_timeout=30,       # Seconds to wait for a connection
    pool_recycle=1800,     # Recycle connections every 30 minutes
    pool_pre_ping=True,    # Test connections before handing out (avoids stale conn errors)
    echo=settings.DEBUG,   # Log SQL in development only
)

# ──────────────────────────────────────────────
# Session factory
# ──────────────────────────────────────────────
SessionLocal = sessionmaker(
    autocommit=False,   # We control commits explicitly
    autoflush=False,    # We control flushes explicitly
    bind=engine,
)

# ──────────────────────────────────────────────
# Declarative base — all models inherit from this
# ──────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ──────────────────────────────────────────────
# FastAPI dependency
# ──────────────────────────────────────────────
def get_db():
    """
    Yield a database session for the duration of a request.

    The finally block guarantees db.close() runs even if the
    route handler raises an exception — preventing connection leaks.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
