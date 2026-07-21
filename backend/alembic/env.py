import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# ── CRITICAL: import ALL models so Alembic's autogenerate sees them ──
from app.database import Base
import app.models  # noqa: F401 — side-effect import registers all ORM models

# Alembic config object
config = context.config

# Override DB URL from pydantic settings which automatically loads the .env file
from app.config import settings
if settings.DATABASE_URL:
    config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# Logging setup
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# The metadata Alembic uses for autogenerate comparisons
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    Offline mode: generate SQL script without connecting to the DB.
    Run with: alembic upgrade head --sql
    Useful for reviewing SQL before applying, or handing to a DBA.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Online mode: connect to the DB and apply migrations directly.
    This is the normal path for: alembic upgrade head
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # No pooling in migrations (single-use)
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,          # Detect column type changes
            compare_server_default=True, # Detect server_default changes
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
