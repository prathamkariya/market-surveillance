"""Add market column to market_data; scope unique constraint to (user_id, symbol, timestamp)

B10: The global (symbol, timestamp) unique constraint is too broad -- the streaming
engine writes MarketData under a shared system user, so any real user submitting
data for the same symbol+timestamp would get a spurious 409 Conflict. Scoping to
(user_id, symbol, timestamp) means each user dataset is independent.

B14/B15: Adds a nullable market VARCHAR(20) column so detect_anomaly() can route
to the correct per-market model registry. Nullable for backward compat -- existing
records will have NULL and detect_anomaly() will raise a clean 400 rather than
silently routing to the wrong model.

Revision ID: 005
Revises: 004
Create Date: 2026-07-17
"""
from alembic import op
import sqlalchemy as sa

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # B14/B15: Add market column (nullable for backward compat)
    op.add_column(
        "market_data",
        sa.Column("market", sa.String(20), nullable=True),
    )
    op.create_index("ix_market_data_market", "market_data", ["market"])

    # B10: Scope the unique constraint from (symbol, timestamp) to (user_id, symbol, timestamp)
    op.drop_constraint("uq_market_data_symbol_timestamp", "market_data", type_="unique")
    op.create_unique_constraint(
        "uq_market_data_user_symbol_timestamp",
        "market_data",
        ["user_id", "symbol", "timestamp"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_market_data_user_symbol_timestamp", "market_data", type_="unique")
    op.create_unique_constraint(
        "uq_market_data_symbol_timestamp", "market_data", ["symbol", "timestamp"]
    )
    op.drop_index("ix_market_data_market", table_name="market_data")
    op.drop_column("market_data", "market")
