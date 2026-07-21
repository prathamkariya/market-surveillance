"""Add watchlists and watchlist_symbols

Revision ID: 002
Revises: 001
Create Date: 2026-06-28
"""
from alembic import op
import sqlalchemy as sa

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── watchlists ───────────────────────────────────────────────
    op.create_table(
        "watchlists",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.String(500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_watchlists_user_id", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_watchlists"),
        sa.UniqueConstraint("user_id", "name", name="uq_watchlists_user_name"),
    )
    op.create_index("ix_watchlists_user_id", "watchlists", ["user_id"])

    # ── watchlist_symbols ────────────────────────────────────────
    op.create_table(
        "watchlist_symbols",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("watchlist_id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("notes", sa.String(500), nullable=True),
        sa.Column("added_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(
            ["watchlist_id"], ["watchlists.id"],
            name="fk_watchlist_symbols_watchlist_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_watchlist_symbols"),
        sa.UniqueConstraint("watchlist_id", "symbol", name="uq_watchlist_symbols_watchlist_symbol"),
    )
    op.create_index("ix_watchlist_symbols_watchlist_id", "watchlist_symbols", ["watchlist_id"])


def downgrade() -> None:
    op.drop_table("watchlist_symbols")
    op.drop_table("watchlists")
