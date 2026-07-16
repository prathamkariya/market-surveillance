"""Initial schema: users, market_data, anomalies, alerts

Revision ID: 001
Revises: None
Create Date: 2026-06-28
"""
from alembic import op
import sqlalchemy as sa

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── users ────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("username", sa.String(50), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.UniqueConstraint("email", name="uq_users_email"),
        sa.UniqueConstraint("username", name="uq_users_username"),
    )

    # ── market_data ──────────────────────────────────────────────
    op.create_table(
        "market_data",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Numeric(15, 6), nullable=False),
        sa.Column("high", sa.Numeric(15, 6), nullable=False),
        sa.Column("low", sa.Numeric(15, 6), nullable=False),
        sa.Column("close", sa.Numeric(15, 6), nullable=False),
        sa.Column("volume", sa.Numeric(20, 2), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_market_data_user_id", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_market_data"),
        sa.UniqueConstraint("symbol", "timestamp", name="uq_market_data_symbol_timestamp"),
    )
    op.create_index("ix_market_data_symbol", "market_data", ["symbol"])
    op.create_index("ix_market_data_user_id", "market_data", ["user_id"])
    op.create_index("ix_market_data_symbol_timestamp", "market_data", ["symbol", "timestamp"])

    # ── anomalies ────────────────────────────────────────────────
    op.create_table(
        "anomalies",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("market_data_id", sa.Integer(), nullable=False),
        sa.Column("anomaly_score", sa.Float(), nullable=False),
        sa.Column("is_anomaly", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("isolation_forest_score", sa.Float(), nullable=True),
        sa.Column("xgboost_score", sa.Float(), nullable=True),
        sa.Column("features", sa.Text(), nullable=True),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["market_data_id"], ["market_data.id"], name="fk_anomalies_market_data_id", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_anomalies"),
    )
    op.create_index("ix_anomalies_market_data_id", "anomalies", ["market_data_id"])

    # ── alerts ───────────────────────────────────────────────────
    op.create_table(
        "alerts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("anomaly_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default=sa.text("'PENDING'")),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["anomaly_id"], ["anomalies.id"], name="fk_alerts_anomaly_id", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_alerts_user_id", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_alerts"),
    )
    op.create_index("ix_alerts_anomaly_id", "alerts", ["anomaly_id"])
    op.create_index("ix_alerts_user_id", "alerts", ["user_id"])


def downgrade() -> None:
    op.drop_table("alerts")
    op.drop_table("anomalies")
    op.drop_table("market_data")
    op.drop_table("users")
