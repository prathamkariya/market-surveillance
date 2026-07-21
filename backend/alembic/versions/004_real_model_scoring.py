"""Replace mock xgboost_score with real per-pattern model output

Phase 7: anomaly_service now scores with a REAL trained
IsolationForestScratch and MultiPatternDetector (mkt_surveillance_ml)
instead of hand-coded mock formulas. isolation_forest_score keeps its
name (it's genuinely an Isolation Forest score now, not a mock).
xgboost_score is renamed rather than kept: it never actually held an
XGBoost score even in the mock version, and MultiPatternDetector's
default estimator is a RandomForestClassifier, not XGBoost either --
keeping a column named "xgboost_score" populated with that value would
be a misleading label carried forward for no reason. pattern_scores adds
the full per-pattern breakdown the old single-float design couldn't
represent, and model_version adds provenance (which trained model
actually produced this score).

Revision ID: 004
Revises: 003
Create Date: 2026-07-13
"""
from alembic import op
import sqlalchemy as sa

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "anomalies", "xgboost_score",
        new_column_name="multi_pattern_max_score",
        existing_type=sa.Float(),
        existing_nullable=True,
    )
    op.add_column(
        "anomalies",
        sa.Column("pattern_scores", sa.Text(), nullable=True),
    )
    op.add_column(
        "anomalies",
        sa.Column("model_version", sa.String(255), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("anomalies", "model_version")
    op.drop_column("anomalies", "pattern_scores")
    op.alter_column(
        "anomalies", "multi_pattern_max_score",
        new_column_name="xgboost_score",
        existing_type=sa.Float(),
        existing_nullable=True,
    )
