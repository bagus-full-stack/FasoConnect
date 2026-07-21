# migrations/versions/001_create_history_table.py
"""create history table

Revision ID: 001
Create Date: 2026-07-19
"""
from alembic import op
import sqlalchemy as sa

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "history",
        sa.Column("id",          sa.String(36),  nullable=False),
        sa.Column("user_id",     sa.String(128), nullable=True),
        sa.Column("action_type", sa.String(32),  nullable=False),
        sa.Column("src_lang",    sa.String(32),  nullable=True),
        sa.Column("tgt_lang",    sa.String(32),  nullable=True),
        sa.Column("source_text", sa.Text(),      nullable=True),
        sa.Column("result_text", sa.Text(),      nullable=True),
        sa.Column("speed",       sa.Float(),     nullable=True),
        sa.Column("created_at",  sa.DateTime(),  nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_history_user_id",     "history", ["user_id"])
    op.create_index("ix_history_action_type", "history", ["action_type"])
    op.create_index("ix_history_created_at",  "history", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_history_created_at",  table_name="history")
    op.drop_index("ix_history_action_type", table_name="history")
    op.drop_index("ix_history_user_id",     table_name="history")
    op.drop_table("history")