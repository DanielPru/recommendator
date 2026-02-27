"""Asset ID refactor - replace decision_id with asset_id

Revision ID: 002_asset_id_refactor
Revises: 001_initial_schema
Create Date: 2024-01-15

This migration:
1. Changes decision_logs PK from UUID 'id' to TEXT 'asset_id'
2. Changes performance_logs FK from 'decision_id' to 'asset_id'
3. Adds new fields: p_attention, p_persuasion, p_final, mode, context_confidence, exploration_weight
4. Removes old fields: ml_score, was_exploration, exploration_bonus

IMPORTANT: This is a destructive migration. Back up data before running.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "002_asset_id_refactor"
down_revision: Union[str, None] = "001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # =========================================================
    # Step 1: Create new decision_logs table with asset_id as PK
    # =========================================================
    op.create_table(
        "decision_logs_new",
        sa.Column("asset_id", sa.String(255), primary_key=True),
        sa.Column(
            "created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")
        ),
        sa.Column("segment_strategy", sa.Text(), nullable=False),
        sa.Column("channel", sa.String(100), nullable=False),
        sa.Column("traffic_type", sa.String(20), nullable=False),
        sa.Column("funnel_stage", sa.String(10), nullable=False),
        sa.Column("content_type", sa.String(10), nullable=False),
        sa.Column("context_hash", sa.String(64), nullable=False),
        sa.Column("structure_hash", sa.String(64), nullable=False),
        sa.Column("structure_features", postgresql.JSON(), nullable=False),
        # New ML scoring fields
        sa.Column("p_attention", sa.Float(), nullable=True),
        sa.Column("p_persuasion", sa.Float(), nullable=True),
        sa.Column("p_final", sa.Float(), nullable=True),
        # New decision metadata
        sa.Column("mode", sa.String(20), nullable=False, server_default="exploration"),
        sa.Column("context_confidence", sa.Float(), server_default="0.0"),
        sa.Column("exploration_weight", sa.Float(), server_default="0.0"),
        sa.Column("model_version", sa.String(50), nullable=True),
        sa.Column("candidates_count", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.String(10), server_default="v1", nullable=False),
    )

    # =========================================================
    # Step 2: Create new performance_logs table with asset_id FK
    # =========================================================
    op.create_table(
        "performance_logs_new",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")
        ),
        sa.Column(
            "asset_id",
            sa.String(255),
            sa.ForeignKey("decision_logs_new.asset_id"),
            nullable=False,
        ),
        sa.Column("context_hash", sa.String(64), nullable=False),
        sa.Column("structure_hash", sa.String(64), nullable=False),
        sa.Column("traffic_type", sa.String(20), nullable=False),
        sa.Column("attention_score", sa.Float(), nullable=False),
        sa.Column("persuasion_score", sa.Float(), nullable=False),
        sa.Column("raw_metrics", postgresql.JSON(), nullable=True),
    )

    # =========================================================
    # Step 3: Drop old tables and rename new ones
    # =========================================================

    # Drop old indexes first
    op.drop_index("ix_performance_logs_decision_id", table_name="performance_logs")
    op.drop_index("ix_performance_logs_context_hash", table_name="performance_logs")
    op.drop_index("ix_performance_logs_structure_hash", table_name="performance_logs")
    op.drop_index("ix_performance_logs_context", table_name="performance_logs")

    op.drop_index("ix_decision_logs_context_hash", table_name="decision_logs")
    op.drop_index("ix_decision_logs_structure_hash", table_name="decision_logs")
    op.drop_index("ix_decision_logs_created_at", table_name="decision_logs")
    op.drop_index("ix_decision_logs_context_structure", table_name="decision_logs")

    # Drop old tables
    op.drop_table("performance_logs")
    op.drop_table("decision_logs")

    # Rename new tables
    op.rename_table("decision_logs_new", "decision_logs")
    op.rename_table("performance_logs_new", "performance_logs")

    # =========================================================
    # Step 4: Create indexes for new tables
    # =========================================================

    # decision_logs indexes
    op.create_index("ix_decision_logs_context_hash", "decision_logs", ["context_hash"])
    op.create_index(
        "ix_decision_logs_structure_hash", "decision_logs", ["structure_hash"]
    )
    op.create_index("ix_decision_logs_created_at", "decision_logs", ["created_at"])
    op.create_index(
        "ix_decision_logs_context_structure",
        "decision_logs",
        ["context_hash", "structure_hash"],
    )

    # performance_logs indexes
    op.create_index("ix_performance_logs_asset_id", "performance_logs", ["asset_id"])
    op.create_index(
        "ix_performance_logs_context_hash", "performance_logs", ["context_hash"]
    )
    op.create_index(
        "ix_performance_logs_structure_hash", "performance_logs", ["structure_hash"]
    )
    op.create_index(
        "ix_performance_logs_context",
        "performance_logs",
        ["context_hash", "structure_hash"],
    )


def downgrade() -> None:
    """
    Downgrade recreates original schema with decision_id.
    WARNING: Data will be lost as asset_id cannot be converted back to UUID.
    """

    # Drop new indexes
    op.drop_index("ix_performance_logs_asset_id", table_name="performance_logs")
    op.drop_index("ix_performance_logs_context_hash", table_name="performance_logs")
    op.drop_index("ix_performance_logs_structure_hash", table_name="performance_logs")
    op.drop_index("ix_performance_logs_context", table_name="performance_logs")

    op.drop_index("ix_decision_logs_context_hash", table_name="decision_logs")
    op.drop_index("ix_decision_logs_structure_hash", table_name="decision_logs")
    op.drop_index("ix_decision_logs_created_at", table_name="decision_logs")
    op.drop_index("ix_decision_logs_context_structure", table_name="decision_logs")

    # Drop new tables
    op.drop_table("performance_logs")
    op.drop_table("decision_logs")

    # Recreate original decision_logs table
    op.create_table(
        "decision_logs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")
        ),
        sa.Column("segment_strategy", sa.Text(), nullable=False),
        sa.Column("channel", sa.String(100), nullable=False),
        sa.Column("traffic_type", sa.String(20), nullable=False),
        sa.Column("funnel_stage", sa.String(10), nullable=False),
        sa.Column("content_type", sa.String(10), nullable=False),
        sa.Column("context_hash", sa.String(64), nullable=False),
        sa.Column("structure_hash", sa.String(64), nullable=False),
        sa.Column("structure_features", postgresql.JSON(), nullable=False),
        sa.Column("ml_score", sa.Float(), nullable=True),
        sa.Column("was_exploration", sa.Boolean(), default=False),
        sa.Column("exploration_bonus", sa.Float(), default=0.0),
        sa.Column("model_version", sa.String(50), nullable=True),
        sa.Column("candidates_count", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.String(10), default="v1", nullable=False),
    )

    # Recreate original performance_logs table
    op.create_table(
        "performance_logs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")
        ),
        sa.Column(
            "decision_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("decision_logs.id"),
            nullable=False,
        ),
        sa.Column("context_hash", sa.String(64), nullable=False),
        sa.Column("structure_hash", sa.String(64), nullable=False),
        sa.Column("traffic_type", sa.String(20), nullable=False),
        sa.Column("attention_score", sa.Float(), nullable=False),
        sa.Column("persuasion_score", sa.Float(), nullable=False),
        sa.Column("raw_metrics", postgresql.JSON(), nullable=True),
    )

    # Recreate original indexes
    op.create_index("ix_decision_logs_context_hash", "decision_logs", ["context_hash"])
    op.create_index(
        "ix_decision_logs_structure_hash", "decision_logs", ["structure_hash"]
    )
    op.create_index("ix_decision_logs_created_at", "decision_logs", ["created_at"])
    op.create_index(
        "ix_decision_logs_context_structure",
        "decision_logs",
        ["context_hash", "structure_hash"],
    )

    op.create_index(
        "ix_performance_logs_decision_id", "performance_logs", ["decision_id"]
    )
    op.create_index(
        "ix_performance_logs_context_hash", "performance_logs", ["context_hash"]
    )
    op.create_index(
        "ix_performance_logs_structure_hash", "performance_logs", ["structure_hash"]
    )
    op.create_index(
        "ix_performance_logs_context",
        "performance_logs",
        ["context_hash", "structure_hash"],
    )
