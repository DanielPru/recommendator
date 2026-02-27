"""Initial schema creation

Revision ID: 001_initial_schema
Revises:
Create Date: 2024-01-15

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgcrypto extension for UUID generation
    op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')

    # Create decision_logs table
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

    # Create performance_logs table
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

    # Create context_stats table
    op.create_table(
        "context_stats",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("context_hash", sa.String(64), unique=True, nullable=False),
        sa.Column("total_decisions", sa.Integer(), default=0),
        sa.Column("total_performances", sa.Integer(), default=0),
        sa.Column("avg_attention", sa.Float(), nullable=True),
        sa.Column("avg_persuasion", sa.Float(), nullable=True),
        sa.Column("std_attention", sa.Float(), nullable=True),
        sa.Column("std_persuasion", sa.Float(), nullable=True),
        sa.Column("first_seen", sa.DateTime(), server_default=sa.text("now()")),
        sa.Column("last_updated", sa.DateTime(), server_default=sa.text("now()")),
    )

    op.create_index("ix_context_stats_context_hash", "context_stats", ["context_hash"])

    # Create structure_stats table
    op.create_table(
        "structure_stats",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("structure_hash", sa.String(64), unique=True, nullable=False),
        sa.Column("structure_features", postgresql.JSON(), nullable=False),
        sa.Column("total_uses", sa.Integer(), default=0),
        sa.Column("total_performances", sa.Integer(), default=0),
        sa.Column("avg_attention", sa.Float(), nullable=True),
        sa.Column("avg_persuasion", sa.Float(), nullable=True),
        sa.Column("first_seen", sa.DateTime(), server_default=sa.text("now()")),
        sa.Column("last_updated", sa.DateTime(), server_default=sa.text("now()")),
    )

    op.create_index(
        "ix_structure_stats_structure_hash", "structure_stats", ["structure_hash"]
    )

    # Create model_registry table
    op.create_table(
        "model_registry",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")
        ),
        sa.Column("version", sa.String(50), nullable=False),
        sa.Column("model_type", sa.String(50), nullable=False),
        sa.Column("model_path", sa.String(500), nullable=False),
        sa.Column("training_samples", sa.Integer(), nullable=False),
        sa.Column("feature_schema_version", sa.String(10), default="v1"),
        sa.Column("auc_score", sa.Float(), nullable=True),
        sa.Column("precision_at_k", sa.Float(), nullable=True),
        sa.Column("is_active", sa.Boolean(), default=True),
    )

    op.create_index("ix_model_registry_version", "model_registry", ["version"])
    op.create_index(
        "ix_model_registry_version_type", "model_registry", ["version", "model_type"]
    )
    op.create_index("ix_model_registry_active", "model_registry", ["is_active"])


def downgrade() -> None:
    op.drop_table("model_registry")
    op.drop_table("structure_stats")
    op.drop_table("context_stats")
    op.drop_table("performance_logs")
    op.drop_table("decision_logs")
    op.execute('DROP EXTENSION IF EXISTS "pgcrypto"')
