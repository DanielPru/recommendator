"""
Database models for CVIE service.
Uses asset_id as the primary lifecycle identifier.
"""

from datetime import datetime
from typing import Optional
import uuid

from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    Boolean,
    DateTime,
    Text,
    JSON,
    Index,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.database import Base


class DecisionLog(Base):
    """
    Logs every structure recommendation decision.
    Uses asset_id as primary key - each asset has exactly one recommendation.
    """

    __tablename__ = "decision_logs"

    # Primary key is now asset_id (provided by caller)
    asset_id = Column(String(255), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Request context
    segment_strategy = Column(Text, nullable=False)
    channel = Column(String(100), nullable=False)
    traffic_type = Column(String(20), nullable=False)  # organic | paid
    funnel_stage = Column(String(10), nullable=False)  # TOFU | MOFU | BOFU
    content_type = Column(String(10), nullable=False)  # image | video

    # Context hash for grouping similar requests
    context_hash = Column(String(64), nullable=False, index=True)

    # Selected structure
    structure_hash = Column(String(64), nullable=False, index=True)
    structure_features = Column(JSON, nullable=False)

    # ML scoring results
    p_attention = Column(Float, nullable=True)
    p_persuasion = Column(Float, nullable=True)
    p_final = Column(Float, nullable=True)

    # Decision metadata
    mode = Column(String(20), nullable=False)  # 'exploration' | 'exploitation'
    context_confidence = Column(Float, default=0.0)
    exploration_weight = Column(Float, default=0.0)
    model_version = Column(String(50), nullable=True)
    candidates_count = Column(Integer, nullable=False)

    # Schema version for future compatibility
    schema_version = Column(String(10), default="v1", nullable=False)

    __table_args__ = (
        Index("ix_decision_logs_created_at", "created_at"),
        Index("ix_decision_logs_context_structure", "context_hash", "structure_hash"),
    )


class PerformanceLog(Base):
    """
    Logs performance outcomes for structure decisions.
    Links to decision_logs via asset_id.
    """

    __tablename__ = "performance_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Link to decision via asset_id
    asset_id = Column(
        String(255), ForeignKey("decision_logs.asset_id"), nullable=False, index=True
    )

    # Context identifiers (denormalized for query efficiency)
    context_hash = Column(String(64), nullable=False, index=True)
    structure_hash = Column(String(64), nullable=False, index=True)
    traffic_type = Column(String(20), nullable=False)

    # Performance metrics
    attention_score = Column(Float, nullable=False)  # 0.0 - 1.0
    persuasion_score = Column(Float, nullable=False)  # 0.0 - 1.0

    # Optional raw metrics for debugging
    raw_metrics = Column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_performance_logs_context", "context_hash", "structure_hash"),
    )


class ContextStats(Base):
    """
    Aggregated statistics per context hash.
    Used for exploration uncertainty calculation.
    """

    __tablename__ = "context_stats"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    context_hash = Column(String(64), unique=True, nullable=False, index=True)

    # Counts
    total_decisions = Column(Integer, default=0)
    total_performances = Column(Integer, default=0)

    # Aggregated scores
    avg_attention = Column(Float, nullable=True)
    avg_persuasion = Column(Float, nullable=True)
    std_attention = Column(Float, nullable=True)
    std_persuasion = Column(Float, nullable=True)

    # Timestamps
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class StructureStats(Base):
    """
    Aggregated statistics per structure hash.
    Used for novelty calculation in exploration.
    """

    __tablename__ = "structure_stats"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    structure_hash = Column(String(64), unique=True, nullable=False, index=True)
    structure_features = Column(JSON, nullable=False)

    # Counts
    total_uses = Column(Integer, default=0)
    total_performances = Column(Integer, default=0)

    # Aggregated scores
    avg_attention = Column(Float, nullable=True)
    avg_persuasion = Column(Float, nullable=True)

    # Timestamps
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ModelRegistry(Base):
    """
    Registry of trained ML models.
    Supports versioning and hot-reload.
    """

    __tablename__ = "model_registry"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Model identification
    version = Column(String(50), nullable=False, index=True)
    model_type = Column(String(50), nullable=False)  # organic_attention, etc.

    # Model metadata
    model_path = Column(String(500), nullable=False)
    training_samples = Column(Integer, nullable=False)
    feature_schema_version = Column(String(10), default="v1")

    # Performance metrics
    auc_score = Column(Float, nullable=True)
    precision_at_k = Column(Float, nullable=True)

    # Status
    is_active = Column(Boolean, default=True)

    __table_args__ = (
        Index("ix_model_registry_version_type", "version", "model_type"),
        Index("ix_model_registry_active", "is_active"),
    )
