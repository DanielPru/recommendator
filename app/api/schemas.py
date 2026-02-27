"""
API schemas using Pydantic.
Uses asset_id as the primary lifecycle identifier.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class TrafficType(str, Enum):
    organic = "organic"
    paid = "paid"


class FunnelStage(str, Enum):
    TOFU = "TOFU"
    MOFU = "MOFU"
    BOFU = "BOFU"


class ContentType(str, Enum):
    image = "image"
    video = "video"


class RecommendRequest(BaseModel):
    """Request for structure recommendation."""

    asset_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Unique asset identifier (generated upstream before calling this endpoint)",
    )
    segment_strategy: str = Field(
        ..., min_length=1, description="Strategic text describing target segment"
    )
    channel: str = Field(
        ...,
        min_length=1,
        description="Distribution channel (e.g., 'tiktok', 'instagram_feed')",
    )
    traffic_type: TrafficType = Field(..., description="Traffic source type")
    funnel_stage: FunnelStage = Field(..., description="Marketing funnel stage")
    content_type: ContentType = Field(..., description="Content format type")

    model_config = {
        "json_schema_extra": {
            "example": {
                "asset_id": "asset_20240115_abc123",
                "segment_strategy": "Target short-form video consumers who respond to authentic UGC content with bold visuals",
                "channel": "tiktok",
                "traffic_type": "organic",
                "funnel_stage": "TOFU",
                "content_type": "video",
            }
        }
    }


class StructureFeatures(BaseModel):
    """Visual structure features."""

    primary_subject_area_ratio: str
    focal_point_count: str
    figure_background_contrast: str
    motion_intensity: str
    text_coverage_ratio: str
    face_presence_scale: str
    product_visibility_ratio: str
    offer_visual_salience: str
    visual_complexity: str


class RecommendResponse(BaseModel):
    """Response with recommended structure."""

    asset_id: str = Field(..., description="Asset identifier for this recommendation")
    structure_hash: str = Field(..., description="Unique identifier for the structure")
    structure: StructureFeatures = Field(
        ..., description="Recommended visual structure"
    )

    # Scoring information
    p_attention: Optional[float] = Field(
        None, description="Attention probability score"
    )
    p_persuasion: Optional[float] = Field(
        None, description="Persuasion probability score"
    )
    p_final: Optional[float] = Field(None, description="Combined final score")

    # Decision metadata
    mode: str = Field(..., description="Decision mode: 'exploration' or 'exploitation'")
    context_confidence: float = Field(
        default=0.0, description="Confidence based on context data"
    )
    exploration_weight: float = Field(
        default=0.0, description="Exploration weight applied"
    )

    # Additional metadata
    model_version: Optional[str] = Field(None, description="Model version used")
    candidates_evaluated: int = Field(..., description="Number of candidates generated")
    context_hash: str = Field(..., description="Hash of the input context")


class PerformanceRequest(BaseModel):
    """Request to log performance data."""

    asset_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Asset ID from recommendation response",
    )
    attention_score: float = Field(
        ..., ge=0.0, le=1.0, description="Attention performance (0-1)"
    )
    persuasion_score: float = Field(
        ..., ge=0.0, le=1.0, description="Persuasion performance (0-1)"
    )
    raw_metrics: Optional[Dict[str, Any]] = Field(
        None, description="Optional raw metrics data"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "asset_id": "asset_20240115_abc123",
                "attention_score": 0.75,
                "persuasion_score": 0.82,
                "raw_metrics": {
                    "view_rate": 0.65,
                    "avg_watch_time": 12.5,
                    "ctr": 0.032,
                },
            }
        }
    }


class PerformanceResponse(BaseModel):
    """Response after logging performance."""

    success: bool
    asset_id: str = Field(..., description="Asset ID that was updated")
    performance_id: str = Field(..., description="Unique performance log ID")
    message: str


class RetrainRequest(BaseModel):
    """Request to trigger model retraining."""

    force: bool = Field(
        default=False, description="Force retrain even with limited data"
    )


class RetrainResponse(BaseModel):
    """Response after retraining."""

    success: bool
    version: Optional[str] = None
    models: Optional[Dict[str, Any]] = None
    thresholds: Optional[Dict[str, Dict[str, float]]] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    models_loaded: bool
    model_version: Optional[str]
    database_connected: bool
